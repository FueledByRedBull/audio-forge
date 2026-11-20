"""What assistive technology and the keyboard get from the Qt Quick scene.

Real Qt, real QML scene and real models: Qt's accessibility layer is switched on
to read the names, roles and states a screen reader would, and keys go through
``QTest`` to the window like a user's. Only blocking dialogs are stood in for.
"""
# pyright: reportMissingImports=false
# (the sibling test modules are found by pytest at run time)

from __future__ import annotations

import sys

import pytest
from PySide6.QtCore import QElapsedTimer, QObject, QPointF, QRectF, Qt, QTimer, qInstallMessageHandler
from PySide6.QtGui import QAccessible
from PySide6.QtQml import QQmlEngine, QQmlExpression, QQmlProperty
from PySide6.QtTest import QTest

from mic_eq.config import DeviceIdentity
from mic_eq.ui.chain_presentation import PREFILTER_HIGHPASS_HZ
from mic_eq.ui.eq_state import EQ_FILTER_OPTIONS
from test_quick_layout import segoe_font  # noqa: F401  (a fixture)
from test_quick_shell import (
    _AccessibleScene,
    _graph_item,
    _tab_stops,
    spoken_announcements,
)
from test_quick_shell import quick_window as shell_window  # noqa: F401  (a fixture)

Role = QAccessible.Role
Text = QAccessible.Text
INTERACTIVE = {
    Role.Button, Role.CheckBox, Role.ComboBox, Role.Slider, Role.EditableText, Role.PageTab,
    Role.RadioButton, Role.SpinBox, Role.ScrollBar,
}
OVERVIEW, HEALTH, SETTINGS, EQUALIZER = 0, 1, 2, 3
# The sidebar rows and the Health chip are bare buttons, whatever base the scene gives them.
PLAIN_BUTTONS = ("KeyButton", "AbstractButton")


@pytest.fixture(autouse=True)
def no_qml_warnings():
    """Fail a test that makes the scene print a warning: a binding loop, a type error, a null read.

    ``quick_window`` only watches the scene while it loads; this watches the whole interaction.
    """
    messages: list[str] = []
    previous = qInstallMessageHandler(
        lambda _mode, _context, message: messages.append(message) if ".qml" in message else None
    )
    yield messages
    qInstallMessageHandler(previous)
    assert messages == [], "the scene printed warnings: " + "; ".join(messages[:3])


@pytest.fixture
def quick_window(shell_window):  # noqa: F811
    """The real main window with its Qt Quick scene, as test_quick_shell builds it.

    It lists a microphone, so Start is offered as it is to most people.
    """
    shell_window.input_choice.addItem("Microphone", DeviceIdentity(name="Microphone", direction="input"))
    return shell_window


def _kind(item) -> str:
    """The QML type of ``item``: ``Toggle`` for ``Toggle_QMLTYPE_12``.

    Never asks an item the app defines in Python (the EQ graph, the meters) for its
    ``metaObject()``: PySide hands back the class's own shared wrapper, which dies with
    that item and then breaks ``Class.staticMetaObject`` for every later test.
    """
    if type(item).__module__.startswith("mic_eq"):
        return type(item).__name__
    return item.metaObject().className().split("_")[0]


def _items(root):
    for child in root.childItems():
        yield child
        yield from _items(child)


def _of_kind(window, *kinds: str) -> list:
    return [item for item in _items(window.centralWidget().rootObject()) if _kind(item) in kinds]


def _with_proxy(window, kind: str, proxy):
    return next(
        item for item in _of_kind(window, kind) if item.property("proxy") is proxy
    )


def _iface(obj):
    return QAccessible.queryAccessibleInterface(obj)


def _named(window, name: str, *kinds: str):
    """The first item of one of ``kinds`` that assistive technology knows as ``name``."""
    return next(
        item for item in _of_kind(window, *kinds) if _iface(item).text(Text.Name) == name
    )


def _accessible_tree(window):
    """Every accessible interface under the window's scene, depth first."""
    pending = [_iface(window.centralWidget())]
    while pending:
        interface = pending.pop()
        yield interface
        pending.extend(
            child for child in (interface.child(i) for i in range(interface.childCount())) if child
        )


def _show(window, qapp, page: int = OVERVIEW, width: int = 1400) -> None:
    window.resize(width, 900)
    window.activateWindow()
    window.centralWidget().setFocus()
    window.shell_page.setIndex(page)
    qapp.processEvents()


def _inspect(window, qapp, key: str) -> None:
    """Open the inspector of the stage ``key`` and let it finish opening."""
    window.quick_bridge.model["inspector"].show(key)
    qapp.processEvents()
    QTest.qWait(400)


def _open_every_disclosure(window, qapp) -> None:
    for disclosure in _of_kind(window, "Disclosure"):
        if disclosure.isVisible():
            disclosure.setProperty("open", True)
    qapp.processEvents()


def _focus(window, item, qapp) -> None:
    """Focus ``item`` the way Tab does, so it shows its keyboard focus."""
    item.forceActiveFocus(Qt.FocusReason.TabFocusReason)
    qapp.processEvents()


def _press(window, key, qapp, modifiers=Qt.KeyboardModifier.NoModifier) -> None:
    QTest.keyClick(window.centralWidget(), key, modifiers)
    qapp.processEvents()


@pytest.fixture
def icon_window(request):
    """The window with the Windows icon font, so icon buttons replace the spelled-out ones."""
    request.getfixturevalue("segoe_font")
    window = request.getfixturevalue("quick_window")
    if not window.quick_bridge.model["theme"]["iconFont"]:
        pytest.skip("No Windows icon font in this environment")
    return window


@pytest.fixture(params=["text-buttons", "icon-buttons"])
def either_window(request):
    """Both looks of the buttons: undo, redo and refresh are words without the icon font."""
    return request.getfixturevalue("icon_window" if request.param == "icon-buttons" else "quick_window")


def _activate(window, item, key, qapp) -> None:
    _focus(window, item, qapp)
    _press(window, key, qapp)


# --- 1. Name, role and state for each control type ---------------------------------


def test_a_toggle_reports_checkbox_role_name_and_whether_it_is_on(quick_window, qapp) -> None:
    window = quick_window
    model = window.quick_bridge.model
    _show(window, qapp, EQUALIZER)
    # The EQ switches on and off from its row in the signal chain.
    eq_switch = _with_proxy(window, "ChainSwitch", model["eq"]["toggle"])
    mute = _with_proxy(window, "MuteBtn", model["top"]["mute"])
    assert model["eq"]["toggle"].checked and not model["top"]["mute"].checked
    with _AccessibleScene():
        # Mute is a button that stays pressed, so it is a checkable button rather than a box.
        for toggle, role, name, on in (
            (eq_switch, Role.CheckBox, "Enable EQ", True), (mute, Role.Button, "Mute output", False),
        ):
            interface = _iface(toggle)
            assert interface.role() == role
            assert interface.text(Text.Name) == name
            assert interface.state().checkable
            assert bool(interface.state().checked) is on, f"{name} starts {'on' if on else 'off'}"

        _activate(window, eq_switch, Qt.Key.Key_Space, qapp)
        _activate(window, mute, Qt.Key.Key_Space, qapp)
        assert not _iface(eq_switch).state().checked
        assert _iface(mute).state().checked
        assert window.engine_session.view.user_muted


def test_a_switch_that_cannot_be_used_is_reported_disabled(quick_window, qapp) -> None:
    window = quick_window
    _open_every_disclosure(window, qapp)
    with _AccessibleScene():
        auto = _named(window, "Auto threshold (gate)", "Toggle")
        assert _iface(auto).state().disabled
        assert not _iface(_named(window, "Enable noise gate", "Toggle")).state().disabled


def test_a_fader_reports_slider_role_name_value_and_range(quick_window, qapp) -> None:
    window = quick_window
    model = window.quick_bridge.model
    _show(window, qapp, EQUALIZER)
    gain = model["eq"]["bands"][0]["gain"]
    fader = _with_proxy(window, "Fader", gain)
    with _AccessibleScene():
        interface = _iface(fader)
        assert interface.role() == Role.Slider
        assert interface.text(Text.Name) == "EQ band 1 gain"
        value = interface.valueInterface()
        # The proxy counts tenths of a dB; the announced value is in dB.
        assert (value.minimumValue(), value.maximumValue()) == (-12.0, 12.0)
        assert value.currentValue() == gain.value / 10

        gain.setValue(25)
        qapp.processEvents()
        assert value.currentValue() == 2.5
        # A band that cannot take gain greys the fader out.
        window.eq_state.set_band_value(0, "filter_type", "high_pass")
        window.eq_state.flush()
        qapp.processEvents()
        assert _iface(fader).state().disabled


def _announced_range(window, fader) -> tuple[float, float, float]:
    """The range and value a screen reader gets from ``fader``, as plain numbers."""
    value = _iface(fader).valueInterface()
    return value.minimumValue(), value.maximumValue(), value.currentValue()


def test_a_value_box_reports_editable_text_with_its_name_and_shown_value(quick_window, qapp) -> None:
    window = quick_window
    model = window.quick_bridge.model
    _show(window, qapp)
    band = model["eq"]["bands"][0]
    boxes = {
        name: next(
            child
            for child in _with_proxy(window, "ValueBox", band[name]).childItems()
            if _kind(child) == "QQuickTextInput"
        )
        for name in ("frequency", "q", "gain")
    }
    with _AccessibleScene():
        frequency = _iface(boxes["frequency"])
        assert frequency.role() == Role.EditableText
        assert frequency.text(Text.Name) == "EQ band 1 center frequency"
        assert frequency.text(Text.Value) == band["frequency"].text == "80 Hz"
        assert frequency.state().editable and not frequency.state().readOnly
        assert _iface(boxes["q"]).text(Text.Name) == "EQ band 1 Q"

        # The gain readout beside the fader is shown, not typed into.
        readout = _iface(boxes["gain"])
        assert readout.state().readOnly
        assert readout.text(Text.Value) == band["gainLabel"].text

        band["frequency"].setValue(90.0)
        band["frequency"].released()
        qapp.processEvents()
        assert frequency.text(Text.Value) == "90 Hz"


def test_a_picker_reports_combo_box_role_name_and_what_is_chosen(quick_window, qapp) -> None:
    window = quick_window
    model = window.quick_bridge.model
    _show(window, qapp)
    mode = _with_proxy(window, "Picker", model["top"]["mode"])
    band_type = _with_proxy(window, "Picker", model["eq"]["bands"][0]["type"])
    with _AccessibleScene() as scene:
        for picker, label, shown in (
            (mode, "Processing mode", "Normal"),
            (band_type, "EQ band 1 filter type", "Low shelf"),
        ):
            assert _iface(picker).role() == Role.ComboBox
            # Screen readers speak a name change but often not a description change, and a
            # combo box has no value channel in QML: the choice goes in the name, after the label.
            assert scene.name(picker) == f"{label}: {shown}"
            assert shown in picker.property("displayText")

        model["top"]["mode"].setIndex(1)
        qapp.processEvents()
        assert scene.name(mode) == "Processing mode: Bypass"


def test_a_button_reports_button_role_its_visible_label_and_whether_it_is_enabled(
    quick_window, qapp
) -> None:
    window = quick_window
    model = window.quick_bridge.model
    _show(window, qapp)
    with _AccessibleScene():
        start = _with_proxy(window, "Btn", model["top"]["start"])
        assert _iface(start).role() == Role.Button
        assert start.property("text") in _iface(start).text(Text.Name)
        assert not _iface(start).state().disabled
        # Nothing to undo yet: the spelled-out Undo button is greyed out.
        undo = _with_proxy(window, "Btn", model["presets"]["undo"])
        assert _iface(undo).text(Text.Name) == "Undo"
        assert _iface(undo).state().disabled
        window.eq_state.set_band_value(0, "gain_db", 3.0)
        window.eq_state.flush()
        window.configuration.commit_edit()
        qapp.processEvents()
        assert not _iface(undo).state().disabled


def test_icon_buttons_are_named_by_what_they_do_not_by_their_tooltip_text(icon_window, qapp) -> None:
    window = icon_window
    _show(window, qapp, EQUALIZER)
    with _AccessibleScene():
        names = {
            _iface(button).text(Text.Name): button
            for button in _of_kind(window, "IconBtn")
            if button.isVisible()
        }
        assert {"Refresh audio devices", "Undo", "Redo", "Settings", "Previous EQ band", "Next EQ band"} <= set(names)
        for name, button in names.items():
            assert _iface(button).role() == Role.Button
            assert len(name) <= 60, f"{name[:40]}... reads as a paragraph, not a name"
        assert _iface(names["Undo"]).state().disabled


def _eq_help_action(window):
    return next(
        action for action in window.eq_presentation.options_menu.actions()
        if action.text() == "How the graph works"
    )


def test_the_equalizer_help_is_named_in_the_more_menu_and_opens_under_it(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp, EQUALIZER)
    model = window.quick_bridge.model
    help_text = model["eq"]["help"]
    more = _with_proxy(window, "Btn", model["eq"]["menu"])
    tip = window.centralWidget().rootObject().findChild(QObject, "tooltip")
    assert more.isVisible() and more.property("text") == "More"
    # The menu keeps every action the header had: the tone presets, the resets and the help.
    menu = window.eq_presentation.options_menu
    titles = [action.text() for action in menu.actions() if not action.isSeparator()]
    assert titles == ["Tone preset", "Clear Auto-EQ", "Reset tone", "How the graph works"]
    assert menu.actions()[0].menu() is window.eq_presentation.tone_menu
    action = _eq_help_action(window)
    assert 0 < len(action.toolTip()) <= 90 and action.toolTip() != help_text
    action.trigger()
    qapp.processEvents()
    assert tip.property("visible") is True and tip.property("text") == help_text
    assert tip.property("parent") == more


def test_chips_read_as_name_then_text_and_follow_the_text(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp, HEALTH)
    with _AccessibleScene() as scene:
        texts = [
            i for i in _items(window.centralWidget().rootObject())
            if _kind(i) == "QQuickText" and _iface(i) is not None
        ]
        names = [scene.name(t) for t in texts if scene.name(t)]
        assert "Noise suppression backend health: Backend: --" in names
        window._set_health_chip("backend_diag_label", "Backend: failing", "bad")
        qapp.processEvents()
        assert "Noise suppression backend health: Backend: failing" in [scene.name(t) for t in texts]


def test_a_disclosure_says_whether_its_section_is_shown(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    _inspect(window, qapp, "compressor")
    disclosure = _named(window, "Show Compressor advanced", "Disclosure")
    advanced = window.quick_bridge.model["stages"][3]["advanced"]
    attack = advanced[0]["proxy"]
    with _AccessibleScene() as scene:
        assert _iface(disclosure).role() == Role.Button
        assert not [i for i in _of_kind(window, "Fader", "ValueBox") if i.property("proxy") is attack]

        _focus(window, disclosure, qapp)
        _press(window, Qt.Key.Key_Space, qapp)

        assert scene.name(disclosure) == "Hide Compressor advanced"
        assert [i for i in _of_kind(window, "ValueBox") if i.property("proxy") is attack]
        _press(window, Qt.Key.Key_Space, qapp)
        assert scene.name(disclosure) == "Show Compressor advanced"
        assert not [i for i in _of_kind(window, "ValueBox") if i.property("proxy") is attack]


def test_the_sidebar_says_which_page_is_open(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    with _AccessibleScene():
        pages = {
            "Overview": _named(window, "Overview", *PLAIN_BUTTONS),
            "Equalizer": _named(window, "Equalizer", *PLAIN_BUTTONS),
            "Settings": _named(window, "Settings", "IconBtn"),
        }
        for button in pages.values():
            assert _iface(button).role() == Role.Button
            assert _iface(button).state().selectable

        def selected() -> list[str]:
            return [name for name, button in pages.items() if _iface(button).state().selected]

        assert selected() == ["Overview"]
        # The Health page is open from the Health button, which says so itself.
        window.shell_page.setIndex(HEALTH)
        qapp.processEvents()
        assert selected() == []
        window.shell_page.setIndex(SETTINGS)
        qapp.processEvents()
        assert selected() == ["Settings"]
        window.shell_page.setIndex(EQUALIZER)
        qapp.processEvents()
        assert selected() == ["Equalizer"]


def test_the_health_chip_exposes_whether_the_health_page_is_open(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    with _AccessibleScene() as scene:
        chip = window.centralWidget().rootObject().findChild(QObject, "healthToggle")
        assert not _iface(chip).state().checked
        assert scene.description(chip) == "Show audio health details"
        window.shell_page.setIndex(HEALTH)
        qapp.processEvents()
        assert _iface(chip).state().checked
        # A click now goes back, so the description says that.
        assert scene.description(chip) == "Hide audio health details"


def test_the_status_strip_texts_are_read_by_assistive_technology_without_being_tab_stops(
    quick_window, qapp
) -> None:
    window = quick_window
    _show(window, qapp)
    root = window.centralWidget().rootObject()
    overview = window.quick_bridge.model["overview"]
    shown = {
        "statusMessage": lambda: window.status_message.text,
        "overviewHeadline": lambda: overview.headline,
        "overviewDetail": lambda: overview.detail,
        "statusCalibration": lambda: window.shell_texts["calibration_status_label"].text,
    }
    with _AccessibleScene() as scene:
        items = {name: root.findChild(QObject, name) for name in shown}
        for name, item in items.items():
            assert _iface(item) is not None, f"{name} is not exposed"
            assert _iface(item).role() == Role.StaticText
            assert scene.name(item) == shown[name]() != ""
        window.status_message.showMessage("Output muted")
        # The microphone goes away from the list, so the Overview says so.
        window.input_choice.clear()
        qapp.processEvents()
        assert scene.name(items["statusMessage"]) == "Output muted"
        assert scene.name(items["overviewHeadline"]) == "No microphone found"

        stopped_on = {item.objectName() for item in _tab_stops(window, qapp)}
        assert not stopped_on & set(shown)


def test_settings_rows_are_switches_or_buttons_and_switches_say_if_they_are_on(
    quick_window, qapp
) -> None:
    window = quick_window
    _show(window, qapp, SETTINGS)
    with _AccessibleScene():
        rows = {_iface(row).text(Text.Name): row for row in _of_kind(window, "SettingRow")}
        switch = rows["Include Measured Route Delay in Latency Estimate"]
        action = rows["Export Diagnostics"]
        assert _iface(switch).role() == Role.CheckBox
        assert _iface(switch).state().checkable
        assert _iface(action).role() == Role.Button and not _iface(action).state().checkable

        proxy = next(
            row["proxy"] for card in window.quick_bridge.model["settings"]["cards"]
            for row in card["rows"] if row["proxy"].text.startswith("Include Measured")
        )
        before = _iface(switch).state().checked
        assert bool(before) is proxy.checked
        _focus(window, switch, qapp)
        _press(window, Qt.Key.Key_Space, qapp)
        assert proxy.checked is not bool(before)
        assert bool(_iface(switch).state().checked) is proxy.checked

        # A switch row is one control: the little switch inside it is not announced again.
        assert _iface(switch).childCount() == 0


# --- 2. Audit: nothing interactive is anonymous ----------------------------------


def _anonymous(window, qapp) -> list[str]:
    found = []
    for page in (OVERVIEW, HEALTH, SETTINGS, EQUALIZER):
        _show(window, qapp, page)
        _open_every_disclosure(window, qapp)
        for interface in _accessible_tree(window):
            if interface.role() not in INTERACTIVE or interface.state().invisible:
                continue
            if not interface.text(Text.Name).strip():
                obj = interface.object()
                found.append(f"page {page}: {interface.role().name} {_kind(obj)}")
    return found


def test_no_interactive_item_in_the_scene_is_without_a_name(either_window, qapp) -> None:
    with _AccessibleScene():
        assert _anonymous(either_window, qapp) == []


def test_names_are_short_labels_and_none_starts_with_a_colon(icon_window, qapp) -> None:
    window = icon_window
    with _AccessibleScene():
        too_long, colon = [], []
        for page in (OVERVIEW, HEALTH, SETTINGS, EQUALIZER):
            _show(window, qapp, page)
            _open_every_disclosure(window, qapp)
            for interface in _accessible_tree(window):
                name = interface.text(Text.Name)
                if name.startswith(":"):
                    colon.append(name)
                if interface.role() in INTERACTIVE and len(name) > 80:
                    too_long.append(name[:60])
        assert colon == []
        assert too_long == []


# --- 4. The EQ graph --------------------------------------------------------------


def test_the_eq_graph_is_named_described_and_follows_the_selected_band(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp, EQUALIZER)
    graph = _graph_item(window.centralWidget().rootObject())
    with _AccessibleScene() as scene:
        interface = _iface(graph)
        assert scene.name(graph) == "Equalizer graph"
        assert interface.role() == Role.Graphic
        assert interface.state().focusable
        description = scene.description(graph)
        for key_hint in ("Left and right change frequency", "left bracket", "Delete"):
            assert key_hint in description
        assert "selected:" not in description

        window.eq_presentation.graph_model.select_band(2)
        qapp.processEvents()
        assert "Band 3 selected: Bell, 320 Hz" in scene.description(graph)
        window.eq_presentation.graph_model.select_band(6)
        qapp.processEvents()
        assert "Band 7 selected" in scene.description(graph)
        assert "Band 3 selected" not in scene.description(graph)


def test_the_eq_graph_says_the_input_analyser_shows_only_while_it_does(quick_window, qapp) -> None:
    import numpy as np

    from mic_eq.ui.eq_graph import ANALYSER_DESCRIPTION

    window = quick_window
    _show(window, qapp, EQUALIZER)
    graph = _graph_item(window.centralWidget().rootObject())
    model = window.eq_presentation.graph_model
    with _AccessibleScene() as scene:
        assert ANALYSER_DESCRIPTION not in scene.description(graph)
        model.set_input_spectrum(
            np.full(2, -40.0, dtype=np.float32), np.array([100.0, 1000.0], dtype=np.float32)
        )
        qapp.processEvents()
        assert ANALYSER_DESCRIPTION in scene.description(graph)
        model.clear_input_spectrum()
        qapp.processEvents()
        assert ANALYSER_DESCRIPTION not in scene.description(graph)


# --- 3. Keyboard -------------------------------------------------------------------


def _tab_cycle(window, qapp, page: int = OVERVIEW, width: int = 1400) -> list:
    """The items real Tab presses focus in turn, until the cycle comes round again."""
    _show(window, qapp, page, width)
    view = window.centralWidget()
    stops = []
    for _ in range(400):
        QTest.keyClick(view, Qt.Key.Key_Tab)
        item = view.quickWindow().activeFocusItem()
        if item in stops:
            assert item is stops[0], "Tab fell into a loop that skips the start of the page"
            return stops
        stops.append(item)
    pytest.fail("Tab never comes round to where it began")


def _tabbable(window, kinds: tuple[str, ...]) -> list:
    return [
        item for item in _of_kind(window, *kinds)
        if item.isVisible() and item.isEnabled()
        and not (_kind(item) == "QQuickTextInput" and item.property("readOnly"))
    ]


KEYBOARD_KINDS = (
    "Toggle", "ChainSwitch", "Fader", "Btn", "IconBtn", "MuteBtn", "LinkBtn", "Picker", "Disclosure",
    "SettingRow", "HealthRow", *PLAIN_BUTTONS, "QQuickTextInput", "EQGraphItem",
)


@pytest.mark.parametrize(
    "page", [OVERVIEW, HEALTH, SETTINGS, EQUALIZER], ids=["overview", "health", "settings", "equalizer"]
)
def test_tab_reaches_every_enabled_control_on_the_page(either_window, qapp, page) -> None:
    window = either_window
    _show(window, qapp, page)
    _open_every_disclosure(window, qapp)
    stops = _tab_cycle(window, qapp, page)
    with _AccessibleScene() as scene:
        missed = [
            f"{_kind(item)} {scene.name(item)!r}"
            for item in _tabbable(window, KEYBOARD_KINDS)
            # A switch row owns the click, so its switch is no stop of its own.
            if item not in stops
            and not (_kind(item) == "Toggle" and _kind(item.parentItem().parentItem()) == "SettingRow")
            # A radio group is one stop, its current choice; the arrow keys reach the others.
            and not (_iface(item).role() == Role.RadioButton and item.property("current") is False)
        ]
    assert missed == []


# The toolbar, then the sidebar from the preset to the Settings gear, then the page.
FRAME_LANDMARKS = [
    "Input audio device", "Output audio device", "Processing mode", "Mute output", "Start Processing",
    "Presets", "Overview", "Input cleanup", "Noise gate", "Equalizer", "Limiter", "Health", "Measure latency",
    "Settings",
]


def _page_names(window, qapp, page: int, width: int, frame: list[str]) -> list[str]:
    """The names Tab visits after the sidebar, checking the frame before them is in order."""
    stops = _tab_cycle(window, qapp, page, width)
    with _AccessibleScene() as scene:
        names = [scene.name(item) for item in stops]
    assert len(set(map(id, stops))) == len(stops)
    positions = _positions(names, frame)
    assert positions == sorted(positions), dict(zip(frame, positions))
    return names[positions[-1] + 1:]


def _positions(names: list[str], landmarks: list[str]) -> list[int]:
    # A picker's name carries its choice after the label, the Health button its reading.
    return [
        next(i for i, name in enumerate(names) if name == landmark or name.startswith(landmark + ": "))
        for landmark in landmarks
    ]


# 1200 is the left edge of `wide`; 1400 and 1800 are wider still.
@pytest.mark.parametrize("width", [1200, 1400, 1800])
def test_tab_visits_the_page_in_reading_order(either_window, qapp, width) -> None:
    _inspect(either_window, qapp, "suppression")
    names = _page_names(either_window, qapp, OVERVIEW, width, FRAME_LANDMARKS)
    landmarks = ["Enable noise suppression", "Noise suppression backend", "Noise suppression strength"]
    positions = _positions(names, landmarks)
    assert positions == sorted(positions), dict(zip(landmarks, positions))
    # Each stage's own switch comes before its controls. Tab starts again from the top.
    _inspect(either_window, qapp, "gate")
    either_window.centralWidget().rootObject().forceActiveFocus()
    names = _page_names(either_window, qapp, OVERVIEW, width, FRAME_LANDMARKS)
    assert names.index("Enable noise gate") < names.index("Gate manual threshold")
    assert names.index("Gate manual threshold") < names.index("Show Noise gate advanced")


@pytest.mark.parametrize("width", [1200, 1400, 1800])
def test_tab_visits_the_equalizer_page_in_reading_order(either_window, qapp, width) -> None:
    names = _page_names(either_window, qapp, EQUALIZER, width, FRAME_LANDMARKS)
    landmarks = [
        "Total", "Equalizer options", "Equalizer graph",
        "Enable EQ band 1", "EQ band 1 filter type", "EQ band 1 gain", "EQ band 1 center frequency",
    ]
    positions = _positions(names, landmarks)
    assert positions == sorted(positions), dict(zip(landmarks, positions))


ACTIVATION_KEYS = [
    pytest.param(Qt.Key.Key_Space, id="space"),
    pytest.param(Qt.Key.Key_Return, id="return"),
    pytest.param(Qt.Key.Key_Enter, id="enter"),
]


@pytest.mark.parametrize("key", ACTIVATION_KEYS)
def test_the_keyboard_activates_a_toggle(quick_window, qapp, key) -> None:
    window = quick_window
    _show(window, qapp, EQUALIZER)
    band = window.quick_bridge.model["eq"]["bands"][0]["enabled"]
    before = window.eq_state.get_band(0).enabled
    _activate(window, _with_proxy(window, "Toggle", band), key, qapp)
    assert window.eq_state.get_band(0).enabled is not before


@pytest.mark.parametrize("key", ACTIVATION_KEYS)
def test_the_keyboard_activates_a_button(quick_window, qapp, monkeypatch, key) -> None:
    window = quick_window
    exported: list[bool] = []
    monkeypatch.setattr(window, "_export_diagnostics", lambda: exported.append(True))
    _show(window, qapp, HEALTH)
    _open_every_disclosure(window, qapp)
    button = next(b for b in _of_kind(window, "Btn") if b.property("text") == "Export diagnostics")
    _activate(window, button, key, qapp)
    assert exported == [True]


@pytest.mark.parametrize("key", ACTIVATION_KEYS)
def test_the_keyboard_opens_a_disclosure(quick_window, qapp, key) -> None:
    window = quick_window
    _show(window, qapp)
    _inspect(window, qapp, "limiter")
    with _AccessibleScene():
        disclosure = _named(window, "Show Limiter advanced", "Disclosure")
    _activate(window, disclosure, key, qapp)
    assert disclosure.property("open") is True


@pytest.mark.parametrize("key", ACTIVATION_KEYS)
def test_the_keyboard_opens_a_page_from_the_sidebar(quick_window, qapp, key) -> None:
    window = quick_window
    _show(window, qapp)
    for name, kinds, page in (
        ("Settings", ("IconBtn",), SETTINGS), ("Equalizer", PLAIN_BUTTONS, EQUALIZER),
        ("Overview", PLAIN_BUTTONS, OVERVIEW),
    ):
        with _AccessibleScene():
            button = _named(window, name, *kinds)
            _activate(window, button, key, qapp)
        assert window.shell_page.currentIndex() == page, name


def _in_view(item, page) -> bool:
    top = item.mapToScene(QPointF(0, 0)).y()
    page_top = page.mapToScene(QPointF(0, 0)).y()
    return page_top <= top and top + item.height() <= page_top + page.height()


def test_each_signal_chain_row_carries_its_stage_switch_under_the_same_name(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    rows = window.quick_bridge.model["sidebar"]["chain"]
    eq_toggle = window.quick_bridge.model["eq"]["toggle"]
    # Input cleanup is a setting with several modes, not a stage that switches on and off.
    assert [row["name"] for row in rows if row["toggle"] is None] == ["Input cleanup"]
    with _AccessibleScene() as scene:
        for row in rows:
            if row["toggle"] is None:
                continue
            switch = _with_proxy(window, "ChainSwitch", row["toggle"])
            assert switch.isVisible(), row["name"]
            assert _iface(switch).role() == Role.CheckBox
            assert bool(_iface(switch).state().checked) is row["toggle"].checked
            assert scene.description(switch).startswith("In the signal chain")
            if row["toggle"] is eq_toggle:
                # The equalizer page has no switch of its own: this row is the EQ's one switch.
                assert not [i for i in _of_kind(window, "Toggle") if i.property("proxy") is eq_toggle]
                assert _iface(switch).text(Text.Name) == "Enable EQ"
                assert _iface(switch).state().checkable and row["toggle"].checked
                continue
            # A screen reader hears the same name here as on the stage's own card.
            card_switch = _with_proxy(window, "Toggle", row["toggle"])
            assert _iface(switch).text(Text.Name) == _iface(card_switch).text(Text.Name) != ""
            # The description tells the two apart.
            assert scene.description(card_switch) != scene.description(switch)


@pytest.mark.parametrize("width", [900, 1400])
def test_every_stage_switch_is_at_least_a_24_px_target(quick_window, qapp, width) -> None:
    window = quick_window
    _show(window, qapp, width=width)
    rows = [row for row in window.quick_bridge.model["sidebar"]["chain"] if row["toggle"]]
    sizes = {}
    for row in rows:
        sizes[f"sidebar {row['name']}"] = _with_proxy(window, "ChainSwitch", row["toggle"])
        if row["card"] is not None:
            _inspect(window, qapp, row["key"])
            sizes[f"inspector {row['name']}"] = _with_proxy(window, "Toggle", row["toggle"])
    assert len(sizes) == 2 * len(rows) - 1  # the Equalizer's own switch is on its page
    small = {name: (item.width(), item.height()) for name, item in sizes.items()
             if item.width() < 24 or item.height() < 24}
    assert small == {}


def test_a_signal_chain_row_opens_its_page_or_its_stage_inspector(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp, EQUALIZER, width=900)
    window.resize(900, 640)
    QTest.qWait(50)
    rows = {row["name"]: row for row in window.quick_bridge.model["sidebar"]["chain"]}
    inspector = window.quick_bridge.model["inspector"]
    shown = window.centralWidget().rootObject().findChild(QObject, "inspector")

    def open_row(name: str) -> None:
        with _AccessibleScene():
            _activate(window, _named(window, name, *PLAIN_BUTTONS), Qt.Key.Key_Space, qapp)
        QTest.qWait(400)

    open_row("Input cleanup")
    assert window.shell_page.currentIndex() == OVERVIEW
    assert (inspector.shown, inspector.stageKey) == (True, "cleanup")
    open_row("Equalizer")
    assert window.shell_page.currentIndex() == EQUALIZER
    assert not inspector.shown
    limiter = _with_proxy(window, "Toggle", rows["Limiter"]["toggle"])
    open_row("Limiter")
    assert window.shell_page.currentIndex() == OVERVIEW
    assert (inspector.shown, inspector.stageKey) == (True, "limiter")
    assert limiter.isVisible() and _in_view(limiter, shown)
    gate = _with_proxy(window, "Toggle", rows["Noise gate"]["toggle"])
    open_row("Noise gate")
    assert inspector.stageKey == "gate"
    assert gate.isVisible() and _in_view(gate, shown) and not limiter.isVisible()


@pytest.mark.parametrize("key", ACTIVATION_KEYS)
def test_the_keyboard_activates_a_settings_row(quick_window, qapp, key) -> None:
    window = quick_window
    _show(window, qapp, SETTINGS)
    proxy = next(
        row["proxy"] for card in window.quick_bridge.model["settings"]["cards"]
        for row in card["rows"] if row["proxy"].text.startswith("Include Measured")
    )
    row = next(r for r in _of_kind(window, "SettingRow") if r.property("p") is proxy)
    before = proxy.checked
    _activate(window, row, key, qapp)
    assert proxy.checked is not before


def test_a_held_key_activates_a_button_once(quick_window, qapp) -> None:
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtGui import QKeyEvent

    window = quick_window
    _show(window, qapp, EQUALIZER)
    band = window.quick_bridge.model["eq"]["bands"][0]["enabled"]
    toggle = _with_proxy(window, "Toggle", band)
    _focus(window, toggle, qapp)
    start = window.eq_state.get_band(0).enabled
    view = window.centralWidget()
    for repeat in (False, True, True, True):
        QCoreApplication.sendEvent(
            view,
            QKeyEvent(
                QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier,
                "\r", repeat, 1,
            ),
        )
    qapp.processEvents()
    assert window.eq_state.get_band(0).enabled is not start


def test_a_picker_chooses_by_keyboard_with_the_arrows_or_the_open_list(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp, EQUALIZER)
    proxy = window.quick_bridge.model["eq"]["bands"][0]["type"]
    picker = _with_proxy(window, "Picker", proxy)
    _focus(window, picker, qapp)

    start = proxy.index
    kinds = [kind for _name, kind in EQ_FILTER_OPTIONS]
    _press(window, Qt.Key.Key_Down, qapp)
    assert window.eq_state.get_band(0).filter_type == kinds[start + 1]
    assert picker.property("displayText") == proxy.items[start + 1]
    with _AccessibleScene() as scene:  # what a screen reader hears for the new choice
        assert scene.name(picker) == f"EQ band 1 filter type: {proxy.items[start + 1]}"
    _press(window, Qt.Key.Key_Up, qapp)
    assert window.eq_state.get_band(0).filter_type == kinds[start]

    # Space opens the list, the arrows move through it and Return chooses.
    _press(window, Qt.Key.Key_Space, qapp)
    assert QQmlProperty.read(picker, "popup.visible")
    _press(window, Qt.Key.Key_Down, qapp)
    _press(window, Qt.Key.Key_Down, qapp)
    assert window.eq_state.get_band(0).filter_type == kinds[start], "moving through the list chooses nothing yet"
    _press(window, Qt.Key.Key_Return, qapp)
    assert not QQmlProperty.read(picker, "popup.visible")
    assert window.eq_state.get_band(0).filter_type == kinds[start + 2]
    assert picker.property("displayText") == proxy.items[start + 2]


def test_a_fader_moves_one_step_per_arrow_key(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp, EQUALIZER)
    gain = window.quick_bridge.model["eq"]["bands"][0]["gain"]
    fader = _with_proxy(window, "Fader", gain)
    _focus(window, fader, qapp)
    assert gain.step == 1.0

    for key, expected in (
        (Qt.Key.Key_Right, 0.1), (Qt.Key.Key_Right, 0.2), (Qt.Key.Key_Left, 0.1),
        (Qt.Key.Key_Up, 0.2), (Qt.Key.Key_Down, 0.1),
    ):
        _press(window, key, qapp)
        assert window.eq_state.get_band(0).gain_db == pytest.approx(expected), key


def test_a_fader_without_a_step_moves_a_hundredth_of_its_range_per_key(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp, EQUALIZER)
    gain = window.quick_bridge.model["eq"]["bands"][0]["gain"]
    gain._view["step"] = 0.0  # a proxy that names no step; the gain proxy counts tenths of a dB
    gain.changed.emit()
    fader = _with_proxy(window, "Fader", gain)
    _focus(window, fader, qapp)

    _press(window, Qt.Key.Key_Right, qapp)

    # 1/100 of the 24 dB range is 0.24 dB, which the proxy rounds to its 0.1 dB grid.
    assert window.eq_state.get_band(0).gain_db == pytest.approx(0.2)


def test_a_value_box_steps_with_up_and_down(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp, EQUALIZER)
    band = window.quick_bridge.model["eq"]["bands"][0]
    for name, step, read in (
        ("frequency", 10, lambda: window.eq_state.get_band(0).frequency_hz),
        ("q", 0.1, lambda: window.eq_state.get_band(0).q),
    ):
        box = _with_proxy(window, "ValueBox", band[name])
        field = next(c for c in box.childItems() if _kind(c) == "QQuickTextInput")
        _focus(window, field, qapp)
        start = round(read(), 1)
        _press(window, Qt.Key.Key_Up, qapp)
        assert read() == pytest.approx(start + step), name
        _press(window, Qt.Key.Key_Down, qapp)
        _press(window, Qt.Key.Key_Down, qapp)
        assert read() == pytest.approx(start - step), name


# The EQ graph's own keys.


@pytest.fixture
def graph(quick_window, qapp):
    window = quick_window
    _show(window, qapp, EQUALIZER)
    item = _graph_item(window.centralWidget().rootObject())
    window.eq_presentation.graph_model.select_band(4)
    _focus(window, item, qapp)
    return window, window.eq_state


def test_graph_brackets_select_the_previous_and_next_band_and_wrap(graph, qapp) -> None:
    window, state = graph
    assert state.selected_band == 4
    _press(window, Qt.Key.Key_BracketRight, qapp)
    assert state.selected_band == 5
    _press(window, Qt.Key.Key_BracketLeft, qapp)
    _press(window, Qt.Key.Key_BracketLeft, qapp)
    assert state.selected_band == 3
    for _ in range(3):
        _press(window, Qt.Key.Key_BracketLeft, qapp)
    assert state.selected_band == 0
    _press(window, Qt.Key.Key_BracketLeft, qapp)
    assert state.selected_band == 9
    _press(window, Qt.Key.Key_BracketRight, qapp)
    assert state.selected_band == 0
    assert window.quick_bridge.model["eq"]["band"].index == 0


def test_graph_arrows_move_the_band_finely_and_shift_moves_it_coarsely(graph, qapp) -> None:
    window, state = graph
    start = state.get_band(4)

    _press(window, Qt.Key.Key_Right, qapp)
    fine = state.get_band(4).frequency_hz
    assert start.frequency_hz < fine < start.frequency_hz * 2 ** (1 / 24)
    _press(window, Qt.Key.Key_Right, qapp, Qt.KeyboardModifier.ShiftModifier)
    coarse = state.get_band(4).frequency_hz
    assert coarse > fine * 2 ** (1 / 24)
    _press(window, Qt.Key.Key_Left, qapp, Qt.KeyboardModifier.ShiftModifier)
    _press(window, Qt.Key.Key_Left, qapp)
    assert state.get_band(4).frequency_hz == pytest.approx(start.frequency_hz, abs=1)

    _press(window, Qt.Key.Key_Up, qapp)
    assert state.get_band(4).gain_db == pytest.approx(0.1)
    _press(window, Qt.Key.Key_Up, qapp, Qt.KeyboardModifier.ShiftModifier)
    assert state.get_band(4).gain_db == pytest.approx(1.1)
    _press(window, Qt.Key.Key_Down, qapp, Qt.KeyboardModifier.ShiftModifier)
    _press(window, Qt.Key.Key_Down, qapp)
    assert state.get_band(4).gain_db == pytest.approx(0.0)


@pytest.mark.parametrize("key", [Qt.Key.Key_Delete, Qt.Key.Key_Backspace], ids=["delete", "backspace"])
def test_graph_delete_and_backspace_reset_the_selected_band(graph, qapp, key) -> None:
    window, state = graph
    default = state.get_band(4)
    for _ in range(3):
        _press(window, Qt.Key.Key_Up, qapp, Qt.KeyboardModifier.ShiftModifier)
    _press(window, Qt.Key.Key_Right, qapp, Qt.KeyboardModifier.ShiftModifier)
    assert state.get_band(4) != default

    _press(window, key, qapp)

    assert state.get_band(4) == default


def test_graph_keys_leave_other_bands_alone_and_pass_unrelated_keys_on(graph, qapp) -> None:
    window, state = graph
    others = [state.get_band(i) for i in range(10) if i != 4]
    _press(window, Qt.Key.Key_Up, qapp)
    _press(window, Qt.Key.Key_Right, qapp)
    assert [state.get_band(i) for i in range(10) if i != 4] == others
    # Tab is not the graph's: it moves on.
    _press(window, Qt.Key.Key_Tab, qapp)
    assert window.centralWidget().quickWindow().activeFocusItem() is not _graph_item(
        window.centralWidget().rootObject()
    )


# --- 6. Focus is visible -------------------------------------------------------------


def _rings(window, control) -> list:
    # The graph is no Control: its ring has none and sits directly on it.
    return [
        ring for ring in _of_kind(window, "FocusRing")
        if (QQmlProperty.read(ring, "control") or ring.parentItem()) == control and ring.isVisible()
    ]


def _shows_focus(window, item) -> bool:
    """Whether the focused ``item`` wears the one focus ring and nothing hover also draws."""
    kind = _kind(item)
    # A value field is wrapped by its box, which wears the ring.
    ringed = item.parentItem() if kind == "QQuickTextInput" else item
    if len(_rings(window, ringed)) != 1:
        return False
    if kind in PLAIN_BUTTONS and not item.property("current"):  # a navigation tab
        QTest.qWait(200)  # its colour fades in
        return item.property("background").property("color").alpha() == 0
    return True


@pytest.mark.parametrize(
    "page", [OVERVIEW, HEALTH, SETTINGS, EQUALIZER], ids=["overview", "health", "settings", "equalizer"]
)
def test_the_focused_control_always_shows_where_focus_is(either_window, qapp, page) -> None:
    window = either_window
    _show(window, qapp, page)
    if page == OVERVIEW:
        _inspect(window, qapp, "gate")
    view = window.centralWidget()
    stops = _tab_cycle(window, qapp, page)
    # Walk the cycle again, judging each stop while it holds focus.
    unmarked = []
    seen = set()
    for _ in stops:
        QTest.keyClick(view, Qt.Key.Key_Tab)
        qapp.processEvents()
        item = view.quickWindow().activeFocusItem()
        seen.add(_kind(item))
        if not _shows_focus(window, item):
            unmarked.append(f"{_kind(item)} {item.objectName()!r}")
        # One ring at a time, wherever focus is.
        assert len([ring for ring in _of_kind(window, "FocusRing") if ring.isVisible()]) == 1
    assert unmarked == []
    expected = {
        OVERVIEW: {
            "Toggle", "ChainSwitch", "Fader", "Picker", "Btn", "MuteBtn", "LinkBtn", "Disclosure", "CloseCross",
        },
        HEALTH: {"HealthRow", "Disclosure"},
        SETTINGS: {"SettingRow", "Picker"},
        EQUALIZER: {"Toggle", "Fader", "Picker", "QQuickTextInput", "EQGraphItem", "KeyButton", "Btn"},
    }
    assert expected[page] <= seen


def test_a_card_title_is_a_heading(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp, HEALTH)
    titles = [t for t in _of_kind(window, "Title") if t.isVisible()]
    assert {t.property("text") for t in titles} >= {"Signal", "Stream"}
    with _AccessibleScene():
        assert all(_iface(t).role() == Role.Heading for t in titles)


def _in_page(item) -> bool:
    while item is not None and _kind(item) != "Page":
        item = item.parentItem()
    return item is not None


def _scrolling_page(item):
    """The scrolling ``Page`` that holds ``item``."""
    while _kind(item) != "Page":
        item = item.parentItem()
    return item


def test_a_health_row_is_one_tab_stop_that_reads_whole_and_shows_its_help_on_focus(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp, HEALTH)
    tip = window.centralWidget().rootObject().findChild(QObject, "tooltip")
    rows = [row for row in _of_kind(window, "HealthRow") if row.isVisible()]
    assert len(rows) >= 6
    stops = _tab_cycle(window, qapp, HEALTH)
    with _AccessibleScene() as scene:
        for row in rows:
            proxy = row.property("proxy")
            assert row in stops
            assert _iface(row).role() == Role.StaticText
            assert scene.name(row) == f"{proxy.name}: {row.property('shownValue')}"
            assert scene.description(row) == proxy.toolTip != ""
            # Its parts do not read again on their own.
            assert [t for t in _items(row) if _kind(t) == "QQuickText" and _iface(t) is not None
                    and scene.name(t)] == []
            _focus(window, row, qapp)
            QTest.qWait(700)  # the tip's delay
            assert tip.property("visible") is True and tip.property("text") == proxy.toolTip
            assert len(_rings(window, row)) == 1


def test_a_caption_with_help_is_a_tab_stop_that_shows_it_on_focus(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp, HEALTH)
    tip = window.centralWidget().rootObject().findChild(QObject, "tooltip")
    # The label of a settings row is no stop: its control shows the same help.
    labels = [c for c in _of_kind(window, "Caption") if c.property("tip") and c.isVisible()]
    assert labels == [] or all(c.property("focusable") is False for c in labels)
    caption = next(c for c in _items(_of_kind(window, "HealthRow")[0]) if _kind(c) == "Caption")
    assert not caption.property("tip")
    assert caption.property("activeFocusOnTab") is False
    caption.setProperty("tip", "Where this comes from")
    assert caption.property("focusable") is True and caption.property("activeFocusOnTab") is True
    _focus(window, caption, qapp)
    QTest.qWait(700)
    assert tip.property("visible") is True and tip.property("text") == "Where this comes from"
    assert caption.property("ringed") is True
    _focus(window, _of_kind(window, "HealthRow")[0], qapp)
    assert caption.property("ringed") is False


def test_page_up_page_down_and_the_arrows_scroll_a_page_and_focus_scrolls_to_itself(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp, HEALTH)
    window.resize(1400, 420)
    qapp.processEvents()
    rows = [row for row in _of_kind(window, "HealthRow") if row.isVisible()]
    page = _scrolling_page(rows[0])
    assert page.property("contentHeight") > page.height()
    _focus(window, rows[0], qapp)
    assert page.property("contentY") == 0
    _press(window, Qt.Key.Key_Down, qapp)
    line = page.property("lineStep")
    assert page.property("contentY") == line
    _press(window, Qt.Key.Key_Up, qapp)
    assert page.property("contentY") == 0
    _press(window, Qt.Key.Key_PageDown, qapp)
    assert page.property("contentY") == min(page.height(), page.property("contentHeight") - page.height())
    _press(window, Qt.Key.Key_PageUp, qapp)
    assert page.property("contentY") == 0
    _press(window, Qt.Key.Key_PageUp, qapp)
    assert page.property("contentY") == 0, "never past the top"
    # Home and End go to the ends, as in a dialog's page.
    _press(window, Qt.Key.Key_End, qapp)
    assert page.property("contentY") == page.property("contentHeight") - page.height()
    _press(window, Qt.Key.Key_End, qapp)
    assert page.property("contentY") == page.property("contentHeight") - page.height(), "never past the end"
    _press(window, Qt.Key.Key_Home, qapp)
    assert page.property("contentY") == 0
    # A modified key is a shortcut for something else.
    _press(window, Qt.Key.Key_Down, qapp, Qt.KeyboardModifier.ControlModifier)
    _press(window, Qt.Key.Key_End, qapp, Qt.KeyboardModifier.AltModifier)
    assert page.property("contentY") == 0
    # Focus on a control that is out of view brings it in.
    _open_every_disclosure(window, qapp)
    last = next(b for b in _of_kind(window, "Btn") if b.property("text") == "Reset drop counter")
    top = last.mapToItem(page.property("contentItem"), 0, 0).y()
    assert top + last.height() > page.height(), "the button starts out of view"
    _focus(window, last, qapp)
    assert page.property("contentY") > 0
    assert page.property("contentY") <= top and top + last.height() <= page.property("contentY") + page.height()


def test_the_arrows_keep_moving_the_cursor_in_a_text_field_and_page_keys_still_scroll(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp, EQUALIZER)
    window.resize(1400, 420)
    qapp.processEvents()
    field = next(
        i for i in _of_kind(window, "QQuickTextInput")
        if i.isVisible() and not i.property("readOnly") and _in_page(i)
    )
    page = _scrolling_page(field)
    assert page.property("contentHeight") > page.height()
    _focus(window, field, qapp)
    before = page.property("contentY")
    _press(window, Qt.Key.Key_Down, qapp)
    assert page.property("contentY") == before


def test_focus_rings_follow_keyboard_focus_not_the_mouse(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    top = window.quick_bridge.model["top"]
    toggle = _with_proxy(window, "MuteBtn", top["mute"])
    _click(window, toggle, qapp)
    assert window.engine_session.view.user_muted
    assert _rings(window, toggle) == [], "a click must not draw the keyboard focus ring"
    _focus(window, _with_proxy(window, "Btn", top["start"]), qapp)  # focus leaves the toggle...
    _focus(window, toggle, qapp)  # ...and Tab brings it back
    assert len(_rings(window, toggle)) == 1


# --- 7. Layout, labels, text size and help ------------------------------------------


def _scene_x(item) -> float:
    return item.mapToScene(item.boundingRect().topLeft()).x()


def _scene_y(item) -> float:
    return item.mapToScene(item.boundingRect().topLeft()).y()


def test_every_visible_label_is_part_of_the_control_name() -> None:
    from mic_eq.ui.control_specs import CONTROL_SPECS

    missing = [
        f"{section}.{key}: {spec.label!r} is not in {spec.accessible_name!r}"
        for section, specs in CONTROL_SPECS.items()
        for key, spec in specs.items()
        if spec.label and spec.label.lower() not in spec.accessible_name.lower()
    ]
    assert missing == []


def test_no_text_in_the_scene_is_a_literal_pixel_size_or_smaller_than_twelve() -> None:
    import re
    from pathlib import Path

    for path in (Path(__file__).parents[1] / "mic_eq" / "ui" / "qml").glob("*.qml"):
        # A literal, or a named size minus something (which can fall under the floor).
        assert re.findall(r"pixelSize:\s*(?:\d|[\w.]*fs\w+\s*-)", path.read_text(encoding="utf-8")) == [], path.name


def test_text_sizes_follow_the_app_font_and_none_is_below_twelve(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    root = window.centralWidget().rootObject()
    sizes = [root.property(name) for name in ("fsCaption", "fsBody", "fsLead", "fsHeading", "fsTitle")]
    assert sizes == sorted(sizes) and sizes[0] >= 12
    assert sizes[1] == round(13 * qapp.font().pointSizeF() / 9)


def test_the_route_stays_readable_and_shares_one_row_with_the_transport_in_a_narrow_window(
    quick_window, qapp
) -> None:
    window = quick_window
    top = window.quick_bridge.model["top"]
    pickers = [_with_proxy(window, "Picker", top[name]) for name in ("input", "output")]
    start = _with_proxy(window, "Btn", top["start"])
    for width in (1400, 900):
        window.resize(width, 800)
        qapp.processEvents()
        centre = _scene_y(start) + start.height() / 2
        assert abs(centre - (_scene_y(pickers[0]) + pickers[0].height() / 2)) < 2, width
        assert all(picker.width() >= 140 for picker in pickers), width
        captions = {
            item.property("text") for item in _of_kind(window, "Caption") if item.isVisible()
        }
        assert {"IN", "OUT"} <= captions, width


def test_an_opened_advanced_section_stays_open_when_the_window_crosses_the_wide_breakpoint(
    quick_window, qapp
) -> None:
    window = quick_window
    window.resize(1400, 900)
    qapp.processEvents()
    _inspect(window, qapp, "compressor")
    opened = [d for d in _of_kind(window, "Disclosure") if d.property("subject") == "Compressor"]
    opened[0].setProperty("open", True)
    qapp.processEvents()
    window.resize(1000, 900)
    qapp.processEvents()
    again = [d for d in _of_kind(window, "Disclosure") if d.property("subject") == "Compressor"]
    assert again and again[0].property("open") is True


def test_a_stage_that_is_off_says_so_and_keeps_its_controls_at_full_strength(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    _inspect(window, qapp, "gate")
    model = window.quick_bridge.model
    gate = model["stages"][1]
    assert gate["toggle"].checked
    row_button = _named(window, "Noise gate", *PLAIN_BUTTONS)

    def says() -> tuple[list[str], str]:
        status = [
            c.property("text") for c in _of_kind(window, "StatusText")
            if c.property("text").startswith("Noise gate is") and c.isVisible()
        ]
        return status, _iface(row_button).text(Text.Description)

    with _AccessibleScene():
        assert says() == (["Noise gate is on"], "-40 dB")
        gate["toggle"].click()
        qapp.processEvents()
        assert says() == (["Noise gate is off"], "Off")
    threshold = gate["rows"][0]["proxy"]
    row = next(
        r for r in _of_kind(window, "ControlRow") if r.property("p") is threshold
    )
    assert row.opacity() == 1 and all(c.opacity() == 1 for c in row.childItems())


@pytest.fixture
def no_icon_font_window(request, monkeypatch):
    monkeypatch.setattr("mic_eq.ui.quick_scene.icon_font", lambda: None)
    return request.getfixturevalue("quick_window")


def test_icon_buttons_are_still_there_without_the_icon_font(no_icon_font_window, qapp) -> None:
    window = no_icon_font_window
    _show(window, qapp, EQUALIZER)
    assert window.quick_bridge.model["theme"]["iconFont"] == ""
    faces = {
        button.property("label"): button.isVisible()
        for button in _of_kind(window, "IconBtn")
        if button.property("label")
    }
    assert faces and all(faces.values())
    assert {"Refresh", "Settings", "Undo", "Prev", "Next"} <= set(faces)


def test_a_warning_banner_sits_under_the_top_bar_and_is_an_alert(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    window.shell_texts["config_warning_banner"].update(text="Settings were reset", shown=True)
    QTest.qWait(50)
    start = _with_proxy(window, "Btn", window.quick_bridge.model["top"]["start"])
    before = _scene_y(start)
    banner = next(
        item for item in _of_kind(window, "QQuickText")
        if item.property("text") == "Settings were reset"
    )
    assert _scene_y(banner) > before, "the transport row does not move down"
    with _AccessibleScene():
        assert _iface(banner).role() == Role.AlertMessage


def test_a_warning_banner_is_announced_when_it_appears_and_not_repeated(quick_window, qapp, monkeypatch) -> None:
    window = quick_window
    _show(window, qapp)
    spoken = spoken_announcements(monkeypatch)
    banner = window.shell_texts["config_warning_banner"]
    banner.update(text="Settings were reset", shown=True)
    banner.update(text="Settings were reset", shown=True)
    assert spoken == ["Configuration warning: Settings were reset"]
    banner.update(shown=False)
    assert len(spoken) == 1
    banner.update(shown=True)
    assert len(spoken) == 2
    window.shell_texts["device_warning_banner"].update(shown=True)
    assert spoken[2].startswith("Audio device warning: Warning: No audio devices")
    # An ordinary label never speaks on its own.
    window.shell_texts["preset_status_label"].update(text="Preset: Other (modified)")
    assert len(spoken) == 3


def test_the_navigation_is_a_named_group_and_the_meters_have_roles(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    with _AccessibleScene():
        interfaces = list(_accessible_tree(window))
        roles = [interface.role() for interface in interfaces]
        assert (Role.Grouping, "Navigation") in [(i.role(), i.text(Text.Name)) for i in interfaces]
        assert Role.ProgressBar in roles


def test_a_help_tip_opened_by_the_keyboard_goes_when_focus_moves_on(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp, EQUALIZER)
    root = window.centralWidget().rootObject()
    tip = root.findChild(QObject, "tooltip")
    help_text = window.quick_bridge.model["eq"]["help"]
    button = _with_proxy(window, "Btn", window.quick_bridge.model["eq"]["menu"])
    _focus(window, button, qapp)
    # The menu's help entry, chosen from the keyboard, opens the help under the button.
    _eq_help_action(window).trigger()
    qapp.processEvents()
    assert tip.property("visible") is True and tip.property("text") == help_text
    _press(window, Qt.Key.Key_Escape, qapp)
    assert tip.property("visible") is False
    _eq_help_action(window).trigger()
    qapp.processEvents()
    assert tip.property("visible") is True
    _press(window, Qt.Key.Key_Tab, qapp)
    # The next control may show its own help; the help that was asked for goes.
    assert not (tip.property("visible") and tip.property("text") == help_text)


@pytest.fixture
def reduced_motion_window(request, monkeypatch):
    monkeypatch.setenv("AUDIOFORGE_REDUCED_MOTION", "1")
    return request.getfixturevalue("quick_window")


@pytest.fixture
def full_motion_window(request, monkeypatch):
    # Windows can report its animations off on its own; the slide must not depend on that.
    monkeypatch.setenv("AUDIOFORGE_REDUCED_MOTION", "0")
    return request.getfixturevalue("quick_window")


def test_reduced_motion_keeps_the_level_trace_running(reduced_motion_window) -> None:
    bridge = reduced_motion_window.quick_bridge
    assert bridge.model["theme"]["reducedMotion"] is True
    assert bridge._trace_timer.isActive()


@pytest.mark.parametrize("motion", ["reduced", "full"])
def test_the_ribbon_takes_each_reading_and_slides_only_with_full_motion(request, qapp, motion) -> None:
    window = request.getfixturevalue(f"{motion}_motion_window")
    _show(window, qapp)
    bridge = window.quick_bridge
    ribbon = window.centralWidget().rootObject().findChild(QObject, "levelRibbon")
    strip = ribbon.parentItem()
    # Readings come by hand here; the sampler would clear them, as no meter is measuring.
    bridge._trace_timer.stop()
    bridge._levels.extend([[-20.0, -24.0]] * 5)
    bridge.levelsChanged.emit()
    QTest.qWait(25)
    assert len(ribbon.property("points")) == 5
    # Reduced motion shows the new reading at once; full motion slides it in over one reading.
    scroll = window.centralWidget().rootObject().findChild(QObject, "ribbonScroll")
    interval = bridge.model["levelIntervalMs"]
    assert scroll.property("duration") == (0 if motion == "reduced" else interval)
    if motion == "reduced":
        assert strip.x() == pytest.approx(-strip.property("step"))


def test_the_ribbon_slides_only_while_the_overview_shows(full_motion_window, qapp) -> None:
    window = full_motion_window
    _show(window, qapp, HEALTH)
    bridge = window.quick_bridge
    bridge._trace_timer.stop()
    scroll = window.centralWidget().rootObject().findChild(QObject, "ribbonScroll")
    # A slide started while the page showed may still be waiting for a frame; only new ones count.
    scroll.setProperty("running", False)
    bridge._levels.extend([[-20.0, -24.0]] * 3)
    bridge.levelsChanged.emit()
    assert scroll.property("running") is False
    window.shell_page.setIndex(OVERVIEW)
    qapp.processEvents()
    bridge.levelsChanged.emit()
    assert scroll.property("running") is True


def test_a_switch_knob_snaps_under_reduced_motion(reduced_motion_window, qapp) -> None:
    window = reduced_motion_window
    _show(window, qapp)
    toggle = next(row["toggle"] for row in window.quick_bridge.model["sidebar"]["chain"] if row["toggle"])
    switch = _with_proxy(window, "ChainSwitch", toggle)
    track = switch.property("contentItem").childItems()[0]
    knob = next(child for child in track.childItems() if child.objectName() == "switchKnob")
    was_on = switch.property("on")
    toggle.click()
    qapp.processEvents()
    assert switch.property("on") is not was_on
    # No frame has run since the change, so only a knob that does not move is already there.
    assert knob.x() == (3 if was_on else track.width() - knob.width() - 3)


# The tight cells are the minimum, the left edge of `wide` at 1200, and 1400.
# The widths are calibrated on Windows; Linux's offscreen platform lays text out in
# the system fonts, which run wider.
@pytest.mark.skipif(sys.platform != "win32", reason="layout widths calibrated on Windows fonts")
@pytest.mark.parametrize("width", [900, 1200, 1400])
@pytest.mark.parametrize("kind", ["low_shelf", "high_pass", "notch"])
def test_the_gain_fader_keeps_room_beside_the_band_fields(quick_window, qapp, width, kind) -> None:
    window = quick_window
    window.shell_page.setIndex(EQUALIZER)
    window.eq_state.set_band_value(0, "filter_type", kind)
    window.eq_state.flush()
    window.resize(width, 900)
    QTest.qWait(50)
    gain = window.quick_bridge.model["eq"]["bands"][0]["gain"]
    fader = _with_proxy(window, "Fader", gain)
    assert fader.width() >= 100


def _eq_item(window, name: str):
    return next(
        item for item in _items(window.centralWidget().rootObject()) if item.objectName() == name
    )


def test_the_legend_names_each_line_style_in_words(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp, EQUALIZER)
    QTest.qWait(50)
    entries = [
        item for item in _items(window.centralWidget().rootObject()) if item.objectName() == "legendEntry"
    ]
    styles = {entry.property("lineStyle"): entry.property("text") for entry in entries}
    assert set(styles) == {"solid", "dashed", "dotted", "dot"}
    assert all(text.strip() for text in styles.values())
    assert styles["dotted"] == f"High-pass {PREFILTER_HIGHPASS_HZ} Hz"
    assert all(entry.isVisible() for entry in entries)


@pytest.mark.parametrize("width", [900, 1400])
def test_the_band_panel_follows_the_selected_band(quick_window, qapp, width) -> None:
    window = quick_window
    _show(window, qapp, EQUALIZER, width)
    QTest.qWait(50)
    page, graph, connector, title = (
        _eq_item(window, name) for name in ("equalizerPage", "eqGraph", "bandConnector", "bandTitle")
    )
    for step in range(3):
        if step:
            window.quick_bridge.stepBand(1)
            QTest.qWait(50)
        band = page.property("bandIndex")
        assert band == step
        assert title.property("text").startswith(f"Band {band + 1} · ")
        handle = graph.mapToScene(QPointF(graph.handleX(band), graph.handleBottom(band)))
        assert abs(_scene_x(connector) - handle.x()) <= 1
        assert abs(_scene_y(connector) - handle.y()) <= 1
    assert connector.height() > 0


@pytest.mark.parametrize("width", [900, 1400])
def test_the_band_panel_docks_under_the_graph(quick_window, qapp, width) -> None:
    window = quick_window
    _show(window, qapp, EQUALIZER, width)
    QTest.qWait(50)
    page, well, graph, panel, footer = (
        _eq_item(window, name) for name in ("equalizerPage", "eqGraphWell", "eqGraph", "bandPanel", "eqFooter")
    )
    assert _scene_y(panel) >= _scene_y(well) + well.height()
    # The graph takes the height the page leaves it, never less than its own, and the page does not scroll.
    assert graph.height() >= graph.implicitHeight() == 320
    assert page.property("contentHeight") <= page.height()
    # The legend and the calibration line sit under the band strip, at the foot of the graph card.
    assert _scene_y(footer) >= _scene_y(panel) + panel.height()


def test_the_band_strip_keeps_its_fields_apart_at_every_width(quick_window, qapp) -> None:
    window = quick_window
    for width in (900, 1400):
        _show(window, qapp, EQUALIZER, width)
        QTest.qWait(50)
        editor = _eq_item(window, "bandEditorRow")
        one_row = editor.property("oneRow")
        rows = [row for row in editor.childItems() if row.isVisible()]
        assert len(rows) == 2
        first, second = (row.mapRectToScene(row.boundingRect()) for row in rows)
        assert not first.intersects(second), width
        if one_row:
            assert abs(first.center().y() - second.center().y()) < 2
        else:
            assert second.top() >= first.bottom()
        for row in rows:
            fields = [item for item in row.childItems() if item.isVisible()]
            for left, right in zip(fields, fields[1:]):
                assert _scene_x(left) + left.width() <= _scene_x(right), (width, left, right)
            panel = _eq_item(window, "bandPanel")
            assert _scene_x(fields[-1]) + fields[-1].width() <= _scene_x(panel) + panel.width()


def test_the_band_strip_names_the_frequency_once(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp, EQUALIZER)
    QTest.qWait(50)
    band = window.quick_bridge.model["eq"]["bands"][0]
    label = band["frequencyLabel"].text
    assert label
    panel = _eq_item(window, "bandPanel")
    # The frequency field holds the value; no caption repeats it beside the title.
    repeated = [
        item for item in _items(panel)
        if item.isVisible() and _kind(item) != "QQuickTextInput" and item.property("text") == label
    ]
    assert repeated == []


def test_the_gain_field_names_its_unit_outside_the_value(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp, EQUALIZER)
    box = _eq_item(window, "gainBox")
    field = next(item for item in _items(box) if _kind(item) == "QQuickTextInput")
    unit = next(item for item in _items(box) if _kind(item) == "QQuickText")
    assert unit.isVisible() and unit.property("text") == "dB"
    assert "dB" not in field.property("text")
    assert field.x() + field.width() <= unit.x()


def test_the_calibration_mark_is_drawn_not_typed(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp, EQUALIZER)
    mark = _eq_item(window, "diagnosticsMark")
    # A glyph can fall back to a box in a font without it; shapes cannot.
    for item in (mark, *_items(mark)):
        assert _kind(item) != "QQuickText", item
        assert "✓" not in str(item.property("text") or "")
    check = next(item for item in _items(mark) if item.objectName() == "checkShape")
    for state, shown in (("ok", True), ("warn", False), ("bad", False), ("idle", False)):
        mark.setProperty("state_", state)
        qapp.processEvents()
        assert check.property("visible") is shown or not mark.isVisible(), state


def test_show_picks_which_layers_the_graph_draws(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp, EQUALIZER)
    QTest.qWait(50)
    graph = _eq_item(window, "eqGraph")
    preview = next(item for item in _of_kind(window, "EQGraphPreview") if item is not graph)
    assert graph.property("show") == "total"
    with _AccessibleScene():
        for key in ("tone", "auto_eq", "total"):
            choice = _eq_item(window, "eqShow_" + key)
            assert _iface(choice).role() == Role.RadioButton
            _click(window, choice, qapp)
            assert graph.property("show") == key
            for other in ("total", "tone", "auto_eq"):
                assert bool(_iface(_eq_item(window, "eqShow_" + other)).state().checked) is (other == key)
        _activate(window, _eq_item(window, "eqShow_tone"), Qt.Key.Key_Space, qapp)
        assert graph.property("show") == "tone"
    # The overview's preview always shows what you hear.
    assert preview.property("show") == "total"


def test_the_show_choices_are_one_tab_stop_that_the_arrow_keys_move(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp, EQUALIZER)
    QTest.qWait(50)
    graph = _eq_item(window, "eqGraph")
    choices = {key: _eq_item(window, "eqShow_" + key) for key in ("total", "tone", "auto_eq")}
    _focus(window, choices["total"], qapp)
    # Right moves to the next choice and wraps; Left goes back. The chosen one alone takes Tab.
    for key, expected in (
        (Qt.Key.Key_Right, "tone"), (Qt.Key.Key_Right, "auto_eq"),
        (Qt.Key.Key_Right, "total"), (Qt.Key.Key_Left, "auto_eq"),
    ):
        _press(window, key, qapp)
        assert graph.property("show") == expected
        assert _focused(window) is choices[expected]
        assert [k for k, choice in choices.items() if choice.property("activeFocusOnTab")] == [expected]
    with _AccessibleScene() as scene:
        for key, name in (("auto_eq", "Auto-EQ"), ("tone", "Tone"), ("total", "Total")):
            _click(window, choices[key], qapp)
            # A screen reader hears which view the graph shows.
            assert scene.description(graph).startswith(f"Showing {name}. "), key


LEGENDS = {
    "total": ["Total (what you hear)", "Auto-EQ", f"High-pass {PREFILTER_HIGHPASS_HZ} Hz", "Bands"],
    "tone": ["Tone (your bands)", "Bands"],
    "auto_eq": ["Auto-EQ", "Auto-EQ bands"],
}


@pytest.mark.parametrize("key", ["total", "tone", "auto_eq"])
def test_the_legend_lists_only_what_the_view_draws(quick_window, qapp, key) -> None:
    window = quick_window
    _show(window, qapp, EQUALIZER)
    _click(window, _eq_item(window, "eqShow_" + key), qapp)
    QTest.qWait(50)
    entries = sorted(
        (item for item in _items(window.centralWidget().rootObject())
         if item.objectName() == "legendEntry" and item.isVisible()),
        key=_scene_x,
    )
    assert [entry.property("text") for entry in entries] == LEGENDS[key]
    # The Auto-EQ view draws no band handles, so nothing runs from one to the band strip.
    assert _eq_item(window, "bandConnector").isVisible() is (key != "auto_eq")


@pytest.mark.parametrize("key", ["total", "tone", "auto_eq"])
def test_the_legend_names_the_input_analyser_first_while_the_stream_runs(quick_window, qapp, key) -> None:
    window = quick_window
    for timer in window.findChildren(QTimer):
        timer.stop()
    _show(window, qapp, EQUALIZER)
    _click(window, _eq_item(window, "eqShow_" + key), qapp)

    def legend() -> list:
        QTest.qWait(50)
        return sorted(
            (item for item in _items(window.centralWidget().rootObject())
             if item.objectName() == "legendEntry" and item.isVisible()),
            key=_scene_x,
        )

    assert [entry.property("text") for entry in legend()] == LEGENDS[key]
    window.stop_btn_action.setEnabled(True)  # what a running stream does
    running = legend()
    assert [entry.property("text") for entry in running] == ["Input analyser", *LEGENDS[key]]
    assert running[0].property("lineStyle") == "fill"
    window.stop_btn_action.setEnabled(False)
    assert [entry.property("text") for entry in legend()] == LEGENDS[key]


def test_the_chosen_view_looks_apart_from_the_others(themed_window, qapp) -> None:
    window = themed_window
    _show(window, qapp, EQUALIZER)
    _click(window, _eq_item(window, "eqShow_tone"), qapp)
    QTest.qWait(50)

    def look(key):
        choice = _eq_item(window, "eqShow_" + key)
        return _colour(choice.property("background"), "color"), _colour(choice.property("contentItem"), "color")

    chosen = look("tone")
    for other in ("total", "auto_eq"):
        fill, text = look(other)
        # The fill and the words both change, under a contrast theme as well.
        assert fill != chosen[0] and text != chosen[1], other


@pytest.mark.parametrize("width", [900, 1199, 1200, 1400])
def test_the_levels_and_the_latency_keep_stop_away_from_mute(quick_window, qapp, width) -> None:
    window = quick_window
    window.resize(width, 900)
    QTest.qWait(50)
    top = window.quick_bridge.model["top"]
    mute = _with_proxy(window, "MuteBtn", top["mute"])
    mode = _with_proxy(window, "Picker", top["mode"])
    start = _with_proxy(window, "Btn", top["start"])

    assert _scene_x(mode) + mode.width() < _scene_x(mute) < _scene_x(start)
    assert _scene_x(start) - (_scene_x(mute) + mute.width()) >= mode.width()


def test_q_and_slope_show_only_for_the_filters_that_use_them(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp, EQUALIZER)
    bands = window.quick_bridge.model["eq"]["bands"]
    for kind, q_shown, slope_shown in (("bell", True, False), ("notch", True, False), ("high_pass", False, True)):
        window.eq_state.set_band_value(0, "filter_type", kind)
        window.eq_state.flush()
        qapp.processEvents()
        assert (bands[0]["q"].shown, bands[0]["slope"].shown) == (q_shown, slope_shown), kind
        assert _with_proxy(window, "ValueBox", bands[0]["q"]).isVisible() is q_shown
        assert _with_proxy(window, "Picker", bands[0]["slope"]).isVisible() is slope_shown


def test_the_meters_tell_assistive_technology_their_reading(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    model = window.quick_bridge.model
    reduction = model["stages"][3]["rows"][2]["proxy"]
    side = _with_source(window, window.input_meter)
    card = _with_source(window, reduction)
    with _AccessibleScene() as scene:
        assert scene.description(side).startswith("no signal")
        window.input_meter.set_levels(-20.4, -12.0)
        reduction.set_gain_reduction(3.2)
        qapp.processEvents()
        assert scene.description(side).startswith("-20 dB")
        assert scene.description(card) == "3 dB"
        window.input_meter.set_muted(True)
        qapp.processEvents()
        assert scene.description(side).startswith("-20 dB. Muted")


def _with_source(window, source):
    return next(
        item for item in _of_kind(window, "QuickMeterItem") if item.property("source") is source
    )


def test_refresh_gives_its_reason_only_while_it_is_disabled(quick_window, qapp) -> None:
    window = quick_window
    refresh = window.quick_bridge.model["top"]["refresh"]
    window.refresh_btn_action.setEnabled(True)
    assert refresh.toolTip == "Refresh audio devices"
    window.refresh_btn_action.setEnabled(False)
    assert "not available while processing" in refresh.toolTip


# --- 8. Every way of switching acts once and the switch follows the model ---------


def _perform(item, action: str) -> None:
    from PySide6.QtGui import QAccessibleActionInterface

    name = getattr(QAccessibleActionInterface, action)()
    interface = _iface(item).actionInterface()
    assert name in interface.actionNames(), (action, interface.actionNames())
    interface.doAction(name)


KEYS = {str(param.id): param.values[0] for param in ACTIVATION_KEYS}


def _click(window, item, qapp) -> None:
    centre = item.mapToScene(item.boundingRect().center()).toPoint()
    QTest.mouseClick(window.centralWidget(), Qt.MouseButton.LeftButton, pos=centre)
    qapp.processEvents()


@pytest.mark.parametrize(
    "how", ["toggleAction", "pressAction", "click", "space", "return", "enter"]
)
def test_a_switch_acts_once_and_shows_the_model_whichever_way_it_is_used(
    quick_window, qapp, how
) -> None:
    window = quick_window
    _show(window, qapp)
    mute = _with_proxy(window, "MuteBtn", window.quick_bridge.model["top"]["mute"])
    with _AccessibleScene():
        for expected in (True, False):
            if how == "click":
                _click(window, mute, qapp)
            elif how in KEYS:
                _activate(window, mute, KEYS[how], qapp)
            else:
                _perform(mute, how)
            qapp.processEvents()
            assert window.engine_session.view.user_muted is expected, f"{how} changes the state once"
            assert bool(_iface(mute).state().checked) is expected
            assert mute.property("checked") is expected


def test_a_switch_row_in_settings_acts_once_whichever_way_it_is_used(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp, SETTINGS)
    proxy = next(
        row["proxy"] for card in window.quick_bridge.model["settings"]["cards"]
        for row in card["rows"] if row["proxy"].text.startswith("Include Measured")
    )
    row = next(r for r in _of_kind(window, "SettingRow") if r.property("p") is proxy)
    with _AccessibleScene():
        for how in ("toggleAction", "pressAction", "toggleAction", "pressAction"):
            before = proxy.checked
            _perform(row, how)
            qapp.processEvents()
            assert proxy.checked is not before, how
            assert bool(_iface(row).state().checked) is proxy.checked


def test_a_refused_switch_goes_back_to_the_models_state(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp, EQUALIZER)
    toggle = _with_proxy(window, "ChainSwitch", window.quick_bridge.model["eq"]["toggle"])
    on = toggle.property("on")
    # The model does not change: the switch must not keep the flipped state.
    toggle.property("proxy").click = lambda: None
    with _AccessibleScene():
        _perform(toggle, "toggleAction")
        qapp.processEvents()
        assert toggle.property("checked") is on


def test_a_disabled_switch_refuses_assistive_technology_and_snaps_back(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    mute = _with_proxy(window, "MuteBtn", window.quick_bridge.model["top"]["mute"])
    clicks: list[bool] = []
    mute.property("proxy").click = lambda: clicks.append(True)
    window.user_mute_checkbox_action.setEnabled(False)
    qapp.processEvents()
    assert mute.property("enabled") is False
    with _AccessibleScene():
        _perform(mute, "toggleAction")
        qapp.processEvents()
    assert clicks == [] and mute.property("checked") is False


def test_the_health_chip_follows_the_page_when_toggled_by_assistive_technology(
    quick_window, qapp
) -> None:
    window = quick_window
    _show(window, qapp)
    chip = window.centralWidget().rootObject().findChild(QObject, "healthToggle")
    with _AccessibleScene():
        _perform(chip, "pressAction")
        qapp.processEvents()
        assert window.shell_page.currentIndex() == HEALTH
        assert bool(_iface(chip).state().checked)
        _perform(chip, "pressAction")
        qapp.processEvents()
        assert window.shell_page.currentIndex() == OVERVIEW
        assert not _iface(chip).state().checked


def test_an_action_row_in_settings_acts_once_on_the_press_action(quick_window, qapp, monkeypatch) -> None:
    window = quick_window
    exported: list[bool] = []
    monkeypatch.setattr(window, "_export_diagnostics", lambda: exported.append(True))
    _show(window, qapp, SETTINGS)
    with _AccessibleScene():
        row = _named(window, "Export Diagnostics", "SettingRow")
        _perform(row, "pressAction")
        qapp.processEvents()
    assert exported == [True]


# --- 9. Mute, mode, help, edges, card order, plain text, the bridge, sliders ------------


def _colour(item, name: str) -> str:
    return QQmlProperty.read(item, name).name().lower()


def _text_format(item) -> int:
    """The item's ``textFormat``, which PySide cannot hand over as a property."""
    expression = QQmlExpression(QQmlEngine.contextForObject(item), item, "textFormat")
    value, _undefined = expression.evaluate()
    assert not expression.hasError()
    return value


def test_mute_is_said_in_words_and_the_window_has_its_own_shortcut(quick_window, qapp) -> None:
    from PySide6.QtGui import QFont

    window = quick_window
    _show(window, qapp)
    mute = _with_proxy(window, "MuteBtn", window.quick_bridge.model["top"]["mute"])
    label = mute.property("contentItem")
    assert "(Ctrl+M)" in window.quick_bridge.model["top"]["mute"].toolTip
    # Holding the keys must not flap the live mute.
    assert window.user_mute_checkbox_action.autoRepeat() is False
    assert mute.property("on") is False

    _press(window, Qt.Key.Key_M, qapp, Qt.KeyboardModifier.ControlModifier)
    assert window.engine_session.view.user_muted
    # Pressed shows in the weight of the word as well as the fill, and the word stays the same.
    assert mute.property("on") is True
    assert label.property("text") == "Mute"
    assert label.property("font").weight() == QFont.Weight.DemiBold
    with _AccessibleScene():
        assert _iface(mute).state().checked

    _press(window, Qt.Key.Key_M, qapp, Qt.KeyboardModifier.ControlModifier)
    assert not window.engine_session.view.user_muted
    assert mute.property("on") is False
    assert label.property("font").weight() == QFont.Weight.Normal


def test_a_mode_that_changes_the_sound_stands_out_in_its_picker(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    mode = _with_proxy(window, "Picker", window.quick_bridge.model["top"]["mode"])
    for index, alert in ((0, False), (1, True), (2, True), (0, False)):
        window.processing_mode_choice.setIndex(index)
        qapp.processEvents()
        assert mode.property("alert") is alert, index


def test_help_is_in_the_accessible_description_not_only_in_the_hover_tip(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    wrong, checked = [], 0
    with _AccessibleScene() as scene:
        # Each stage's controls show only in its inspector.
        for key in ("cleanup", "gate", "suppression", "deesser", "compressor", "limiter"):
            _inspect(window, qapp, key)
            _open_every_disclosure(window, qapp)
            for item in _of_kind(window, "Toggle", "Fader", "Picker", "Btn"):
                tip = getattr(item.property("proxy"), "toolTip", "")
                if item.isVisible() and tip:
                    checked += 1
                    if scene.description(item) != tip:
                        wrong.append(f"{_kind(item)} {scene.name(item)!r}")
        band = window.quick_bridge.model["eq"]["bands"][0]["frequency"]
        box = next(
            child for child in _with_proxy(window, "ValueBox", band).childItems()
            if _kind(child) == "QQuickTextInput"
        )
        assert band.toolTip and scene.description(box) == band.toolTip
        chip = window.quick_bridge.model["health"]["signal"][0]["proxy"]
        window.shell_page.setIndex(HEALTH)
        qapp.processEvents()
        row = _with_proxy(window, "HealthRow", chip)
        assert scene.name(row).startswith(chip.name)
        assert chip.toolTip and scene.description(row) == chip.toolTip
        window.shell_page.setIndex(SETTINGS)
        qapp.processEvents()
        helped = [
            (row, row.property("p")) for row in _of_kind(window, "SettingRow")
            if row.isVisible() and row.property("p").toolTip not in ("", row.property("p").text)
        ]
        assert helped and [r for r, p in helped if scene.description(r) != p.toolTip] == []
        # Buttons with no proxy say their own help: the tip and the description are one string.
        window.shell_page.setIndex(HEALTH)
        _open_every_disclosure(window, qapp)
        for text in ("Export diagnostics", "Reset drop counter"):
            button = next(b for b in _of_kind(window, "Btn") if b.property("text") == text)
            assert button.property("help") and scene.description(button) == button.property("help"), text
    assert checked >= 15 and wrong == []


def test_keyboard_focus_shows_the_help_of_a_control_and_escape_closes_it(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    tip = window.centralWidget().rootObject().findChild(QObject, "tooltip")
    mode = window.quick_bridge.model["top"]["mode"]
    picker = _with_proxy(window, "Picker", mode)
    _focus(window, picker, qapp)
    QTest.qWait(700)  # the tip's delay
    assert tip.property("visible") is True and tip.property("text") == mode.toolTip != ""
    _press(window, Qt.Key.Key_Escape, qapp)
    assert tip.property("visible") is False
    # Focus given by a click is not a request for help.
    _click(window, _with_proxy(window, "MuteBtn", window.quick_bridge.model["top"]["mute"]), qapp)
    QTest.qWait(700)
    assert tip.property("visible") is False


def test_control_edges_and_tracks_draw_from_the_control_edge_token(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    _inspect(window, qapp, "compressor")
    _open_every_disclosure(window, qapp)
    model = window.quick_bridge.model
    edge = model["theme"]["control_edge"].lower()
    # A switch that is off is an outline with its knob in the edge colour.
    off = next(
        toggle for toggle in _of_kind(window, "Toggle")
        if toggle.isVisible() and toggle.property("live") and not toggle.property("on")
    )
    track = off.property("contentItem").childItems()[0]
    assert _colour(track, "border.color") == edge
    assert _colour(next(c for c in track.childItems() if c.objectName() == "switchKnob"), "color") == edge
    mute = _with_proxy(window, "MuteBtn", model["top"]["mute"])
    assert _colour(mute.property("background"), "border.color") == edge
    band = model["eq"]["bands"][0]
    assert _colour(_with_proxy(window, "Fader", band["gain"]).property("background"), "color") == edge
    assert _colour(_with_proxy(window, "ValueBox", band["frequency"]), "border.color") == edge
    picker = _with_proxy(window, "Picker", model["top"]["mode"])
    assert _colour(picker.property("background"), "border.color") == edge
    presets = _with_proxy(window, "Btn", model["presets"]["menu"])
    assert _colour(presets.property("background"), "border.color") == edge
    disclosure = next(d for d in _of_kind(window, "Disclosure") if d.isVisible())
    assert _colour(disclosure.property("background"), "border.color") == edge


@pytest.mark.parametrize("width", [900, 1200, 1400, 1800])
def test_the_stage_inspectors_are_opened_and_stacked_in_signal_order(quick_window, qapp, width) -> None:
    window = quick_window
    _show(window, qapp, width=width)
    root = window.centralWidget().rootObject()
    stack = root.findChild(QObject, "inspectorStack")
    inspector = window.quick_bridge.model["inspector"]
    chain = [row for row in window.quick_bridge.model["sidebar"]["chain"] if row["card"]]
    panels = [item for item in stack.childItems() if _kind(item) == "InspectorPanel"]
    assert [panel.property("card")["key"] for panel in panels] == [row["key"] for row in chain] == [
        "cleanup", "gate", "suppression", "deesser", "compressor", "limiter",
    ]
    view = window.centralWidget()
    with _AccessibleScene() as scene:
        for row in chain:
            _activate(window, _named(window, row["name"], *PLAIN_BUTTONS), Qt.Key.Key_Return, qapp)
            QTest.qWait(400)
            assert inspector.stageKey == row["key"]
            assert [panel.isVisible() for panel in panels] == [p is panels[chain.index(row)] for p in panels]
            # The stage's own switch comes first; Input cleanup has none, so its close button does.
            focused = view.quickWindow().activeFocusItem()
            expected = f"Enable {row['name'].lower()}" if row["toggle"] else f"Close {row['name']}"
            assert scene.name(focused) == expected
            assert QRectF(0, 0, width, 900).contains(
                focused.mapRectToScene(QRectF(0, 0, focused.width(), focused.height()))
            )


def _focused(window):
    return window.centralWidget().quickWindow().activeFocusItem()


@pytest.mark.parametrize("how", ["mouse", "space", "return"])
def test_a_stage_row_opens_its_inspector_and_escape_gives_focus_back(quick_window, qapp, how) -> None:
    window = quick_window
    _show(window, qapp)
    inspector = window.quick_bridge.model["inspector"]
    compressor = window.quick_bridge.model["stages"][3]
    with _AccessibleScene() as scene:
        row = _named(window, "Compressor", *PLAIN_BUTTONS)
        if how == "mouse":
            _click(window, row, qapp)
        else:
            _activate(window, row, Qt.Key.Key_Space if how == "space" else Qt.Key.Key_Return, qapp)
        QTest.qWait(400)
        assert (inspector.shown, inspector.stageKey) == (True, "compressor")
        assert _iface(row).state().selected
        panel = window.centralWidget().rootObject().findChild(QObject, "inspector")
        assert scene.name(panel) == "Compressor inspector" and _iface(panel).role() == Role.Grouping
        # Focus goes to the stage's switch, wearing its ring only when the keyboard put it there.
        switch = _with_proxy(window, "Toggle", compressor["toggle"])
        assert _focused(window) is switch
        assert (len(_rings(window, switch)) == 1) is (how != "mouse")

        _press(window, Qt.Key.Key_Escape, qapp)
        assert not inspector.shown
        assert _focused(window) is row and len(_rings(window, row)) == 1
        assert not _iface(row).state().selected
        QTest.qWait(400)
        assert not panel.isVisible()


def test_the_close_button_is_named_for_its_stage_and_gives_focus_back(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    inspector = window.quick_bridge.model["inspector"]
    with _AccessibleScene() as scene:
        row = _named(window, "Compressor", *PLAIN_BUTTONS)
        _activate(window, row, Qt.Key.Key_Return, qapp)
        close = _named(window, "Close Compressor", "CloseCross")
        assert _iface(close).role() == Role.Button and scene.description(close) == "Esc"
        # Its ring shows on the glass.
        _focus(window, close, qapp)
        assert len(_rings(window, close)) == 1
        _press(window, Qt.Key.Key_Space, qapp)
        assert not inspector.shown
        assert _focused(window) is row


def test_one_inspector_shows_at_a_time_and_its_row_closes_it(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    inspector = window.quick_bridge.model["inspector"]
    with _AccessibleScene():
        gate_row = _named(window, "Noise gate", *PLAIN_BUTTONS)
        compressor_row = _named(window, "Compressor", *PLAIN_BUTTONS)
        _click(window, gate_row, qapp)
        _click(window, compressor_row, qapp)
        QTest.qWait(400)
        assert inspector.stageKey == "compressor"
        shown = [panel for panel in _of_kind(window, "InspectorPanel") if panel.isVisible()]
        assert [panel.property("card")["key"] for panel in shown] == ["compressor"]
        assert [_iface(row).state().selected for row in (gate_row, compressor_row)] == [False, True]
        _click(window, compressor_row, qapp)
        assert not inspector.shown
        assert _focused(window) is compressor_row


def test_a_stage_keeps_its_open_sections_when_another_is_inspected(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    gate = window.quick_bridge.model["stages"][1]
    _inspect(window, qapp, "gate")
    disclosure = _named(window, "Show Noise gate advanced", "Disclosure")
    disclosure.setProperty("open", True)
    threshold = _with_proxy(window, "Fader", gate["rows"][0]["proxy"])
    _inspect(window, qapp, "deesser")
    assert not threshold.isVisible()
    _inspect(window, qapp, "gate")
    assert disclosure.property("open") is True
    assert _with_proxy(window, "Fader", gate["rows"][0]["proxy"]) is threshold and threshold.isVisible()


def test_the_inspector_closes_when_another_page_opens(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    inspector = window.quick_bridge.model["inspector"]
    _inspect(window, qapp, "limiter")
    window.shell_page.setIndex(HEALTH)
    qapp.processEvents()
    assert not inspector.shown
    window.shell_page.setIndex(OVERVIEW)
    _inspect(window, qapp, "limiter")
    with _AccessibleScene():
        overview = _named(window, "Overview", *PLAIN_BUTTONS)
        assert not _iface(overview).state().selected
        _activate(window, overview, Qt.Key.Key_Space, qapp)
        assert not inspector.shown and _iface(overview).state().selected


def _inside(item, ancestor) -> bool:
    while item is not None:
        if item is ancestor:
            return True
        item = item.parentItem()
    return False


@pytest.mark.parametrize("how", ["page", "overview-click", "unopened"])
def test_focus_leaves_the_inspector_however_it_closes(quick_window, qapp, how) -> None:
    window = quick_window
    _show(window, qapp)
    inspector = window.quick_bridge.model["inspector"]
    panel = window.centralWidget().rootObject().findChild(QObject, "inspector")
    with _AccessibleScene():
        row = _named(window, "Compressor", *PLAIN_BUTTONS)
        overview = _named(window, "Overview", *PLAIN_BUTTONS)
        if how == "unopened":
            # Opened without a row, as the model can be: focus falls back to the Overview row.
            _inspect(window, qapp, "compressor")
            _focus(window, _with_proxy(window, "Toggle", window.quick_bridge.model["stages"][3]["toggle"]), qapp)
        else:
            _activate(window, row, Qt.Key.Key_Return, qapp)
            QTest.qWait(400)
        assert inspector.shown and _inside(_focused(window), panel)
        if how == "overview-click":
            _click(window, overview, qapp)
        else:
            window.shell_page.setIndex(HEALTH if how == "page" else OVERVIEW)
            if how == "unopened":
                inspector.close()
            qapp.processEvents()
        assert not inspector.shown
        focused = _focused(window)
        assert focused is not None and not _inside(focused, panel)
        expected = {"page": row, "overview-click": overview, "unopened": overview}[how]
        assert focused is expected


def test_the_input_cleanup_inspector_holds_the_settings_pickers_and_no_switch(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    _inspect(window, qapp, "cleanup")
    panel = next(p for p in _of_kind(window, "InspectorPanel") if p.isVisible())
    shown = [item for item in _items(panel) if item.isVisible()]
    assert not [item for item in shown if _kind(item) == "Toggle"]
    pickers = [item for item in shown if _kind(item) == "Picker"]
    assert [item.property("proxy") for item in pickers] == [
        window.input_cleanup_mode_choice, window.input_channel_mode_choice,
    ]
    # The Settings page shows the same two choices.
    for proxy in (window.input_cleanup_mode_choice, window.input_channel_mode_choice):
        assert len([item for item in _of_kind(window, "Picker") if item.property("proxy") is proxy]) == 2
    before = window.input_cleanup_mode_choice.index
    window.input_cleanup_mode_choice.setIndex((before + 1) % len(window.input_cleanup_mode_choice.items))
    qapp.processEvents()
    assert pickers[0].property("currentIndex") == window.input_cleanup_mode_choice.index != before


@pytest.mark.parametrize("key", ["cleanup", "gate", "suppression", "deesser", "compressor", "limiter"])
def test_every_control_in_a_stage_inspector_is_reached_and_named(quick_window, qapp, key) -> None:
    window = quick_window
    _show(window, qapp)
    _inspect(window, qapp, key)
    _open_every_disclosure(window, qapp)
    panel = next(p for p in _of_kind(window, "InspectorPanel") if p.isVisible())
    inside = {id(item) for item in _items(panel)}
    controls = [item for item in _tabbable(window, KEYBOARD_KINDS + ("CloseCross",)) if id(item) in inside]
    assert controls
    stops = _tab_cycle(window, qapp)
    with _AccessibleScene() as scene:
        assert [f"{_kind(item)} {scene.name(item)!r}" for item in controls if item not in stops] == []
        assert [_kind(item) for item in controls if not scene.name(item)] == []


def _slide(window, qapp, act, limit_ms: int) -> tuple[int, list[tuple[int, float]]]:
    """How long the inspector takes to slide after ``act``, and its progress along the way."""
    panel = window.centralWidget().rootObject().findChild(QObject, "inspector")
    start = panel.property("progress")
    timer = QElapsedTimer()
    timer.start()
    act()
    samples = []
    while panel.property("progress") == start or 0 < panel.property("progress") < 1:
        assert timer.elapsed() < limit_ms, samples[-3:]
        QTest.qWait(5)
        samples.append((timer.elapsed(), panel.property("progress")))
    return timer.elapsed(), samples


def test_the_inspector_slides_on_the_panel_tokens(full_motion_window, qapp) -> None:
    window = full_motion_window
    theme = window.quick_bridge.model["theme"]
    inspector = window.quick_bridge.model["inspector"]
    _show(window, qapp)
    opening, samples = _slide(window, qapp, lambda: inspector.show("gate"), 2000)
    assert theme["motionPanelOpen"] * 0.8 <= opening <= theme["motionPanelOpen"] + 300
    # Opening eases out: most of the way is covered in the first half of the time.
    early = [progress for elapsed, progress in samples if elapsed <= theme["motionPanelOpen"] * 0.4]
    assert early and max(early) > 0.75
    closing, samples = _slide(window, qapp, inspector.close, 2000)
    assert theme["motionPanelClose"] * 0.8 <= closing <= theme["motionPanelClose"] + 300
    # Closing eases in: it starts slowly and leaves fast.
    early = [progress for elapsed, progress in samples if elapsed <= theme["motionPanelClose"] * 0.4]
    assert early and min(early) > 0.6


def test_under_reduced_motion_the_inspector_fades_without_sliding(reduced_motion_window, qapp) -> None:
    window = reduced_motion_window
    _show(window, qapp)
    panel = window.centralWidget().rootObject().findChild(QObject, "inspector")
    window.quick_bridge.model["inspector"].show("gate")
    QTest.qWait(30)
    assert panel.property("progress") == 1
    assert 0 < panel.property("opacity") < 1
    window.quick_bridge.model["inspector"].close()
    QTest.qWait(30)
    # It fades where it stands, then goes.
    assert panel.property("progress") == 1
    assert 0 < panel.property("opacity") < 1
    assert panel.isVisible()
    QTest.qWait(window.quick_bridge.model["theme"]["motionPanelClose"] + 100)
    assert panel.property("progress") == 0
    assert panel.property("opacity") == 0 and not panel.isVisible()


def _text_blocks(source: str):
    """Every ``Text { ... }`` the file declares, as (line number, its body)."""
    import re

    for match in re.finditer(r"(?<![\w.])Text \{", source):
        depth, end = 0, match.end() - 1
        while True:
            depth += {"{": 1, "}": -1}.get(source[end], 0)
            if depth == 0:
                break
            end += 1
        yield source.count("\n", 0, match.start()) + 1, source[match.end() - 1:end]


def test_no_text_in_the_scene_reads_the_strings_it_shows_as_markup() -> None:
    from pathlib import Path

    files = sorted((Path(__file__).parents[1] / "mic_eq" / "ui" / "qml").glob("*.qml"))
    blocks = [(path.name, line, body) for path in files for line, body in _text_blocks(path.read_text(encoding="utf-8"))]
    assert len(blocks) > 25
    assert [(name, line) for name, line, body in blocks if "textFormat: Text.PlainText" not in body] == []


def test_names_and_messages_with_markup_are_shown_as_typed(quick_window, qapp) -> None:
    from mic_eq.config import DeviceIdentity

    window = quick_window
    _show(window, qapp)
    markup = ("<b>x</b>", "<i>mic</i>", "<u>boom</u>", "<s>warn</s>", "<b>drops</b>")
    window.input_choice.addItem(markup[1], DeviceIdentity(name="mic", direction="input"))
    window.input_choice.setCurrentIndex(window.input_choice.count() - 1)
    window.shell_texts["config_warning_banner"].update(text=markup[3], shown=True)
    window._set_health_chip("dropped_label", f"Drops: {markup[4]}", "warn")
    window.shell_texts["preset_status_label"].update(data={"name": markup[0], "modified": True})
    window.status_message.showMessage(markup[2])
    qapp.processEvents()
    shown = [
        item for item in _of_kind(window, "QQuickText", "Caption", "StatusText", "Title")
        if any(mark in item.property("text") for mark in markup)
    ]
    assert len(shown) >= 5, [item.property("text") for item in shown]
    plain = Qt.TextFormat.PlainText.value
    assert [item.property("text") for item in shown if _text_format(item) != plain] == []
    picker = _with_proxy(window, "Picker", window.quick_bridge.model["top"]["input"])
    assert picker.property("displayText") == markup[1]


def test_a_value_set_through_assistive_technology_is_committed_once(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp, EQUALIZER)
    gain = window.quick_bridge.model["eq"]["bands"][0]["gain"]
    fader = _with_proxy(window, "Fader", gain)
    set_value, commits = gain.setValue, []
    gain.setValue = lambda value: (commits.append(value), set_value(value))[1]
    with _AccessibleScene():
        _iface(fader).valueInterface().setCurrentValue(2.5)
        qapp.processEvents()
        assert commits == [25.0]
        assert gain.value == 25
        window.eq_state.flush()
        assert window.eq_state.get_band(0).gain_db == 2.5
        assert _announced_range(window, fader) == (-12.0, 12.0, 2.5)
        # Setting what the model already holds is no new edit.
        _iface(fader).valueInterface().setCurrentValue(2.5)
        qapp.processEvents()
        assert commits == [25.0]


def test_dragging_a_fader_commits_each_move_once(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp, EQUALIZER)
    QTest.qWait(50)  # the newly shown page lays out on the next frame
    view = window.centralWidget()
    gain = window.quick_bridge.model["eq"]["bands"][0]["gain"]
    fader = _with_proxy(window, "Fader", gain)
    set_value, commits = gain.setValue, []
    gain.setValue = lambda value: (commits.append(value), set_value(value))[1]
    middle = fader.mapToScene(QPointF(fader.width() * 0.5, fader.height() / 2)).toPoint()
    stops = [
        fader.mapToScene(QPointF(fader.width() * share, fader.height() / 2)).toPoint()
        for share in (0.6, 0.7, 0.8)
    ]
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=middle)
    for point in stops:
        QTest.mouseMove(view, point)
        qapp.processEvents()
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=stops[-1])
    # One commit per move, one for the press and one the slider sends on release; a second path
    # would double each move.
    assert len(stops) <= len(commits) <= len(stops) + 2 and commits == sorted(commits), commits
    assert abs(gain.value - commits[-1]) <= 0.5 and gain.value > 0  # the control rounds to a tenth of a dB


CONTRAST_COLORS = {
    "window": "#102030", "window_text": "#f0e0d0", "gray_text": "#80ff80",
    "highlight": "#ffee00", "highlight_text": "#000011",
}


@pytest.fixture(params=["dark", "contrast"])
def themed_window(request, monkeypatch):
    """The window with the dark palette, then with a Windows contrast theme's colours."""
    from mic_eq.ui.theme import build_palette

    if request.param == "contrast":
        monkeypatch.setattr("mic_eq.ui.quick_scene.PALETTE", build_palette(CONTRAST_COLORS))
    window = request.getfixturevalue("quick_window")
    window.contrast = request.param == "contrast"
    return window


def test_the_alert_look_of_a_mode_picker_does_not_rely_on_colour(themed_window, qapp) -> None:
    window = themed_window
    _show(window, qapp)
    picker = _with_proxy(window, "Picker", window.quick_bridge.model["top"]["mode"])
    background, text = picker.property("background"), picker.property("contentItem")
    looks = {}
    for index in (0, 1):
        window.processing_mode_choice.setIndex(index)
        qapp.processEvents()
        looks[index] = (
            QQmlProperty.read(background, "border.width"),
            _colour(background, "border.color"),
            text.property("font").weight(),
            _colour(text, "color"),
        )
    normal, bypass = looks[0], looks[1]
    # A thicker edge and bolder text mark Bypass whatever the colours are.
    assert bypass[0] > normal[0] and bypass[2] > normal[2]
    if window.contrast:
        # The theme has one text colour: the colours of the two looks are the same.
        assert (bypass[1], bypass[3]) == (normal[1], normal[3])


def use_system_theme(monkeypatch, system: dict | None):
    """Have the scenes loaded from now on drawn as under Windows contrast theme ``system`` (None: dark).

    Returns the palette they read.
    """
    from mic_eq.ui.theme import PALETTE, build_palette

    if system is None:
        return PALETTE
    palette = build_palette(system)
    monkeypatch.setattr("mic_eq.ui.quick_scene.PALETTE", palette)
    monkeypatch.setattr("mic_eq.ui.quick_scene.SYSTEM_CONTRAST_COLORS", system)
    return palette


def _system_themes() -> dict[str, dict | None]:
    from test_theme_tokens import HC_BLACK, HC_WHITE

    return {"dark": None, "HC Black": HC_BLACK, "HC White": HC_WHITE}


@pytest.fixture(params=list(_system_themes()))
def system_window(request, monkeypatch):
    """The window in the dark look, then in the Windows black and white contrast themes."""
    palette = use_system_theme(monkeypatch, _system_themes()[request.param])
    window = request.getfixturevalue("quick_window")
    vars(window)["system_palette"] = palette
    vars(window)["system_colors"] = _system_themes()[request.param]
    return window


def test_the_selected_sidebar_row_reads_and_rings_on_its_fill_in_every_theme(system_window, qapp) -> None:
    from mic_eq.ui.theme import contrast_ratio, flatten

    window, palette, system = system_window, system_window.system_palette, system_window.system_colors
    _show(window, qapp, OVERVIEW)
    row = _named(window, "Overview", *PLAIN_BUTTONS)
    assert row.property("current") is True
    fill = flatten(palette.accent_row_start, palette.sidebar_top)
    name = next(
        item for item in _items(row.property("contentItem"))
        if _kind(item) == "QQuickText" and item.property("text") == "Overview"
    )
    assert _colour(name, "color") == palette.hover_text.lower()
    assert contrast_ratio(_colour(name, "color"), fill) >= 4.5

    _focus(window, row, qapp)
    (ring,) = _rings(window, row)
    ring_colour = _colour(ring, "border.color")
    # Highlight on a Highlight fill would vanish: the ring takes the text colour on the fill instead.
    assert ring_colour == (palette.hover_text if system else palette.focus_ring).lower()
    assert contrast_ratio(ring_colour, fill) >= 3
    if system:
        assert system["highlight"] != ring_colour

    # A row that is not selected keeps the usual ring and text.
    other = _named(window, "Equalizer", *PLAIN_BUTTONS)
    _focus(window, other, qapp)
    (other_ring,) = _rings(window, other)
    assert _colour(other_ring, "border.color") == palette.focus_ring.lower()


def test_a_selected_stage_that_is_off_keeps_its_dim_name_in_the_dark_look(system_window, qapp) -> None:
    from mic_eq.ui.theme import contrast_ratio, flatten

    window, palette, system = system_window, system_window.system_palette, system_window.system_colors
    equalizer = next(row for row in window.quick_bridge.model["sidebar"]["chain"] if row["name"] == "Equalizer")
    if equalizer["toggle"].checked:
        equalizer["toggle"].click()
    _show(window, qapp, EQUALIZER)
    QTest.qWait(400)  # the name's colour fades
    row = _named(window, "Equalizer", *PLAIN_BUTTONS)
    assert row.property("current") is True and not equalizer["toggle"].checked
    name = next(
        item for item in _items(row.property("contentItem"))
        if _kind(item) == "QQuickText" and item.property("text") == "Equalizer"
    )
    # Dark: the off stage's dim colour, as before. A contrast theme reads the highlighted text.
    assert _colour(name, "color") == (palette.hover_text if system else palette.text_dim).lower()
    if system:
        assert contrast_ratio(_colour(name, "color"), flatten(palette.accent_row_start, palette.sidebar_top)) >= 4.5
