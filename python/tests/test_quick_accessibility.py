"""What assistive technology and the keyboard get from the Qt Quick scene.

Real Qt, real QML scene and real models: Qt's accessibility layer is switched on
to read the names, roles and states a screen reader would, and keys go through
``QTest`` to the window like a user's. Only blocking dialogs are stood in for.
"""
# pyright: reportMissingImports=false
# (the sibling test modules are found by pytest at run time)

from __future__ import annotations

import pytest
from PySide6.QtCore import QObject, QPointF, Qt, qInstallMessageHandler
from PySide6.QtGui import QAccessible
from PySide6.QtQml import QQmlEngine, QQmlExpression, QQmlProperty
from PySide6.QtTest import QTest

from mic_eq.ui.eq_state import EQ_FILTER_OPTIONS
from test_quick_layout import segoe_font  # noqa: F401  (a fixture)
from test_quick_shell import (
    _AccessibleScene,
    _graph_item,
    _tab_stops,
)
from test_quick_shell import quick_window as shell_window  # noqa: F401  (a fixture)

Role = QAccessible.Role
Text = QAccessible.Text
INTERACTIVE = {
    Role.Button, Role.CheckBox, Role.ComboBox, Role.Slider, Role.EditableText, Role.PageTab,
    Role.RadioButton, Role.SpinBox, Role.ScrollBar,
}
MIC, HEALTH, SETTINGS = 0, 1, 2
# The navigation tabs and the Health chip are bare buttons, whatever base the scene gives them.
PLAIN_BUTTONS = ("KeyButton", "AbstractButton")


@pytest.fixture(autouse=True)
def no_qml_warnings():
    """Fail a test that makes the scene print a warning: a binding loop, a type error, a null read.

    ``quick_window`` only watches the scene while it loads; this watches the whole interaction.
    """
    messages: list[str] = []
    previous = qInstallMessageHandler(
        lambda _mode, _context, message: messages.append(message) if "Main.qml" in message else None
    )
    yield messages
    qInstallMessageHandler(previous)
    assert messages == [], "the scene printed warnings: " + "; ".join(messages[:3])


@pytest.fixture
def quick_window(shell_window):  # noqa: F811
    """The real main window with its Qt Quick scene, as test_quick_shell builds it."""
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


def _show(window, qapp, page: int = MIC, width: int = 1400) -> None:
    window.resize(width, 900)
    window.activateWindow()
    window.centralWidget().setFocus()
    window.shell_page.setIndex(page)
    qapp.processEvents()


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
    _show(window, qapp)
    eq_switch = _with_proxy(window, "Toggle", model["eq"]["toggle"])
    mute = _with_proxy(window, "Toggle", model["top"]["mute"])
    assert model["eq"]["toggle"].checked and not model["top"]["mute"].checked
    with _AccessibleScene():
        for toggle, name, on in ((eq_switch, "Enable EQ", True), (mute, "Mute output", False)):
            interface = _iface(toggle)
            assert interface.role() == Role.CheckBox
            assert interface.text(Text.Name) == name
            assert interface.state().checkable
            assert bool(interface.state().checked) is on, f"{name} starts {'on' if on else 'off'}"

        _activate(window, eq_switch, Qt.Key.Key_Space, qapp)
        _activate(window, mute, Qt.Key.Key_Space, qapp)
        assert not _iface(eq_switch).state().checked
        assert _iface(mute).state().checked
        assert window.user_muted


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
    _show(window, qapp)
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
        window._commit_pending_configuration_snapshot()
        qapp.processEvents()
        assert not _iface(undo).state().disabled


def test_icon_buttons_are_named_by_what_they_do_not_by_their_tooltip_text(icon_window, qapp) -> None:
    window = icon_window
    _show(window, qapp)
    with _AccessibleScene():
        names = {
            _iface(button).text(Text.Name): button
            for button in _of_kind(window, "IconBtn")
            if button.isVisible()
        }
        assert {"Refresh audio devices", "Undo", "Redo", "Previous EQ band", "Next EQ band"} <= set(names)
        for name, button in names.items():
            assert _iface(button).role() == Role.Button
            assert len(name) <= 60, f"{name[:40]}... reads as a paragraph, not a name"
        assert _iface(names["Undo"]).state().disabled


def test_the_help_buttons_name_what_they_explain_and_describe_it(icon_window, qapp) -> None:
    window = icon_window
    _show(window, qapp)
    stages = window.quick_bridge.model["stages"]
    help_text = stages[1]["help"]
    with _AccessibleScene() as scene:
        button = next(
            button for button in _of_kind(window, "IconBtn")
            if scene.description(button) == help_text
        )
        assert "noise gate" in scene.name(button).lower()
        assert len(scene.name(button)) <= 60


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
    disclosure = _named(window, "Show COMPRESSOR advanced", "Disclosure")
    advanced = window.quick_bridge.model["stages"][3]["advanced"]
    attack = advanced[0]["proxy"]
    with _AccessibleScene() as scene:
        assert _iface(disclosure).role() == Role.Button
        assert not [i for i in _of_kind(window, "Fader", "ValueBox") if i.property("proxy") is attack]

        _focus(window, disclosure, qapp)
        _press(window, Qt.Key.Key_Space, qapp)

        assert scene.name(disclosure) == "Hide COMPRESSOR advanced"
        assert [i for i in _of_kind(window, "ValueBox") if i.property("proxy") is attack]
        _press(window, Qt.Key.Key_Space, qapp)
        assert scene.name(disclosure) == "Show COMPRESSOR advanced"
        assert not [i for i in _of_kind(window, "ValueBox") if i.property("proxy") is attack]


def test_the_navigation_tabs_expose_which_page_is_current(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    with _AccessibleScene():
        tabs = {
            _iface(tab).text(Text.Name): tab
            for tab in _of_kind(window, *PLAIN_BUTTONS)
            if _iface(tab).role() == Role.PageTab
        }
        assert list(tabs) == ["Mic page", "Health page", "Settings page"]

        def selected() -> list[str]:
            return [name for name, tab in tabs.items() if _iface(tab).state().selected]

        assert selected() == ["Mic page"]
        window.shell_page.setIndex(HEALTH)
        qapp.processEvents()
        assert selected() == ["Health page"]
        window.shell_page.setIndex(SETTINGS)
        qapp.processEvents()
        assert selected() == ["Settings page"]


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
    shown = {
        "statusMessage": window.status_message,
        "statusTransmission": window.shell_texts["transmission_status_label"],
        "statusCalibration": window.shell_texts["calibration_status_label"],
    }
    with _AccessibleScene() as scene:
        items = {name: root.findChild(QObject, name) for name in shown}
        for name, item in items.items():
            assert _iface(item) is not None, f"{name} is not exposed"
            assert _iface(item).role() == Role.StaticText
            assert scene.name(item) == shown[name].text != ""
        window.status_message.showMessage("Output muted")
        window.shell_texts["transmission_status_label"].update(text="Transmission: Running / Normal / Muted")
        qapp.processEvents()
        assert scene.name(items["statusMessage"]) == "Output muted"
        assert scene.name(items["statusTransmission"]) == "Transmission: Running / Normal / Muted"

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
    for page in (MIC, HEALTH, SETTINGS):
        _show(window, qapp, page)
        _open_every_disclosure(window, qapp)
        for interface in _accessible_tree(window):
            if interface.role() not in INTERACTIVE or interface.state().invisible:
                continue
            if not interface.text(Text.Name).strip():
                obj = interface.object()
                found.append(f"page {page}: {interface.role().name} {_kind(obj)}")
    return found


def test_no_interactive_item_in_the_scene_is_without_a_name(quick_window, qapp) -> None:
    with _AccessibleScene():
        assert _anonymous(quick_window, qapp) == []


def test_no_interactive_item_is_without_a_name_with_the_icon_font(icon_window, qapp) -> None:
    with _AccessibleScene():
        assert _anonymous(icon_window, qapp) == []


def test_names_are_short_labels_and_none_starts_with_a_colon(icon_window, qapp) -> None:
    window = icon_window
    with _AccessibleScene():
        too_long, colon = [], []
        for page in (MIC, HEALTH, SETTINGS):
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
    _show(window, qapp)
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


# --- 3. Keyboard -------------------------------------------------------------------


def _tab_cycle(window, qapp, page: int = MIC, width: int = 1400) -> list:
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
    "Toggle", "Fader", "Btn", "IconBtn", "Picker", "Disclosure", "SettingRow", *PLAIN_BUTTONS,
    "QQuickTextInput", "EQGraphItem",
)


@pytest.mark.parametrize("page", [MIC, HEALTH, SETTINGS], ids=["mic", "health", "settings"])
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
        ]
    assert missed == []


# 1200 is wide but not roomy (one band-editor column); 1400 and up has two.
@pytest.mark.parametrize("width", [1200, 1400, 1800])
def test_tab_visits_the_page_in_reading_order(either_window, qapp, width) -> None:
    window = either_window
    stops = _tab_cycle(window, qapp, width=width)
    with _AccessibleScene() as scene:
        names = [scene.name(item) for item in stops]
    assert len(set(map(id, stops))) == len(stops)
    landmarks = [
        "Mic page", "Health page", "Settings page",
        "Input audio device", "Output audio device", "Mute output", "Processing mode",
        "Start Processing",
        "Presets",
        "Enable EQ", "Tone preset", "Equalizer graph",
        "Enable EQ band 1", "EQ band 1 filter type", "EQ band 1 gain", "EQ band 1 center frequency",
        "Enable noise suppression", "Noise suppression backend", "Noise suppression strength",
    ]
    # A picker's name carries its choice after the label.
    positions = [
        next(i for i, name in enumerate(names) if name == landmark or name.startswith(landmark + ": "))
        for landmark in landmarks
    ]
    assert positions == sorted(positions), dict(zip(landmarks, positions))
    # Each card's own switch comes before its controls, and the Health chip closes the page.
    assert names.index("Enable noise gate") < names.index("Gate manual threshold")
    assert names.index("Gate manual threshold") < names.index("Show NOISE GATE advanced")
    assert names[-1].startswith("Health")


ACTIVATION_KEYS = [
    pytest.param(Qt.Key.Key_Space, id="space"),
    pytest.param(Qt.Key.Key_Return, id="return"),
    pytest.param(Qt.Key.Key_Enter, id="enter"),
]


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
    with _AccessibleScene():
        disclosure = _named(window, "Show LIMITER advanced", "Disclosure")
    _activate(window, disclosure, key, qapp)
    assert disclosure.property("open") is True


@pytest.mark.parametrize("key", ACTIVATION_KEYS)
def test_the_keyboard_picks_a_navigation_tab(quick_window, qapp, key) -> None:
    window = quick_window
    _show(window, qapp)
    with _AccessibleScene():
        tab = _named(window, "Settings page", *PLAIN_BUTTONS)
        _activate(window, tab, key, qapp)
    assert window.shell_page.currentIndex() == SETTINGS


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
    _show(window, qapp)
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
    _show(window, qapp)
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
    _show(window, qapp)
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
    _show(window, qapp)
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
    _show(window, qapp)
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
    _show(window, qapp)
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


@pytest.mark.parametrize("page", [MIC, SETTINGS], ids=["mic", "settings"])
def test_the_focused_control_always_shows_where_focus_is(either_window, qapp, page) -> None:
    window = either_window
    _show(window, qapp, page)
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
    expected = {"Toggle", "Fader", "Picker", "Btn", "QQuickTextInput", "EQGraphItem", "Disclosure"}
    assert (expected if page == MIC else {"SettingRow", "Picker"}) <= seen


def test_focus_rings_follow_keyboard_focus_not_the_mouse(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    toggle = _with_proxy(window, "Toggle", window.quick_bridge.model["top"]["mute"])
    view = window.centralWidget()
    _click(window, toggle, qapp)
    assert window.user_muted
    assert _rings(window, toggle) == [], "a click must not draw the keyboard focus ring"
    _focus(window, _graph_item(view.rootObject()), qapp)  # focus leaves the toggle...
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

    qml = (Path(__file__).parents[1] / "mic_eq" / "ui" / "qml" / "Main.qml").read_text(encoding="utf-8")
    assert re.findall(r"pixelSize:\s*\d", qml) == []


def test_text_sizes_follow_the_app_font_and_none_is_below_twelve(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    root = window.centralWidget().rootObject()
    sizes = [root.property(name) for name in ("fsCaption", "fsBody", "fsLead", "fsHeading", "fsTitle")]
    assert sizes == sorted(sizes) and sizes[0] >= 12
    assert sizes[1] == round(13 * qapp.font().pointSizeF() / 9)


def test_the_route_stays_readable_and_sits_above_the_transport_in_a_narrow_window(
    quick_window, qapp
) -> None:
    window = quick_window
    top = window.quick_bridge.model["top"]
    window.resize(1400, 900)
    qapp.processEvents()
    pickers = [_with_proxy(window, "Picker", top[name]) for name in ("input", "output")]
    start = _with_proxy(window, "Btn", top["start"])
    assert abs(_scene_y(start) - _scene_y(pickers[0])) < 10, "one row when there is room"

    window.resize(900, 800)
    qapp.processEvents()
    assert _scene_y(start) > _scene_y(pickers[0]) + pickers[0].height()
    assert all(picker.width() >= 140 for picker in pickers)
    captions = {
        item.property("text") for item in _of_kind(window, "Caption") if item.isVisible()
    }
    assert {"In", "Out"} <= captions


def test_an_opened_advanced_section_stays_open_when_the_window_crosses_the_wide_breakpoint(
    quick_window, qapp
) -> None:
    window = quick_window
    window.resize(1400, 900)
    qapp.processEvents()
    opened = [d for d in _of_kind(window, "Disclosure") if d.property("subject") == "COMPRESSOR"]
    opened[0].setProperty("open", True)
    qapp.processEvents()
    window.resize(1000, 900)
    qapp.processEvents()
    again = [d for d in _of_kind(window, "Disclosure") if d.property("subject") == "COMPRESSOR"]
    assert again and again[0].property("open") is True


def test_a_stage_that_is_off_says_so_and_keeps_its_controls_at_full_strength(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    model = window.quick_bridge.model
    gate = model["stages"][1]
    assert gate["toggle"].checked
    badges = lambda: [  # noqa: E731
        c for c in _of_kind(window, "Caption") if c.property("text") == "Off" and c.isVisible()
    ]
    already_off = len(badges())
    gate["toggle"].click()
    qapp.processEvents()
    assert len(badges()) == already_off + 1
    threshold = gate["rows"][0]["proxy"]
    row = next(
        r for r in _of_kind(window, "ControlRow") if r.property("p") is threshold
    )
    assert row.opacity() == 1 and all(c.opacity() == 1 for c in row.childItems())


@pytest.fixture
def no_icon_font_window(request, monkeypatch):
    monkeypatch.setattr("mic_eq.ui.quick_shell.icon_font", lambda: None)
    return request.getfixturevalue("quick_window")


def test_icon_buttons_are_still_there_without_the_icon_font(no_icon_font_window, qapp) -> None:
    window = no_icon_font_window
    _show(window, qapp)
    assert window.quick_bridge.model["theme"]["iconFont"] == ""
    faces = {
        button.property("label"): button.isVisible()
        for button in _of_kind(window, "IconBtn")
        if button.property("label")
    }
    assert faces and all(faces.values())
    assert {"Refresh", "More", "Prev", "Next", "?"} <= set(faces)


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


def test_the_navigation_rail_is_a_tab_list_and_the_meters_have_roles(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    with _AccessibleScene():
        roles = [interface.role() for interface in _accessible_tree(window)]
        assert Role.PageTabList in roles
        assert Role.ProgressBar in roles


def test_a_help_tip_opened_by_the_keyboard_goes_when_focus_moves_on(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    root = window.centralWidget().rootObject()
    tip = root.findChild(QObject, "tooltip")
    help_text = window.quick_bridge.model["stages"][1]["help"]
    button = next(b for b in _of_kind(window, "IconBtn") if b.property("hint") == help_text)
    _focus(window, button, qapp)
    _press(window, Qt.Key.Key_Space, qapp)
    assert tip.property("visible") is True and tip.property("text") == help_text
    _press(window, Qt.Key.Key_Escape, qapp)
    assert tip.property("visible") is False
    _press(window, Qt.Key.Key_Space, qapp)
    assert tip.property("visible") is True
    _press(window, Qt.Key.Key_Tab, qapp)
    # The next control may show its own help; the help that was asked for goes.
    assert not (tip.property("visible") and tip.property("text") == help_text)


@pytest.fixture
def reduced_motion_window(request, monkeypatch):
    monkeypatch.setenv("AUDIOFORGE_REDUCED_MOTION", "1")
    return request.getfixturevalue("quick_window")


def test_reduced_motion_keeps_the_level_trace_running(reduced_motion_window) -> None:
    bridge = reduced_motion_window.quick_bridge
    assert bridge.model["theme"]["motion"] == 0
    assert bridge._trace_timer.isActive()


# The tight cells are the left edge of each layout regime: the minimum, `wide` at 1200 and `roomy` at 1400.
@pytest.mark.parametrize("width", [900, 1200, 1400])
@pytest.mark.parametrize("kind", ["low_shelf", "high_pass", "notch"])
def test_the_gain_fader_keeps_room_beside_the_band_fields(quick_window, qapp, width, kind) -> None:
    window = quick_window
    window.eq_state.set_band_value(0, "filter_type", kind)
    window.eq_state.flush()
    window.resize(width, 900)
    QTest.qWait(50)
    gain = window.quick_bridge.model["eq"]["bands"][0]["gain"]
    fader = _with_proxy(window, "Fader", gain)
    assert fader.width() >= 100


@pytest.mark.parametrize("width", [900, 1199, 1200, 1400])
def test_the_mode_picker_keeps_stop_away_from_mute(quick_window, qapp, width) -> None:
    window = quick_window
    window.resize(width, 900)
    QTest.qWait(50)
    top = window.quick_bridge.model["top"]
    mute = _with_proxy(window, "Toggle", top["mute"])
    mode = _with_proxy(window, "Picker", top["mode"])
    start = _with_proxy(window, "Btn", top["start"])

    assert _scene_x(mute) + mute.width() < _scene_x(mode) < _scene_x(start)
    assert _scene_x(start) - (_scene_x(mute) + mute.width()) >= mode.width()


def test_q_and_slope_show_only_for_the_filters_that_use_them(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
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
    mute = _with_proxy(window, "Toggle", window.quick_bridge.model["top"]["mute"])
    with _AccessibleScene():
        for expected in (True, False):
            if how == "click":
                _click(window, mute, qapp)
            elif how in KEYS:
                _activate(window, mute, KEYS[how], qapp)
            else:
                _perform(mute, how)
            qapp.processEvents()
            assert window.user_muted is expected, f"{how} changes the state once"
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
    _show(window, qapp)
    toggle = _with_proxy(window, "Toggle", window.quick_bridge.model["eq"]["toggle"])
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
    mute = _with_proxy(window, "Toggle", window.quick_bridge.model["top"]["mute"])
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
        assert window.shell_page.currentIndex() == MIC
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
    window = quick_window
    _show(window, qapp)
    badge = window.centralWidget().rootObject().findChild(QObject, "mutedBadge")
    assert "(Ctrl+M)" in window.quick_bridge.model["top"]["mute"].toolTip
    # Holding the keys must not flap the live mute.
    assert window.user_mute_checkbox_action.autoRepeat() is False
    assert badge.property("visible") is False

    _press(window, Qt.Key.Key_M, qapp, Qt.KeyboardModifier.ControlModifier)
    assert window.user_muted
    assert badge.property("visible") is True
    assert [child.property("text") for child in badge.childItems()] == ["Muted"]

    _press(window, Qt.Key.Key_M, qapp, Qt.KeyboardModifier.ControlModifier)
    assert not window.user_muted
    assert badge.property("visible") is False


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
    _open_every_disclosure(window, qapp)
    wrong, checked = [], 0
    with _AccessibleScene() as scene:
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
        row = next(
            text for text in _of_kind(window, "QQuickText")
            if _iface(text) is not None and scene.name(text).startswith(chip.name)
        )
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
    _click(window, _with_proxy(window, "Toggle", window.quick_bridge.model["top"]["mute"]), qapp)
    QTest.qWait(700)
    assert tip.property("visible") is False


def test_control_edges_and_tracks_draw_from_the_control_edge_token(quick_window, qapp) -> None:
    window = quick_window
    _show(window, qapp)
    _open_every_disclosure(window, qapp)
    model = window.quick_bridge.model
    edge = model["theme"]["control_edge"].lower()
    track = _with_proxy(window, "Toggle", model["top"]["mute"]).property("contentItem").childItems()[0]
    assert _colour(track, "color") == edge
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
def test_the_stage_cards_are_visited_and_laid_out_in_signal_order(quick_window, qapp, width) -> None:
    window = quick_window
    stops = _tab_cycle(window, qapp, width=width)
    with _AccessibleScene() as scene:
        names = [scene.name(item) for item in stops]
    switches = [
        "Enable noise suppression", "Enable noise gate", "Enable de-esser", "Enable compressor", "Enable limiter",
    ]
    positions = [names.index(name) for name in switches]
    assert positions == sorted(positions), dict(zip(switches, positions))

    grid = window.centralWidget().rootObject().findChild(QObject, "stageGrid")
    cards = [item for item in grid.childItems() if _kind(item) == "StageCard"]
    assert len(cards) == 5
    rows = [(card.x(), card.y(), card.x() + card.width(), card.y() + card.height()) for card in cards]
    assert grid.property("implicitHeight") >= max(row[3] for row in rows)
    # One column under 1200 px, otherwise two: left to right, then down, each under the card above it.
    wide = width >= 1200
    assert len({round(row[0]) for row in rows}) == (2 if wide else 1)
    for first, second in ((0, 1), (0, 2), (1, 3), (2, 4)) if wide else ((0, 1), (1, 2), (2, 3), (3, 4)):
        above, below = rows[first], rows[second]
        if above[0] == below[0]:
            assert below[1] > above[3], (first, second)
        else:
            assert below[0] >= above[2] and below[1] == above[1] == 0, (first, second)
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1], (a, b)


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

    qml = (Path(__file__).parents[1] / "mic_eq" / "ui" / "qml" / "Main.qml").read_text(encoding="utf-8")
    blocks = list(_text_blocks(qml))
    assert len(blocks) > 25
    assert [line for line, body in blocks if "textFormat: Text.PlainText" not in body] == []


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
    _show(window, qapp)
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
    _show(window, qapp)
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
        monkeypatch.setattr("mic_eq.ui.quick_shell.PALETTE", build_palette(CONTRAST_COLORS))
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
