// Main window scene. Controls consume shared state from quick_shell.py;
// values, control metadata and commands live in the Python presentation models.
pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import QtQuick.Templates as T
import AudioForge

Rectangle {
    id: root

    required property QuickBridge ui
    readonly property var m: ui.model
    readonly property var t: Look.t
    readonly property bool hasIcons: Look.hasIcons
    readonly property bool wide: width >= 1200
    readonly property bool running: m.top.stop.enabled
    readonly property bool pressing: pointer.active
    readonly property int fsCaption: Look.fsCaption
    readonly property int fsBody: Look.fsBody
    readonly property int fsLead: Look.fsLead
    readonly property int fsHeading: Look.fsHeading
    readonly property int fsTitle: Look.fsTitle
    readonly property int fsDisplay: Look.fsDisplay
    // When the last tooltip closed; the next one opens at once if it follows closely.
    property double tipClosedAt: 0

    color: t.window_ground

    onPressingChanged: if (pressing) tip.close()
    // Another page or the Overview row closes the inspector too; focus never stays on the hidden panel.
    onInspectingChanged: {
        if (inspecting)
            return
        for (let item = Window.activeFocusItem; item; item = item.parent) {
            if (item === inspector) {
                returnFocus(Qt.OtherFocusReason)
                return
            }
        }
    }
    // Esc closes a tip that keyboard focus opened, as it closes one the help button opened;
    // without a tip it closes the inspector.
    Keys.onEscapePressed: event => {
        if (tip.visible)
            tip.close()
        else if (inspecting)
            closeInspector(Qt.TabFocusReason)
        else
            event.accepted = false
    }

    // One shared tooltip, so long help text wraps and matches the theme.
    Tip {
        id: tip
        objectName: "tooltip"
        delay: 500
        onClosed: root.tipClosedAt = Date.now()
    }

    // A tip asked for by a click or key opens at once and goes away by itself.
    function showTip(item, text, asked) {
        if (text === "" || (pressing && !asked))
            return
        tip.parent = item
        tip.text = text
        tip.delay = asked || Date.now() - tipClosedAt < 600 ? 0 : 500
        tip.timeout = asked ? 10000 : -1
        tip.open()
    }

    // Menus that belong to a button open under it, also from the keyboard.
    function press(item, proxy) {
        const below = item.mapToGlobal(0, item.height + 4)
        proxy.clickAt(below.x, below.y)
    }

    function hideTip(item) {
        if (tip.parent === item)
            tip.close()
    }

    // Hovering the parent or moving keyboard focus onto it shows the help; a click does not.
    component Hint: HoverHandler {
        property string text: ""
        property bool focused: (parent as T.Control)?.visualFocus === true
        onHoveredChanged: sync()
        onFocusedChanged: sync()
        function sync() {
            if (hovered || focused)
                root.showTip(parent, text)
            else
                root.hideTip(parent)
        }
    }

    component FadeIn: NumberAnimation {
        from: 0
        to: 1
        duration: root.t.motionColor
        easing.type: Easing.OutCubic
    }

    component Caption: MutedText {
        property string tip: ""
        Hint { text: parent.tip }
    }

    // A caption that screen readers read, for text that carries a state (not a label of a control).
    component StatusText: Caption {
        Accessible.role: Accessible.StaticText
        Accessible.name: text
    }

    component Toggle: SwitchButton {
        id: toggle
        property var proxy
        on: proxy ? proxy.checked : false
        enabled: proxy ? proxy.enabled : false
        onRequested: proxy.click()
        Accessible.name: proxy && proxy.name !== "" ? proxy.name : text
        Accessible.description: proxy ? proxy.toolTip : ""
        Hint { text: toggle.proxy ? toggle.proxy.toolTip : "" }
    }

    component Fader: Slider {
        id: fader
        property var proxy
        // The slider's units per proxy unit. A screen reader announces the slider's own value,
        // so a proxy that counts in tenths of a dB sets 0.1 here and is announced in dB.
        property real unit: 1
        from: proxy ? proxy.minimum * unit : 0
        to: proxy ? proxy.maximum * unit : unit
        value: proxy ? proxy.value * unit : 0
        stepSize: 0
        enabled: proxy ? proxy.enabled : false
        implicitHeight: 24
        onMoved: proxy.setValue(value / unit)
        onPressedChanged: if (!pressed) proxy.released()
        // Assistive technology (UI Automation RangeValue) sets `value` without `moved`. Dragging commits
        // through onMoved, and a value the model already holds is only the binding following it.
        onValueChanged: if (!pressed && proxy && Math.abs(value - proxy.value * unit) > 1e-9) {
            proxy.setValue(value / unit)
            proxy.released()
        }
        Keys.onLeftPressed: nudge(-1)
        Keys.onRightPressed: nudge(1)
        Keys.onUpPressed: nudge(1)
        Keys.onDownPressed: nudge(-1)
        function nudge(direction) {
            proxy.setValue(proxy.value + direction * (proxy.step > 0 ? proxy.step : (to - from) / unit / 100))
            proxy.released()
        }
        Accessible.name: proxy ? proxy.name : ""
        Accessible.description: proxy ? proxy.toolTip : ""
        Hint { text: fader.proxy ? fader.proxy.toolTip : "" }
        background: Rectangle {
            x: fader.leftPadding
            y: fader.topPadding + fader.availableHeight / 2 - 2
            width: fader.availableWidth
            height: 4
            radius: 2
            color: root.t.control_edge
            Rectangle {
                width: fader.visualPosition * parent.width
                height: 4
                radius: 2
                color: root.t.accent
            }
        }
        handle: Rectangle {
            x: fader.leftPadding + fader.visualPosition * (fader.availableWidth - width)
            y: fader.topPadding + fader.availableHeight / 2 - height / 2
            width: 16
            height: 16
            radius: 8
            color: root.t.text_primary
            scale: fader.pressed ? 1.15 : 1
            Behavior on scale { NumberAnimation { duration: Look.movement(root.t.motionColor / 2) } }
            FocusRing { control: fader }
        }
    }

    component ValueBox: Rectangle {
        id: box
        property var proxy
        property var display: null
        readonly property var source: display ? display : proxy
        implicitWidth: 92
        implicitHeight: 30
        radius: root.t.radiusControl
        color: root.t.control_surface
        border.color: root.t.control_edge
        Hint { text: box.proxy ? box.proxy.toolTip : ""; focused: input.activeFocus && !input.readOnly }
        FocusRing { visible: input.activeFocus && !input.readOnly }
        TextInput {
            id: input
            anchors.fill: parent
            anchors.margins: 6
            horizontalAlignment: TextInput.AlignHCenter
            verticalAlignment: TextInput.AlignVCenter
            text: box.source ? box.source.text : ""
            readOnly: box.display !== null
            // A display is read, not edited: no Tab stop. A click still focuses it, so Ctrl+C copies.
            activeFocusOnTab: !readOnly
            enabled: box.proxy ? box.proxy.enabled : false
            color: root.t.text_primary
            selectionColor: root.t.accent
            selectedTextColor: root.t.text_on_accent
            selectByMouse: true
            clip: true
            font.pixelSize: root.fsBody
            font.features: { "tnum": 1 }
            Accessible.role: Accessible.EditableText
            Accessible.name: box.proxy ? box.proxy.name : ""
            Accessible.description: box.proxy ? box.proxy.toolTip : ""
            onEditingFinished: if (!readOnly && text !== box.source.text) box.proxy.setText(text)
            Keys.onUpPressed: step(1)
            Keys.onDownPressed: step(-1)
            // The field claims Ctrl+Z and Ctrl+Shift+Z whatever its history holds, so the keys arrive
            // here. With no typing to take back, forward exactly the window's own shortcuts
            // (Edit > Undo, Edit > Redo in main_window.py), so a box behaves like the page around it.
            Keys.onPressed: event => {
                if (event.key !== Qt.Key_Z)
                    return
                const undo = event.modifiers === Qt.ControlModifier && !canUndo
                const redo = event.modifiers === (Qt.ControlModifier | Qt.ShiftModifier) && !canRedo
                if (undo || redo) {
                    (undo ? root.m.presets.undo : root.m.presets.redo).click()
                    event.accepted = true
                }
            }
            // Put the shown value back; the abandoned text is never applied.
            Keys.onEscapePressed: event => {
                if (readOnly || text === box.source.text)
                    event.accepted = false
                else
                    text = Qt.binding(() => box.source ? box.source.text : "")
            }
            // Only a focused editable field steps, so the page can scroll past the rest.
            WheelHandler {
                enabled: input.activeFocus && !input.readOnly
                onWheel: event => input.step(event.angleDelta.y > 0 ? 1 : -1)
            }
            function step(direction) {
                if (!readOnly) {
                    box.proxy.setValue(box.proxy.value + direction * box.proxy.step)
                    box.proxy.released()
                }
            }
        }
    }

    component Picker: ChoiceBox {
        id: picker
        property var proxy
        model: proxy ? proxy.items : []
        currentIndex: proxy ? proxy.index : -1
        enabled: proxy ? proxy.enabled : false
        onActivated: index => proxy.setIndex(index)
        // Replacing the model resets the index; read it again afterwards.
        onModelChanged: currentIndex = Qt.binding(() => proxy ? proxy.index : -1)
        label: proxy ? proxy.name : ""
        Accessible.description: proxy ? proxy.toolTip : ""
        prompting: currentIndex < 0 && !!proxy
        prompt: proxy ? (proxy.placeholder ?? "") : ""
        expandGlyph: root.m.glyphs.expand
        // A name cut short by a narrow window is shown whole in the tooltip.
        Hint {
            text: {
                const tip = picker.proxy ? picker.proxy.toolTip : ""
                return picker.shown.truncated ? picker.shown.text + (tip !== "" ? "\n" + tip : "") : tip
            }
        }
    }

    // A PushButton that acts through a shared command, `look` as there.
    component Btn: PushButton {
        id: btn
        property var proxy
        property string help: proxy ? proxy.toolTip : ""
        text: proxy ? proxy.text : ""
        enabled: proxy ? proxy.enabled : false
        onClicked: if (proxy) root.press(this, proxy)
        Accessible.name: proxy && proxy.name !== "" ? proxy.name : text
        Accessible.description: help
        Hint { text: btn.help }
    }

    // Held by the mouse, by Space or by the B on its keycap, it plays the voice without the
    // equalizer; letting go, or losing the press, the focus or the view, returns it. Only the
    // item that began a hold ends it.
    component HoldBtn: PushButton {
        id: hold
        property var proxy
        property bool holding: false
        // A hold ended elsewhere (an edit, Stop, the window going away) is no longer this button's.
        readonly property bool held: proxy ? proxy.held : false
        onHeldChanged: if (!held) holding = false
        function begin() {
            if (!holding && proxy && proxy.enabled) {
                proxy.press()
                holding = proxy.held
            }
        }
        function end() {
            if (holding) {
                holding = false
                proxy.release()
            }
        }
        function holdKey(event) {
            return event.key === Qt.Key_Space
                || (event.key === Qt.Key_B && event.modifiers === Qt.NoModifier)
        }
        // Measured in the held look's weight, so a hold does not reflow the header.
        TextMetrics {
            id: fullLabel
            text: hold.proxy ? hold.proxy.text : ""
            font.pixelSize: Look.fsBody
            font.weight: Font.DemiBold
        }
        TextMetrics {
            id: shortLabel
            text: hold.proxy ? hold.proxy.shortText : ""
            font: fullLabel.font
        }
        readonly property real shortWidth: leftPadding + Math.ceil(shortLabel.advanceWidth) + rightPadding
        // Short of room for the whole label, it says Hold beside its keycap; never less.
        text: !proxy ? "" : availableWidth >= Math.ceil(fullLabel.advanceWidth) ? proxy.text : proxy.shortText
        enabled: proxy ? proxy.enabled : false
        look: held ? "soft" : "plain"
        leftPadding: 12
        rightPadding: 8 + keycap.width + 8
        implicitWidth: leftPadding + Math.ceil(fullLabel.advanceWidth) + rightPadding
        onPressedChanged: pressed ? begin() : end()
        onCanceled: end()
        onActiveFocusChanged: if (!activeFocus) end()
        onVisibleChanged: if (!visible) end()
        onEnabledChanged: if (!enabled) end()
        Keys.onPressed: event => {
            if (holdKey(event)) {
                event.accepted = true
                if (!event.isAutoRepeat)
                    begin()
            }
        }
        Keys.onReleased: event => {
            if (event.key === Qt.Key_Space || event.key === Qt.Key_B) {
                event.accepted = true
                if (!event.isAutoRepeat)
                    end()
            }
        }
        Accessible.name: proxy ? proxy.name : ""
        Accessible.description: proxy ? proxy.toolTip : ""
        Hint { text: hold.proxy ? hold.proxy.toolTip : "" }
        EscChip {
            id: keycap
            objectName: "holdKeycap"
            text: hold.proxy ? hold.proxy.keycap : ""
            anchors.right: parent.right
            anchors.rightMargin: 8
            anchors.verticalCenter: parent.verticalCenter
        }
    }

    component IconBtn: KeyButton {
        id: iconBtn
        property var proxy: null
        property string glyph: ""
        // Shown in place of the glyph when the system has no icon font.
        property string label: ""
        property string hint: proxy ? proxy.toolTip : ""
        enabled: proxy ? proxy.enabled : true
        opacity: enabled ? 1 : 0.4
        implicitWidth: Math.max(30, face.implicitWidth + 16)
        implicitHeight: 30
        onClicked: if (proxy) root.press(this, proxy)
        Accessible.role: Accessible.Button
        Accessible.name: proxy && proxy.name !== "" ? proxy.name : proxy && proxy.text !== "" ? proxy.text : hint
        Accessible.description: proxy ? hint : ""
        Hint { text: iconBtn.hint }
        background: Rectangle {
            radius: root.t.radiusControl
            color: iconBtn.hovered ? root.t.control_hover : "transparent"
            FocusRing { control: iconBtn }
        }
        contentItem: Text {
            id: face
            text: root.hasIcons ? iconBtn.glyph : iconBtn.label
            textFormat: Text.PlainText
            font.family: root.hasIcons ? root.t.iconFont : Application.font.family
            font.pixelSize: root.hasIcons ? root.fsLead : root.fsBody
            color: iconBtn.hovered ? root.t.hover_text : root.t.text_muted
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
        }
    }

    component Chip: Rectangle {
        id: chip
        property var proxy
        // False when a button over the chip carries the text for assistive technology.
        property bool announced: true
        // One line, cut short with "..." when the chip is limited in width.
        property bool single: false
        readonly property string state_: proxy && proxy.state !== "" ? proxy.state : "idle"
        implicitHeight: chipText.implicitHeight + 12
        implicitWidth: chipText.implicitWidth + 24
        radius: root.t.radiusControl
        color: root.t["status_" + state_ + "_surface"] ?? root.t.status_idle_surface
        border.color: root.t["status_" + state_ + "_border"] ?? root.t.status_idle_border
        Hint {
            text: chip.proxy
                  ? [chipText.truncated ? chipText.text : "", chip.proxy.toolTip].filter(t => t !== "").join("\n\n")
                  : ""
        }
        Text {
            id: chipText
            textFormat: Text.PlainText
            anchors.centerIn: parent
            width: Math.min(implicitWidth, chip.width - 24)
            text: chip.proxy ? chip.proxy.text : ""
            color: root.t["status_" + chip.state_ + "_text"] ?? root.t.status_idle_text
            font.pixelSize: root.fsBody
            wrapMode: chip.single ? Text.NoWrap : Text.Wrap
            elide: Text.ElideRight
            Accessible.ignored: !chip.announced
            Accessible.name: chip.proxy && chip.proxy.name !== "" ? chip.proxy.name + ": " + text : text
            Accessible.description: chip.proxy ? chip.proxy.toolTip : ""
        }
    }

    component ControlRow: RowLayout {
        id: line
        property var row
        readonly property var p: row.proxy
        visible: p.shown
        spacing: 12
        Caption {
            text: line.row.label
            tip: line.p.toolTip
            visible: text !== "" && line.row.kind !== "toggle"
            opacity: line.p.enabled ? 1 : 0.45
            Layout.preferredWidth: 108
        }
        Loader {
            Layout.fillWidth: true
            sourceComponent: ({
                field: fieldRow, number: numberRow, combo: comboRow, toggle: toggleRow,
                text: textRow, meter: meterBar
            })[line.row.kind]
        }
        Component {
            id: fieldRow
            RowLayout {
                spacing: 12
                opacity: line.p.enabled ? 1 : 0.45
                Fader { proxy: line.p; Layout.fillWidth: true }
                ValueBox { proxy: line.p; display: line.row.display }
            }
        }
        Component {
            id: numberRow
            RowLayout {
                opacity: line.p.enabled ? 1 : 0.45
                Item { Layout.fillWidth: true }
                ValueBox { proxy: line.p }
            }
        }
        Component { id: comboRow; Picker { proxy: line.p } }
        Component {
            id: toggleRow
            RowLayout {
                Toggle { proxy: line.p; text: line.row.label !== "" ? line.row.label : line.p.text }
                Item { Layout.fillWidth: true }
            }
        }
        Component {
            id: textRow
            Text {
                textFormat: Text.PlainText
                text: line.p.text
                color: line.row.label !== "" ? root.t.text_primary : root.t.text_muted
                font.pixelSize: root.fsBody
                wrapMode: Text.Wrap
                horizontalAlignment: line.row.label !== "" ? Text.AlignRight : Text.AlignLeft
                Accessible.name: line.p.name !== "" ? line.p.name + ": " + text : text
            }
        }
        Component {
            id: meterBar
            QuickMeterItem {
                source: line.p
                implicitHeight: 18
                Accessible.role: Accessible.ProgressBar
                Accessible.name: line.row.label
                Accessible.description: line.p.valueText
            }
        }
    }

    component CardHeader: RowLayout {
        id: header
        property var card
        spacing: 12
        Toggle { proxy: header.card.toggle; visible: header.card.toggle !== null }
        Title { text: header.card.title }
        // A stage that is off keeps its controls at full strength; this says why nothing happens.
        Caption {
            text: "Off"
            visible: header.card.toggle !== null && !header.card.toggle.checked
        }
        IconBtn {
            glyph: root.m.glyphs.help
            hint: header.card.help
            label: "?"
            visible: hint !== ""
            onClicked: root.showTip(this, hint, true)
            onActiveFocusChanged: if (!activeFocus) root.hideTip(this)
            Keys.onEscapePressed: root.hideTip(this)
            Accessible.name: "Help for " + header.card.title.toLowerCase()
            Accessible.description: hint
        }
        Item { Layout.fillWidth: true }
    }

    // A full-width row that opens the section under it.
    component Disclosure: KeyButton {
        id: disclosure
        property bool open: false
        property string subject: ""
        implicitHeight: 38
        onClicked: open = !open
        Accessible.role: Accessible.Button
        Accessible.name: (open ? "Hide " : "Show ") + subject + " " + text.toLowerCase()
        background: Rectangle {
            radius: root.t.radiusControl
            color: disclosure.hovered || disclosure.open
                   ? root.t.control_surface : root.t.control_surface_alt
            border.color: root.t.control_edge
            Behavior on color { ColorAnimation { duration: root.t.motionColor } }
            FocusRing { control: disclosure }
        }
        contentItem: RowLayout {
            spacing: 8
            Text {
                textFormat: Text.PlainText
                text: disclosure.text
                color: root.t.text_primary
                font.pixelSize: root.fsBody
                font.weight: Font.Medium
                leftPadding: 12
                Layout.fillWidth: true
            }
            Text {
                textFormat: Text.PlainText
                text: disclosure.open ? "Hide" : "Show"
                color: root.t.text_muted
                font.pixelSize: root.fsCaption
            }
            Text {
                textFormat: Text.PlainText
                text: root.m.glyphs.expand
                visible: root.hasIcons
                font.family: root.t.iconFont
                font.pixelSize: root.fsCaption
                color: root.t.text_primary
                Layout.rightMargin: 14
                rotation: disclosure.open ? 180 : 0
                Behavior on rotation { NumberAnimation { duration: Look.movement(root.t.motionColor) } }
            }
        }
    }

    // One line of the Settings page: a switch, an action, or a menu.
    component SettingRow: ModelButton {
        id: setting
        property var row
        readonly property var p: row.proxy
        readonly property bool isSwitch: row.kind === "toggle"
        visible: p.shown
        enabled: p.enabled
        implicitHeight: 44
        // A switch row acts on its check state (see ModelButton); the other rows on the click.
        onClicked: if (!isSwitch) root.press(this, p)
        checkable: isSwitch
        on: isSwitch && p.checked
        onRequested: root.press(this, p)
        Accessible.role: isSwitch ? Accessible.CheckBox : Accessible.Button
        Accessible.name: p.text
        Accessible.description: p.toolTip !== p.text ? p.toolTip : ""
        Hint { text: setting.p.toolTip !== setting.p.text ? setting.p.toolTip : "" }
        background: Rectangle {
            radius: root.t.radiusControl
            color: setting.hovered && setting.enabled ? root.t.control_surface_alt : "transparent"
            FocusRing { control: setting; outset: 0 }
        }
        contentItem: RowLayout {
            spacing: 12
            Text {
                textFormat: Text.PlainText
                text: setting.p.text
                color: setting.enabled ? root.t.text_primary : root.t.action_disabled_text
                font.pixelSize: root.fsBody
                elide: Text.ElideRight
                leftPadding: 10
                Layout.fillWidth: true
            }
            Toggle {
                visible: setting.isSwitch
                proxy: setting.p
                // The row is the control; the switch only shows the state.
                enabled: false
                live: setting.enabled
                activeFocusOnTab: false
                Accessible.ignored: true
                Layout.rightMargin: 10
            }
            Text {
                textFormat: Text.PlainText
                visible: !setting.isSwitch && setting.enabled && root.hasIcons
                text: setting.row.kind === "menu" ? root.m.glyphs.expand : root.m.glyphs.next
                Layout.rightMargin: 14
                font.family: root.t.iconFont
                font.pixelSize: root.fsCaption
                color: root.t.text_muted
            }
        }
    }

    // One line of the Health page. The chip text is "Title: value".
    component HealthRow: RowLayout {
        id: health
        property var row
        readonly property var proxy: row.proxy
        readonly property string value: proxy.text.startsWith(row.title + ":")
                                        ? proxy.text.slice(row.title.length + 1).trim() : proxy.text
        readonly property string state_: proxy.state !== "" ? proxy.state : "idle"
        spacing: 10
        Rectangle {
            implicitWidth: 8
            implicitHeight: 8
            radius: 4
            color: root.t["status_" + health.state_ + "_text"] ?? root.t.status_idle_text
        }
        Caption {
            text: health.row.title
            tip: health.proxy.toolTip
            color: root.t.text_primary
            Layout.preferredWidth: 110
        }
        Text {
            textFormat: Text.PlainText
            text: health.value.charAt(0).toUpperCase() + health.value.slice(1)
            color: root.t.text_primary
            font.pixelSize: root.fsBody
            wrapMode: Text.Wrap
            Layout.fillWidth: true
            Accessible.name: health.proxy.name + ": " + text
            Accessible.description: health.proxy.toolTip
        }
    }

    component Page: Flickable {
        id: page
        default property alias content: column.data
        property alias spacing: column.spacing
        // The room left of the content, and right of it where the scroll bar runs.
        property int leftInset: 8
        property int rightInset: 16
        property string scrollName: "Page scroll bar"
        // The page's ground behind its content; it stays put while the content scrolls.
        property color ground: "transparent"
        contentHeight: column.implicitHeight + 24
        boundsBehavior: Flickable.StopAtBounds
        // A mouse drag belongs to the controls; wheel, scroll bar and touch still scroll.
        acceptedButtons: Qt.NoButton
        clip: true
        // Keep the control that has keyboard focus on screen.
        readonly property Item focusItem: Window.activeFocusItem
        onFocusItemChanged: {
            var item = focusItem
            for (var p = item; p && p !== column; p = p.parent) {}
            if (!visible || !p || root.pressing)
                return
            var top = item.mapToItem(contentItem, 0, 0).y
            if (top < contentY + 12)
                contentY = Math.max(0, top - 12)
            else if (top + item.height > contentY + height - 12)
                contentY = Math.min(contentHeight - height, top + item.height - height + 12)
        }
        ScrollBar.vertical: ScrollBar {
            id: bar
            Accessible.name: page.scrollName
            policy: page.contentHeight > page.height ? ScrollBar.AlwaysOn : ScrollBar.AlwaysOff
            contentItem: Rectangle {
                implicitWidth: 6
                radius: 3
                color: bar.pressed || bar.hovered ? root.t.text_muted : root.t.control_edge
            }
        }
        Rectangle {
            parent: page
            anchors.fill: parent
            z: -1
            radius: root.t.radiusCard
            color: page.ground
        }
        ColumnLayout {
            id: column
            x: page.leftInset
            y: 4
            width: parent.width - page.leftInset - page.rightInset
            spacing: 16
        }
    }

    // The cross that closes the inspector, drawn as two strokes so it needs no icon font.
    component CloseBtn: KeyButton {
        id: closer
        implicitWidth: 30
        implicitHeight: 30
        Accessible.role: Accessible.Button
        Accessible.description: "Esc"
        background: Rectangle {
            radius: root.t.radiusControl
            color: closer.hovered ? root.t.control_hover : "transparent"
            FocusRing { control: closer }
        }
        contentItem: Item {
            Repeater {
                model: [45, -45]
                delegate: Rectangle {
                    required property int modelData
                    anchors.centerIn: parent
                    width: 14
                    height: 2
                    radius: 1
                    rotation: modelData
                    color: closer.hovered ? root.t.hover_text : root.t.text_muted
                }
            }
        }
    }

    // One stage in the inspector: its switch, its controls and readings, and what it is for.
    component InspectorPanel: ColumnLayout {
        id: panel
        // A stage of `m.stages`, or the Input cleanup card.
        required property var modelData
        readonly property alias card: panel.modelData
        // Where keyboard focus lands when the panel opens.
        readonly property Item first: card.toggle ? onSwitch : closeButton
        signal closeRequested(int reason)
        spacing: 0
        RowLayout {
            spacing: 10
            Layout.fillWidth: true
            Layout.margins: 18
            Layout.bottomMargin: 10
            Text {
                text: panel.card.name
                textFormat: Text.PlainText
                color: root.t.text_primary
                font.pixelSize: root.fsLead
                font.weight: Font.DemiBold
                elide: Text.ElideRight
                Layout.fillWidth: true
                Accessible.role: Accessible.Heading
                Accessible.name: text
            }
            // The key that does what the cross does.
            EscChip {}
            CloseBtn {
                id: closeButton
                Accessible.name: "Close " + panel.card.name
                onClicked: panel.closeRequested(visualFocus ? Qt.TabFocusReason : Qt.MouseFocusReason)
            }
        }
        RowLayout {
            visible: !!panel.card.toggle
            spacing: 12
            Layout.fillWidth: true
            Layout.leftMargin: 18
            Layout.rightMargin: 18
            Layout.bottomMargin: 12
            // Said in words, so the state does not rest on the switch's colour.
            StatusText {
                text: panel.card.name + (panel.card.toggle && panel.card.toggle.checked ? " is on" : " is off")
                color: root.t.text_primary
                Layout.fillWidth: true
            }
            Toggle { id: onSwitch; proxy: panel.card.toggle }
        }
        Rectangle {
            implicitHeight: 1
            color: root.t.glass_edge
            Layout.fillWidth: true
        }
        Page {
            leftInset: 18
            rightInset: 12
            spacing: 14
            scrollName: panel.card.name + " scroll bar"
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.topMargin: 10
            Chip {
                proxy: panel.card.alert
                visible: !!proxy && (proxy.state === "warn" || proxy.state === "bad")
                Layout.fillWidth: true
            }
            Repeater {
                model: panel.card.rows
                delegate: ControlRow {
                    required property var modelData
                    row: modelData
                    Layout.fillWidth: true
                }
            }
            ColumnLayout {
                visible: panel.card.trace && root.ui.levels.length > 1
                spacing: 6
                Layout.fillWidth: true
                LevelTrace {
                    points: root.ui.levels
                    Layout.fillWidth: true
                    implicitHeight: 52
                    Accessible.role: Accessible.Graphic
                    Accessible.name: "Input and output level history"
                }
                Row {
                    spacing: 14
                    Repeater {
                        model: [["Input", root.t.text_muted], ["Output", root.t.accent]]
                        delegate: Row {
                            required property var modelData
                            spacing: 6
                            Rectangle {
                                width: 10
                                height: 2
                                color: parent.modelData[1]
                                anchors.verticalCenter: parent.verticalCenter
                            }
                            Text {
                                textFormat: Text.PlainText
                                text: parent.modelData[0]
                                color: root.t.text_muted
                                font.pixelSize: root.fsCaption
                            }
                        }
                    }
                }
            }
            Disclosure {
                id: more
                text: "Advanced"
                subject: panel.card.name
                visible: panel.card.advanced.length > 0
                Layout.fillWidth: true
            }
            Repeater {
                model: more.open ? panel.card.advanced : []
                delegate: ControlRow {
                    required property var modelData
                    row: modelData
                    Layout.fillWidth: true
                    FadeIn on opacity {}
                }
            }
            Chip {
                proxy: panel.card.readout ?? null
                visible: !!proxy
                Layout.fillWidth: true
            }
            BodyText {
                text: panel.card.help
                visible: text !== ""
                color: root.t.text_muted
                Layout.bottomMargin: 8
            }
        }
        Rectangle {
            implicitHeight: 1
            color: root.t.glass_edge
            Layout.fillWidth: true
        }
        RowLayout {
            spacing: 8
            Layout.fillWidth: true
            Layout.margins: 12
            IconBtn { proxy: root.m.presets.undo; glyph: root.m.glyphs.undo; visible: root.hasIcons }
            IconBtn { proxy: root.m.presets.redo; glyph: root.m.glyphs.redo; visible: root.hasIcons }
            Btn { proxy: root.m.presets.undo; visible: !root.hasIcons }
            Btn { proxy: root.m.presets.redo; visible: !root.hasIcons }
            Item { Layout.fillWidth: true }
        }
    }

    // The shared Mute button, with the scene's hover help.
    component MuteBtn: MuteButton {
        id: mute
        Hint { text: mute.proxy ? mute.proxy.toolTip : "" }
    }

    // A command shown as underlined text beside the note it belongs to.
    component LinkBtn: KeyButton {
        id: link
        property var proxy
        text: proxy ? proxy.text : ""
        enabled: proxy ? proxy.enabled : false
        opacity: enabled ? 1 : 0.45
        implicitWidth: linkText.implicitWidth
        implicitHeight: 24
        onClicked: if (proxy) root.press(this, proxy)
        Accessible.role: Accessible.Button
        Accessible.name: text
        Accessible.description: proxy ? proxy.toolTip : ""
        Hint { text: link.proxy ? link.proxy.toolTip : "" }
        background: Item { FocusRing { control: link; outset: 0 } }
        contentItem: Text {
            id: linkText
            text: link.text
            textFormat: Text.PlainText
            leftPadding: 4
            rightPadding: 4
            color: link.hovered ? root.t.hover_text : root.t.accent_text
            font.pixelSize: root.fsCaption
            font.underline: true
            verticalAlignment: Text.AlignVCenter
        }
    }

    // A stage's switch on its sidebar row; the same model as the switch on the stage card. Its name
    // is that switch's too, so the description says which of the two this is.
    component ChainSwitch: Toggle {
        Accessible.description: ["In the signal chain", proxy ? proxy.toolTip : ""].filter(part => part !== "").join(". ")
    }

    // A heading over a group in the sidebar, or over a card on a page.
    component SectionLabel: Text {
        textFormat: Text.PlainText
        color: root.t.text_primary
        font.pixelSize: root.fsBody
        font.weight: Font.DemiBold
    }

    // A line icon drawn from SVG path data on a square grid of `grid` units, so it needs no
    // icon font and follows the theme's colours.
    component LineIcon: Canvas {
        id: icon
        property string path
        property color stroke: root.t.tile_icon
        property real grid: 18
        property real lineWidth: 1.5
        implicitWidth: 16
        implicitHeight: 16
        Accessible.ignored: true
        onPathChanged: requestPaint()
        onStrokeChanged: requestPaint()
        onPaint: {
            const ctx = getContext("2d")
            ctx.reset()
            ctx.scale(width / grid, height / grid)
            ctx.lineWidth = lineWidth
            ctx.lineCap = "round"
            ctx.lineJoin = "round"
            ctx.strokeStyle = stroke
            ctx.path = path
            ctx.stroke()
        }
    }

    // The check drawn wherever something passed, on LineIcon's 12-unit grid.
    readonly property string checkPath: "M2.5 6.5L5 9l4.5-5.5"

    // A state told by shape as well as colour: a check disc when fine, a triangle with "!" to
    // look at, a disc with "!" for a fault, a ring while idle.
    component StateMark: Canvas {
        property string level: "idle"
        implicitWidth: 14
        implicitHeight: 14
        Accessible.ignored: true
        onLevelChanged: requestPaint()
        onPaint: {
            const ctx = getContext("2d")
            const s = width
            const mark = root.t.app_surface
            ctx.reset()
            ctx.lineCap = "round"
            ctx.lineJoin = "round"
            if (level === "ok" || level === "bad") {
                ctx.fillStyle = level === "ok" ? root.t.status_ok_dot : root.t.status_bad_text
                ctx.beginPath()
                ctx.arc(s / 2, s / 2, s / 2, 0, 2 * Math.PI)
                ctx.fill()
            } else if (level === "warn") {
                ctx.fillStyle = root.t.status_warn_border
                ctx.beginPath()
                ctx.moveTo(s / 2, 0.04 * s)
                ctx.lineTo(0.98 * s, 0.92 * s)
                ctx.lineTo(0.02 * s, 0.92 * s)
                ctx.closePath()
                ctx.fill()
            } else {
                ctx.strokeStyle = root.t.text_dim
                ctx.lineWidth = 2
                ctx.beginPath()
                ctx.arc(s / 2, s / 2, s / 2 - 1, 0, 2 * Math.PI)
                ctx.stroke()
                return
            }
            ctx.strokeStyle = mark
            ctx.lineWidth = Math.max(1.4, s / 9)
            ctx.beginPath()
            if (level === "ok") {
                ctx.moveTo(0.24 * s, 0.52 * s)
                ctx.lineTo(0.42 * s, 0.70 * s)
                ctx.lineTo(0.78 * s, 0.30 * s)
                ctx.stroke()
                return
            }
            const top = level === "warn" ? 0.36 : 0.24
            ctx.moveTo(s / 2, top * s)
            ctx.lineTo(s / 2, (top + 0.28) * s)
            ctx.stroke()
            ctx.fillStyle = mark
            ctx.beginPath()
            ctx.arc(s / 2, (top + 0.44) * s, ctx.lineWidth * 0.6, 0, 2 * Math.PI)
            ctx.fill()
        }
    }

    // The round mark at the start of a route check line: a check that passed, "!" to act on, or
    // "i" for a note.
    component CheckMark: Rectangle {
        property string kind: "ok"
        implicitWidth: 26
        implicitHeight: 26
        radius: 13
        color: kind === "ok" ? root.t.status_ok_fill : kind === "warn" ? root.t.status_warn_surface
                                                                       : root.t.status_note_fill
        border.color: kind === "warn" ? root.t.status_warn_border : root.t.tile_edge
        Accessible.ignored: true
        LineIcon {
            anchors.centerIn: parent
            width: 12
            height: 12
            grid: 12
            stroke: parent.kind === "ok" ? root.t.status_ok_text : parent.kind === "warn" ? root.t.status_warn_text
                                                                                           : root.t.text_muted
            path: parent.kind === "ok" ? root.checkPath : parent.kind === "warn" ? "M6 2.5v4.5M6 9.5v.2"
                                                                                       : "M6 5.5v3.5M6 3v.2"
        }
    }

    // A raised, rounded panel on the window ground: the sidebar and the main pane.
    component Panel: Rectangle {
        id: panel
        property color topColor: root.t.sidebar_top
        property color bottomColor: root.t.sidebar_bottom
        radius: root.t.radiusPanel
        border.color: root.t.panel_edge
        gradient: Gradient {
            GradientStop { position: 0; color: panel.topColor }
            GradientStop { position: 1; color: panel.bottomColor }
        }
        // The light catching the top edge.
        Rectangle {
            x: panel.radius
            y: 1
            width: parent.width - 2 * panel.radius
            height: 1
            color: root.t.panel_highlight
        }
    }

    // The sidebar row icons, as path data on an 18-unit grid, keyed as the rows' lines are.
    readonly property var rowIcons: ({
        overview: "M4 2.5h2.5a1.5 1.5 0 0 1 1.5 1.5v2.5a1.5 1.5 0 0 1 -1.5 1.5h-2.5a1.5 1.5 0 0 1 -1.5 -1.5v-2.5"
                  + "a1.5 1.5 0 0 1 1.5 -1.5zM11.5 2.5h2.5a1.5 1.5 0 0 1 1.5 1.5v2.5a1.5 1.5 0 0 1 -1.5 1.5h-2.5"
                  + "a1.5 1.5 0 0 1 -1.5 -1.5v-2.5a1.5 1.5 0 0 1 1.5 -1.5zM4 10h10a1.5 1.5 0 0 1 1.5 1.5v2.5"
                  + "a1.5 1.5 0 0 1 -1.5 1.5h-10a1.5 1.5 0 0 1 -1.5 -1.5v-2.5a1.5 1.5 0 0 1 1.5 -1.5z",
        cleanup: "M9 2a2.5 2.5 0 0 1 2.5 2.5v4a2.5 2.5 0 0 1 -2.5 2.5a2.5 2.5 0 0 1 -2.5 -2.5v-4"
                 + "a2.5 2.5 0 0 1 2.5 -2.5zM3.5 8.5a5.5 5.5 0 0 0 11 0M9 14v2.5",
        gate: "M2 12h3V6h3v8h3V4h3v8h2",
        suppression: "M2 9h2l1.5-5 2.5 10 2.5-8L12 11l1-2h3",
        deesser: "M12.5 5.5C12 4 10.5 3.5 9 3.5 7 3.5 6 4.5 6 5.8c0 3 6.5 1.5 6.5 4.6 0 1.4-1.4 2.2-3.2 2.2"
                 + "-1.6 0-3-.6-3.5-2.1",
        eq: "M5 2.5v13M9 2.5v13M13 2.5v13M6.6 11a1.6 1.6 0 1 1 -3.2 0a1.6 1.6 0 1 1 3.2 0z"
            + "M10.6 6a1.6 1.6 0 1 1 -3.2 0a1.6 1.6 0 1 1 3.2 0zM14.6 10a1.6 1.6 0 1 1 -3.2 0a1.6 1.6 0 1 1 3.2 0z",
        compressor: "M3 3.5l5 5.5-5 5.5M15 3.5L10 9l5 5.5",
        limiter: "M2.5 4.5h13M3 15c1.5 0 2-4 3.5-4s1.5 3 3 3 2-9 3.5-9"
    })

    // One sidebar row: an icon tile, a button with the name and a state line, and the stage's
    // switch if it has one.
    component NavRow: Item {
        id: nav
        property string name
        property string detail: ""
        property bool current: false
        property bool dim: false
        property var toggle: null
        // How hard the stage works, 0 to 1, drawn as a bar after the state line when metered.
        property real activity: 0
        property bool metered: false
        // The row's icon as path data; a narrow window leaves the tiles out, so names stay whole.
        property string icon: ""
        readonly property bool tiled: icon !== "" && root.wide
        readonly property real textLeft: tiled ? 54 : 12
        readonly property real textRight: !!toggle ? 54 : 12
        // The row's button, where focus goes back to when what it opened closes.
        readonly property alias button: navButton
        signal opened()
        implicitHeight: 52
        KeyButton {
            id: navButton
            readonly property bool current: nav.current
            anchors.fill: parent
            onClicked: nav.opened()
            Accessible.role: Accessible.Button
            Accessible.name: nav.name
            Accessible.description: nav.detail
            Accessible.selectable: true
            Accessible.selected: nav.current
            // Hover fills the row; keyboard focus only draws the ring.
            background: Rectangle {
                radius: 14
                color: navButton.hovered && !navButton.current ? root.t.control_hover : "transparent"
                Behavior on color { ColorAnimation { duration: root.t.motionRow } }
                // The selected row lights from its left end.
                Rectangle {
                    anchors.fill: parent
                    radius: 14
                    opacity: navButton.current ? 1 : 0
                    border.color: root.t.accent_edge
                    gradient: Gradient {
                        orientation: Gradient.Horizontal
                        GradientStop { position: 0; color: root.t.accent_row_start }
                        GradientStop { position: 1; color: root.t.accent_row_end }
                    }
                    Behavior on opacity { NumberAnimation { duration: root.t.motionRow } }
                }
                FocusRing { control: navButton; outset: -2 }
            }
            contentItem: Item {
                Rectangle {
                    visible: nav.tiled
                    x: 12
                    width: 32
                    height: 32
                    radius: 10
                    anchors.verticalCenter: parent.verticalCenter
                    // Off, the tile keeps only its edge.
                    border.color: navButton.current ? root.t.accent_tile_edge : root.t.tile_edge
                    gradient: Gradient {
                        GradientStop {
                            position: 0
                            color: navButton.current ? root.t.accent_tile_top : nav.dim ? "transparent" : root.t.tile_top
                        }
                        GradientStop {
                            position: 1
                            color: navButton.current ? root.t.accent_tile_bottom
                                                     : nav.dim ? "transparent" : root.t.tile_bottom
                        }
                    }
                    LineIcon {
                        anchors.centerIn: parent
                        path: nav.icon
                        stroke: navButton.current ? root.t.accent_bright : nav.dim ? root.t.text_dim : root.t.tile_icon
                    }
                }
                Column {
                    x: nav.textLeft
                    width: parent.width - nav.textLeft - nav.textRight
                    anchors.verticalCenter: parent.verticalCenter
                    spacing: 1
                    Text {
                        width: parent.width
                        text: nav.name
                        textFormat: Text.PlainText
                        color: nav.dim ? root.t.text_dim : root.t.text_primary
                        font.pixelSize: root.fsBody
                        font.weight: Font.DemiBold
                        elide: Text.ElideRight
                        Behavior on color { ColorAnimation { duration: root.t.motionRow } }
                    }
                    Row {
                        visible: nav.detail !== ""
                        width: parent.width
                        spacing: 8
                        Text {
                            id: navDetail
                            width: Math.min(implicitWidth, parent.width - (activityBar.visible ? activityBar.width + 8 : 0))
                            text: nav.detail
                            textFormat: Text.PlainText
                            color: nav.current ? root.t.accent_soft : nav.dim ? root.t.text_dim : root.t.text_muted
                            font.pixelSize: root.fsCaption
                            elide: Text.ElideRight
                        }
                        // Empty while the stage is idle; a stage that is off shows none.
                        Rectangle {
                            id: activityBar
                            visible: nav.metered && !nav.dim
                            width: 36
                            height: 4
                            radius: 2
                            anchors.verticalCenter: navDetail.verticalCenter
                            color: root.t.activity_track
                            Accessible.ignored: true
                            Rectangle {
                                width: parent.width * nav.activity
                                height: parent.height
                                radius: parent.radius
                                gradient: Gradient {
                                    orientation: Gradient.Horizontal
                                    GradientStop { position: 0; color: root.t.activity_low }
                                    GradientStop { position: 1; color: root.t.activity_high }
                                }
                            }
                        }
                    }
                }
            }
        }
        ChainSwitch {
            visible: !!nav.toggle
            proxy: nav.toggle
            anchors.right: parent.right
            anchors.rightMargin: 12
            anchors.verticalCenter: parent.verticalCenter
        }
    }

    // A level meter in the toolbar: its name, a thin bar, and the reading when there is room.
    component LevelReadout: RowLayout {
        id: readout
        property var meter
        property string label
        property bool valueShown: true
        spacing: 6
        // A clip is said by a word on a filled tag, as on the upright meter, not by colour alone.
        Rectangle {
            implicitWidth: root.readoutLabelWidth
            implicitHeight: readoutLabel.implicitHeight + 2
            radius: 3
            color: readout.meter.clipTagShown ? root.t.meter_clip : "transparent"
            Caption {
                id: readoutLabel
                anchors.centerIn: parent
                text: readout.meter.clipTagShown ? "CLIP" : readout.label
                color: readout.meter.clipTagShown ? root.t.text_on_data_color : root.t.text_muted
                font.pixelSize: root.fsCaption
                font.weight: Font.DemiBold
                Accessible.ignored: true
            }
        }
        QuickMeterItem {
            source: readout.meter
            horizontal: true
            implicitHeight: 6
            Layout.preferredWidth: root.readoutBarWidth
            Accessible.role: Accessible.ProgressBar
            Accessible.name: readout.meter.name
            // The reading first; it changes only when the whole-dB value or the words do.
            Accessible.description: readout.meter.accessibleText
        }
        Caption {
            visible: readout.valueShown
            text: readout.meter.valueText
            font.pixelSize: root.fsCaption
            font.features: { "tnum": 1 }
            Layout.preferredWidth: root.readoutValueWidth
            Accessible.ignored: true
        }
    }

    // A chain row opens its own page, or else its stage's inspector beside the Overview; the row of
    // the open inspector closes it.
    function openChainRow(row, button) {
        const reason = button.visualFocus ? Qt.TabFocusReason : Qt.MouseFocusReason
        if (row.page >= 0) {
            m.page.setIndex(row.page)
        } else if (inspecting && m.inspector.stageKey === row.key) {
            closeInspector(reason)
        } else {
            m.page.setIndex(0)
            inspectorOrigin = button
            m.inspector.show(row.key)
            const panel = inspectorPanel(row.key)
            if (panel)
                panel.first.forceActiveFocus(reason)
        }
    }

    function inspectorPanel(key) {
        for (let i = 0; i < inspectorPanels.count; ++i) {
            const panel = inspectorPanels.itemAt(i) as InspectorPanel
            if (panel?.card?.key === key)
                return panel
        }
        return null
    }

    // Focus goes back to the row that opened the inspector, or else to the Overview row. It moves
    // before the close, so the close finds it outside the panel.
    function closeInspector(reason) {
        returnFocus(reason)
        m.inspector.close()
    }

    function returnFocus(reason) {
        (inspectorOrigin ?? overviewRow.button).forceActiveFocus(reason)
    }

    // The toolbar meters keep their size whatever they read, so the row does not shift.
    TextMetrics { id: clipMetrics; text: "CLIP"; font.pixelSize: root.fsCaption; font.weight: Font.DemiBold }
    TextMetrics { id: outMetrics; text: "OUT"; font.pixelSize: root.fsCaption; font.weight: Font.DemiBold }
    TextMetrics { id: valueMetrics; text: "no signal"; font.pixelSize: root.fsCaption }
    readonly property real readoutLabelWidth: Math.ceil(Math.max(clipMetrics.width, outMetrics.width)) + 8
    readonly property real readoutValueWidth: Math.ceil(valueMetrics.width) + 2
    readonly property int readoutBarWidth: 72
    readonly property real readoutWidth: readoutLabelWidth + readoutBarWidth + readoutValueWidth + 12

    // Toolbar: the route, the mode, Mute, the levels, the latency and the transport in one row.
    Rectangle {
        id: toolbar
        objectName: "toolbar"
        readonly property int gap: 10
        // The row with every part shown. In less room the brand, the mode's key word, the dB
        // readings and the latency caption give way, and the two meters stack.
        readonly property real fullWidth: toolRow.anchors.leftMargin + toolRow.anchors.rightMargin + 10 * gap
            + brand.implicitWidth + 2 * 140 + refresh.implicitWidth
            + modeKey.width + 6 + modeMetrics.width + 36
            + muteButton.implicitWidth + 2 * root.readoutWidth + meterGrid.columnSpacing
            + Math.max(latencyFigure.implicitWidth, latencyCaption.implicitWidth)
            + 1 + transport.implicitWidth + reasonWidth
        readonly property bool full: fullWidth <= width
        // Measured on one line, so the reason's own wrapping cannot feed back into the layout. The
        // laid-out line can run a pixel past the metric, which would wrap its last word.
        readonly property real reasonWidth: startBlocked.visible ? Math.ceil(reasonMetrics.width) + 2 : 0
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.top: parent.top
        // Taller only when large text stacks the meters past the usual height.
        height: Math.max(56, toolRow.implicitHeight + 8)
        gradient: Gradient {
            GradientStop { position: 0; color: root.t.toolbar_top }
            GradientStop { position: 1; color: root.t.toolbar_bottom }
        }
        Rectangle {
            anchors.bottom: parent.bottom
            width: parent.width
            height: 1
            color: root.t.border
        }
        TextMetrics { id: modeKey; text: "Mode"; font.pixelSize: root.fsBody }
        TextMetrics { id: reasonMetrics; text: startBlocked.text; font: startBlocked.font }
        // The widest choice, bold as a choice that changes the sound is shown.
        TextMetrics {
            id: modeMetrics
            text: root.m.top.mode.items.reduce((wide, item) => item.length > wide.length ? item : wide, "")
            font.pixelSize: root.fsBody
            font.weight: Font.DemiBold
        }
        RowLayout {
            id: toolRow
            objectName: "toolRow"
            anchors.fill: parent
            anchors.leftMargin: 16
            anchors.rightMargin: 12
            spacing: toolbar.gap
            Text {
                id: brand
                visible: toolbar.full
                text: "AudioForge"
                textFormat: Text.PlainText
                color: root.t.text_primary
                font.pixelSize: root.fsLead
                font.weight: Font.DemiBold
            }
            Picker {
                proxy: root.m.top.input
                glyph: root.hasIcons ? root.m.glyphs.mic : ""
                Layout.fillWidth: true
                Layout.preferredWidth: 240
                Layout.maximumWidth: 260
                // The pickers give up width first; a cut name is whole in the tooltip.
                Layout.minimumWidth: toolbar.full ? 140 : 100
            }
            Picker {
                proxy: root.m.top.output
                glyph: root.hasIcons ? root.m.glyphs.speaker : ""
                Layout.fillWidth: true
                Layout.preferredWidth: 240
                Layout.maximumWidth: 260
                Layout.minimumWidth: toolbar.full ? 140 : 100
            }
            IconBtn { id: refresh; proxy: root.m.top.refresh; glyph: root.m.glyphs.refresh; label: "Refresh" }
            Picker {
                proxy: root.m.top.mode
                key: toolbar.full ? "Mode" : ""
                alert: root.m.top.mode.index > 0
                implicitWidth: (key !== "" ? modeKey.width + 6 : 0) + modeMetrics.width + 36
            }
            // The levels and the latency sit between Mute and Start, so one slip cannot hit both.
            MuteBtn { id: muteButton; proxy: root.m.top.mute }
            GridLayout {
                id: meterGrid
                columns: toolbar.full ? 2 : 1
                columnSpacing: 14
                rowSpacing: 4
                Repeater {
                    model: root.m.meters
                    delegate: LevelReadout {
                        required property var modelData
                        required property int index
                        meter: modelData
                        label: index === 0 ? "IN" : "OUT"
                        valueShown: toolbar.full
                    }
                }
            }
            Item {
                implicitWidth: latencyColumn.implicitWidth
                implicitHeight: latencyColumn.implicitHeight
                Hint { text: root.m.top.latency.toolTip }
                Column {
                    id: latencyColumn
                    Text {
                        id: latencyFigure
                        text: root.m.top.latency.text
                        textFormat: Text.PlainText
                        color: root.t.text_primary
                        font.pixelSize: root.fsBody
                        font.weight: Font.DemiBold
                        font.features: { "tnum": 1 }
                        Accessible.role: Accessible.StaticText
                        Accessible.name: latencyCaption.text + ": " + text
                    }
                    Text {
                        id: latencyCaption
                        visible: toolbar.full
                        text: root.m.top.latency.data.measured ? "Latency" : "Latency, not measured"
                        textFormat: Text.PlainText
                        color: root.t.text_muted
                        font.pixelSize: root.fsCaption
                        Accessible.ignored: true
                    }
                }
            }
            Item {
                id: startReason
                Layout.fillWidth: true
                // A row that fits keeps the reason on one line. In a narrow window it wraps onto two
                // lines rather than vanish; cut short with large text, it is whole in the tooltip.
                Layout.minimumWidth: toolbar.full ? toolbar.reasonWidth : Math.min(toolbar.reasonWidth, 80)
                implicitWidth: toolbar.reasonWidth
                implicitHeight: startBlocked.height
                StatusText {
                    id: startBlocked
                    objectName: "startBlockedReason"
                    visible: root.m.overview.state === "no_mic" && !root.running
                    text: root.m.startBlockedReason
                    width: Math.min(toolbar.reasonWidth, parent.width)
                    anchors.right: parent.right
                    anchors.verticalCenter: parent.verticalCenter
                    horizontalAlignment: Text.AlignRight
                    font.pixelSize: root.fsCaption
                    wrapMode: Text.WordWrap
                    maximumLineCount: 2
                    elide: Text.ElideRight
                    tip: truncated ? text : ""
                }
            }
            Rectangle {
                implicitWidth: 1
                implicitHeight: 28
                color: root.t.border_strong
            }
            Btn {
                id: transport
                proxy: root.running ? root.m.top.stop : root.m.top.start
                text: root.running ? "Stop" : "Start"
                look: root.running ? "danger" : "primary"
                implicitWidth: Math.max(92, contentItem.implicitWidth + 28)
                implicitHeight: 34
                Accessible.name: proxy.text
            }
        }
    }

    // The sidebar and the main pane stand apart on the window ground by this gap.
    readonly property int panelGap: 10

    // The Health line in the sidebar footer: a mark whose shape says the state, and the reading.
    component HealthChip: Item {
        id: chip
        property var proxy
        readonly property string level: proxy && proxy.state !== "" ? proxy.state : "idle"
        readonly property string shown: proxy ? proxy.text : ""
        implicitHeight: 28
        implicitWidth: mark.width + 8 + healthText.implicitWidth
        Hint {
            text: chip.proxy
                  ? [healthText.truncated ? healthText.text : "", chip.proxy.toolTip].filter(t => t !== "").join("\n\n")
                  : ""
        }
        Text {
            id: healthText
            x: mark.width + 8
            width: Math.min(implicitWidth, chip.width - x)
            anchors.verticalCenter: parent.verticalCenter
            textFormat: Text.PlainText
            text: chip.shown
            color: chip.level === "warn" || chip.level === "bad" ? root.t["status_" + chip.level + "_text"]
                                                                 : root.t.text_primary
            font.pixelSize: root.fsCaption
            font.weight: Font.DemiBold
            elide: Text.ElideRight
            Accessible.ignored: true
        }
        StateMark {
            id: mark
            level: chip.level
            anchors.verticalCenter: parent.verticalCenter
        }
    }

    // Sidebar: the preset, the pages and the signal chain, and the health of the stream.
    Panel {
        id: sidebar
        objectName: "sidebar"
        anchors.left: parent.left
        anchors.top: toolbar.bottom
        anchors.bottom: parent.bottom
        anchors.margins: root.panelGap
        width: root.wide ? 256 : 220
        ColumnLayout {
            anchors.fill: parent
            anchors.topMargin: 22
            anchors.leftMargin: 8
            anchors.rightMargin: 8
            anchors.bottomMargin: 12
            spacing: 12

            ColumnLayout {
                id: presetBlock
                objectName: "presetBlock"
                readonly property var preset: root.m.presets.status.data
                spacing: 8
                Layout.fillWidth: true
                Layout.leftMargin: 12
                Layout.rightMargin: 12
                Accessible.name: root.m.presets.status.name + ": " + root.m.presets.status.text
                SectionLabel { text: "Preset" }
                RowLayout {
                    spacing: 8
                    Layout.fillWidth: true
                    Text {
                        objectName: "presetName"
                        textFormat: Text.PlainText
                        text: presetBlock.preset.name
                        color: root.t.text_primary
                        font.pixelSize: root.fsLead
                        font.weight: Font.DemiBold
                        elide: Text.ElideRight
                        Layout.fillWidth: true
                        Layout.maximumWidth: implicitWidth
                        // A name cut short is shown whole in the tooltip.
                        Hint { text: parent.truncated ? parent.text : "" }
                    }
                    // Saved is a check and a word; unsaved changes keep their warning chip.
                    Row {
                        visible: !presetBlock.preset.modified
                        spacing: 4
                        LineIcon {
                            width: 11
                            height: 11
                            grid: 12
                            stroke: root.t.status_ok_text
                            path: root.checkPath
                            anchors.verticalCenter: parent.verticalCenter
                        }
                        Text {
                            textFormat: Text.PlainText
                            text: "Saved"
                            color: root.t.status_ok_text
                            font.pixelSize: root.fsCaption
                        }
                    }
                    Rectangle {
                        visible: presetBlock.preset.modified
                        implicitWidth: stateText.implicitWidth + 16
                        implicitHeight: 20
                        radius: 10
                        color: root.t.status_warn_surface
                        border.color: root.t.status_warn_border
                        Text {
                            id: stateText
                            textFormat: Text.PlainText
                            anchors.centerIn: parent
                            text: "Unsaved changes"
                            color: root.t.status_warn_text
                            font.pixelSize: root.fsCaption
                        }
                    }
                    Item { Layout.fillWidth: true }
                }
                Flow {
                    id: presetButtons
                    spacing: 8
                    Layout.fillWidth: true
                    Layout.topMargin: 4
                    // With the undo and redo icons beside it, Presets takes the rest of the row.
                    Btn {
                        proxy: root.m.presets.menu
                        width: root.hasIcons ? Math.max(implicitWidth, presetButtons.width - undoIcon.width - redoIcon.width
                                                        - 2 * presetButtons.spacing)
                                             : implicitWidth
                    }
                    IconBtn { id: undoIcon; proxy: root.m.presets.undo; glyph: root.m.glyphs.undo; visible: root.hasIcons }
                    IconBtn { id: redoIcon; proxy: root.m.presets.redo; glyph: root.m.glyphs.redo; visible: root.hasIcons }
                    // Without a system icon font the two are spelled out.
                    Btn { proxy: root.m.presets.undo; visible: !root.hasIcons }
                    Btn { proxy: root.m.presets.redo; visible: !root.hasIcons }
                }
            }

            Page {
                id: navigation
                leftInset: 0
                // Room for the scroll bar only when there is one.
                rightInset: contentHeight > height ? 8 : 0
                spacing: 2
                scrollName: "Navigation scroll bar"
                Layout.fillWidth: true
                Layout.fillHeight: true
                Layout.topMargin: 8
                Accessible.role: Accessible.Grouping
                Accessible.name: "Navigation"
                NavRow {
                    id: overviewRow
                    name: "Overview"
                    detail: root.m.overview.navLine
                    icon: root.rowIcons.overview
                    current: root.m.page.index === 0 && !root.inspecting
                    onOpened: {
                        root.m.inspector.close()
                        root.m.page.setIndex(0)
                    }
                    Layout.fillWidth: true
                }
                SectionLabel {
                    text: "Signal chain"
                    Layout.topMargin: 16
                    Layout.bottomMargin: 6
                    Layout.leftMargin: 12
                }
                Repeater {
                    model: root.m.sidebar.chain
                    delegate: NavRow {
                        id: chainRow
                        required property var modelData
                        readonly property var line: root.m.sidebar.lines.lines[modelData.line]
                        name: modelData.name
                        toggle: modelData.toggle
                        detail: line.line
                        activity: line.activity
                        metered: line.metered
                        icon: root.rowIcons[modelData.line] ?? ""
                        dim: !!modelData.toggle && !modelData.toggle.checked
                        current: modelData.page >= 0 ? root.m.page.index === modelData.page
                                                     : root.inspecting && root.m.inspector.stageKey === modelData.key
                        onOpened: root.openChainRow(modelData, chainRow.button)
                        Layout.fillWidth: true
                    }
                }
            }

            ColumnLayout {
                id: footer
                objectName: "sidebarFooter"
                spacing: 8
                Layout.fillWidth: true
                Rectangle {
                    implicitHeight: 1
                    color: root.t.panel_edge
                    Layout.fillWidth: true
                    Layout.bottomMargin: 4
                }
                HealthChip {
                    id: healthChip
                    objectName: "healthChip"
                    proxy: root.m.status.health
                    // A long reading is cut short; the tooltip and the name carry all of it.
                    Layout.maximumWidth: footer.width - 24
                    Layout.leftMargin: 12
                    ModelButton {
                        id: healthToggle
                        objectName: "healthToggle"
                        anchors.fill: parent
                        on: root.m.page.index === root.m.healthPage
                        onRequested: root.m.page.setIndex(checked ? root.m.healthPage : 0)
                        Accessible.role: Accessible.Button
                        // Named by what it shows; the action goes in the description.
                        Accessible.name: healthChip.shown
                        Accessible.description: checked ? "Hide audio health details" : "Show audio health details"
                        background: Item { FocusRing { control: healthToggle } }
                    }
                }
                StatusText {
                    objectName: "statusCalibration"
                    text: root.m.status.calibration.text
                    font.pixelSize: root.fsCaption
                    wrapMode: Text.Wrap
                    Layout.fillWidth: true
                    Layout.topMargin: 6
                    Layout.leftMargin: 12
                    Layout.rightMargin: 12
                    tip: truncated ? text : ""
                }
                // Tight enough that the link and the gear share the narrow sidebar at a large font.
                RowLayout {
                    spacing: 4
                    Layout.fillWidth: true
                    Layout.leftMargin: 12
                    LinkBtn { proxy: root.m.sidebar.measureLatency; Layout.leftMargin: -4 }
                    Item { Layout.fillWidth: true }
                    IconBtn {
                        glyph: root.m.glyphs.settings
                        label: "Settings"
                        hint: "Settings"
                        onClicked: root.m.page.setIndex(root.m.settingsPage)
                        Accessible.selectable: true
                        Accessible.selected: root.m.page.index === root.m.settingsPage
                    }
                }
            }
        }
    }

    // The main pane's raised panel, with a cool sheen across its top. It gives way to the rail
    // of a docked inspector.
    Panel {
        id: mainPanel
        objectName: "mainPanel"
        anchors.left: sidebar.right
        anchors.right: parent.right
        anchors.top: toolbar.bottom
        anchors.bottom: parent.bottom
        anchors.margins: root.panelGap
        anchors.rightMargin: root.panelGap + root.inspectorRoom
        topColor: root.t.pane_top
        bottomColor: root.t.pane_bottom
        Canvas {
            id: sheen
            readonly property color glow: root.t.pane_glow
            anchors.fill: parent
            Accessible.ignored: true
            // A resize repaints a Canvas by itself; a theme change does not.
            onGlowChanged: requestPaint()
            onPaint: {
                const ctx = getContext("2d")
                ctx.reset()
                const r = root.t.radiusPanel
                ctx.beginPath()
                ctx.roundedRect(0, 0, width, height, r, r)
                ctx.clip()
                // An ellipse 700 by 260 across, centred above the top edge.
                const squash = 260 / 700
                ctx.translate(width * 0.3, -60)
                ctx.scale(1, squash)
                const sheenFill = ctx.createRadialGradient(0, 0, 0, 0, 0, 700)
                sheenFill.addColorStop(0, glow)
                sheenFill.addColorStop(0.7, "transparent")
                ctx.fillStyle = sheenFill
                ctx.fillRect(-width * 0.3, 0, width, (height + 60) / squash)
            }
        }
    }

    ColumnLayout {
        id: mainPane
        objectName: "mainPane"
        anchors.left: mainPanel.left
        anchors.right: mainPanel.right
        anchors.top: mainPanel.top
        anchors.bottom: mainPanel.bottom
        anchors.leftMargin: root.wide ? 24 : 12
        anchors.rightMargin: root.wide ? 16 : 4
        anchors.topMargin: root.wide ? 26 : 16
        anchors.bottomMargin: 8
        spacing: 12

        // Under the toolbar, so Start and Mute stay where they are when one appears.
        Repeater {
            model: root.m.banners
            delegate: Rectangle {
                id: banner
                required property var modelData
                visible: modelData.shown
                Layout.fillWidth: true
                Layout.leftMargin: 8
                Layout.rightMargin: 16
                implicitHeight: bannerText.implicitHeight + 16
                radius: root.t.radiusControl
                color: root.t.status_warn_surface
                border.color: root.t.status_warn_border
                Text {
                    id: bannerText
                    textFormat: Text.PlainText
                    anchors.fill: parent
                    anchors.margins: 8
                    text: banner.modelData.text
                    color: root.t.status_warn_text
                    font.pixelSize: root.fsBody
                    wrapMode: Text.Wrap
                    Accessible.role: Accessible.AlertMessage
                    Accessible.name: banner.modelData.name + ": " + text
                }
            }
        }

        StackLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            currentIndex: root.m.page.index

            Page {
                id: overviewPage
                objectName: "overviewPage"
                readonly property var ov: root.m.overview
                spacing: 20

                // The status sentence: the state in words, beside a mark whose shape and colour only
                // repeat them.
                ColumnLayout {
                    spacing: 4
                    Layout.fillWidth: true
                    Layout.topMargin: 4
                    RowLayout {
                        spacing: 12
                        Layout.fillWidth: true
                        Item {
                            implicitWidth: 20
                            implicitHeight: 20
                            Layout.alignment: Qt.AlignTop
                            Layout.topMargin: Math.max(0, (headline.font.pixelSize * 1.3 - 20) / 2)
                            Accessible.ignored: true
                            // A soft halo once there is a state to tell.
                            Rectangle {
                                visible: !statusDot.idle
                                anchors.fill: parent
                                radius: width / 2
                                color: Qt.rgba(statusDot.color.r, statusDot.color.g, statusDot.color.b, 0.14)
                            }
                            // Idle is an open ring; fine a disc, an issue a triangle, a fault or lost
                            // callers a diamond, muted a square.
                            Canvas {
                                id: statusDot
                                objectName: "overviewToneDot"
                                property color color: ({
                                    ok: root.t.status_ok_text,
                                    warn: root.t.status_warn_text,
                                    bad: root.t.status_bad_text,
                                    mute: root.t.mute,
                                    callers: root.t.callers_affected
                                })[overviewPage.ov.tone] ?? root.t.control_edge
                                readonly property string tone: overviewPage.ov.tone
                                readonly property bool idle: !["ok", "warn", "bad", "mute", "callers"].includes(tone)
                                anchors.centerIn: parent
                                width: idle ? 16 : 11
                                height: width
                                Behavior on color { ColorAnimation { duration: root.t.motionStatus } }
                                onColorChanged: requestPaint()
                                onIdleChanged: requestPaint()
                                onToneChanged: requestPaint()
                                onPaint: {
                                    const ctx = getContext("2d")
                                    const s = width
                                    ctx.reset()
                                    ctx.fillStyle = color
                                    ctx.strokeStyle = color
                                    ctx.beginPath()
                                    if (idle) {
                                        ctx.lineWidth = 2
                                        ctx.arc(s / 2, s / 2, s / 2 - 1, 0, 2 * Math.PI)
                                        ctx.stroke()
                                        return
                                    }
                                    if (tone === "warn") {
                                        ctx.moveTo(s / 2, 0)
                                        ctx.lineTo(s, s)
                                        ctx.lineTo(0, s)
                                    } else if (tone === "bad" || tone === "callers") {
                                        ctx.moveTo(s / 2, 0)
                                        ctx.lineTo(s, s / 2)
                                        ctx.lineTo(s / 2, s)
                                        ctx.lineTo(0, s / 2)
                                    } else if (tone === "mute") {
                                        ctx.rect(0.5, 0.5, s - 1, s - 1)
                                    } else {
                                        ctx.arc(s / 2, s / 2, s / 2, 0, 2 * Math.PI)
                                    }
                                    ctx.closePath()
                                    ctx.fill()
                                }
                            }
                        }
                        StatusText {
                            id: headline
                            objectName: "overviewHeadline"
                            text: overviewPage.ov.headline
                            color: overviewPage.ov.callersAffected ? root.t.callers_affected_text : root.t.text_primary
                            font.pixelSize: root.fsDisplay
                            font.weight: Font.DemiBold
                            wrapMode: Text.Wrap
                            Layout.fillWidth: true
                            onTextChanged: settle.restart()
                            NumberAnimation on opacity {
                                id: settle
                                from: 0.35
                                to: 1
                                duration: root.t.motionStatus
                                easing.type: Easing.OutCubic
                            }
                        }
                    }
                    StatusText {
                        objectName: "overviewDetail"
                        text: overviewPage.ov.detail
                        font.pixelSize: root.fsBody
                        wrapMode: Text.Wrap
                        Layout.fillWidth: true
                        Layout.leftMargin: 32
                    }
                    // The way to where clipping can be fixed, or to leave it for this run.
                    Rectangle {
                        objectName: "overviewClippingNotice"
                        visible: overviewPage.ov.showClippingNotice
                        implicitHeight: noticeBody.implicitHeight + 24
                        radius: root.t.radiusControl
                        color: root.t.status_bad_surface
                        border.color: root.t.status_bad_border
                        Accessible.role: Accessible.Grouping
                        Accessible.name: "Clipping actions"
                        Layout.fillWidth: true
                        Layout.leftMargin: 32
                        Layout.rightMargin: 16
                        Layout.topMargin: 10
                        ColumnLayout {
                            id: noticeBody
                            spacing: 8
                            anchors { left: parent.left; right: parent.right; top: parent.top; margins: 12 }
                            StatusText {
                                objectName: "overviewSoundSettingsLine"
                                visible: overviewPage.ov.offerSoundSettings
                                text: overviewPage.ov.soundSettingsLine
                                color: root.t.status_bad_text
                                font.pixelSize: root.fsCaption
                                wrapMode: Text.Wrap
                                Layout.fillWidth: true
                            }
                            StatusText {
                                objectName: "overviewSoundSettingsFailure"
                                visible: text !== ""
                                text: overviewPage.ov.soundSettingsFailure
                                color: root.t.status_bad_text
                                font.pixelSize: root.fsBody
                                font.weight: Font.DemiBold
                                wrapMode: Text.Wrap
                                Accessible.role: Accessible.AlertMessage
                                Layout.fillWidth: true
                            }
                            RowLayout {
                                spacing: 8
                                Layout.fillWidth: true
                                PushButton {
                                    objectName: "overviewShowMeHow"
                                    visible: overviewPage.ov.offerSoundSettings
                                    text: "Show me how"
                                    look: "primary"
                                    onClicked: overviewPage.ov.openSoundSettings()
                                }
                                PushButton {
                                    objectName: "overviewRemindLater"
                                    text: "Remind me later"
                                    onClicked: overviewPage.ov.dismissNotice()
                                }
                            }
                        }
                    }
                    // The words carry the news; the pink fill only repeats it.
                    Rectangle {
                        objectName: "overviewCallersNotice"
                        visible: overviewPage.ov.callersAffected
                        Layout.fillWidth: true
                        Layout.leftMargin: 32
                        Layout.topMargin: 10
                        implicitHeight: callersNoticeText.implicitHeight + 16
                        radius: root.t.radiusControl
                        color: root.t.callers_affected_fill
                        border.color: root.t.callers_affected_edge
                        Accessible.role: Accessible.AlertMessage
                        Accessible.name: callersNoticeText.text
                        Text {
                            id: callersNoticeText
                            objectName: "overviewCallersNoticeText"
                            textFormat: Text.PlainText
                            x: 8
                            y: 8
                            width: parent.width - 16
                            text: overviewPage.ov.callersLine
                            color: root.t.callers_affected_text
                            font.pixelSize: root.fsBody
                            wrapMode: Text.Wrap
                            Accessible.ignored: true
                        }
                    }
                    // Callers hear nothing here already, so neither action asks for confirmation.
                    ColumnLayout {
                        visible: overviewPage.ov.offerRestart || overviewPage.ov.offerChooseMicrophone
                        spacing: 6
                        Layout.fillWidth: true
                        Layout.leftMargin: 32
                        Layout.topMargin: 10
                        RowLayout {
                            spacing: 8
                            Btn {
                                objectName: "overviewRestart"
                                visible: overviewPage.ov.offerRestart
                                proxy: root.m.top.restart
                                look: "primary"
                            }
                            Btn {
                                objectName: "overviewChooseMicrophone"
                                visible: overviewPage.ov.offerChooseMicrophone
                                proxy: root.m.top.chooseMicrophone
                            }
                        }
                        Caption {
                            objectName: "overviewRestartLine"
                            visible: overviewPage.ov.offerRestart
                            text: "Stops the audio and starts it again with the same microphone and output."
                            wrapMode: Text.Wrap
                            Layout.fillWidth: true
                        }
                        Caption {
                            objectName: "overviewChooseMicrophoneLine"
                            visible: overviewPage.ov.offerChooseMicrophone
                            text: "Pick another microphone from the Input list; the audio starts on it, and it stays your choice."
                            wrapMode: Text.Wrap
                            Layout.fillWidth: true
                        }
                    }
                }

                // What AudioForge checked, and for a missing microphone the way out. The heading
                // stands over the card; a status after its dot ("All set") reads as a check.
                ColumnLayout {
                    id: checksBlock
                    // The heading, and after its first separator the status ("All set").
                    readonly property int cut: overviewPage.ov.checksTitle.indexOf(" \u00b7 ")
                    readonly property string heading: cut < 0 ? overviewPage.ov.checksTitle
                                                              : overviewPage.ov.checksTitle.slice(0, cut)
                    readonly property string status: cut < 0 ? "" : overviewPage.ov.checksTitle.slice(cut + 3)
                    visible: overviewPage.ov.checks.length > 0
                    spacing: 10
                    Layout.fillWidth: true
                    RowLayout {
                        spacing: 14
                        Layout.leftMargin: 2
                        SectionLabel {
                            text: checksBlock.heading
                            Accessible.role: Accessible.Heading
                            Accessible.name: overviewPage.ov.checksTitle
                        }
                        Row {
                            visible: checksBlock.status !== ""
                            spacing: 5
                            LineIcon {
                                width: 12
                                height: 12
                                grid: 12
                                stroke: root.t.status_ok_text
                                path: root.checkPath
                                anchors.verticalCenter: parent.verticalCenter
                            }
                            Text {
                                textFormat: Text.PlainText
                                text: checksBlock.status
                                color: root.t.status_ok_text
                                font.pixelSize: root.fsCaption
                                Accessible.ignored: true
                            }
                        }
                    }
                    Surface {
                        objectName: "overviewChecks"
                        pad: 0
                        spacing: 0
                        Layout.fillWidth: true
                        Repeater {
                            model: overviewPage.ov.checks
                            delegate: Item {
                                id: check
                                required property var modelData
                                required property int index
                                // The device a line names reads quieter than its label.
                                readonly property bool split: modelData.detail !== ""
                                implicitHeight: Math.max(46, checkLine.implicitHeight + 20)
                                Layout.fillWidth: true
                                Rectangle {
                                    visible: check.index > 0
                                    x: 18
                                    width: parent.width - 36
                                    height: 1
                                    color: root.t.panel_edge
                                }
                                CheckMark {
                                    id: checkMark
                                    kind: check.modelData.kind
                                    x: 18
                                    anchors.verticalCenter: parent.verticalCenter
                                }
                                RowLayout {
                                    id: checkLine
                                    spacing: 6
                                    anchors.left: checkMark.right
                                    anchors.leftMargin: 12
                                    anchors.right: parent.right
                                    anchors.rightMargin: 18
                                    anchors.verticalCenter: parent.verticalCenter
                                    StatusText {
                                        id: checkText
                                        text: check.modelData.label
                                        color: check.modelData.kind === "note" ? root.t.text_muted : root.t.text_primary
                                        font.pixelSize: root.fsLead
                                        wrapMode: Text.Wrap
                                        elide: Text.ElideNone
                                        Accessible.name: check.modelData.text
                                        Layout.maximumWidth: checkLine.width
                                    }
                                    Text {
                                        visible: check.split
                                        text: "\u00b7 " + check.modelData.detail
                                        textFormat: Text.PlainText
                                        color: root.t.text_muted
                                        font.pixelSize: root.fsLead
                                        wrapMode: Text.Wrap
                                        Layout.fillWidth: true
                                        Accessible.ignored: true
                                    }
                                    Item { visible: !check.split; Layout.fillWidth: true }
                                }
                            }
                        }
                        ColumnLayout {
                            visible: overviewPage.ov.state === "no_mic"
                            spacing: 10
                            Layout.fillWidth: true
                            Layout.margins: 18
                            Layout.topMargin: 6
                            Btn {
                                objectName: "overviewRefresh"
                                proxy: root.m.top.refresh
                                text: "Refresh audio devices"
                                look: "primary"
                            }
                            Picker {
                                proxy: root.m.top.input
                                glyph: root.hasIcons ? root.m.glyphs.mic : ""
                                Layout.preferredWidth: 320
                                Layout.maximumWidth: 320
                            }
                            Caption {
                                text: "Pick your microphone here once it appears, then press Start."
                                wrapMode: Text.Wrap
                                Layout.fillWidth: true
                            }
                        }
                    }
                }

                // Ten seconds of level: the microphone above the line, what others hear below.
                ColumnLayout {
                    spacing: 10
                    Layout.fillWidth: true
                    RowLayout {
                        spacing: 16
                        Layout.fillWidth: true
                        Layout.leftMargin: 2
                        StatusText {
                            objectName: "ribbonCaption"
                            text: overviewPage.ov.ribbonCaption
                            color: root.t.text_primary
                            font.pixelSize: root.fsBody
                            font.weight: Font.DemiBold
                            elide: Text.ElideRight
                            Layout.fillWidth: true
                        }
                        // Which colour is which, with the level while the audio runs; the card
                        // spells the halves out.
                        Repeater {
                            model: [[root.t.signal_in, "IN"], [root.t.signal_out, "OUT"]]
                            delegate: Row {
                                id: key
                                required property var modelData
                                required property int index
                                spacing: 6
                                Rectangle {
                                    width: 8
                                    height: 8
                                    radius: 4
                                    color: key.modelData[0]
                                    anchors.verticalCenter: parent.verticalCenter
                                }
                                Caption {
                                    text: key.modelData[1] + " " + (root.running ? root.m.meters[key.index].valueText
                                                                                 : "\u2014")
                                    color: key.modelData[0]
                                    font.pixelSize: root.fsCaption
                                    font.features: { "tnum": 1 }
                                }
                            }
                        }
                    }
                    Surface {
                        objectName: "levelRibbonCard"
                        pad: 12
                        spacing: 0
                        Layout.fillWidth: true
                        Item {
                            id: ribbonBox
                            implicitHeight: root.wide ? 90 : 72
                            clip: true
                            Layout.fillWidth: true
                            // One sample wider than the box, slid by that sample between readings.
                            Item {
                                id: ribbonStrip
                                readonly property real step: ribbonBox.width / (root.m.ribbonPoints - 1)
                                readonly property var levels: root.ui.levels
                                width: ribbonBox.width + step
                                height: ribbonBox.height
                                // Hidden with its page, it takes each reading without sliding.
                                onLevelsChanged: if (visible) scroll.restart()
                                NumberAnimation on x {
                                    id: scroll
                                    objectName: "ribbonScroll"
                                    running: false
                                    from: 0
                                    to: -ribbonStrip.step
                                    duration: Look.movement(root.m.levelIntervalMs)
                                }
                                LevelRibbon {
                                    objectName: "levelRibbon"
                                    // Nothing is measured while the audio is stopped.
                                    visible: root.running
                                    width: ribbonStrip.width
                                    height: ribbonStrip.height
                                    points: ribbonStrip.levels
                                    Accessible.role: Accessible.Graphic
                                    Accessible.name: "Level, last 10 seconds"
                                }
                            }
                            // Which half is which: the microphone above the line, what others hear below.
                            Caption {
                                objectName: "ribbonInLabel"
                                text: "IN \u00b7 your microphone"
                                color: root.t.signal_in
                                font.pixelSize: root.fsCaption
                            }
                            Caption {
                                objectName: "ribbonOutLabel"
                                anchors.bottom: parent.bottom
                                text: "OUT \u00b7 what others hear"
                                color: root.t.signal_out
                                font.pixelSize: root.fsCaption
                            }
                            // Stopped: the centre line and a note of when the trace begins.
                            Rectangle {
                                visible: !root.running
                                width: parent.width
                                height: 1
                                anchors.verticalCenter: parent.verticalCenter
                                color: root.t.data_grid
                            }
                            Rectangle {
                                objectName: "ribbonWaiting"
                                visible: !root.running
                                anchors.centerIn: parent
                                implicitWidth: waitingRow.implicitWidth + 32
                                implicitHeight: 34
                                radius: height / 2
                                color: root.t.pill_surface
                                border.color: root.t.pill_edge
                                Row {
                                    id: waitingRow
                                    anchors.centerIn: parent
                                    spacing: 10
                                    Rectangle {
                                        width: 22
                                        height: 22
                                        radius: 11
                                        color: root.t.accent_fill
                                        anchors.verticalCenter: parent.verticalCenter
                                        LineIcon {
                                            anchors.centerIn: parent
                                            width: 10
                                            height: 10
                                            grid: 12
                                            stroke: root.t.accent_bright
                                            path: "M3.5 2l6.5 4-6.5 4z"
                                        }
                                    }
                                    StatusText {
                                        text: "Starts when you press Start"
                                        color: root.t.text_primary
                                        font.pixelSize: root.fsCaption
                                        anchors.verticalCenter: parent.verticalCenter
                                    }
                                }
                            }
                        }
                    }
                }

                // A look at the curve; the Equalizer page edits it.
                ColumnLayout {
                    spacing: 10
                    Layout.fillWidth: true
                    RowLayout {
                        spacing: 14
                        Layout.fillWidth: true
                        Layout.leftMargin: 2
                        SectionLabel { text: "Equalizer"; Accessible.role: Accessible.Heading; Accessible.name: text }
                        Caption { text: root.m.eq.layers.text; Layout.fillWidth: true; elide: Text.ElideRight }
                        Btn {
                            objectName: "openEqualizer"
                            text: "Open equalizer"
                            enabled: true
                            onClicked: root.m.page.setIndex(root.m.equalizerPage)
                        }
                    }
                    Surface {
                        objectName: "eqPreviewCard"
                        pad: 14
                        spacing: 0
                        Layout.fillWidth: true
                        EQGraphPreview {
                            objectName: "eqPreview"
                            graph: root.m.eq.curve
                            implicitHeight: root.wide ? 150 : 130
                            Layout.fillWidth: true
                            Accessible.role: Accessible.Graphic
                            Accessible.name: "Equalizer preview"
                            Accessible.description: root.m.eq.curve ? root.m.eq.curve.accessibleDescription : ""
                        }
                    }
                }

                ColumnLayout {
                    spacing: 8
                    Layout.fillWidth: true
                    Flow {
                        spacing: 12
                        Layout.fillWidth: true
                        Btn {
                            objectName: "voiceSetupButton"
                            proxy: root.m.presets.voiceSetup
                            look: overviewPage.ov.recommendSetup ? "soft" : "plain"
                        }
                        Btn { proxy: root.m.presets.testSound }
                        Btn { proxy: root.m.presets.autoEq }
                    }
                    StatusText {
                        objectName: "setupLine"
                        text: overviewPage.ov.setupLine
                        font.pixelSize: root.fsCaption
                        wrapMode: Text.Wrap
                        Layout.fillWidth: true
                    }
                }
            }

            // Health page
            Page {
                Surface {
                    Layout.fillWidth: true
                    Text {
                        textFormat: Text.PlainText
                        text: root.m.health.advice.text
                        color: root.t.text_primary
                        font.pixelSize: root.fsHeading
                        wrapMode: Text.Wrap
                        Layout.fillWidth: true
                        Accessible.name: root.m.health.advice.name + ": " + text
                    }
                    Caption {
                        text: root.m.health.route.text
                        wrapMode: Text.Wrap
                        Layout.fillWidth: true
                    }
                }
                Repeater {
                    model: [["Signal", root.m.health.signal], ["Stream", root.m.health.stream]]
                    delegate: Surface {
                        id: group
                        required property var modelData
                        spacing: 10
                        Layout.fillWidth: true
                        Title { text: group.modelData[0] }
                        Repeater {
                            model: group.modelData[1]
                            delegate: HealthRow {
                                required property var modelData
                                row: modelData
                                Layout.fillWidth: true
                            }
                        }
                    }
                }
                Surface {
                    Layout.fillWidth: true
                    Disclosure {
                        id: details
                        text: "Technical details"
                        subject: "health"
                        Layout.fillWidth: true
                    }
                    Caption {
                        visible: details.open
                        text: "Hover Drops or Recovery for the engine's raw counters, which a support report can include."
                        wrapMode: Text.Wrap
                        Layout.fillWidth: true
                    }
                    RowLayout {
                        visible: details.open
                        spacing: 12
                        onVisibleChanged: if (visible) fadeIn.restart()
                        FadeIn on opacity { id: fadeIn }
                        Btn {
                            text: "Export diagnostics"
                            help: "Save a support snapshot without audio or device names (Ctrl+Shift+D)"
                            enabled: true
                            onClicked: root.ui.exportDiagnostics()
                        }
                        Btn {
                            text: "Reset drop counter"
                            help: "Set the dropped-audio count back to zero."
                            enabled: true
                            onClicked: root.ui.resetDrops()
                        }
                        Item { Layout.fillWidth: true }
                    }
                }
            }

            // Settings page
            Page {
                Surface {
                    Layout.fillWidth: true
                    Title { text: "Audio input" }
                    Repeater {
                        model: root.m.settings.input
                        delegate: ControlRow {
                            required property var modelData
                            row: modelData
                            Layout.maximumWidth: 560
                            Layout.leftMargin: 10
                        }
                    }
                }
                Repeater {
                    model: root.m.settings.cards
                    delegate: Surface {
                        id: menuCard
                        required property var modelData
                        spacing: 2
                        Layout.fillWidth: true
                        Title { text: menuCard.modelData.title; bottomPadding: 10 }
                        Repeater {
                            model: menuCard.modelData.rows
                            delegate: SettingRow {
                                required property var modelData
                                row: modelData
                                Layout.fillWidth: true
                            }
                        }
                    }
                }
            }

            Page {
                id: equalizerPage
                objectName: "equalizerPage"
                readonly property var eq: root.m.eq
                readonly property int bandIndex: eq.band.index
                readonly property var band: eq.bands[bandIndex]
                readonly property color bandColor: root.t.bandColors[bandIndex]
                readonly property bool calibrated: eq.diagnostics.data.calibrated
                ground: root.t.pane_surface
                onBandIndexChanged: connector.follow()

                // The page header: the switch, the title over what each layer holds, then the actions.
                RowLayout {
                    spacing: 12
                    Layout.fillWidth: true
                    Layout.topMargin: 4
                    Toggle { proxy: equalizerPage.eq.toggle; Layout.alignment: Qt.AlignTop; Layout.topMargin: 4 }
                    ColumnLayout {
                        spacing: 2
                        Layout.fillWidth: true
                        // The layer line elides before the actions beside it give way.
                        Layout.preferredWidth: equalizerTitleRow.implicitWidth
                        RowLayout {
                            id: equalizerTitleRow
                            spacing: 8
                            Text {
                                objectName: "equalizerTitle"
                                textFormat: Text.PlainText
                                text: "Equalizer"
                                color: root.t.text_primary
                                font.pixelSize: root.fsTitle
                                font.weight: Font.DemiBold
                            }
                            // An EQ that is off keeps its controls at full strength; this says why nothing happens.
                            Caption { text: "Off"; visible: !equalizerPage.eq.toggle.checked }
                            IconBtn {
                                glyph: root.m.glyphs.help
                                hint: equalizerPage.eq.help
                                label: "?"
                                onClicked: root.showTip(this, hint, true)
                                onActiveFocusChanged: if (!activeFocus) root.hideTip(this)
                                Keys.onEscapePressed: root.hideTip(this)
                                Accessible.name: "Help for " + equalizerPage.eq.title.toLowerCase()
                                Accessible.description: hint
                            }
                        }
                        Caption {
                            objectName: "equalizerLayers"
                            text: equalizerPage.eq.layers.text
                            tip: truncated ? text : ""
                            Layout.fillWidth: true
                        }
                    }
                    Btn { proxy: equalizerPage.eq.tone; Layout.alignment: Qt.AlignTop }
                    Btn { proxy: root.m.presets.autoEq; Layout.alignment: Qt.AlignTop }
                    IconBtn {
                        proxy: equalizerPage.eq.menu
                        glyph: root.m.glyphs.more
                        label: "More"
                        Layout.alignment: Qt.AlignTop
                    }
                }
                // Under the header: hold to compare on the left, the edit history on the right.
                RowLayout {
                    spacing: 12
                    Layout.fillWidth: true
                    HoldBtn {
                        objectName: "holdOriginal"
                        proxy: equalizerPage.eq.compare
                        Layout.alignment: Qt.AlignTop
                        // Short of room the label shortens before anything overlaps; the tip and name keep it whole.
                        Layout.fillWidth: true
                        Layout.minimumWidth: shortWidth
                        Layout.preferredWidth: implicitWidth
                        Layout.maximumWidth: implicitWidth
                    }
                    Item { Layout.fillWidth: true; Layout.minimumWidth: 0 }
                    // Undo, then how many edits are unsaved and the latest one in words, with its age.
                    ColumnLayout {
                        id: undoHistory
                        objectName: "undoHistory"
                        readonly property var history: root.m.presets.history
                        spacing: 2
                        Layout.alignment: Qt.AlignTop
                        // A fixed column at the right edge; the chip and the caption elide inside it.
                        Layout.minimumWidth: 30 + 6 + 48
                        Layout.preferredWidth: 240
                        Layout.maximumWidth: 240
                        // The caption's line stays reserved, so the graph below never moves when it appears.
                        Layout.minimumHeight: undoRow.implicitHeight + spacing + undoCaption.implicitHeight
                        RowLayout {
                            id: undoRow
                            spacing: 6
                            Layout.alignment: Qt.AlignRight
                            Layout.fillWidth: false
                            Layout.maximumWidth: undoHistory.width
                            IconBtn {
                                objectName: "equalizerUndo"
                                proxy: root.m.presets.undo
                                glyph: root.m.glyphs.undo
                                label: "Undo"
                            }
                            Chip {
                                objectName: "undoChip"
                                proxy: undoHistory.history
                                visible: undoHistory.history.count > 0
                                single: true
                                Layout.fillWidth: true
                                Layout.maximumWidth: implicitWidth
                            }
                        }
                        Caption {
                            id: undoCaption
                            objectName: "undoCaption"
                            text: undoHistory.history.caption
                            visible: text !== ""
                            font.pixelSize: root.fsCaption
                            horizontalAlignment: Text.AlignRight
                            tip: truncated ? text : ""
                            Layout.fillWidth: true
                        }
                    }
                }

                // The graph in its well; the connector runs from the band's handle down to its panel.
                Rectangle {
                    id: graphWell
                    objectName: "eqGraphWell"
                    readonly property int inset: 6
                    // B held on the focused graph plays the original; the graph ends only its own hold.
                    property bool holdingOriginal: false
                    readonly property bool originalHeld: equalizerPage.eq.compare.held
                    onOriginalHeldChanged: if (!originalHeld) holdingOriginal = false
                    function endOriginal() {
                        if (holdingOriginal) {
                            holdingOriginal = false
                            equalizerPage.eq.compare.release()
                        }
                    }
                    implicitHeight: eqGraph.implicitHeight + 2 * inset
                    radius: root.t.radiusWell
                    color: root.t.data_surface
                    border.color: root.t.border
                    Layout.fillWidth: true
                    EQGraphItem {
                        id: eqGraph
                        objectName: "eqGraph"
                        graph: equalizerPage.eq.curve
                        x: graphWell.inset
                        y: graphWell.inset
                        width: graphWell.width - 2 * graphWell.inset
                        height: graphWell.height - 2 * graphWell.inset
                        implicitHeight: 320
                        activeFocusOnTab: true
                        Accessible.role: Accessible.Graphic
                        Accessible.name: "Equalizer graph"
                        Accessible.description: eqGraph.graph ? eqGraph.graph.accessibleDescription : ""
                        onHandlesChanged: connector.follow()
                        onActiveFocusChanged: if (!eqGraph.activeFocus) graphWell.endOriginal()
                        onVisibleChanged: if (!eqGraph.visible) graphWell.endOriginal()
                        Keys.onPressed: event => {
                            if (event.key !== Qt.Key_B || event.modifiers !== Qt.NoModifier)
                                return
                            event.accepted = true
                            if (!event.isAutoRepeat && !graphWell.holdingOriginal && equalizerPage.eq.compare.enabled) {
                                equalizerPage.eq.compare.press()
                                graphWell.holdingOriginal = equalizerPage.eq.compare.held
                            }
                        }
                        Keys.onReleased: event => {
                            if (event.key !== Qt.Key_B)
                                return
                            event.accepted = true
                            if (!event.isAutoRepeat)
                                graphWell.endOriginal()
                        }
                        FocusRing { visible: parent.activeFocus }
                    }
                    // While the original plays, in words: the colour is never the only sign. It sits
                    // on the region strip, above the plot, so no handle is ever under it.
                    Rectangle {
                        objectName: "originalPill"
                        visible: equalizerPage.eq.compare.held
                        x: Math.round((graphWell.width - width) / 2)
                        y: graphWell.inset + Math.max(0, Math.round((equalizerPage.eq.regionStrip - height) / 2))
                        width: Math.min(pillText.implicitWidth + 24, graphWell.width - 24)
                        height: pillText.implicitHeight + 8
                        radius: height / 2
                        color: root.t.status_info_surface
                        border.color: root.t.status_info_border
                        Accessible.role: Accessible.AlertMessage
                        Accessible.name: pillText.text
                        Text {
                            id: pillText
                            x: 12
                            y: 4
                            width: parent.width - 24
                            text: equalizerPage.eq.compare.pill
                            textFormat: Text.PlainText
                            color: root.t.status_info_text
                            font.pixelSize: root.fsCaption
                            font.weight: Font.DemiBold
                            wrapMode: Text.Wrap
                            horizontalAlignment: Text.AlignHCenter
                            Accessible.ignored: true
                        }
                    }
                    Rectangle {
                        id: connector
                        property real handleX: 0
                        property real handleBottom: 0
                        objectName: "bandConnector"
                        function follow() {
                            handleX = eqGraph.handleX(equalizerPage.bandIndex)
                            handleBottom = eqGraph.handleBottom(equalizerPage.bandIndex)
                        }
                        x: eqGraph.x + Math.round(handleX)
                        y: eqGraph.y + handleBottom
                        width: 1
                        height: Math.max(0, bandPanel.y - graphWell.y - y)
                        color: equalizerPage.bandColor
                        Accessible.ignored: true
                        Component.onCompleted: follow()
                    }
                }

                // The selected band, docked under the graph in the page's flow; it never covers the graph.
                GlassPanel {
                    id: bandPanel
                    objectName: "bandPanel"
                    holdsWheel: false
                    implicitHeight: panelColumn.implicitHeight + 2 * 16
                    Layout.fillWidth: true
                    Layout.maximumWidth: 760
                    Layout.alignment: Qt.AlignHCenter
                    Accessible.role: Accessible.Grouping
                    Accessible.name: "EQ band " + (equalizerPage.bandIndex + 1)
                    ColumnLayout {
                        id: panelColumn
                        x: 20
                        y: 16
                        width: parent.width - 40
                        spacing: 12
                        RowLayout {
                            spacing: 10
                            Layout.fillWidth: true
                            // The band's number on its colour, as the graph prints it on the selected handle.
                            Rectangle {
                                implicitWidth: 22
                                implicitHeight: 22
                                radius: 11
                                color: equalizerPage.bandColor
                                Accessible.ignored: true
                                Text {
                                    textFormat: Text.PlainText
                                    anchors.centerIn: parent
                                    text: equalizerPage.bandIndex + 1
                                    color: root.t.text_on_data_color
                                    font.pixelSize: root.fsCaption
                                    font.weight: Font.Bold
                                }
                            }
                            Text {
                                objectName: "bandTitle"
                                textFormat: Text.PlainText
                                text: "Band " + (equalizerPage.bandIndex + 1) + " \u00b7 " + equalizerPage.band.type.text
                                color: root.t.text_primary
                                font.pixelSize: root.fsLead
                                font.weight: Font.DemiBold
                                elide: Text.ElideRight
                                Layout.fillWidth: true
                            }
                            Caption {
                                text: equalizerPage.band.frequencyLabel.text
                                tip: equalizerPage.band.frequencyLabel.toolTip
                            }
                            IconBtn {
                                glyph: root.m.glyphs.previous
                                label: "Prev"
                                hint: "Previous EQ band"
                                onClicked: root.ui.stepBand(-1)
                            }
                            IconBtn {
                                glyph: root.m.glyphs.next
                                label: "Next"
                                hint: "Next EQ band"
                                onClicked: root.ui.stepBand(1)
                            }
                            Toggle { proxy: equalizerPage.band.enabled }
                        }
                        // Two rows: the panel's width leaves the gain fader too little room beside the fields.
                        ColumnLayout {
                            spacing: 10
                            Layout.fillWidth: true
                            RowLayout {
                                objectName: "bandEditorRow"
                                spacing: 12
                                Layout.fillWidth: true
                                Picker { proxy: equalizerPage.band.type; implicitWidth: 130 }
                                Caption { text: "Gain" }
                                Fader { proxy: equalizerPage.band.gain; unit: 0.1; Layout.fillWidth: true }
                                ValueBox {
                                    objectName: "gainBox"
                                    proxy: equalizerPage.band.gain
                                    display: equalizerPage.band.gainLabel
                                    implicitWidth: 64
                                }
                            }
                            RowLayout {
                                spacing: 12
                                Caption { text: "Freq" }
                                ValueBox { proxy: equalizerPage.band.frequency; implicitWidth: 84 }
                                Caption { text: "Width (Q)"; visible: equalizerPage.band.q.shown }
                                ValueBox {
                                    proxy: equalizerPage.band.q
                                    visible: proxy.shown
                                    opacity: proxy.enabled ? 1 : 0.45
                                    implicitWidth: 56
                                }
                                Caption { text: "Slope"; visible: equalizerPage.band.slope.shown }
                                Picker {
                                    proxy: equalizerPage.band.slope
                                    visible: proxy.shown
                                    implicitWidth: 120
                                }
                            }
                        }
                        // What the band does to the voice where it sits, in one sentence.
                        StatusText {
                            objectName: "bandHint"
                            text: equalizerPage.band.hint.text
                            wrapMode: Text.Wrap
                            elide: Text.ElideNone
                            Layout.fillWidth: true
                        }
                        Rectangle {
                            implicitHeight: 1
                            color: root.t.glass_edge
                            Layout.fillWidth: true
                        }
                        // What calibration found, a mark beside the words, then how each line is drawn.
                        RowLayout {
                            spacing: 8
                            visible: equalizerPage.eq.diagnostics.shown && equalizerPage.calibrated
                            Layout.fillWidth: true
                            Text {
                                objectName: "diagnosticsMark"
                                readonly property string state_: equalizerPage.eq.diagnostics.state
                                textFormat: Text.PlainText
                                text: ({ ok: "\u2713", warn: "!", bad: "\u00d7" })[state_] ?? "\u2013"
                                color: root.t["status_" + state_ + "_text"] ?? root.t.text_muted
                                font.pixelSize: root.fsBody
                                font.weight: Font.Bold
                                Accessible.ignored: true
                            }
                            Chip { proxy: equalizerPage.eq.diagnostics; Layout.fillWidth: true }
                        }
                        Flow {
                            spacing: 20
                            Layout.fillWidth: true
                            Repeater {
                                model: equalizerPage.eq.legend
                                delegate: Row {
                                    id: entry
                                    objectName: "legendEntry"
                                    required property var modelData
                                    readonly property string lineStyle: modelData.lineStyle
                                    readonly property string text: modelData.text
                                    readonly property color sample: ({
                                        solid: root.t.data_curve,
                                        dashed: root.t.data_text,
                                        dotted: root.t.data_text_muted,
                                        dot: equalizerPage.bandColor
                                    })[lineStyle]
                                    spacing: 7
                                    // The line drawn as the graph draws it: whole, in dashes, in dots, or a handle's dot.
                                    Item {
                                        width: 16
                                        height: entryText.height
                                        Row {
                                            anchors.verticalCenter: parent.verticalCenter
                                            spacing: 2
                                            Repeater {
                                                model: ({ solid: 1, dashed: 3, dotted: 4, dot: 1 })[entry.lineStyle]
                                                delegate: Rectangle {
                                                    width: ({ solid: 16, dashed: 4, dotted: 2, dot: 8 })[entry.lineStyle]
                                                    height: entry.lineStyle === "dot" ? 8 : 2
                                                    radius: entry.lineStyle === "dot" ? 4 : entry.lineStyle === "dotted" ? 1 : 0
                                                    color: entry.sample
                                                }
                                            }
                                        }
                                    }
                                    Caption {
                                        id: entryText
                                        text: entry.text
                                        font.pixelSize: root.fsCaption
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // The latest message, in one line at the foot of the main pane.
        StatusText {
            objectName: "statusMessage"
            text: root.m.status.message.text
            Layout.fillWidth: true
            Layout.leftMargin: 8
            Layout.rightMargin: 16
            tip: truncated ? text : ""
        }
    }

    // The inspector sits at the right under the toolbar. A wide window gives it a rail of its own
    // and the main pane narrows; a narrower one lays it over the pane as glass.
    readonly property bool inspecting: m.inspector.shown
    readonly property bool docked: wide
    readonly property int inspectorGap: 12
    readonly property real inspectorRoom: docked ? inspector.progress * inspector.travel : 0
    property Item inspectorOrigin: null

    Rectangle {
        objectName: "inspectorRail"
        anchors.right: parent.right
        anchors.top: toolbar.bottom
        anchors.bottom: parent.bottom
        width: root.inspectorRoom
        visible: width > 0
        color: root.t.rail_surface
        Rectangle {
            width: 1
            height: parent.height
            color: root.t.border
        }
    }

    GlassPanel {
        id: inspector
        objectName: "inspector"
        // 0 while closed and 1 while open; the panel slides in by it.
        property real progress: 0
        readonly property real travel: width + 2 * root.inspectorGap
        readonly property InspectorPanel panel: root.inspectorPanel(root.m.inspector.stageKey)
        width: 400
        x: root.width - width - root.inspectorGap
        y: toolbar.y + toolbar.height + root.inspectorGap
        height: root.height - y - root.inspectorGap
        opacity: 0
        visible: root.inspecting || opacity > 0
        backdrop: root.docked ? null : mainPane
        backdropRect: Qt.rect(x + slide.x - mainPane.x, y - mainPane.y, width, height)
        transform: Translate { id: slide; x: (1 - inspector.progress) * inspector.travel }
        Accessible.role: Accessible.Grouping
        Accessible.name: panel ? panel.card.name + " inspector" : ""
        states: State {
            name: "open"
            when: root.inspecting
            PropertyChanges { inspector.progress: 1; inspector.opacity: 1 }
        }
        // Under reduced motion the panel does not slide; it only fades.
        transitions: [
            Transition {
                to: "open"
                NumberAnimation {
                    objectName: "inspectorOpenMove"
                    property: "progress"
                    duration: Look.movement(root.t.motionPanelOpen)
                    easing.type: Easing.BezierSpline
                    easing.bezierCurve: root.t.easePanelOpen
                }
                NumberAnimation {
                    objectName: "inspectorOpenFade"
                    property: "opacity"
                    duration: root.t.motionPanelOpen
                    easing.type: Easing.BezierSpline
                    easing.bezierCurve: root.t.easePanelOpen
                }
            },
            // The panel stays in place while it fades, then leaves.
            Transition {
                from: "open"
                enabled: Look.reducedMotion
                SequentialAnimation {
                    NumberAnimation {
                        objectName: "inspectorReducedCloseFade"
                        property: "opacity"
                        duration: root.t.motionPanelClose
                        easing.type: Easing.BezierSpline
                        easing.bezierCurve: root.t.easePanelClose
                    }
                    PropertyAction { property: "progress" }
                }
            },
            Transition {
                from: "open"
                enabled: !Look.reducedMotion
                NumberAnimation {
                    objectName: "inspectorCloseMove"
                    property: "progress"
                    duration: Look.movement(root.t.motionPanelClose)
                    easing.type: Easing.BezierSpline
                    easing.bezierCurve: root.t.easePanelClose
                }
                NumberAnimation {
                    objectName: "inspectorCloseFade"
                    property: "opacity"
                    duration: root.t.motionPanelClose
                    easing.type: Easing.BezierSpline
                    easing.bezierCurve: root.t.easePanelClose
                }
            }
        ]
        // Every panel stays alive, so a stage keeps its open sections; only the current one shows.
        Item {
            objectName: "inspectorStack"
            anchors.fill: parent
            Repeater {
                id: inspectorPanels
                model: root.m.sidebar.chain.filter(row => !!row.card).map(row => row.card)
                delegate: InspectorPanel {
                    readonly property bool current: card.key === root.m.inspector.stageKey
                    anchors.fill: parent
                    visible: current
                    onCurrentChanged: if (current) crossfade.restart()
                    onCloseRequested: reason => root.closeInspector(reason)
                    FadeIn on opacity { id: crossfade; duration: root.t.motionCrossfade }
                }
            }
        }
    }

    // Ctrl+K opens it (its window action is in command_palette.py); it lies over every pane.
    CommandPalette {
        objectName: "commandPalette"
        anchors.fill: parent
        commands: root.m.palette
        backdrop: mainPane
        fallbackFocus: overviewRow.button
    }

    // On top, so a press is seen before the control under it takes focus:
    // a clicked control is already under the pointer and must not be scrolled away.
    Item {
        anchors.fill: parent
        z: 1
        PointHandler { id: pointer }
    }
}
