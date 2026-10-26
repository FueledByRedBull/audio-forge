// Main window scene. Controls consume shared state from quick_shell.py;
// values, control metadata and commands live in the Python presentation models.
pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Effects
import QtQuick.Layouts
import QtQuick.Templates as T
import AudioForge

Rectangle {
    id: root

    required property QuickBridge ui
    readonly property var m: ui.model
    readonly property var t: m.theme
    readonly property bool hasIcons: t.iconFont !== ""
    readonly property bool effects: GraphicsInfo.api !== GraphicsInfo.Software
    readonly property bool wide: width >= 1200
    // The band editor needs more room than the rest of the page for two columns.
    readonly property bool roomy: width >= 1400
    readonly property bool running: m.top.stop.enabled
    readonly property bool pressing: pointer.active
    // The sizes of text, in pixels, follow the app font (9 pt is 12 px); nothing is smaller than 12.
    readonly property real zoom: Application.font.pointSize > 0 ? Application.font.pointSize / 9 : 1
    readonly property int fsCaption: Math.round(12 * zoom)
    readonly property int fsBody: Math.round(13 * zoom)
    readonly property int fsLead: Math.round(15 * zoom)
    readonly property int fsHeading: Math.round(18 * zoom)
    readonly property int fsTitle: Math.round(20 * zoom)
    // When the last tooltip closed; the next one opens at once if it follows closely.
    property double tipClosedAt: 0

    color: t.app_surface

    onPressingChanged: if (pressing) tip.close()
    // Esc closes a tip that keyboard focus opened, as it closes one the help button opened.
    Keys.onEscapePressed: event => {
        if (tip.visible)
            tip.close()
        else
            event.accepted = false
    }

    // One shared tooltip, so long help text wraps and matches the theme.
    ToolTip {
        id: tip
        objectName: "tooltip"
        delay: 500
        padding: 10
        // Hover help is not animated, and the ToolTip style's own fade is not either.
        enter: null
        exit: null
        onClosed: root.tipClosedAt = Date.now()
        contentItem: Text {
            text: tip.text
            textFormat: Text.PlainText
            color: root.t.text_primary
            font.pixelSize: root.fsCaption
            wrapMode: Text.Wrap
            width: Math.min(implicitWidth, 360)
        }
        background: Rectangle {
            color: root.t.control_surface
            border.color: root.t.border_strong
            radius: root.t.radiusControl
        }
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
        property bool focused: parent.visualFocus === true
        onHoveredChanged: sync()
        onFocusedChanged: sync()
        function sync() {
            if (hovered || focused)
                root.showTip(parent, text)
            else
                root.hideTip(parent)
        }
    }

    // The one focus indicator: a ring that hover never draws. `outset` is how far it sits outside
    // the control; a negative value draws it inside.
    component FocusRing: Rectangle {
        property T.Control control
        property int outset: 3
        anchors.fill: parent
        anchors.margins: -outset
        radius: height / 2 < 9 ? height / 2 : 9
        color: "transparent"
        border.width: 2
        border.color: root.t.accent
        visible: control.visualFocus
    }

    component FadeIn: NumberAnimation {
        from: 0
        to: 1
        duration: root.t.motion
        easing.type: Easing.OutCubic
    }

    component Caption: Text {
        property string tip: ""
        textFormat: Text.PlainText
        color: root.t.text_muted
        font.pixelSize: root.fsBody
        elide: Text.ElideRight
        Hint { text: parent.tip }
    }

    // A caption that screen readers read, for text that carries a state (not a label of a control).
    component StatusText: Caption {
        Accessible.role: Accessible.StaticText
        Accessible.name: text
    }

    component Title: Text {
        textFormat: Text.PlainText
        color: root.t.text_primary
        font.pixelSize: root.fsBody
        font.weight: Font.DemiBold
        font.letterSpacing: 1.1
        font.capitalization: Font.AllUppercase
    }

    component Surface: Item {
        id: surface
        default property alias content: body.data
        property int pad: 18
        property alias spacing: body.spacing
        implicitHeight: body.implicitHeight + 2 * pad
        Rectangle {
            anchors.fill: parent
            radius: root.t.radiusCard + 2
            color: root.t.card_surface
            border.color: root.t.border
            layer.enabled: root.effects
            layer.effect: MultiEffect {
                shadowEnabled: true
                shadowColor: "black"
                shadowOpacity: 0.4
                shadowBlur: 0.8
                shadowVerticalOffset: 5
            }
        }
        ColumnLayout {
            id: body
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.top: parent.top
            anchors.margins: surface.pad
            spacing: 14
        }
    }

    // Return and Enter press a button as Space does; a held key presses it once.
    component KeyButton: AbstractButton {
        Keys.onReturnPressed: event => { if (!event.isAutoRepeat) click() }
        Keys.onEnterPressed: event => { if (!event.isAutoRepeat) click() }
    }

    // A button whose check state belongs to the model. Every way of switching it (pointer, keys,
    // assistive technology) changes `checked`, so requested() follows that; the model decides,
    // and the button goes back to the model's state `on`.
    component ModelButton: KeyButton {
        property bool on
        signal requested()
        // The real check state, so assistive technology also hears it before the first change.
        checkable: true
        checked: on
        onCheckedChanged: if (checked !== on) {
            if (enabled) requested()
            checked = Qt.binding(() => on)
        }
        Accessible.onPressAction: click()
    }

    component Toggle: ModelButton {
        id: toggle
        property var proxy
        on: proxy ? proxy.checked : false
        // Colors follow this rather than `enabled`, so a row can own the click.
        property bool live: enabled
        enabled: proxy ? proxy.enabled : false
        implicitHeight: 22
        implicitWidth: track.width + (label.text === "" ? 0 : label.implicitWidth + 10)
        onRequested: proxy.click()
        Accessible.role: Accessible.CheckBox
        Accessible.name: proxy && proxy.name !== "" ? proxy.name : text
        Accessible.description: proxy ? proxy.toolTip : ""
        Hint { text: toggle.proxy ? toggle.proxy.toolTip : "" }
        background: null
        contentItem: Item {
            Rectangle {
                id: track
                width: 40
                height: 22
                radius: 11
                color: !toggle.live ? root.t.action_disabled
                       : toggle.on ? root.t.accent : root.t.control_edge
                Behavior on color { ColorAnimation { duration: root.t.motion } }
                Rectangle {
                    width: 16
                    height: 16
                    radius: 8
                    y: 3
                    x: toggle.on ? 21 : 3
                    color: toggle.live ? root.t.text_primary : root.t.action_disabled_text
                    Behavior on x {
                        NumberAnimation { duration: root.t.motion; easing.type: Easing.OutCubic }
                    }
                }
                FocusRing { control: toggle }
            }
            Text {
                id: label
                anchors.left: track.right
                anchors.leftMargin: 10
                anchors.verticalCenter: track.verticalCenter
                text: toggle.text
                textFormat: Text.PlainText
                color: toggle.enabled ? root.t.text_primary : root.t.action_disabled_text
                font.pixelSize: root.fsBody
            }
        }
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
            Behavior on scale { NumberAnimation { duration: root.t.motion / 2 } }
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

    component Picker: ComboBox {
        id: picker
        property var proxy
        // Set for a choice that changes what the sound does, so it does not read as the default.
        property bool alert: false
        model: proxy ? proxy.items : []
        currentIndex: proxy ? proxy.index : -1
        enabled: proxy ? proxy.enabled : false
        opacity: enabled ? 1 : 0.45
        implicitHeight: 32
        implicitWidth: 170
        leftPadding: 10
        rightPadding: 26
        onActivated: index => proxy.setIndex(index)
        // Replacing the model resets the index; read it again afterwards.
        onModelChanged: currentIndex = Qt.binding(() => proxy ? proxy.index : -1)
        // A combo box has no value channel in QML, and a name change is spoken where a description
        // change often is not: the label stays first, so the visible text is still in the name.
        Accessible.name: proxy ? proxy.name + (shown.text !== "" ? ": " + shown.text : "") : ""
        Accessible.description: proxy ? proxy.toolTip : ""
        // A name cut short by a narrow window is shown whole in the tooltip.
        Hint {
            text: {
                const tip = picker.proxy ? picker.proxy.toolTip : ""
                return shown.truncated ? shown.text + (tip !== "" ? "\n" + tip : "") : tip
            }
        }
        // With nothing chosen the box shows its prompt, set apart from a real choice.
        readonly property bool prompting: currentIndex < 0 && !!proxy
        contentItem: Text {
            id: shown
            text: picker.prompting ? (picker.proxy.placeholder ?? "") : picker.displayText
            textFormat: Text.PlainText
            color: picker.hovered || picker.down ? root.t.hover_text
                 : picker.prompting ? root.t.text_muted
                 : picker.alert ? root.t.status_warn_text : root.t.text_primary
            font.pixelSize: root.fsBody
            font.italic: picker.prompting
            elide: Text.ElideRight
            verticalAlignment: Text.AlignVCenter
        }
        indicator: Text {
            x: picker.width - width - 11
            y: (picker.height - height) / 2
            text: root.m.glyphs.expand
            textFormat: Text.PlainText
            font.family: root.t.iconFont
            font.pixelSize: root.fsCaption
            color: picker.hovered || picker.down ? root.t.hover_text : root.t.text_muted
        }
        background: Rectangle {
            radius: root.t.radiusControl
            color: picker.hovered || picker.down ? root.t.control_hover : root.t.control_surface
            border.color: picker.prompting ? root.t.status_warn_border
                        : picker.alert ? root.t.status_warn_text : root.t.control_edge
            FocusRing { control: picker }
        }
        delegate: ItemDelegate {
            id: option
            required property int index
            required property var modelData
            width: ListView.view ? ListView.view.width : 0
            height: 30
            highlighted: picker.highlightedIndex === index
            contentItem: Text {
                text: option.modelData
                textFormat: Text.PlainText
                color: option.highlighted ? root.t.hover_text : root.t.text_primary
                font.pixelSize: root.fsBody
                elide: Text.ElideRight
                verticalAlignment: Text.AlignVCenter
            }
            background: Rectangle {
                radius: 4
                color: option.highlighted ? root.t.control_hover : "transparent"
            }
        }
        popup: Popup {
            y: picker.height + 4
            width: Math.max(picker.width, 240)
            padding: 4
            // Kept inside the window, as the stock combo box does.
            topMargin: 6
            bottomMargin: 6
            implicitHeight: Math.min(contentItem.implicitHeight + 8, 320, root.height - 12)
            contentItem: ListView {
                clip: true
                implicitHeight: contentHeight
                model: picker.delegateModel
                currentIndex: picker.highlightedIndex
                footer: Text {
                    width: ListView.view ? ListView.view.width : 0
                    height: picker.count === 0 ? 30 : 0
                    visible: picker.count === 0
                    text: "No choices available"
                    textFormat: Text.PlainText
                    color: root.t.text_muted
                    font.pixelSize: root.fsBody
                    leftPadding: 8
                    verticalAlignment: Text.AlignVCenter
                }
            }
            background: Rectangle {
                color: root.t.control_surface
                border.color: root.t.border_strong
                radius: root.t.radiusControl
            }
        }
    }

    component Btn: KeyButton {
        id: btn
        property var proxy
        // "plain", "primary" or "danger"
        property string look: "plain"
        text: proxy ? proxy.text : ""
        enabled: proxy ? proxy.enabled : false
        opacity: enabled ? 1 : 0.45
        implicitHeight: 32
        implicitWidth: caption.implicitWidth + 28
        onClicked: if (proxy) root.press(this, proxy)
        Accessible.role: Accessible.Button
        Accessible.name: proxy && proxy.name !== "" ? proxy.name : text
        Accessible.description: proxy ? proxy.toolTip : ""
        Hint { text: btn.proxy ? btn.proxy.toolTip : "" }
        background: Rectangle {
            radius: root.t.radiusControl
            color: btn.look === "primary" ? (btn.hovered ? root.t.accent_hover : root.t.accent)
                 : btn.look === "danger" ? root.t.action_destructive
                 : btn.hovered ? root.t.control_hover : root.t.control_surface
            border.color: btn.look === "plain" ? root.t.control_edge
                        : btn.look === "danger" ? root.t.action_destructive_border : "transparent"
            scale: btn.pressed ? 0.98 : 1
            FocusRing { control: btn }
        }
        contentItem: Text {
            id: caption
            text: btn.text
            textFormat: Text.PlainText
            color: btn.look === "primary" ? root.t.text_on_accent
                 : btn.look === "danger" ? root.t.text_on_emphasis
                 : btn.hovered ? root.t.hover_text : root.t.text_primary
            font.pixelSize: root.fsBody
            font.weight: btn.look === "plain" ? Font.Normal : Font.DemiBold
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            elide: Text.ElideRight
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
            Behavior on color { ColorAnimation { duration: root.t.motion } }
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
                Behavior on rotation { NumberAnimation { duration: root.t.motion } }
            }
        }
    }

    component StageCard: Surface {
        id: stage
        property var card
        CardHeader { card: stage.card; Layout.fillWidth: true }
        Chip {
            proxy: stage.card.alert
            visible: !!proxy && (proxy.state === "warn" || proxy.state === "bad")
            Layout.fillWidth: true
        }
        Repeater {
            model: stage.card.rows
            delegate: ControlRow {
                required property var modelData
                row: modelData
                Layout.fillWidth: true
            }
        }
        ColumnLayout {
            visible: stage.card.trace && root.ui.levels.length > 1
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
            subject: stage.card.title
            visible: stage.card.advanced.length > 0
            Layout.fillWidth: true
        }
        Repeater {
            model: more.open ? stage.card.advanced : []
            delegate: ControlRow {
                required property var modelData
                row: modelData
                Layout.fillWidth: true
                FadeIn on opacity {}
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
            Accessible.name: "Page scroll bar"
            policy: page.contentHeight > page.height ? ScrollBar.AlwaysOn : ScrollBar.AlwaysOff
            contentItem: Rectangle {
                implicitWidth: 6
                radius: 3
                color: bar.pressed || bar.hovered ? root.t.text_muted : root.t.control_edge
            }
        }
        ColumnLayout {
            id: column
            x: 8
            y: 4
            width: parent.width - 24
            spacing: 16
        }
    }

    // Navigation rail
    Rectangle {
        id: rail
        width: 76
        height: parent.height
        color: root.t.rail_surface
        Column {
            y: 14
            Accessible.role: Accessible.PageTabList
            Repeater {
                model: [[root.m.glyphs.mic, "Mic"], [root.m.glyphs.health, "Health"],
                        [root.m.glyphs.settings, "Settings"]]
                delegate: KeyButton {
                    id: nav
                    required property var modelData
                    required property int index
                    readonly property bool current: root.m.page.index === index
                    width: 76
                    height: 66
                    onClicked: root.m.page.setIndex(index)
                    Accessible.role: Accessible.PageTab
                    Accessible.name: modelData[1] + " page"
                    Accessible.selectable: true
                    Accessible.selected: current
                    background: Rectangle {
                        color: nav.current || nav.hovered ? root.t.card_surface : "transparent"
                        Behavior on color { ColorAnimation { duration: root.t.motion } }
                        Rectangle {
                            width: 3
                            height: nav.current ? parent.height : 0
                            anchors.verticalCenter: parent.verticalCenter
                            radius: 1.5
                            color: root.t.accent
                            Behavior on height {
                                NumberAnimation { duration: root.t.motion; easing.type: Easing.OutCubic }
                            }
                        }
                        FocusRing { control: nav; outset: -4 }
                    }
                    contentItem: Column {
                        topPadding: 12
                        spacing: 6
                        Text {
                            textFormat: Text.PlainText
                            anchors.horizontalCenter: parent.horizontalCenter
                            text: nav.modelData[0]
                            visible: root.hasIcons
                            color: nav.current ? root.t.accent
                                 : nav.hovered ? root.t.text_primary : root.t.text_muted
                            font.family: root.t.iconFont
                            font.pixelSize: root.fsTitle
                        }
                        Text {
                            textFormat: Text.PlainText
                            anchors.horizontalCenter: parent.horizontalCenter
                            text: nav.modelData[1]
                            color: nav.current || nav.hovered ? root.t.text_primary : root.t.text_muted
                            font.pixelSize: root.fsCaption
                            font.weight: nav.current ? Font.DemiBold : Font.Normal
                        }
                    }
                }
            }
        }
    }

    // Level meters, in a panel of their own so they read as one instrument.
    Rectangle {
        id: meters
        objectName: "meterPanel"
        anchors.right: parent.right
        anchors.top: parent.top
        anchors.bottom: parent.bottom
        anchors.margins: 18
        anchors.leftMargin: 0
        anchors.bottomMargin: 46
        width: meterRow.width + 20
        radius: root.t.radiusCard + 2
        color: root.t.card_surface
        border.color: root.t.border
        Row {
            id: meterRow
            x: 10
            y: 12
            height: parent.height - 24
            spacing: 6
            Repeater {
                model: root.m.meters
                delegate: QuickMeterItem {
                    required property var modelData
                    source: modelData
                    width: 50
                    height: meterRow.height
                    Accessible.role: Accessible.ProgressBar
                    Accessible.name: modelData.name
                    // The reading first; it changes only when the whole-dB value or the words do.
                    Accessible.description: modelData.accessibleText
                }
            }
        }
    }

    ColumnLayout {
        id: content
        objectName: "contentColumn"
        anchors.left: rail.right
        anchors.right: meters.left
        anchors.top: parent.top
        anchors.bottom: parent.bottom
        anchors.margins: 18
        anchors.leftMargin: 10
        anchors.rightMargin: 0
        anchors.bottomMargin: 10
        spacing: 12

        // Top bar: route and transport
        Surface {
            Layout.fillWidth: true
            Layout.leftMargin: 8
            Layout.rightMargin: 16
            pad: 14
            // The route and the transport share a row when there is room, else the route is on top.
            GridLayout {
                columns: root.wide ? 2 : 1
                columnSpacing: 12
                rowSpacing: 10
                RowLayout {
                    spacing: 12
                    Layout.fillWidth: true
                    Caption { text: root.wide ? "Input" : "In" }
                    Picker {
                        proxy: root.m.top.input
                        Layout.fillWidth: true
                        Layout.preferredWidth: 200
                        Layout.minimumWidth: 140
                    }
                    Text {
                        textFormat: Text.PlainText
                        text: root.m.glyphs.next
                        visible: root.hasIcons
                        font.family: root.t.iconFont
                        font.pixelSize: root.fsCaption
                        color: root.t.text_muted
                    }
                    Caption { text: root.wide ? "Output" : "Out" }
                    Picker {
                        proxy: root.m.top.output
                        Layout.fillWidth: true
                        Layout.preferredWidth: 200
                        Layout.minimumWidth: 140
                    }
                    IconBtn { proxy: root.m.top.refresh; glyph: root.m.glyphs.refresh; label: "Refresh" }
                }
                RowLayout {
                    spacing: 12
                    Layout.alignment: Qt.AlignRight
                    Rectangle {
                        implicitWidth: 1
                        visible: root.wide
                        color: root.t.border
                        Layout.fillHeight: true
                        Layout.topMargin: 4
                        Layout.bottomMargin: 4
                    }
                    // Said in words as well as by the switch: a mute nobody notices leaves the other
                    // end hearing nothing.
                    Rectangle {
                        objectName: "mutedBadge"
                        visible: root.m.top.mute.checked
                        implicitWidth: mutedText.implicitWidth + 20
                        implicitHeight: 26
                        radius: 13
                        color: root.t.status_warn_surface
                        border.color: root.t.status_warn_border
                        Text {
                            id: mutedText
                            anchors.centerIn: parent
                            text: "Muted"
                            textFormat: Text.PlainText
                            color: root.t.status_warn_text
                            font.pixelSize: root.fsBody
                            font.weight: Font.DemiBold
                            // The switch already says it, and the status line announces the change.
                            Accessible.ignored: true
                        }
                    }
                    // The mode picker sits between Mute and Stop, so one slip cannot hit both.
                    Toggle { proxy: root.m.top.mute; text: "Mute" }
                    Picker { proxy: root.m.top.mode; implicitWidth: 118; alert: root.m.top.mode.index > 0 }
                    Btn {
                        proxy: root.running ? root.m.top.stop : root.m.top.start
                        text: root.running ? "Stop" : "Start"
                        look: root.running ? "danger" : "primary"
                        implicitWidth: 112
                        implicitHeight: 34
                        Accessible.name: proxy.text
                    }
                }
            }
        }

        // Under the top bar, so Start and Mute stay where they are when one appears.
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

            // Mic page
            Page {
                objectName: "micPage"
                // The buttons drop under the name when the two do not fit side by side.
                GridLayout {
                    id: presetBar
                    objectName: "presetBar"
                    readonly property var preset: root.m.presets.status.data
                    columns: presetTitle.implicitWidth + presetActions.implicitWidth + columnSpacing <= width ? 2 : 1
                    columnSpacing: 12
                    rowSpacing: 8
                    Layout.fillWidth: true
                    Layout.topMargin: 2
                    Accessible.name: root.m.presets.status.name + ": " + root.m.presets.status.text
                    ColumnLayout {
                        id: presetTitle
                        spacing: 2
                        Layout.fillWidth: true
                        Text {
                            textFormat: Text.PlainText
                            text: "PRESET"
                            color: root.t.text_muted
                            font.pixelSize: root.fsCaption
                            font.letterSpacing: 1.2
                            font.weight: Font.DemiBold
                        }
                        RowLayout {
                            spacing: 10
                            Text {
                                objectName: "presetName"
                                textFormat: Text.PlainText
                                text: presetBar.preset.name
                                color: root.t.text_primary
                                font.pixelSize: root.fsHeading
                                font.weight: Font.DemiBold
                                elide: Text.ElideRight
                                Layout.fillWidth: true
                                Layout.maximumWidth: Math.min(implicitWidth, 420)
                                // A name cut short is shown whole in the tooltip.
                                Hint { text: parent.truncated ? parent.text : "" }
                            }
                            Rectangle {
                                implicitWidth: stateText.implicitWidth + 16
                                implicitHeight: 20
                                radius: 10
                                color: presetBar.preset.modified ? root.t.status_warn_surface
                                                                 : root.t.status_idle_surface
                                border.color: presetBar.preset.modified ? root.t.status_warn_border
                                                                        : root.t.status_idle_border
                                Text {
                                    id: stateText
                                    textFormat: Text.PlainText
                                    anchors.centerIn: parent
                                    text: presetBar.preset.modified ? "Unsaved changes" : "Saved"
                                    color: presetBar.preset.modified ? root.t.status_warn_text
                                                                     : root.t.status_idle_text
                                    font.pixelSize: root.fsCaption
                                }
                            }
                            Item { Layout.fillWidth: true }
                        }
                    }
                    RowLayout {
                        id: presetActions
                        spacing: 12
                        Layout.fillWidth: false
                        Layout.alignment: Qt.AlignRight | Qt.AlignVCenter
                        Btn { proxy: root.m.presets.menu }
                        IconBtn { proxy: root.m.presets.undo; glyph: root.m.glyphs.undo; visible: root.hasIcons }
                        IconBtn { proxy: root.m.presets.redo; glyph: root.m.glyphs.redo; visible: root.hasIcons }
                        // Without a system icon font the two are spelled out.
                        Btn { proxy: root.m.presets.undo; visible: !root.hasIcons }
                        Btn { proxy: root.m.presets.redo; visible: !root.hasIcons }
                        Btn { proxy: root.m.presets.testSound }
                        Btn { proxy: root.m.presets.autoEq }
                        Btn { objectName: "voiceSetupButton"; proxy: root.m.presets.voiceSetup }
                    }
                }

                Surface {
                    id: eqCard
                    readonly property var eq: root.m.eq
                    readonly property var band: eq.bands[eq.band.index]
                    readonly property bool calibrated: eq.diagnostics.data.calibrated
                    Layout.fillWidth: true
                    CardHeader {
                        card: eqCard.eq
                        Layout.fillWidth: true
                        Caption { text: eqCard.eq.layers.text; visible: root.wide }
                        Btn { proxy: eqCard.eq.tone }
                        IconBtn { proxy: eqCard.eq.menu; glyph: root.m.glyphs.more; label: "More" }
                    }
                    EQGraphItem {
                        graph: eqCard.eq.curve
                        Layout.fillWidth: true
                        implicitHeight: 300
                        activeFocusOnTab: true
                        Accessible.role: Accessible.Graphic
                        Accessible.name: "Equalizer graph"
                        Accessible.description: graph ? graph.accessibleDescription : ""
                        FocusRing { visible: parent.activeFocus }
                    }
                    Caption {
                        text: eqCard.eq.layers.text
                        visible: !root.wide
                        Layout.fillWidth: true
                    }
                    Chip {
                        proxy: eqCard.eq.diagnostics
                        visible: proxy.shown && eqCard.calibrated
                        Layout.fillWidth: true
                    }
                    // One row when there is room, two when the window is narrow.
                    GridLayout {
                        columns: root.roomy ? 2 : 1
                        columnSpacing: 12
                        rowSpacing: 10
                        Layout.fillWidth: true
                        RowLayout {
                            objectName: "bandEditorRow"
                            spacing: 12
                            Layout.fillWidth: true
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
                            // The band the editor shows; the graph prints this number on its handle once selected.
                            Rectangle {
                                implicitWidth: 20
                                implicitHeight: 20
                                radius: 10
                                color: root.t.bandColors[eqCard.eq.band.index]
                                Text {
                                    textFormat: Text.PlainText
                                    anchors.centerIn: parent
                                    text: eqCard.eq.band.index + 1
                                    color: root.t.text_on_data_color
                                    font.pixelSize: root.fsCaption
                                    font.weight: Font.DemiBold
                                }
                            }
                            Text {
                                textFormat: Text.PlainText
                                text: eqCard.band.frequencyLabel.text
                                color: root.t.text_primary
                                font.pixelSize: root.fsLead
                                font.weight: Font.DemiBold
                                Layout.preferredWidth: 58
                            }
                            Toggle { proxy: eqCard.band.enabled }
                            Picker { proxy: eqCard.band.type; implicitWidth: 130 }
                            Caption { text: "Gain" }
                            Fader { proxy: eqCard.band.gain; unit: 0.1; Layout.fillWidth: true }
                            ValueBox {
                                objectName: "gainBox"
                                proxy: eqCard.band.gain
                                display: eqCard.band.gainLabel
                                implicitWidth: 64
                            }
                        }
                        RowLayout {
                            spacing: 12
                            Caption { text: "Freq" }
                            ValueBox { proxy: eqCard.band.frequency; implicitWidth: 84 }
                            Caption { text: "Q"; visible: eqCard.band.q.shown }
                            ValueBox {
                                proxy: eqCard.band.q
                                visible: proxy.shown
                                opacity: proxy.enabled ? 1 : 0.45
                                implicitWidth: 56
                            }
                            Caption { text: "Slope"; visible: eqCard.band.slope.shown }
                            Picker {
                                proxy: eqCard.band.slope
                                visible: proxy.shown
                                implicitWidth: 120
                            }
                        }
                    }
                }

                // Signal order reads left to right, then down. The cards are siblings in that order, so Tab
                // and screen readers follow it; each sits under the card above it in its column.
                Item {
                    id: stageGrid
                    objectName: "stageGrid"
                    readonly property int lanes: root.wide ? 2 : 1
                    readonly property int gap: 16
                    Layout.fillWidth: true
                    implicitHeight: {
                        let bottom = 0
                        for (let i = 0; i < stageCards.count; ++i) {
                            const card = stageCards.itemAt(i)
                            if (card)
                                bottom = Math.max(bottom, card.y + card.height)
                        }
                        return bottom
                    }
                    Repeater {
                        id: stageCards
                        model: root.m.stages
                        delegate: StageCard {
                            required property int index
                            required property var modelData
                            readonly property Item above: index >= stageGrid.lanes
                                                          ? stageCards.itemAt(index - stageGrid.lanes) : null
                            card: modelData
                            width: (stageGrid.width - (stageGrid.lanes - 1) * stageGrid.gap) / stageGrid.lanes
                            height: implicitHeight
                            x: (index % stageGrid.lanes) * (width + stageGrid.gap)
                            y: above ? above.y + above.height + stageGrid.gap : 0
                        }
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
                            enabled: true
                            Hint { text: "Save a support snapshot without audio or device names (Ctrl+Shift+D)" }
                            onClicked: root.ui.exportDiagnostics()
                        }
                        Btn {
                            text: "Reset drop counter"
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
        }

        // Status strip
        RowLayout {
            objectName: "statusStrip"
            // The room the texts share, read from the column: sizes taken from the strip's
            // own width would feed back into its minimum width.
            readonly property real room: content.width - 24
            Layout.fillWidth: true
            Layout.leftMargin: 8
            Layout.rightMargin: 16
            spacing: 14
            StatusText {
                objectName: "statusMessage"
                text: root.m.status.message.text
                Layout.fillWidth: true
                // The other texts give way before the message does.
                Layout.minimumWidth: parent.room / 4
                tip: truncated ? text : ""
            }
            // Heard (green), muted or waiting for a mute (amber), or stopped.
            Rectangle {
                implicitWidth: 8
                implicitHeight: 8
                radius: 4
                color: ({ ok: root.t.status_ok_text, warn: root.t.status_warn_text })[
                    root.m.status.transmission.state] ?? root.t.control_edge
            }
            StatusText {
                objectName: "statusTransmission"
                text: root.m.status.transmission.text
                color: root.t.text_primary
                Layout.maximumWidth: parent.room * 0.3
                tip: truncated ? text : ""
            }
            StatusText {
                objectName: "statusCalibration"
                text: root.m.status.calibration.text
                Layout.fillWidth: true
                Layout.maximumWidth: implicitWidth
                tip: truncated ? text : ""
            }
            Chip {
                proxy: root.m.status.health
                announced: false
                single: true
                implicitHeight: 28
                // A long reading must not push the other texts out of the strip.
                Layout.maximumWidth: parent.room * 0.3
                ModelButton {
                    id: healthToggle
                    objectName: "healthToggle"
                    anchors.fill: parent
                    on: root.m.page.index === root.m.healthPage
                    onRequested: root.m.page.setIndex(checked ? root.m.healthPage : 0)
                    Accessible.role: Accessible.Button
                    // Named by what it shows; the action goes in the description.
                    Accessible.name: root.m.status.health.text
                    Accessible.description: checked ? "Hide audio health details" : "Show audio health details"
                    background: Item { FocusRing { control: healthToggle } }
                }
            }
        }
    }

    // On top, so a press is seen before the control under it takes focus:
    // a clicked control is already under the pointer and must not be scrolled away.
    Item {
        anchors.fill: parent
        z: 1
        PointHandler { id: pointer }
    }
}
