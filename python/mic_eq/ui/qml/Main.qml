// Main window scene. Every control is bound to a proxy from quick_shell.py;
// values, ranges, tooltips and handlers live in the Python panels.
import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Effects
import QtQuick.Layouts
import AudioForge

Rectangle {
    id: root

    readonly property var m: ui.model
    readonly property var t: m.theme
    readonly property bool effects: GraphicsInfo.api !== GraphicsInfo.Software
    readonly property bool wide: width >= 1200
    readonly property bool running: m.top.stop.enabled

    color: t.app_surface

    // One shared tooltip, so long help text wraps and matches the theme.
    ToolTip {
        id: tip
        delay: 500
        padding: 10
        contentItem: Text {
            text: tip.text
            color: root.t.text_primary
            font.pixelSize: 12
            wrapMode: Text.Wrap
            width: Math.min(implicitWidth, 360)
        }
        background: Rectangle {
            color: root.t.control_surface
            border.color: root.t.border_strong
            radius: root.t.radiusControl
        }
    }

    function showTip(item, text) {
        if (text === "")
            return
        tip.parent = item
        tip.text = text
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

    component Hint: HoverHandler {
        property string text: ""
        onHoveredChanged: hovered ? root.showTip(parent, text) : root.hideTip(parent)
    }

    component FocusRing: Rectangle {
        property Item control: parent
        anchors.fill: parent
        anchors.margins: -3
        radius: height / 2 < 9 ? height / 2 : 9
        color: "transparent"
        border.width: 2
        border.color: root.t.accent
        visible: control.visualFocus
    }

    component Caption: Text {
        property string tip: ""
        color: root.t.text_muted
        font.pixelSize: 13
        elide: Text.ElideRight
        Hint { text: parent.tip }
    }

    component Title: Text {
        color: root.t.text_primary
        font.pixelSize: 13
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

    component Toggle: AbstractButton {
        id: toggle
        property var proxy
        readonly property bool on: proxy ? proxy.checked : false
        // Colors follow this rather than `enabled`, so a row can own the click.
        property bool live: enabled
        enabled: proxy ? proxy.enabled : false
        implicitHeight: 22
        implicitWidth: track.width + (label.text === "" ? 0 : label.implicitWidth + 10)
        onClicked: proxy.click()
        Accessible.role: Accessible.CheckBox
        Accessible.name: proxy && proxy.name !== "" ? proxy.name : text
        Accessible.checkable: true
        Accessible.checked: on
        Hint { text: toggle.proxy ? toggle.proxy.toolTip : "" }
        background: null
        contentItem: Item {
            Rectangle {
                id: track
                width: 40
                height: 22
                radius: 11
                color: !toggle.live ? root.t.action_disabled
                       : toggle.on ? root.t.accent : root.t.border_strong
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
                color: toggle.enabled ? root.t.text_primary : root.t.action_disabled_text
                font.pixelSize: 13
            }
        }
    }

    component Fader: Slider {
        id: fader
        property var proxy
        from: proxy ? proxy.minimum : 0
        to: proxy ? proxy.maximum : 1
        value: proxy ? proxy.value : 0
        stepSize: 0
        enabled: proxy ? proxy.enabled : false
        implicitHeight: 24
        onMoved: proxy.setValue(value)
        onPressedChanged: if (!pressed) proxy.released()
        Keys.onLeftPressed: nudge(-1)
        Keys.onRightPressed: nudge(1)
        function nudge(direction) {
            proxy.setValue(proxy.value + direction * (proxy.step > 0 ? proxy.step : (to - from) / 100))
            proxy.released()
        }
        Accessible.name: proxy ? proxy.name : ""
        Hint { text: fader.proxy ? fader.proxy.toolTip : "" }
        background: Rectangle {
            x: fader.leftPadding
            y: fader.topPadding + fader.availableHeight / 2 - 2
            width: fader.availableWidth
            height: 4
            radius: 2
            color: root.t.border_strong
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
        border.color: input.activeFocus ? root.t.accent : root.t.border
        Hint { text: box.proxy ? box.proxy.toolTip : "" }
        TextInput {
            id: input
            anchors.fill: parent
            anchors.margins: 6
            horizontalAlignment: TextInput.AlignHCenter
            verticalAlignment: TextInput.AlignVCenter
            text: box.source ? box.source.text : ""
            readOnly: box.display !== null
            activeFocusOnTab: !readOnly
            enabled: box.proxy ? box.proxy.enabled : false
            color: root.t.text_primary
            selectionColor: root.t.accent
            selectedTextColor: root.t.text_on_accent
            selectByMouse: true
            clip: true
            font.pixelSize: 13
            font.features: { "tnum": 1 }
            Accessible.role: Accessible.EditableText
            Accessible.name: box.proxy ? box.proxy.name : ""
            onEditingFinished: if (!readOnly && text !== box.source.text) box.proxy.setText(text)
            Keys.onUpPressed: step(1)
            Keys.onDownPressed: step(-1)
            // Only a focused field scrolls, so the page can scroll past it.
            WheelHandler {
                enabled: input.activeFocus
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
        model: proxy ? proxy.items : []
        currentIndex: proxy ? proxy.index : -1
        enabled: proxy ? proxy.enabled : false
        opacity: enabled ? 1 : 0.45
        implicitHeight: 32
        implicitWidth: 170
        leftPadding: 12
        rightPadding: 30
        onActivated: index => proxy.setIndex(index)
        // Replacing the model resets the index; read it again afterwards.
        onModelChanged: currentIndex = Qt.binding(() => proxy ? proxy.index : -1)
        Accessible.name: proxy ? proxy.name : ""
        Hint { text: picker.proxy ? picker.proxy.toolTip : "" }
        contentItem: Text {
            text: picker.displayText
            color: root.t.text_primary
            font.pixelSize: 13
            elide: Text.ElideRight
            verticalAlignment: Text.AlignVCenter
        }
        indicator: Text {
            x: picker.width - width - 11
            y: (picker.height - height) / 2
            text: root.m.glyphs.expand
            font.family: root.t.iconFont
            font.pixelSize: 10
            color: root.t.text_muted
        }
        background: Rectangle {
            radius: root.t.radiusControl
            color: picker.hovered || picker.down ? root.t.control_hover : root.t.control_surface
            border.color: picker.visualFocus ? root.t.accent : root.t.border
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
                color: root.t.text_primary
                font.pixelSize: 13
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
            implicitHeight: Math.min(contentItem.implicitHeight + 8, 320)
            contentItem: ListView {
                clip: true
                implicitHeight: contentHeight
                model: picker.delegateModel
                currentIndex: picker.highlightedIndex
            }
            background: Rectangle {
                color: root.t.control_surface
                border.color: root.t.border_strong
                radius: root.t.radiusControl
            }
        }
    }

    component Btn: AbstractButton {
        id: btn
        property var proxy
        // "plain", "primary", "danger" or "quiet"
        property string look: "plain"
        text: proxy ? proxy.text : ""
        enabled: proxy ? proxy.enabled : false
        opacity: enabled ? 1 : 0.45
        implicitHeight: 32
        implicitWidth: caption.implicitWidth + (look === "quiet" ? 0 : 28)
        onClicked: if (proxy) root.press(this, proxy)
        Accessible.role: Accessible.Button
        Accessible.name: proxy && proxy.name !== "" ? proxy.name : text
        Hint { text: btn.proxy ? btn.proxy.toolTip : "" }
        background: Rectangle {
            radius: root.t.radiusControl
            visible: btn.look !== "quiet"
            color: btn.look === "primary" ? (btn.hovered ? root.t.accent_hover : root.t.accent)
                 : btn.look === "danger" ? root.t.action_destructive
                 : btn.hovered ? root.t.control_hover : root.t.control_surface
            border.color: btn.look === "plain" ? root.t.border
                        : btn.look === "danger" ? root.t.action_destructive_border : "transparent"
            scale: btn.pressed ? 0.98 : 1
            FocusRing { control: btn }
        }
        contentItem: Text {
            id: caption
            text: btn.text
            color: btn.look === "primary" ? root.t.text_on_accent
                 : btn.look === "danger" ? root.t.text_on_emphasis
                 : btn.look === "quiet" && !btn.hovered && !btn.visualFocus ? root.t.text_muted
                 : root.t.text_primary
            font.pixelSize: 13
            font.weight: btn.look === "plain" || btn.look === "quiet" ? Font.Normal : Font.DemiBold
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            elide: Text.ElideRight
        }
    }

    component IconBtn: AbstractButton {
        id: icon
        property var proxy: null
        property string glyph: ""
        property string hint: proxy ? proxy.toolTip : ""
        visible: root.t.iconFont !== ""
        enabled: proxy ? proxy.enabled : true
        implicitWidth: 30
        implicitHeight: 30
        onClicked: if (proxy) root.press(this, proxy)
        Accessible.role: Accessible.Button
        Accessible.name: proxy && proxy.name !== "" ? proxy.name : hint
        Hint { text: icon.hint }
        background: Rectangle {
            radius: root.t.radiusControl
            color: icon.hovered ? root.t.control_hover : "transparent"
            FocusRing { control: icon }
        }
        contentItem: Text {
            text: icon.glyph
            font.family: root.t.iconFont
            font.pixelSize: 15
            color: icon.hovered ? root.t.text_primary : root.t.text_muted
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
        }
    }

    component Chip: Rectangle {
        id: chip
        property var proxy
        readonly property string state_: proxy && proxy.state !== "" ? proxy.state : "idle"
        implicitHeight: chipText.implicitHeight + 12
        implicitWidth: chipText.implicitWidth + 24
        radius: root.t.radiusControl
        color: root.t["status_" + state_ + "_surface"] ?? root.t.status_idle_surface
        border.color: root.t["status_" + state_ + "_border"] ?? root.t.status_idle_border
        Hint { text: chip.proxy ? chip.proxy.toolTip : "" }
        Text {
            id: chipText
            anchors.centerIn: parent
            width: Math.min(implicitWidth, chip.width - 24)
            text: chip.proxy ? chip.proxy.text : ""
            color: root.t["status_" + chip.state_ + "_text"] ?? root.t.status_idle_text
            font.pixelSize: 13
            wrapMode: Text.Wrap
            Accessible.name: chip.proxy ? chip.proxy.name + ": " + text : text
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
                text: textRow, meter: meterRow
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
                text: line.p.text
                color: line.row.label !== "" ? root.t.text_primary : root.t.text_muted
                font.pixelSize: 13
                wrapMode: Text.Wrap
                horizontalAlignment: line.row.label !== "" ? Text.AlignRight : Text.AlignLeft
                Accessible.name: line.p.name !== "" ? line.p.name + ": " + text : text
            }
        }
        Component { id: meterRow; WidgetItem { proxy: line.p; implicitHeight: 18 } }
    }

    component CardHeader: RowLayout {
        id: header
        property var card
        spacing: 12
        Toggle { proxy: header.card.toggle; visible: header.card.toggle !== null }
        Title { text: header.card.title }
        IconBtn {
            glyph: root.m.glyphs.help
            hint: header.card.help
            visible: hint !== "" && root.t.iconFont !== ""
            onClicked: root.showTip(this, hint)
        }
        Item { Layout.fillWidth: true }
    }

    // A full-width row that opens the section under it.
    component Disclosure: AbstractButton {
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
            border.color: disclosure.visualFocus ? root.t.accent : root.t.border
            Behavior on color { ColorAnimation { duration: root.t.motion } }
        }
        contentItem: RowLayout {
            spacing: 8
            Text {
                text: disclosure.text
                color: root.t.text_primary
                font.pixelSize: 13
                font.weight: Font.Medium
                leftPadding: 12
                Layout.fillWidth: true
            }
            Text {
                text: disclosure.open ? "Hide" : "Show"
                color: root.t.text_muted
                font.pixelSize: 12
            }
            Text {
                text: root.m.glyphs.expand
                visible: root.t.iconFont !== ""
                font.family: root.t.iconFont
                font.pixelSize: 10
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
        readonly property bool on: card.toggle === null || card.toggle.checked
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
                opacity: stage.on ? 1 : 0.5
                Layout.fillWidth: true
            }
        }
        ColumnLayout {
            visible: stage.card.trace && ui.levels.length > 1
            spacing: 6
            Layout.fillWidth: true
            LevelTrace {
                points: ui.levels
                Layout.fillWidth: true
                implicitHeight: 52
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
                            text: parent.modelData[0]
                            color: root.t.text_muted
                            font.pixelSize: 11
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
                opacity: stage.on ? 1 : 0.5
                Layout.fillWidth: true
            }
        }
    }

    // One line of the Settings page: a switch, an action, or a menu.
    component SettingRow: AbstractButton {
        id: setting
        property var row
        readonly property var p: row.proxy
        readonly property bool isSwitch: row.kind === "toggle"
        visible: p.shown
        enabled: p.enabled
        implicitHeight: 44
        onClicked: root.press(this, p)
        Accessible.role: isSwitch ? Accessible.CheckBox : Accessible.Button
        Accessible.name: p.text
        Accessible.checkable: isSwitch
        Accessible.checked: p.checked
        Hint { text: setting.p.toolTip !== setting.p.text ? setting.p.toolTip : "" }
        background: Rectangle {
            radius: root.t.radiusControl
            color: setting.hovered && setting.enabled ? root.t.control_surface_alt : "transparent"
            border.width: setting.visualFocus ? 2 : 0
            border.color: root.t.accent
        }
        contentItem: RowLayout {
            spacing: 12
            Text {
                text: setting.p.text
                color: setting.enabled ? root.t.text_primary : root.t.action_disabled_text
                font.pixelSize: 13
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
                Layout.rightMargin: 10
            }
            Text {
                visible: !setting.isSwitch && setting.enabled && root.t.iconFont !== ""
                text: setting.row.kind === "menu" ? root.m.glyphs.expand : root.m.glyphs.next
                Layout.rightMargin: 14
                font.family: root.t.iconFont
                font.pixelSize: 10
                color: root.t.text_muted
            }
        }
    }

    // One line of the Health page. The chip text is "Title: value".
    component HealthRow: RowLayout {
        id: health
        property var row
        property bool technical: false
        readonly property var proxy: row.proxy
        readonly property string value: proxy.text.startsWith(row.title + ":")
                                        ? proxy.text.slice(row.title.length + 1).trim() : proxy.text
        readonly property string state_: proxy.state !== "" ? proxy.state : "idle"
        // Counter rows show a plain state until Technical details is open.
        readonly property bool coded: row.coded
        spacing: 10
        Rectangle {
            width: 8
            height: 8
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
            text: health.coded && !health.technical
                  ? ({ok: "OK", warn: "Needs attention", bad: "Problem", info: "Info", idle: "Idle"})[health.state_] ?? "--"
                  : health.coded ? health.value
                  : health.value.charAt(0).toUpperCase() + health.value.slice(1)
            color: health.coded && health.technical ? root.t.data_text_muted : root.t.text_primary
            font.pixelSize: 13
            font.family: health.coded && health.technical ? "Cascadia Mono"
                                                          : Qt.application.font.family
            wrapMode: Text.Wrap
            Layout.fillWidth: true
            Accessible.name: health.proxy.name + ": " + text
        }
    }

    component Page: Flickable {
        id: page
        default property alias content: column.data
        contentHeight: column.implicitHeight + 24
        boundsBehavior: Flickable.StopAtBounds
        clip: true
        ScrollBar.vertical: ScrollBar {
            id: bar
            policy: page.contentHeight > page.height ? ScrollBar.AlwaysOn : ScrollBar.AlwaysOff
            contentItem: Rectangle {
                implicitWidth: 6
                radius: 3
                color: bar.pressed || bar.hovered ? root.t.text_muted : root.t.border_strong
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
            Repeater {
                model: [[root.m.glyphs.mic, "Mic"], [root.m.glyphs.health, "Health"],
                        [root.m.glyphs.settings, "Settings"]]
                delegate: AbstractButton {
                    id: nav
                    required property var modelData
                    required property int index
                    readonly property bool current: root.m.page.index === index
                    width: 76
                    height: 66
                    onClicked: root.m.page.setIndex(index)
                    Accessible.role: Accessible.PageTab
                    Accessible.name: modelData[1] + " page"
                    background: Rectangle {
                        color: nav.current || nav.hovered || nav.visualFocus
                               ? root.t.card_surface : "transparent"
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
                    }
                    contentItem: Column {
                        topPadding: 12
                        spacing: 6
                        Text {
                            anchors.horizontalCenter: parent.horizontalCenter
                            text: nav.modelData[0]
                            visible: root.t.iconFont !== ""
                            color: nav.current ? root.t.accent
                                 : nav.hovered ? root.t.text_primary : root.t.text_muted
                            font.family: root.t.iconFont
                            font.pixelSize: 20
                        }
                        Text {
                            anchors.horizontalCenter: parent.horizontalCenter
                            text: nav.modelData[1]
                            color: nav.current || nav.hovered ? root.t.text_primary : root.t.text_muted
                            font.pixelSize: 11
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
                delegate: WidgetItem {
                    required property var modelData
                    proxy: modelData
                    width: 50
                    height: meterRow.height
                    Accessible.name: modelData.name
                }
            }
        }
    }

    ColumnLayout {
        anchors.left: rail.right
        anchors.right: meters.left
        anchors.top: parent.top
        anchors.bottom: parent.bottom
        anchors.margins: 18
        anchors.leftMargin: 10
        anchors.rightMargin: 0
        anchors.bottomMargin: 10
        spacing: 12

        Repeater {
            model: root.m.banners
            delegate: Rectangle {
                required property var modelData
                visible: modelData.shown
                Layout.fillWidth: true
                Layout.leftMargin: 8
                Layout.rightMargin: 16
                implicitHeight: bannerText.implicitHeight + 16
                radius: root.t.radiusControl
                color: root.t.warning_banner_surface
                Text {
                    id: bannerText
                    anchors.fill: parent
                    anchors.margins: 8
                    text: parent.modelData.text
                    color: root.t.warning_banner_text
                    font.pixelSize: 13
                    wrapMode: Text.Wrap
                    Accessible.name: parent.modelData.name + ": " + text
                }
            }
        }

        // Top bar: route and transport
        Surface {
            Layout.fillWidth: true
            Layout.leftMargin: 8
            Layout.rightMargin: 16
            pad: 14
            RowLayout {
                spacing: 12
                Caption { text: "Input"; visible: root.wide }
                Picker { proxy: root.m.top.input; Layout.fillWidth: true; Layout.preferredWidth: 200 }
                Text {
                    text: root.m.glyphs.next
                    visible: root.t.iconFont !== ""
                    font.family: root.t.iconFont
                    font.pixelSize: 11
                    color: root.t.text_muted
                }
                Caption { text: "Output"; visible: root.wide }
                Picker { proxy: root.m.top.output; Layout.fillWidth: true; Layout.preferredWidth: 200 }
                IconBtn { proxy: root.m.top.refresh; glyph: root.m.glyphs.refresh }
                Rectangle {
                    width: 1
                    color: root.t.border
                    Layout.fillHeight: true
                    Layout.topMargin: 4
                    Layout.bottomMargin: 4
                }
                Picker { proxy: root.m.top.mode; implicitWidth: 140 }
                Toggle { proxy: root.m.top.mute; text: "Mute" }
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

        StackLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            currentIndex: root.m.page.index

            // Mic page
            Page {
                objectName: "micPage"
                RowLayout {
                    id: presetBar
                    readonly property var preset: root.m.presets.status.data
                    Layout.fillWidth: true
                    Layout.topMargin: 2
                    spacing: 12
                    Accessible.name: root.m.presets.status.name + ": " + root.m.presets.status.text
                    ColumnLayout {
                        spacing: 2
                        Layout.fillWidth: true
                        Text {
                            text: "PRESET"
                            color: root.t.text_muted
                            font.pixelSize: 10
                            font.letterSpacing: 1.2
                            font.weight: Font.DemiBold
                        }
                        RowLayout {
                            spacing: 10
                            Text {
                                text: presetBar.preset.name
                                color: root.t.text_primary
                                font.pixelSize: 19
                                font.weight: Font.DemiBold
                                elide: Text.ElideRight
                                Layout.maximumWidth: 420
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
                                    anchors.centerIn: parent
                                    text: presetBar.preset.modified ? "Unsaved changes" : "Saved"
                                    color: presetBar.preset.modified ? root.t.status_warn_text
                                                                     : root.t.status_idle_text
                                    font.pixelSize: 11
                                }
                            }
                            Item { Layout.fillWidth: true }
                        }
                    }
                    Btn { proxy: root.m.presets.menu }
                    Btn { proxy: root.m.presets.undo }
                    Btn { proxy: root.m.presets.testSound }
                    Btn { proxy: root.m.presets.autoEq }
                    Btn { proxy: root.m.presets.voiceSetup; look: "primary" }
                }

                Surface {
                    id: eqCard
                    readonly property var eq: root.m.eq
                    readonly property var band: eq.bands[eq.band.index]
                    readonly property bool on: eq.toggle.checked
                    readonly property bool calibrated: eq.diagnostics.data.calibrated
                    Layout.fillWidth: true
                    CardHeader {
                        card: eqCard.eq
                        Layout.fillWidth: true
                        Caption { text: eqCard.eq.layers.text; visible: root.wide }
                        Btn { proxy: eqCard.eq.tone }
                        IconBtn { proxy: eqCard.eq.menu; glyph: root.m.glyphs.more }
                    }
                    WidgetItem {
                        proxy: eqCard.eq.curve
                        Layout.fillWidth: true
                        implicitHeight: 300
                        activeFocusOnTab: true
                        opacity: eqCard.on ? 1 : 0.5
                        Accessible.name: "Equalizer graph"
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
                        columns: root.wide ? 2 : 1
                        columnSpacing: 12
                        rowSpacing: 10
                        opacity: eqCard.on ? 1 : 0.5
                        Layout.fillWidth: true
                        RowLayout {
                            spacing: 12
                            Layout.fillWidth: true
                            IconBtn {
                                glyph: root.m.glyphs.previous
                                hint: "Previous EQ band"
                                onClicked: ui.stepBand(-1)
                            }
                            IconBtn {
                                glyph: root.m.glyphs.next
                                hint: "Next EQ band"
                                onClicked: ui.stepBand(1)
                            }
                            Rectangle {
                                width: 12
                                height: 12
                                radius: 6
                                color: root.t.bandColors[eqCard.eq.band.index]
                            }
                            Text {
                                text: eqCard.band.frequencyLabel.text
                                color: root.t.text_primary
                                font.pixelSize: 15
                                font.weight: Font.DemiBold
                                Layout.preferredWidth: 58
                            }
                            Toggle { proxy: eqCard.band.enabled }
                            Picker { proxy: eqCard.band.type; implicitWidth: 130 }
                            Caption { text: "Gain" }
                            Fader { proxy: eqCard.band.gain; Layout.fillWidth: true }
                            ValueBox {
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

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 16
                    Repeater {
                        // Signal order reads left to right, then down.
                        model: root.wide ? [[0, 2, 4], [1, 3]] : [[0, 1, 2, 3, 4]]
                        delegate: ColumnLayout {
                            id: lane
                            required property var modelData
                            spacing: 16
                            Layout.alignment: Qt.AlignTop
                            Layout.fillWidth: true
                            Layout.preferredWidth: 1
                            Repeater {
                                model: lane.modelData
                                delegate: StageCard {
                                    required property int modelData
                                    card: root.m.stages[modelData]
                                    Layout.fillWidth: true
                                }
                            }
                        }
                    }
                }
            }

            // Health page
            Page {
                Surface {
                    Layout.fillWidth: true
                    Text {
                        text: root.m.health.advice.text
                        color: root.t.text_primary
                        font.pixelSize: 18
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
                                technical: details.open
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
                        text: "Raw counters are shown in the rows above. Hover a row for what its codes mean."
                        wrapMode: Text.Wrap
                        Layout.fillWidth: true
                    }
                    RowLayout {
                        visible: details.open
                        spacing: 12
                        Btn {
                            text: "Export diagnostics"
                            enabled: true
                            onClicked: ui.exportDiagnostics()
                        }
                        Btn {
                            text: "Reset drop counter"
                            enabled: true
                            onClicked: ui.resetDrops()
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
            Layout.fillWidth: true
            Layout.leftMargin: 8
            Layout.rightMargin: 16
            spacing: 14
            Caption {
                text: root.m.status.message.text
                Layout.fillWidth: true
            }
            Rectangle {
                width: 8
                height: 8
                radius: 4
                color: root.running ? root.t.status_ok_text : root.t.border_strong
            }
            Caption { text: root.m.status.transmission.text; color: root.t.text_primary }
            Caption { text: root.m.status.calibration.text }
            Chip {
                proxy: root.m.status.health
                implicitHeight: 28
                TapHandler {
                    onTapped: root.m.page.setIndex(
                        root.m.page.index === root.m.healthPage ? 0 : root.m.healthPage)
                }
            }
        }
    }
}
