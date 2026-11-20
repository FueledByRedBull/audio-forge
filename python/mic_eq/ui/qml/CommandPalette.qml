// The command palette over the main scene: a search field above the window's commands. Typing
// filters the rows, Up and Down select one, Enter runs it and Esc closes without changing
// anything. Left and Right step the draft of a numeric setting, which only Enter applies.
// `commands` is the model (command_palette.CommandPalette); `backdrop` is the pane the glass
// blurs, a sibling laid out in this item's own coordinates.
pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls.Basic

Item {
    id: overlay
    required property var commands
    property Item backdrop: null
    // Where focus was when the palette opened; it goes back there when the palette closes.
    property Item origin: null
    // Where focus goes instead when the origin is gone or hidden by then.
    property Item fallbackFocus: null
    readonly property bool shown: commands.shown
    // Whether the running transition opens the panel; set before it starts, so its length is
    // the opening or the closing one.
    property bool opening: false
    // 0 while closed and 1 while open; the panel drops in by it.
    property real progress: 0
    readonly property real drop: 12

    visible: shown || opacity > 0
    opacity: 0

    Behavior on opacity {
        NumberAnimation { id: fade; objectName: "paletteFade"; duration: Look.t.motionCrossfade }
    }
    // Under reduced motion the panel does not drop in; it only fades.
    Behavior on progress {
        NumberAnimation {
            id: move
            objectName: "paletteMove"
            duration: Look.movement(overlay.opening ? Look.t.motionPanelOpen : Look.t.motionPanelClose)
            easing.type: Easing.BezierSpline
            easing.bezierCurve: Look.t.easePanelOpen
        }
    }

    onShownChanged: {
        opening = shown
        progress = shown ? 1 : 0
        opacity = shown ? 1 : 0
        if (shown) {
            origin = Window.activeFocusItem
            field.text = ""
            field.forceActiveFocus(Qt.ShortcutFocusReason)
        } else {
            const back = origin && origin.visible && origin.enabled ? origin : fallbackFocus
            origin = null
            // As a keyboard move, so the control's focus ring shows where focus landed.
            if (back)
                back.forceActiveFocus(Qt.TabFocusReason)
        }
    }

    Connections {
        target: overlay.commands
        // The rows are a new list after each change; keep the selected one in view.
        function onChanged() {
            Qt.callLater(overlay.reveal)
        }
    }
    function plain(event) {
        return (event.modifiers & (Qt.ShiftModifier | Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier)) === 0
    }
    function reveal() {
        if (commands.current >= 0)
            list.positionViewAtIndex(commands.current, ListView.Contain)
    }

    // A press outside the panel closes it, and nothing under it is reached while it shows.
    MouseArea {
        anchors.fill: parent
        enabled: overlay.shown
        acceptedButtons: Qt.AllButtons
        hoverEnabled: true
        onPressed: overlay.commands.close()
        onWheel: wheel => wheel.accepted = true
        Accessible.ignored: true
    }

    GlassPanel {
        id: panel
        objectName: "paletteGlass"
        readonly property Item area: overlay.backdrop
        modal: true
        width: Math.min(560, (area ? area.width : overlay.width) - 32)
        height: content.implicitHeight + 24
        x: area ? area.x + (area.width - width) / 2 : (overlay.width - width) / 2
        y: (area ? area.y : 0) + 24
        backdrop: area
        backdropRect: area ? Qt.rect(x - area.x, y + slide.y - area.y, width, height) : Qt.rect(0, 0, 0, 0)
        transform: Translate { id: slide; y: (overlay.progress - 1) * overlay.drop }
        Accessible.role: Accessible.Dialog
        Accessible.name: "Command palette"

        Column {
            id: content
            x: 12
            y: 12
            width: panel.width - 24
            spacing: 8

            TextField {
                id: field
                objectName: "paletteSearch"
                width: parent.width
                implicitHeight: 40
                leftPadding: 12
                rightPadding: 12
                placeholderText: "Search commands"
                placeholderTextColor: Look.t.text_muted
                color: Look.t.text_primary
                selectionColor: Look.t.accent
                selectedTextColor: Look.t.text_on_accent
                font.pixelSize: Look.fsLead
                Accessible.name: "Search commands"
                Accessible.description: "Up and Down select a command, Enter runs it, Esc closes."
                onTextChanged: overlay.commands.setQuery(text)
                Keys.onUpPressed: overlay.commands.move(-1)
                Keys.onDownPressed: overlay.commands.move(1)
                // On a numeric row a plain arrow steps its value; elsewhere, or with Shift or
                // Ctrl held, the arrows move the cursor.
                Keys.onLeftPressed: event => event.accepted = overlay.plain(event) && overlay.commands.adjust(-1)
                Keys.onRightPressed: event => event.accepted = overlay.plain(event) && overlay.commands.adjust(1)
                Keys.onReturnPressed: event => { if (!event.isAutoRepeat) overlay.commands.run() }
                Keys.onEnterPressed: event => { if (!event.isAutoRepeat) overlay.commands.run() }
                Keys.onEscapePressed: overlay.commands.close()
                // Focus stays in the palette until it closes.
                Keys.onTabPressed: event => event.accepted = true
                Keys.onBacktabPressed: event => event.accepted = true
                background: Rectangle {
                    radius: Look.t.radiusControl
                    color: Look.t.control_surface
                    border.color: Look.t.control_edge
                    FocusRing { visible: field.activeFocus }
                }
            }

            ListView {
                id: list
                objectName: "paletteList"
                width: parent.width
                height: Math.min(contentHeight, Math.max(120, overlay.height * 0.55))
                visible: count > 0
                clip: true
                interactive: contentHeight > height
                boundsBehavior: Flickable.StopAtBounds
                keyNavigationEnabled: false
                activeFocusOnTab: false
                model: overlay.commands.rows
                Accessible.role: Accessible.List
                Accessible.name: "Commands"
                delegate: Rectangle {
                    id: row
                    required property var modelData
                    required property int index
                    readonly property bool selected: index === overlay.commands.current
                    // The selected and hovered fills are the highlight under a contrast theme: all their text is the highlighted text.
                    readonly property bool plainInk: (selected || pointer.containsMouse) && Look.t.highContrast
                    width: ListView.view.width
                    height: Math.max(44, texts.implicitHeight + 12)
                    radius: Look.t.radiusControl
                    color: selected ? Look.t.accent_fill : pointer.containsMouse ? Look.t.control_hover : "transparent"
                    border.color: selected ? Look.t.accent_edge : "transparent"
                    enabled: modelData.enabled
                    Accessible.role: Accessible.ListItem
                    Accessible.name: modelData.value ? modelData.title + ": " + modelData.value : modelData.title
                    Accessible.description: modelData.description
                    Accessible.selectable: true
                    Accessible.selected: selected
                    // Activating a row from assistive technology runs it, as Enter does.
                    Accessible.onPressAction: {
                        overlay.commands.select(row.index)
                        overlay.commands.run()
                    }
                    MouseArea {
                        id: pointer
                        anchors.fill: parent
                        hoverEnabled: true
                        // A click selects a row; a click on the selected row runs it.
                        onClicked: overlay.commands.clickAt(row.index)
                        Accessible.ignored: true
                    }
                    Column {
                        id: texts
                        anchors.left: parent.left
                        anchors.right: trailing.left
                        anchors.leftMargin: 10
                        anchors.rightMargin: 8
                        anchors.verticalCenter: parent.verticalCenter
                        spacing: 2
                        Text {
                            width: parent.width
                            text: row.modelData.title
                            textFormat: Text.PlainText
                            elide: Text.ElideRight
                            color: row.plainInk ? Look.t.hover_text
                                : !row.enabled ? (row.selected ? Look.t.text_muted : Look.t.action_disabled_text)
                                : row.selected ? Look.t.hover_text : Look.t.text_primary
                            font.pixelSize: Look.fsBody
                            Accessible.ignored: true
                        }
                        Row {
                            width: parent.width
                            spacing: 6
                            Text {
                                width: Math.min(implicitWidth, parent.width - (callers.visible ? callers.implicitWidth + 6 : 0))
                                text: row.modelData.detail
                                textFormat: Text.PlainText
                                elide: Text.ElideRight
                                color: row.plainInk ? Look.t.hover_text : Look.t.text_muted
                                font.pixelSize: Look.fsCaption
                                Accessible.ignored: true
                            }
                            Text {
                                id: callers
                                objectName: "paletteCallers"
                                visible: text !== ""
                                text: row.modelData.callers
                                textFormat: Text.PlainText
                                color: row.plainInk ? Look.t.hover_text : Look.t.callers_affected_text
                                font.pixelSize: Look.fsCaption
                                Accessible.ignored: true
                            }
                        }
                    }
                    Row {
                        id: trailing
                        anchors.right: parent.right
                        anchors.rightMargin: 10
                        anchors.verticalCenter: parent.verticalCenter
                        spacing: 4
                        Text {
                            anchors.verticalCenter: parent.verticalCenter
                            text: row.modelData.value
                            textFormat: Text.PlainText
                            color: row.plainInk ? Look.t.hover_text
                                : row.modelData.draft ? Look.t.accent_text : Look.t.text_muted
                            font.pixelSize: Look.fsBody
                            font.weight: row.modelData.draft ? Font.DemiBold : Font.Normal
                            font.features: { "tnum": 1 }
                            Accessible.ignored: true
                        }
                        Text {
                            anchors.verticalCenter: parent.verticalCenter
                            leftPadding: 6
                            text: row.modelData.group
                            textFormat: Text.PlainText
                            color: row.plainInk ? Look.t.hover_text : row.selected ? Look.t.text_muted : Look.t.text_dim
                            font.pixelSize: Look.fsCaption
                            Accessible.ignored: true
                        }
                    }
                }
            }

            Text {
                objectName: "paletteEmpty"
                width: parent.width
                visible: list.count === 0
                topPadding: 8
                bottomPadding: 8
                horizontalAlignment: Text.AlignHCenter
                text: overlay.commands.noMatchText
                textFormat: Text.PlainText
                color: Look.t.text_muted
                font.pixelSize: Look.fsBody
            }

            // The keys, named as they are on the keyboard.
            Flow {
                width: parent.width
                spacing: 12
                Repeater {
                    model: [
                        { "keys": ["Up", "Down"], "text": "select" },
                        { "keys": ["Enter"], "text": "run" },
                        { "keys": ["Left", "Right"], "text": "change a value" },
                        { "keys": ["Esc"], "text": "close" },
                    ]
                    delegate: Row {
                        id: hint
                        required property var modelData
                        spacing: 4
                        Accessible.role: Accessible.StaticText
                        Accessible.name: modelData.keys.join(" and ") + ": " + modelData.text
                        Repeater {
                            model: hint.modelData.keys
                            delegate: EscChip {
                                required property string modelData
                                text: modelData
                            }
                        }
                        Text {
                            anchors.verticalCenter: parent.verticalCenter
                            text: hint.modelData.text
                            textFormat: Text.PlainText
                            color: Look.t.text_muted
                            font.pixelSize: Look.fsCaption
                            Accessible.ignored: true
                        }
                    }
                }
            }
        }
    }
}
