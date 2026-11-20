// The cross that closes a dialog or a panel, drawn as two strokes so it needs no icon font.
// Esc does the same, so that is its description.
pragma ComponentBehavior: Bound
import QtQuick

KeyButton {
    id: closer
    implicitWidth: 30
    implicitHeight: 30
    Accessible.role: Accessible.Button
    Accessible.name: "Close"
    Accessible.description: "Esc"
    background: Rectangle {
        radius: Look.t.radiusControl
        color: closer.hovered ? Look.t.control_hover : "transparent"
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
                color: closer.hovered ? Look.t.hover_text : Look.t.text_muted
            }
        }
    }
}
