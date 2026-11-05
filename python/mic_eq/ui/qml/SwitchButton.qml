// An on/off switch with its caption (`text`). The state is the model's `on`; see ModelButton.
import QtQuick

ModelButton {
    id: toggle
    // Colors follow this rather than `enabled`, so a row can own the click.
    property bool live: enabled
    implicitHeight: 22
    implicitWidth: track.width + (label.text === "" ? 0 : label.implicitWidth + 10)
    Accessible.role: Accessible.CheckBox
    background: null
    contentItem: Item {
        Rectangle {
            id: track
            width: 40
            height: 22
            radius: 11
            color: !toggle.live ? Look.t.action_disabled
                   : toggle.on ? Look.t.accent : Look.t.control_edge
            Behavior on color { ColorAnimation { duration: Look.t.motion } }
            Rectangle {
                width: 16
                height: 16
                radius: 8
                y: 3
                x: toggle.on ? 21 : 3
                color: toggle.live ? Look.t.text_primary : Look.t.action_disabled_text
                Behavior on x {
                    NumberAnimation { duration: Look.t.motion; easing.type: Easing.OutCubic }
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
            color: toggle.enabled ? Look.t.text_primary : Look.t.action_disabled_text
            font.pixelSize: Look.fsBody
        }
    }
}
