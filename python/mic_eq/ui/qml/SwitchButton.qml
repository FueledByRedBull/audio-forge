// An on/off switch with its caption (`text`). The state is the model's `on`; see ModelButton.
import QtQuick

ModelButton {
    id: toggle
    // Colors follow this rather than `enabled`, so a row can own the click.
    property bool live: enabled
    // The track is smaller than the hit target, which stays 24 px high.
    implicitHeight: 24
    implicitWidth: track.width + (label.text === "" ? 0 : label.implicitWidth + 10)
    Accessible.role: Accessible.CheckBox
    background: null
    contentItem: Item {
        Rectangle {
            id: track
            width: 30
            height: 18
            radius: 9
            y: (parent.height - height) / 2
            // On is a filled light track, off an outline: the two differ by more than color.
            color: !toggle.live ? Look.t.action_disabled : toggle.on ? Look.t.switch_on : "transparent"
            border.width: toggle.live && !toggle.on ? 1 : 0
            border.color: Look.t.control_edge
            Behavior on color { ColorAnimation { duration: Look.t.motionSwitch } }
            Rectangle {
                objectName: "switchKnob"
                width: 12
                height: 12
                radius: 6
                y: 3
                x: toggle.on ? track.width - width - 3 : 3
                color: !toggle.live ? Look.t.action_disabled_text
                       : toggle.on ? Look.t.app_surface : Look.t.control_edge
                Behavior on x {
                    NumberAnimation {
                        duration: Look.movement(Look.t.motionSwitch)
                        easing.type: Easing.OutCubic
                    }
                }
                Behavior on color { ColorAnimation { duration: Look.t.motionSwitch } }
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
