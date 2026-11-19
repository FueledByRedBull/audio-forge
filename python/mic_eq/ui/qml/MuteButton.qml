// Mute: it stays pressed while the output is muted, in the mute violet; its words do not change.
// `proxy` is the shared mute command.
import QtQuick

ModelButton {
    id: mute
    property var proxy
    on: proxy ? proxy.checked : false
    enabled: proxy ? proxy.enabled : false
    onRequested: proxy.click()
    opacity: enabled ? 1 : 0.45
    implicitWidth: muteMetrics.width + 28
    implicitHeight: 32
    Accessible.role: Accessible.Button
    Accessible.name: proxy ? proxy.name : ""
    Accessible.description: proxy ? proxy.toolTip : ""
    // A button that stays pressed toggles, so assistive technology can switch it as a toggle too.
    Accessible.onToggleAction: click()
    // Measured bold, so pressing it does not change its width.
    TextMetrics { id: muteMetrics; text: muteLabel.text; font.pixelSize: Look.fsBody; font.weight: Font.DemiBold }
    background: Rectangle {
        radius: Look.t.radiusControl
        color: mute.on ? Look.t.mute_fill : mute.hovered ? Look.t.control_hover : Look.t.control_surface
        border.color: mute.on ? Look.t.mute_edge : Look.t.control_edge
        Behavior on color { ColorAnimation { duration: Look.t.motionRow } }
        FocusRing { control: mute }
    }
    contentItem: Text {
        id: muteLabel
        text: "Mute"
        textFormat: Text.PlainText
        color: mute.on ? Look.t.mute_text : mute.hovered ? Look.t.hover_text : Look.t.text_primary
        font.pixelSize: Look.fsBody
        font.weight: mute.on ? Font.DemiBold : Font.Normal
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
    }
}
