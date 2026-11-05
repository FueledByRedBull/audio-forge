// A text button. `look` is "plain", "primary" or "danger".
import QtQuick

KeyButton {
    id: btn
    property string look: "plain"
    opacity: enabled ? 1 : 0.45
    implicitHeight: 32
    implicitWidth: caption.implicitWidth + 28
    Accessible.role: Accessible.Button
    Accessible.name: text
    background: Rectangle {
        radius: Look.t.radiusControl
        color: btn.look === "primary" ? (btn.hovered ? Look.t.accent_hover : Look.t.accent)
             : btn.look === "danger" ? Look.t.action_destructive
             : btn.hovered ? Look.t.control_hover : Look.t.control_surface
        border.color: btn.look === "plain" ? Look.t.control_edge
                    : btn.look === "danger" ? Look.t.action_destructive_border : "transparent"
        scale: btn.pressed ? 0.98 : 1
        FocusRing { control: btn }
    }
    contentItem: Text {
        id: caption
        text: btn.text
        textFormat: Text.PlainText
        color: btn.look === "primary" ? Look.t.text_on_accent
             : btn.look === "danger" ? Look.t.text_on_emphasis
             : btn.hovered ? Look.t.hover_text : Look.t.text_primary
        font.pixelSize: Look.fsBody
        font.weight: btn.look === "plain" ? Font.Normal : Font.DemiBold
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }
}
