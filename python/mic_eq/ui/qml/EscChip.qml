// A keycap that names the key doing what the control beside it does.
import QtQuick

Rectangle {
    id: chip
    property string text: "Esc"
    implicitWidth: cap.implicitWidth + 12
    implicitHeight: 22
    radius: Look.t.radiusControl
    color: Look.t.control_surface
    border.color: Look.t.control_edge
    Text {
        id: cap
        anchors.centerIn: parent
        text: chip.text
        textFormat: Text.PlainText
        color: Look.t.text_muted
        font.pixelSize: Look.fsTick
        Accessible.ignored: true
    }
}
