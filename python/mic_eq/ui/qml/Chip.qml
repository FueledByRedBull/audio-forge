// A short message set off in the colours of its state: `tone` is ok, warn, bad, info or idle.
// A `tip` is shown on hover and read as the chip's description.
import QtQuick

Rectangle {
    id: chip
    property string text
    property string tone: "idle"
    property string tip
    implicitHeight: words.implicitHeight + 12
    radius: Look.t.radiusControl
    color: Look.t["status_" + tone + "_surface"]
    border.color: Look.t["status_" + tone + "_border"]
    Accessible.role: Accessible.StaticText
    Accessible.name: text
    Accessible.description: tip
    HoverHandler { id: hover; enabled: chip.tip !== "" }
    Tip {
        text: chip.tip
        delay: 500
        visible: chip.tip !== "" && hover.hovered
    }
    Text {
        id: words
        anchors.fill: parent
        anchors.margins: 6
        anchors.leftMargin: 10
        anchors.rightMargin: 10
        text: chip.text
        textFormat: Text.PlainText
        color: Look.t["status_" + chip.tone + "_text"]
        font.pixelSize: Look.fsCaption
        font.weight: Font.DemiBold
        wrapMode: Text.Wrap
        verticalAlignment: Text.AlignVCenter
        Accessible.ignored: true
    }
}
