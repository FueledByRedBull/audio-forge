// The live output is muted: the mute's own colours with the words leading, so the state is
// not told by colour alone. `text` is the notice that follows the lead.
import QtQuick

Rectangle {
    id: banner
    property string text
    // Plain anchors rather than nested layouts, so the height is right from the first frame
    // and a window sized from it at load fits it.
    implicitHeight: 10 + lead.implicitHeight + 2 + notice.implicitHeight + 10
    radius: Look.t.radiusControl
    color: Look.t.mute_fill
    border.color: Look.t.mute_edge
    Accessible.role: Accessible.StaticText
    Accessible.name: text
    Rectangle {
        width: 10
        height: 10
        radius: 5
        color: Look.t.mute
        anchors { left: parent.left; leftMargin: 12; top: parent.top; topMargin: 14 }
    }
    Text {
        id: lead
        objectName: "mutedLead"
        text: "Live output muted."
        textFormat: Text.PlainText
        color: Look.t.mute_text
        font.pixelSize: Look.fsBody
        font.weight: Font.Bold
        anchors { left: parent.left; leftMargin: 32; right: parent.right; rightMargin: 10; top: parent.top; topMargin: 10 }
        Accessible.ignored: true
    }
    Text {
        id: notice
        text: banner.text
        textFormat: Text.PlainText
        color: Look.t.mute_text
        font.pixelSize: Look.fsBody
        wrapMode: Text.Wrap
        anchors { left: lead.left; right: lead.right; top: lead.bottom; topMargin: 2 }
        Accessible.ignored: true
    }
}
