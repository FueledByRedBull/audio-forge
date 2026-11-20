// The mark beside a step or a check, drawn so it needs no icon font: `shape` is "check",
// "cross", "clock", "dot", "ring" or "dash". Words always go beside it; the mark is not read.
pragma ComponentBehavior: Bound
import QtQuick

Item {
    id: mark
    property string shape: "ring"
    property color color: Look.t.text_muted
    readonly property bool round: shape !== "cross" && shape !== "dash"
    implicitWidth: 16
    implicitHeight: 16
    Accessible.ignored: true

    // A check sits on a filled disc; the dot, the ring and the clock in a ring.
    Rectangle {
        anchors.fill: parent
        radius: width / 2
        visible: mark.round
        color: mark.shape === "check" ? mark.color : "transparent"
        border.width: mark.shape === "check" ? 0 : 1.5
        border.color: mark.color
    }
    Rectangle {
        visible: mark.shape === "dot"
        anchors.centerIn: parent
        width: 8
        height: 8
        radius: 4
        color: mark.color
    }
    Text {
        visible: mark.shape === "check"
        anchors.centerIn: parent
        text: "\u2713"
        textFormat: Text.PlainText
        color: Look.t.text_on_mark
        font.pixelSize: Look.fsCaption
        font.weight: Font.Bold
    }
    // The hands of the clock: one up, one to the right.
    Rectangle {
        visible: mark.shape === "clock"
        x: mark.width / 2 - 0.75
        y: 4
        width: 1.5
        height: mark.height / 2 - 4
        color: mark.color
    }
    Rectangle {
        visible: mark.shape === "clock"
        x: mark.width / 2 - 0.75
        y: mark.height / 2 - 0.75
        width: mark.width / 2 - 4
        height: 1.5
        color: mark.color
    }
    Repeater {
        model: mark.shape === "cross" ? [45, -45] : mark.shape === "dash" ? [0] : []
        delegate: Rectangle {
            required property int modelData
            anchors.centerIn: parent
            width: 12
            height: 2
            radius: 1
            rotation: modelData
            color: mark.color
        }
    }
}
