// The one focus indicator: a ring that hover never draws. `outset` is how far it sits outside
// the control; a negative value draws it inside.
import QtQuick
import QtQuick.Templates as T

Rectangle {
    property T.Control control
    property int outset: 3
    anchors.fill: parent
    anchors.margins: -outset
    radius: height / 2 < 9 ? height / 2 : 9
    color: "transparent"
    border.width: 2
    border.color: Look.t.accent
    visible: control.visualFocus
}
