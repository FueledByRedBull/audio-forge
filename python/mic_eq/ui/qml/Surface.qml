// A card: children are laid out in a column inside a bordered, shadowed surface.
import QtQuick
import QtQuick.Effects
import QtQuick.Layouts

Item {
    id: surface
    default property alias content: body.data
    property int pad: 18
    property alias spacing: body.spacing
    // Software rendering draws no shadow effect.
    readonly property bool shadows: GraphicsInfo.api !== GraphicsInfo.Software
    implicitHeight: body.implicitHeight + 2 * pad
    Rectangle {
        anchors.fill: parent
        radius: Look.t.radiusCard + 2
        color: Look.t.card_surface
        border.color: Look.t.border
        layer.enabled: surface.shadows
        layer.effect: MultiEffect {
            shadowEnabled: true
            shadowColor: "black"
            shadowOpacity: 0.4
            shadowBlur: 0.8
            shadowVerticalOffset: 5
        }
    }
    ColumnLayout {
        id: body
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.top: parent.top
        anchors.margins: surface.pad
        spacing: 14
    }
}
