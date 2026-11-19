// A panel of glass: what lies behind it on `backdrop` shows through, blurred. It is drawn solid,
// without the blur, when the renderer cannot run the effect, when the user turned transparency
// effects off, or under a contrast theme.
pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Effects

Item {
    id: glass
    default property alias content: body.data
    // The item behind the panel, never one that holds it; without one the glass lies on the ground.
    property Item backdrop: null
    // The part of `backdrop` the panel covers, in the backdrop's coordinates.
    property rect backdropRect
    readonly property bool blurAllowed: GraphicsInfo.api !== GraphicsInfo.Software
    readonly property bool effectsOn: blurAllowed && !Look.t.reducedTransparency && !Look.t.highContrast
    // A panel in a scrolling page's flow passes the wheel on, so the page still scrolls under it.
    property bool holdsWheel: true

    // A press, hover or wheel on bare glass stays with the panel instead of reaching what is under it.
    MouseArea {
        anchors.fill: parent
        hoverEnabled: true
        acceptedButtons: Qt.AllButtons
        onWheel: wheel => wheel.accepted = glass.holdsWheel
    }
    Rectangle {
        anchors.fill: parent
        radius: Look.t.radiusGlass
        color: Look.t.app_surface
        layer.enabled: glass.effectsOn
        layer.effect: MultiEffect {
            shadowEnabled: true
            shadowColor: Look.t.glass_shadow
            shadowBlur: 1
            shadowVerticalOffset: 12
        }
    }
    Loader {
        objectName: "glassBlur"
        anchors.fill: parent
        // Hidden glass shows nothing, so nothing behind it is captured or blurred.
        active: glass.effectsOn && glass.backdrop !== null && glass.visible
        sourceComponent: Item {
            ShaderEffectSource {
                id: behind
                anchors.fill: parent
                sourceItem: glass.backdrop
                sourceRect: glass.backdropRect
                visible: false
            }
            Rectangle {
                id: shape
                anchors.fill: parent
                radius: Look.t.radiusGlass
                visible: false
                layer.enabled: true
            }
            MultiEffect {
                anchors.fill: parent
                source: behind
                blurEnabled: true
                blurMax: Look.t.glassBlur
                blur: 1
                // The effect's saturation is a change from the source, where the token is a factor.
                saturation: Look.t.glassSaturation - 1
                maskEnabled: true
                maskSource: shape
            }
        }
    }
    Rectangle {
        objectName: "glassFill"
        anchors.fill: parent
        radius: Look.t.radiusGlass
        color: glass.effectsOn ? Look.t.glass : Look.t.glass_solid
        border.color: Look.t.glass_edge
    }
    // The light catching the top edge.
    Rectangle {
        x: Look.t.radiusGlass
        y: 1
        width: parent.width - 2 * Look.t.radiusGlass
        height: 1
        color: Look.t.glass_highlight
    }
    Item {
        id: body
        anchors.fill: parent
    }
}
