// A tooltip in the theme: long help text wraps, and it does not fade.
import QtQuick
import QtQuick.Controls.Basic

ToolTip {
    id: tip
    padding: 10
    // Hover help is not animated, and the ToolTip style's own fade is not either.
    enter: null
    exit: null
    contentItem: Text {
        text: tip.text
        textFormat: Text.PlainText
        color: Look.t.text_primary
        font.pixelSize: Look.fsCaption
        wrapMode: Text.Wrap
        width: Math.min(implicitWidth, 360)
    }
    background: Rectangle {
        color: Look.t.control_surface
        border.color: Look.t.border_strong
        radius: Look.t.radiusControl
    }
}
