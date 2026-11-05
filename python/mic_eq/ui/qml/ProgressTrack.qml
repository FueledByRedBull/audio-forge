// A progress bar: `fraction` (0 to 1) of the track is filled.
import QtQuick

Rectangle {
    id: bar
    property real fraction: 0
    implicitHeight: 14
    radius: Look.t.radiusControl
    color: Look.t.control_surface
    border.color: Look.t.control_edge
    Accessible.role: Accessible.ProgressBar
    Rectangle {
        x: 1
        y: 1
        width: (bar.width - 2) * bar.fraction
        height: bar.height - 2
        radius: Look.t.radiusControl
        color: Look.t.accent
    }
}
