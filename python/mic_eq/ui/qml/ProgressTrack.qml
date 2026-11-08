// A progress bar: `fraction` (0 to 1) of the track is filled. While `busy` it shows a block
// sliding along the track instead (still, when animations are off).
import QtQuick

Rectangle {
    id: bar
    property real fraction: 0
    property bool busy: false
    implicitHeight: 14
    radius: Look.t.radiusControl
    color: Look.t.control_surface
    border.color: Look.t.control_edge
    Accessible.role: Accessible.ProgressBar
    Rectangle {
        visible: !bar.busy
        x: 1
        y: 1
        width: (bar.width - 2) * bar.fraction
        height: bar.height - 2
        radius: Look.t.radiusControl
        color: Look.t.accent
    }
    Rectangle {
        id: sweep
        // Where the block is along its run, 0 to 1.
        property real phase: 0.5
        visible: bar.busy
        x: 1 + (bar.width - 2 - width) * phase
        y: 1
        width: (bar.width - 2) / 3
        height: bar.height - 2
        radius: Look.t.radiusControl
        color: Look.t.accent
        SequentialAnimation on phase {
            running: bar.busy && Look.t.motion > 0
            loops: Animation.Infinite
            NumberAnimation { from: 0; to: 1; duration: 900; easing.type: Easing.InOutQuad }
            NumberAnimation { from: 1; to: 0; duration: 900; easing.type: Easing.InOutQuad }
        }
    }
}
