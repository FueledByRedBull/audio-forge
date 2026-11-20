// A progress bar: `fraction` (0 to 1) of the track is filled. While `busy` it shows a block
// sliding along the track instead; under reduced motion nothing slides, and the word "Working"
// stands in the track. Assistive technology reads `value` as a percentage (0 to 100).
import QtQuick

Rectangle {
    id: bar
    property real fraction: 0
    property bool busy: false
    readonly property real value: fraction * 100
    readonly property real from: 0
    readonly property real to: 100
    implicitHeight: 14
    radius: Look.t.radiusControl
    color: Look.t.control_surface
    border.color: Look.t.control_edge
    Accessible.role: Accessible.ProgressBar
    Text {
        objectName: "workingLabel"
        visible: bar.busy && Look.reducedMotion
        anchors.centerIn: parent
        text: "Working"
        textFormat: Text.PlainText
        font.pixelSize: Look.fsCaption
        color: Look.t.text_primary
        Accessible.ignored: true
    }
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
        visible: bar.busy && !Look.reducedMotion
        x: 1 + (bar.width - 2 - width) * phase
        y: 1
        width: (bar.width - 2) / 3
        height: bar.height - 2
        radius: Look.t.radiusControl
        color: Look.t.accent
        SequentialAnimation on phase {
            running: bar.busy && bar.visible && !Look.reducedMotion
            loops: Animation.Infinite
            NumberAnimation { from: 0; to: 1; duration: Look.t.motionBusy; easing.type: Easing.InOutQuad }
            NumberAnimation { from: 1; to: 0; duration: Look.t.motionBusy; easing.type: Easing.InOutQuad }
        }
    }
}
