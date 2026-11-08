// The Test my sound recording window. `vm` is its RecordingProgressView (listening_comparison_view.py):
// the progress shown here, and Cancel sent back.
import QtQuick
import QtQuick.Layouts

Rectangle {
    id: root

    required property var vm
    // As wide as the prompt on one line, as the window was; it wraps only when made narrower.
    implicitWidth: {
        promptFont.font
        return Math.max(320, Math.ceil(promptFont.advanceWidth(vm.prompt)) + 2 * 16 + 2)
    }
    implicitHeight: column.implicitHeight + 2 * 16

    color: Look.t.app_surface

    FontMetrics {
        id: promptFont
        font.pixelSize: Look.fsBody
    }

    ColumnLayout {
        id: column
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.top: parent.top
        anchors.margins: 16
        spacing: 12

        BodyText {
            objectName: "promptText"
            text: root.vm.prompt
            Accessible.role: Accessible.StaticText
            Accessible.name: text
        }
        Chip {
            objectName: "mutedNotice"
            text: root.vm.mutedNotice
            tone: "warn"
            Layout.fillWidth: true
        }
        ProgressTrack {
            objectName: "progressBar"
            fraction: root.vm.progress / 100
            Layout.fillWidth: true
            Accessible.name: "Recording progress"
            Accessible.description: root.vm.progress + " percent"
        }
        // The only control, so it has the keyboard: Space and Enter cancel, as Escape does.
        PushButton {
            objectName: "cancelButton"
            text: "Cancel"
            focus: true
            Layout.fillWidth: true
            onClicked: root.vm.cancel()
        }
    }
}
