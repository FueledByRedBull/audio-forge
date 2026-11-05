// Latency calibration dialog scene. The dialog (latency_calibration_dialog.py) owns the work;
// `vm` is its LatencyCalibrationView: what is shown here, and the commands sent back.
pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Layouts
import AudioForge

Rectangle {
    id: root

    required property var vm
    readonly property var readings: vm.readings
    // What each reading is called. The label column is as wide as the longest name, so none is
    // cut short at a large text size.
    readonly property var mainRows: [
        { key: "total", label: "Total latency" },
        { key: "confidence", label: "Confidence" }
    ]
    readonly property var detailRows: [
        { key: "roundTrip", label: "Measured route" },
        { key: "oneWay", label: "Directional estimate" },
        { key: "comp", label: "Route delay for reporting" },
        { key: "agreement", label: "Probe agreement" },
        { key: "ambiguity", label: "Echo ambiguity" },
        { key: "engine", label: "Engine latency" }
    ]
    readonly property int labelWidth: {
        let widest = 0
        for (const row of mainRows.concat(detailRows))
            widest = Math.max(widest, labelFont.advanceWidth(row.label))
        return Math.ceil(widest) + 2
    }
    // The room a label needs beside a value: the page and card margins, the gap and a short value.
    readonly property int readingMargins: 40 + 32 + 16 + 140
    // The narrowest the dialog can be without cutting off its actions or its labels.
    readonly property int minimumWidth: Math.max(Math.ceil(actions.implicitWidth) + 24,
                                                 labelWidth + readingMargins)

    color: Look.t.app_surface
    focus: true

    FontMetrics {
        id: labelFont
        font.pixelSize: Look.fsBody
    }

    // Enter outside a button does what the main button says; Escape is left to the dialog.
    Keys.onReturnPressed: event => activatePrimary(event)
    Keys.onEnterPressed: event => activatePrimary(event)
    function activatePrimary(event) {
        if (event.isAutoRepeat)
            return
        if (vm.acceptEnabled)
            vm.accept()
        else if (vm.runEnabled)
            vm.run()
    }

    // One reading: its name, then its value.
    component Reading: RowLayout {
        id: reading
        property string label
        property string value
        spacing: 16
        MutedText {
            objectName: "readingLabel"
            text: reading.label
            // The value is read with its name; the name alone would be heard twice.
            Accessible.ignored: true
            Layout.preferredWidth: root.labelWidth
        }
        BodyText {
            text: reading.value
            Accessible.role: Accessible.StaticText
            Accessible.name: reading.label + ": " + text
        }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        ScrollPage {
            label: "Latency calibration page"
            Layout.fillWidth: true
            Layout.fillHeight: true

            BodyText {
                text: "Measure the delay of the selected microphone and output. For the best "
                      + "result, use a loopback cable or a speaker-to-microphone path in a "
                      + "quiet room. The measured delay corrects the latency AudioForge "
                      + "reports. It does not reduce audio delay."
            }

            Surface {
                Layout.fillWidth: true
                pad: 16
                Title { text: "Measured latency" }
                ColumnLayout {
                    spacing: 8
                    Layout.fillWidth: true
                    Repeater {
                        model: root.mainRows
                        delegate: Reading {
                            required property var modelData
                            objectName: modelData.key + "Reading"
                            label: modelData.label
                            value: root.readings[modelData.key]
                        }
                    }
                }
                // The other readings are for checking a result; they stay out of the way until asked for.
                PushButton {
                    objectName: "detailsButton"
                    text: root.vm.detailsShown ? "Hide details" : "Show details"
                    onClicked: root.vm.toggleDetails()
                }
                ColumnLayout {
                    objectName: "detailsPanel"
                    visible: root.vm.detailsShown
                    spacing: 8
                    Layout.fillWidth: true
                    Repeater {
                        model: root.detailRows
                        delegate: Reading {
                            required property var modelData
                            objectName: modelData.key + "Reading"
                            label: modelData.label
                            value: root.readings[modelData.key]
                        }
                    }
                }
            }

            RowLayout {
                spacing: 10
                Layout.fillWidth: true
                ProgressTrack {
                    objectName: "progressBar"
                    fraction: root.vm.progress / 100
                    Layout.fillWidth: true
                    Accessible.name: "Latency calibration progress"
                    Accessible.description: root.vm.progress + " percent"
                }
                MutedText {
                    text: root.vm.progress + "%"
                    Accessible.ignored: true
                }
            }

            BodyText {
                objectName: "statusText"
                text: root.vm.status
                Accessible.role: Accessible.StaticText
                Accessible.name: text
            }

            QuickMeterItem {
                objectName: "levelMeter"
                source: root.vm.meter
                Layout.fillWidth: true
                Layout.preferredHeight: 120
                Accessible.role: Accessible.ProgressBar
                Accessible.name: "Latency calibration capture level"
                Accessible.description: root.vm.meter.accessibleText
            }
        }

        // The actions stay in view while the page above them scrolls.
        RowLayout {
            id: actions
            spacing: 8
            Layout.fillWidth: true
            Layout.margins: 12
            PushButton {
                objectName: "closeButton"
                text: "Close"
                Accessible.name: "Close latency calibration"
                onClicked: root.vm.close()
            }
            PushButton {
                objectName: "resetButton"
                text: "Reset"
                Accessible.name: "Reset latency calibration"
                onClicked: root.vm.reset()
            }
            Item { Layout.fillWidth: true }
            PushButton {
                objectName: "runButton"
                text: "Run calibration"
                enabled: root.vm.runEnabled
                look: root.vm.acceptEnabled ? "plain" : "primary"
                onClicked: root.vm.run()
            }
            PushButton {
                objectName: "acceptButton"
                text: "Accept"
                Accessible.name: "Accept latency calibration"
                enabled: root.vm.acceptEnabled
                look: "primary"
                onClicked: root.vm.accept()
            }
        }
    }
}
