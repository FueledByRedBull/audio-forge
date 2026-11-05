// Guided Setup dialog scene. The dialog (first_run_setup_dialog.py) owns the flow; `vm` is its
// FirstRunSetupView: what is shown here, and the commands sent back.
pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Layouts
import AudioForge

Rectangle {
    id: root

    required property var vm
    readonly property var page: vm.page
    readonly property var status: vm.status
    readonly property var health: vm.health
    // The device captions share a column as wide as the longer one, so neither is cut short at a
    // large text size. Reading `font` makes each width follow the font once it is set.
    readonly property int captionWidth: {
        plainFont.font
        return Math.ceil(Math.max(plainFont.advanceWidth("Microphone"),
                                  plainFont.advanceWidth("Destination"))) + 2
    }
    // The action buttons stay in view whatever the step, so the widest main button counts.
    readonly property int actionsWidth: {
        plainFont.font
        boldFont.font
        let widest = 0
        for (const text of vm.actionLabels)
            widest = Math.max(widest, Math.ceil(boldFont.advanceWidth(text)))
        const plain = ["Pause setup", "Back", "Skip this step"]
            .reduce((sum, text) => sum + Math.ceil(plainFont.advanceWidth(text)) + 28, 0)
        return plain + widest + 28 + 3 * 8 + 24 + 2
    }
    // The room a caption needs beside a choice: the page and card margins, the gap and a short choice.
    readonly property int deviceMargins: 40 + 32 + 12 + 140
    // The narrowest the dialog can be without cutting off its actions or its captions.
    readonly property int minimumWidth: Math.max(actionsWidth, captionWidth + deviceMargins)

    color: Look.t.app_surface
    focus: true

    FontMetrics {
        id: plainFont
        font.pixelSize: Look.fsBody
    }
    FontMetrics {
        id: boldFont
        font.pixelSize: Look.fsBody
        font.weight: Font.DemiBold
    }

    // Enter outside a button does what the main button says; Escape is left to the dialog.
    Keys.onReturnPressed: event => activatePrimary(event)
    Keys.onEnterPressed: event => activatePrimary(event)
    function activatePrimary(event) {
        if (!event.isAutoRepeat && page.actionEnabled)
            vm.act()
    }

    // The colour of a message follows its state: ok, warn, bad, info or idle.
    function stateColor(state) {
        return Look.t["status_" + state + "_text"]
    }

    // A live reading from the main window, set off in the colours of its state.
    component Chip: Rectangle {
        id: chip
        property string text
        property string tone: "idle"
        implicitHeight: words.implicitHeight + 12
        radius: Look.t.radiusControl
        color: Look.t["status_" + tone + "_surface"]
        border.color: Look.t["status_" + tone + "_border"]
        Accessible.role: Accessible.StaticText
        Accessible.name: text
        Text {
            id: words
            anchors.fill: parent
            anchors.margins: 6
            anchors.leftMargin: 10
            anchors.rightMargin: 10
            text: chip.text
            textFormat: Text.PlainText
            color: root.stateColor(chip.tone)
            font.pixelSize: Look.fsCaption
            font.weight: Font.DemiBold
            wrapMode: Text.Wrap
            verticalAlignment: Text.AlignVCenter
            Accessible.ignored: true
        }
    }

    // One choice of the route: its caption, then the list.
    component DeviceRow: RowLayout {
        id: row
        property string caption
        property string name
        property string pickerName
        property var choices
        property int current
        signal picked(int index)
        spacing: 12
        MutedText {
            text: row.caption
            // The picker is named with it; the caption alone would be heard twice.
            Accessible.ignored: true
            Layout.preferredWidth: root.captionWidth
        }
        ChoiceBox {
            objectName: row.pickerName
            label: row.name
            model: row.choices
            currentIndex: row.current
            expandGlyph: root.vm.expandGlyph
            onActivated: index => row.picked(index)
            Layout.fillWidth: true
            Layout.minimumWidth: 120
        }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        ScrollPage {
            label: "Audio setup page"
            Layout.fillWidth: true
            Layout.fillHeight: true

            ColumnLayout {
                spacing: 4
                Layout.fillWidth: true
                ProgressTrack {
                    objectName: "progressBar"
                    fraction: root.page.progressCount / root.page.progressTotal
                    Layout.fillWidth: true
                    Accessible.name: "Setup progress"
                    Accessible.description: root.page.progressText
                }
                MutedText {
                    objectName: "progressText"
                    text: root.page.progressText
                    font.pixelSize: Look.fsCaption
                    // The bar is read with it; the text alone would be heard twice.
                    Accessible.ignored: true
                }
            }

            BodyText {
                objectName: "titleText"
                text: root.page.title
                font.pixelSize: Look.fsHeading
                font.weight: Font.Bold
                Accessible.role: Accessible.StaticText
                Accessible.name: text
            }

            MutedText {
                objectName: "descriptionText"
                text: root.page.description
                wrapMode: Text.Wrap
                Layout.fillWidth: true
                Accessible.role: Accessible.StaticText
                Accessible.name: text
            }

            BodyText {
                objectName: "statusText"
                text: root.status.text
                color: root.stateColor(root.status.state)
                font.pixelSize: Look.fsLead
                Accessible.role: Accessible.StaticText
                Accessible.name: text
            }

            Surface {
                objectName: "routePanel"
                visible: root.page.routeShown
                Layout.fillWidth: true
                pad: 16
                Title { text: "Speech and clipping" }
                GridLayout {
                    columns: 2
                    columnSpacing: 12
                    rowSpacing: 8
                    uniformCellWidths: true
                    Layout.fillWidth: true
                    QuickMeterItem {
                        objectName: "inputMeter"
                        source: root.vm.inputMeter
                        Layout.preferredWidth: 60
                        Layout.preferredHeight: 120
                        Layout.alignment: Qt.AlignHCenter
                        Accessible.role: Accessible.ProgressBar
                        Accessible.name: "Live microphone level"
                        Accessible.description: root.vm.inputMeter.accessibleText
                    }
                    QuickMeterItem {
                        objectName: "outputMeter"
                        source: root.vm.outputMeter
                        Layout.preferredWidth: 60
                        Layout.preferredHeight: 120
                        Layout.alignment: Qt.AlignHCenter
                        Accessible.role: Accessible.ProgressBar
                        Accessible.name: "Live destination level"
                        Accessible.description: root.vm.outputMeter.accessibleText
                    }
                    Chip {
                        objectName: "inputHealth"
                        text: root.health.input.text
                        tone: root.health.input.state
                        Layout.fillWidth: true
                        Layout.alignment: Qt.AlignTop
                    }
                    Chip {
                        objectName: "outputHealth"
                        text: root.health.output.text
                        tone: root.health.output.state
                        Layout.fillWidth: true
                        Layout.alignment: Qt.AlignTop
                    }
                }
            }

            SwitchButton {
                id: mute
                objectName: "muteSwitch"
                text: "Mute output"
                on: root.vm.muted
                onRequested: root.vm.requestMute(checked)
                Accessible.name: "Mute output during setup"
                Accessible.description: root.vm.muteTip
                Tip {
                    text: root.vm.muteTip
                    delay: 500
                    visible: mute.hovered || mute.visualFocus
                }
            }

            Surface {
                objectName: "devicesPanel"
                visible: root.page.devicesShown
                Layout.fillWidth: true
                pad: 16
                Title { text: "Audio route" }
                ColumnLayout {
                    spacing: 12
                    Layout.fillWidth: true
                    DeviceRow {
                        caption: "Microphone"
                        name: "Setup microphone"
                        pickerName: "inputPicker"
                        choices: root.vm.inputs
                        current: root.vm.inputIndex
                        onPicked: index => root.vm.pickInput(index)
                        Layout.fillWidth: true
                    }
                    DeviceRow {
                        caption: "Destination"
                        name: "Setup destination"
                        pickerName: "outputPicker"
                        choices: root.vm.outputs
                        current: root.vm.outputIndex
                        onPicked: index => root.vm.pickOutput(index)
                        Layout.fillWidth: true
                    }
                }
            }

            PushButton {
                objectName: "latencyButton"
                text: "Advanced latency calibration..."
                visible: root.page.latencyShown
                enabled: root.page.latencyEnabled
                Accessible.name: "Open advanced latency calibration"
                Layout.alignment: Qt.AlignLeft
                onClicked: root.vm.openLatency()
            }
        }

        // The actions stay in view while the page above them scrolls.
        RowLayout {
            id: actions
            spacing: 8
            Layout.fillWidth: true
            Layout.margins: 12
            PushButton {
                objectName: "pauseButton"
                text: "Pause setup"
                onClicked: root.vm.pause()
            }
            PushButton {
                objectName: "backButton"
                text: "Back"
                enabled: root.page.backEnabled
                onClicked: root.vm.back()
            }
            PushButton {
                objectName: "skipButton"
                text: "Skip this step"
                onClicked: root.vm.skip()
            }
            Item { Layout.fillWidth: true }
            PushButton {
                objectName: "actionButton"
                text: root.page.actionText
                enabled: root.page.actionEnabled
                look: "primary"
                onClicked: root.vm.act()
            }
        }
    }
}
