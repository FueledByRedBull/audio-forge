// Auto-EQ calibration dialog scene. The dialog (calibration_dialog.py) owns the recording and the
// analysis; `vm` is its CalibrationView: what is shown here, and the commands sent back.
pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Layouts
import AudioForge

Rectangle {
    id: root

    required property var vm
    readonly property var page: vm.page
    readonly property var status: vm.status
    readonly property string tipMode: "Subtle aims for small tone changes, never more than 2 dB from flat. "
        + "Full strength aims for the tone preset's whole curve, up to 5 dB. Your recording decides "
        + "whether a correction is applied and how far headroom lets it go."
    readonly property string tipSmoothing: "Conservative smoothing ignores narrow dips and peaks in the "
        + "recording. Broad is safest but less detailed."
    readonly property string tipCompare: "Listen to this same passage with current and proposed processing"
    readonly property string stepName: vm.steps[page.step]
    // The widest a row of actions gets: Cancel, Retake, Compare and the main button in the review,
    // or Cancel, the override and the main button after the level check, each with the gap before
    // the spacer and the one after it. The main button is bold.
    readonly property int actionsWidth: {
        plainFont.font
        boldFont.font
        let start = 0
        for (const text of vm.startLabels)
            start = Math.max(start, Math.ceil(boldFont.advanceWidth(text)) + 28)
        let override = 0
        for (const text of vm.overrideLabels)
            override = Math.max(override, Math.ceil(plainFont.advanceWidth(text)) + 28)
        const plain = text => Math.ceil(plainFont.advanceWidth(text)) + 28
        const review = plain("Cancel") + plain("Retake") + plain("Compare Recording") + start + 4 * 8
        const level = plain("Cancel") + override + start + 3 * 8
        return Math.max(review, level) + 24 + 2
    }
    // The steps in a row, with the room the header margins take.
    readonly property int stepsWidth: {
        plainFont.font
        let width = 2 * 16
        for (const name of vm.steps)
            width += Math.ceil(boldFont.advanceWidth(name)) + 16
        return width
    }
    // The room a picker needs: the page and card margins around it at its narrowest.
    readonly property int minimumWidth: Math.max(actionsWidth, stepsWidth, 120 + 40 + 32)

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
        if (!event.isAutoRepeat && page.startEnabled)
            vm.start()
    }
    // A list takes Enter itself while it is open, to choose; closed, Enter does what the main button says.
    function activateUnlessChoosing(event, picker) {
        if (picker.popup.visible)
            event.accepted = false
        else
            activatePrimary(event)
    }

    // The caption of a field; its picker is named with it, so the caption alone is not read.
    component Caption: MutedText {
        Accessible.ignored: true
    }

    // A button that hides or disables itself loses the keyboard; pass it on to the page, as a dialog does.
    component ActionButton: PushButton {
        onActiveFocusChanged: if (!activeFocus && !enabled) scrollPage.focusPage()
        onVisibleChanged: if (!visible && activeFocus) scrollPage.focusPage()
    }

    // One field: its caption above the list, as the form wrapped every row.
    component Field: ColumnLayout {
        id: field
        property string caption
        property string pickerName
        property string tip
        property alias model: picker.model
        property alias currentIndex: picker.currentIndex
        signal picked(int index)
        spacing: 4
        Layout.fillWidth: true
        Caption { text: field.caption + ":" }
        ChoiceBox {
            id: picker
            objectName: field.pickerName
            label: field.caption
            enabled: root.page.choicesEnabled
            expandGlyph: root.vm.expandGlyph
            onActivated: index => field.picked(index)
            Keys.onReturnPressed: event => root.activateUnlessChoosing(event, picker)
            Keys.onEnterPressed: event => root.activateUnlessChoosing(event, picker)
            Layout.fillWidth: true
            Layout.minimumWidth: 120
            Accessible.description: field.tip
            Tip {
                text: field.tip
                delay: 500
                visible: field.tip !== "" && (picker.hovered || picker.visualFocus)
            }
        }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        // Where the run is: set up, record, review.
        Row {
            objectName: "stepHeader"
            spacing: 16
            Layout.fillWidth: true
            Layout.margins: 16
            Layout.bottomMargin: 0
            Accessible.role: Accessible.StaticText
            Accessible.name: "Step " + (root.page.step + 1) + " of " + root.vm.steps.length + ": " + root.stepName
            Repeater {
                model: root.vm.steps
                delegate: Column {
                    id: stepItem
                    required property int index
                    required property string modelData
                    readonly property bool current: index === root.page.step
                    spacing: 4
                    Text {
                        text: stepItem.modelData
                        textFormat: Text.PlainText
                        color: stepItem.current ? Look.t.text_primary : Look.t.text_muted
                        font.pixelSize: Look.fsBody
                        font.weight: stepItem.current ? Font.DemiBold : Font.Normal
                        Accessible.ignored: true
                    }
                    Rectangle {
                        width: parent.width
                        height: 2
                        color: stepItem.current ? Look.t.accent : "transparent"
                    }
                }
            }
        }

        ScrollPage {
            id: scrollPage
            label: root.page.pageName
            Layout.fillWidth: true
            Layout.fillHeight: true

            Surface {
                objectName: "curveCard"
                visible: root.page.curveShown
                Layout.fillWidth: true
                pad: 16
                Accessible.role: Accessible.Grouping
                Accessible.name: curveTitle.text
                Title { id: curveTitle; text: "Tone preset"; Accessible.ignored: true }
                Field {
                    caption: "Tone preset"
                    pickerName: "curvePicker"
                    model: root.vm.curves
                    currentIndex: root.vm.curveIndex
                    onPicked: index => root.vm.pickCurve(index)
                }
                Field {
                    caption: "Matching"
                    pickerName: "modePicker"
                    tip: root.tipMode
                    model: root.vm.modes
                    currentIndex: root.vm.modeIndex
                    onPicked: index => root.vm.pickMode(index)
                }
                Field {
                    caption: "Smoothing"
                    pickerName: "smoothingPicker"
                    tip: root.tipSmoothing
                    model: root.vm.smoothings
                    currentIndex: root.vm.smoothingIndex
                    onPicked: index => root.vm.pickSmoothing(index)
                }
                MutedText {
                    objectName: "curveDescription"
                    text: root.page.curveDescription
                    font.pixelSize: Look.fsCaption
                    wrapMode: Text.Wrap
                    Layout.fillWidth: true
                    Accessible.role: Accessible.StaticText
                    Accessible.name: text
                }
            }

            Surface {
                objectName: "instructionsCard"
                visible: root.page.instructionsShown
                Layout.fillWidth: true
                pad: 16
                Accessible.role: Accessible.Grouping
                Accessible.name: instructionsTitle.text
                Title { id: instructionsTitle; text: "Read this passage aloud"; Accessible.ignored: true }
                // The text grows with the passage, so no line is cut and nothing scrolls inside the page.
                BodyText {
                    objectName: "passageText"
                    text: root.vm.passage
                    Accessible.role: Accessible.StaticText
                    Accessible.name: text
                    Accessible.description: "Read-only passage to speak while Auto-EQ records."
                }
            }

            Chip {
                objectName: "mutedNotice"
                visible: root.page.mutedShown
                text: root.vm.mutedNotice
                tone: "warn"
                Layout.fillWidth: true
            }

            Surface {
                objectName: "recordingCard"
                visible: root.page.recordingShown
                Layout.fillWidth: true
                pad: 16
                Accessible.role: Accessible.Grouping
                Accessible.name: recordingTitle.text
                Title { id: recordingTitle; text: root.page.recordingTitle; Accessible.ignored: true }
                ProgressTrack {
                    objectName: "progressBar"
                    visible: root.page.meterShown
                    fraction: root.page.progress / 100
                    busy: root.page.progressBusy
                    implicitHeight: 20
                    Layout.fillWidth: true
                    Accessible.name: "Auto-EQ recording progress"
                    Accessible.description: busy ? "Working" : root.page.progress + " percent"
                }
                BodyText {
                    objectName: "timeText"
                    visible: root.page.meterShown
                    text: root.page.timeText
                    color: root.page.timeDone ? Look.t.status_ok_text : Look.t.text_primary
                    font.pixelSize: Look.fsLead
                    font.weight: Font.Bold
                    horizontalAlignment: Text.AlignHCenter
                    Accessible.role: Accessible.StaticText
                    Accessible.name: text
                }
                QuickMeterItem {
                    objectName: "levelMeter"
                    visible: root.page.meterShown
                    source: root.vm.meter
                    Layout.fillWidth: true
                    Layout.preferredHeight: 120
                    Accessible.role: Accessible.ProgressBar
                    Accessible.name: "Auto-EQ input level"
                    Accessible.description: root.vm.meter.accessibleText
                }
                BodyText {
                    id: warning
                    objectName: "warningText"
                    text: root.status.text
                    color: Look.t["status_" + root.status.state + "_text"]
                    font.pixelSize: Look.fsLead
                    font.weight: root.status.strong ? Font.Bold : Font.Normal
                    horizontalAlignment: Text.AlignHCenter
                    Accessible.role: Accessible.StaticText
                    Accessible.name: text
                    // The analysis's own words, for whoever wants the numbers behind the sentence.
                    Accessible.description: root.status.tip
                    HoverHandler { id: warningHover; enabled: root.status.tip !== "" }
                    Tip {
                        text: root.status.tip
                        delay: 500
                        visible: root.status.tip !== "" && warningHover.hovered
                    }
                }
            }

            Surface {
                objectName: "resultCard"
                visible: root.page.recordingShown && root.page.diagnosticsShown
                Layout.fillWidth: true
                pad: 16
                Accessible.role: Accessible.Grouping
                Accessible.name: resultTitle.text
                Title { id: resultTitle; text: "Result"; Accessible.ignored: true }
                Repeater {
                    model: root.vm.chips
                    delegate: Chip {
                        required property var modelData
                        objectName: modelData.key + "Chip"
                        text: modelData.text
                        tone: modelData.state
                        tip: modelData.tip
                        Layout.fillWidth: true
                    }
                }
                EQComparisonItem {
                    objectName: "eqCompare"
                    visible: root.page.eqCompareShown
                    source: root.vm.eqCompare
                    Layout.fillWidth: true
                    Layout.preferredHeight: 180
                    Accessible.role: Accessible.Graphic
                    Accessible.name: "EQ now compared with the proposed EQ"
                    Accessible.description: root.vm.eqCompare.descriptionText
                }
                MutedText {
                    objectName: "resultHint"
                    text: "The first line rates the recording, not your voice or your microphone. "
                          + "Hover a line for the numbers behind it."
                    font.pixelSize: Look.fsCaption
                    wrapMode: Text.Wrap
                    Layout.fillWidth: true
                    Accessible.role: Accessible.StaticText
                    Accessible.name: text
                }
            }
        }

        // The actions stay in view while the page above them scrolls.
        RowLayout {
            id: actions
            spacing: 8
            Layout.fillWidth: true
            Layout.margins: 12
            PushButton {
                objectName: "cancelButton"
                text: "Cancel"
                Accessible.name: "Cancel Auto-EQ"
                onClicked: root.vm.cancel()
            }
            ActionButton {
                objectName: "retakeButton"
                text: "Retake"
                visible: root.page.retakeShown
                enabled: root.page.retakeEnabled
                Accessible.name: "Retake Auto-EQ recording"
                onClicked: root.vm.retake()
            }
            ActionButton {
                id: compareButton
                objectName: "compareButton"
                text: "Compare Recording"
                visible: root.page.compareShown
                enabled: root.page.compareEnabled
                Accessible.description: root.tipCompare
                onClicked: root.vm.compare()
                Tip {
                    text: root.tipCompare
                    delay: 500
                    visible: compareButton.hovered || compareButton.visualFocus
                }
            }
            Item { Layout.fillWidth: true }
            ActionButton {
                objectName: "overrideButton"
                text: root.page.overrideText
                visible: root.page.overrideShown
                onClicked: root.vm.override()
            }
            ActionButton {
                objectName: "startButton"
                text: root.page.startText
                enabled: root.page.startEnabled
                look: "primary"
                onClicked: root.vm.start()
            }
        }
    }
}
