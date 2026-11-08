// Auto Voice Setup dialog scene. The dialog (voice_setup_dialog.py) owns the recordings and the
// analysis; `vm` is its VoiceSetupView: what is shown here, and the commands sent back.
pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Layouts
import AudioForge

Rectangle {
    id: root

    required property var vm
    readonly property var page: vm.page
    readonly property var status: vm.status
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
    // The room a field needs: the page and card margins around it at its narrowest.
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

    // The caption of a field; its control is named with it, so the caption alone is not read.
    component Caption: MutedText {
        Accessible.ignored: true
    }

    // A button that hides or disables itself loses the keyboard; pass it on to the page, as a dialog does.
    component ActionButton: PushButton {
        onActiveFocusChanged: if (!activeFocus && !enabled) scrollPage.focusPage()
        onVisibleChanged: if (!visible && activeFocus) scrollPage.focusPage()
    }

    // One list: its caption above it, as the form wrapped every row.
    component Field: ColumnLayout {
        id: field
        property string caption
        property string pickerName
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
            expandGlyph: root.vm.expandGlyph
            onActivated: index => field.picked(index)
            Keys.onReturnPressed: event => root.activateUnlessChoosing(event, picker)
            Keys.onEnterPressed: event => root.activateUnlessChoosing(event, picker)
            // Enter here can start the run, which locks the pickers and hides their card.
            onActiveFocusChanged: if (!activeFocus && !enabled) scrollPage.focusPage()
            onVisibleChanged: if (!visible && activeFocus) scrollPage.focusPage()
            Layout.fillWidth: true
            Layout.minimumWidth: 120
        }
    }

    // One number: its caption above it.
    component NumberField: ColumnLayout {
        id: numberField
        required property var spec
        spacing: 4
        Layout.fillWidth: true
        Caption { text: numberField.spec.caption + ":" }
        NumberBox {
            id: box
            objectName: numberField.spec.key + "Number"
            label: numberField.spec.name
            low: numberField.spec.low
            high: numberField.spec.high
            decimals: root.vm.decimals
            suffix: numberField.spec.suffix
            number: root.vm.numbers[numberField.spec.key]
            onModified: value => root.vm.setNumber(numberField.spec.key, value)
            // Enter here can start the run, which locks the field and hides its card.
            onActiveFocusChanged: if (!activeFocus && !enabled) scrollPage.focusPage()
            onVisibleChanged: if (!visible && activeFocus) scrollPage.focusPage()
            Layout.fillWidth: true
            Layout.minimumWidth: 120
            Accessible.description: numberField.spec.tip
            Tip {
                text: numberField.spec.tip
                delay: 500
                visible: numberField.spec.tip !== "" && (box.hovered || box.visualFocus)
            }
        }
    }

    // A line of the review, read from the view model by its key.
    component Line: Chip {
        required property string key
        readonly property var info: root.vm.chipMap[key]
        objectName: key + "Chip"
        text: info.text
        tone: info.state
        tip: info.tip
        Layout.fillWidth: true
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        // Where the run is: target, level, room noise, voice, review.
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
                visible: root.page.targetShown
                enabled: root.page.curveGroupEnabled
                Layout.fillWidth: true
                pad: 16
                Accessible.role: Accessible.Grouping
                Accessible.name: curveTitle.text
                Title { id: curveTitle; text: "Tone preset"; Accessible.ignored: true }
                Field {
                    caption: "Tone preset"
                    pickerName: "curvePicker"
                    enabled: root.page.curveEnabled
                    model: root.vm.curves
                    currentIndex: root.vm.curveIndex
                    onPicked: index => root.vm.pickCurve(index)
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
                objectName: "dynamicsCard"
                visible: root.page.targetShown
                enabled: root.page.dynamicsEnabled
                Layout.fillWidth: true
                pad: 16
                Accessible.role: Accessible.Grouping
                Accessible.name: dynamicsTitle.text
                Title { id: dynamicsTitle; text: "Dynamics"; Accessible.ignored: true }
                Field {
                    caption: "Compression"
                    pickerName: "dynamicsPicker"
                    model: root.vm.dynamics
                    currentIndex: root.vm.dynamicsIndex
                    onPicked: index => root.vm.pickDynamics(index)
                }
                Repeater {
                    model: root.vm.numberFields
                    delegate: NumberField {
                        id: numberItem
                        required property var modelData
                        spec: modelData
                        // The two limits belong to the Custom setting; the loudness target does not.
                        enabled: modelData.setting ? root.vm.dynamicsCustom : root.page.targetEnabled
                    }
                }
                MutedText {
                    objectName: "dynamicsHint"
                    text: "Compression decides how much louder and quieter words are evened out. "
                          + "Tone and target loudness are set separately."
                    font.pixelSize: Look.fsCaption
                    wrapMode: Text.Wrap
                    Layout.fillWidth: true
                    Accessible.role: Accessible.StaticText
                    Accessible.name: text
                }
            }

            Surface {
                objectName: "noiseCard"
                visible: root.page.noiseShown
                Layout.fillWidth: true
                pad: 16
                Accessible.role: Accessible.Grouping
                Accessible.name: noiseTitle.text
                Title { id: noiseTitle; text: "Capture room noise"; Accessible.ignored: true }
                BodyText {
                    objectName: "noiseHint"
                    text: "Stay quiet for 2 seconds so the wizard can measure room noise "
                          + "and set the gate safely."
                    Accessible.role: Accessible.StaticText
                    Accessible.name: text
                }
            }

            Surface {
                objectName: "voiceCard"
                visible: root.page.voiceShown
                Layout.fillWidth: true
                pad: 16
                Accessible.role: Accessible.Grouping
                Accessible.name: voiceTitle.text
                Title { id: voiceTitle; text: "Read this passage aloud"; Accessible.ignored: true }
                // The text grows with the passage, so no line is cut and nothing scrolls inside the page.
                BodyText {
                    objectName: "passageText"
                    text: root.vm.passage
                    Accessible.role: Accessible.StaticText
                    Accessible.name: text
                    Accessible.description: "Read-only passage to speak during Auto Voice Setup."
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
                BodyText {
                    objectName: "phaseText"
                    text: root.page.phaseText
                    font.pixelSize: Look.fsLead
                    font.weight: Font.Bold
                    horizontalAlignment: Text.AlignHCenter
                    Accessible.role: Accessible.StaticText
                    Accessible.name: text
                }
                ProgressTrack {
                    objectName: "progressBar"
                    visible: root.page.meterShown
                    fraction: root.page.progress / 100
                    implicitHeight: 20
                    Layout.fillWidth: true
                    Accessible.name: "Voice Setup progress"
                    Accessible.description: root.page.progress + " percent"
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
                    Accessible.name: "Voice Setup input level"
                    Accessible.description: root.vm.meter.accessibleText
                }
                BodyText {
                    objectName: "warningText"
                    text: root.status.text
                    color: Look.t["status_" + root.status.state + "_text"]
                    font.pixelSize: Look.fsLead
                    font.weight: root.status.strong ? Font.Bold : Font.Normal
                    horizontalAlignment: Text.AlignHCenter
                    Accessible.role: Accessible.StaticText
                    Accessible.name: text
                }
            }

            Surface {
                objectName: "reviewCard"
                visible: root.page.reviewShown && root.page.summaryShown
                Layout.fillWidth: true
                pad: 16
                Accessible.role: Accessible.Grouping
                Accessible.name: reviewTitle.text
                Title { id: reviewTitle; text: "Recommended Settings"; Accessible.ignored: true }
                Line { key: "overall" }
                Repeater {
                    model: root.vm.issues
                    delegate: Chip {
                        required property int index
                        required property string modelData
                        objectName: "issueChip" + index
                        text: modelData
                        tone: "warn"
                        Layout.fillWidth: true
                    }
                }
                Line { key: "eq" }
                Line { key: "gate" }
                Line { key: "deesser" }
                Line { key: "compressor" }
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
                    objectName: "reviewHint"
                    text: "Voice Setup tunes tone, noise suppression, the noise gate, the "
                          + "de-esser and the compressor. The limiter is left as it is. Hover a "
                          + "line for the numbers behind it."
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
                Accessible.name: "Cancel Auto Voice Setup"
                onClicked: root.vm.cancel()
            }
            ActionButton {
                objectName: "retakeButton"
                text: "Retake"
                visible: root.page.retakeShown
                enabled: root.page.retakeEnabled
                Accessible.name: "Retake Voice Setup recording"
                onClicked: root.vm.retake()
            }
            ActionButton {
                objectName: "compareButton"
                text: "Compare Recording"
                visible: root.page.compareShown
                enabled: root.page.compareEnabled
                onClicked: root.vm.compare()
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
                look: root.page.startLook
                onClicked: root.vm.start()
            }
        }
    }
}
