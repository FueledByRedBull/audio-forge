// Listening comparison dialog scene. The dialog (listening_comparison_dialog.py) owns rendering
// and playback; `vm` is its ListeningComparisonView: what is shown here, and the commands sent back.
pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Layouts

Rectangle {
    id: root

    required property var vm
    readonly property var page: vm.page
    readonly property var status: vm.status
    readonly property string tipOutput: "Choose the speakers or headphones that should receive the preview."
    readonly property string tipRefresh: "Refresh available playback devices."
    readonly property string tipStop: "Stop the current preview."
    readonly property string tipLevelMatch: "Apply a bounded preview-only gain so level does not dominate the comparison."
    readonly property string levelMatchText: "Match speech level (actual level difference remains visible)"
    // The row captions share a column as wide as the longest, so none is cut short at a large text
    // size. Reading `font` makes each width follow the font once it is set.
    readonly property int captionWidth: {
        plainFont.font
        let widest = Math.max(plainFont.advanceWidth("Output device:"), plainFont.advanceWidth("Playback:"))
        for (const clip of vm.clips)
            widest = Math.max(widest, plainFont.advanceWidth(clip.caption))
        return Math.ceil(widest) + 2
    }
    // A play button reads Stop while its clip plays; both labels count.
    readonly property int playWidth: {
        plainFont.font
        let widest = 0
        for (const clip of vm.clips) {
            const name = clip.caption.slice(0, -1)
            widest = Math.max(widest, plainFont.advanceWidth("Play " + name), plainFont.advanceWidth("Stop " + name))
        }
        return Math.ceil(widest) + 28
    }
    readonly property int actionsWidth: {
        plainFont.font
        boldFont.font
        const reject = Math.ceil(plainFont.advanceWidth(page.rejectText)) + 28
        const keep = page.keepShown ? Math.ceil(boldFont.advanceWidth("Keep Proposed")) + 28 + 8 : 0
        return reject + keep + 24 + 2
    }
    // A row: its caption, the gap, then a play button beside a short detail, or the device list
    // at its narrowest beside Refresh.
    readonly property int rowWidth: {
        plainFont.font
        const refresh = Math.ceil(plainFont.advanceWidth("Refresh")) + 28
        return captionWidth + 12 + Math.max(playWidth + 12 + 140, 120 + 8 + refresh)
    }
    // The narrowest the dialog can be without cutting off its actions or its rows: the page and
    // card margins around the widest row.
    readonly property int minimumWidth: Math.max(actionsWidth, rowWidth + 40 + 32)

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

    // Enter outside a button keeps the proposal once it can be kept; Escape is left to the dialog.
    Keys.onReturnPressed: event => activatePrimary(event)
    Keys.onEnterPressed: event => activatePrimary(event)
    function activatePrimary(event) {
        if (!event.isAutoRepeat && page.keepShown && page.keepEnabled)
            vm.keep()
    }
    // The device list takes Enter itself while it is open, to choose; closed, Enter keeps.
    function activateUnlessChoosing(event) {
        if (picker.popup.visible)
            event.accepted = false
        else
            activatePrimary(event)
    }

    // The caption of a row; its control is named with it, so the caption alone is not read.
    component Caption: MutedText {
        Accessible.ignored: true
        Layout.preferredWidth: root.captionWidth
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        ScrollPage {
            label: root.page.pageName
            Layout.fillWidth: true
            Layout.fillHeight: true

            MutedText {
                objectName: "scopeText"
                text: root.page.scope
                font.pixelSize: Look.fsCaption
                wrapMode: Text.Wrap
                Layout.fillWidth: true
                Accessible.role: Accessible.StaticText
                Accessible.name: text
            }

            Chip {
                objectName: "scopeWarning"
                text: root.page.scopeWarning
                tone: "info"
                Layout.fillWidth: true
            }

            Surface {
                objectName: "playbackCard"
                Layout.fillWidth: true
                pad: 16
                Accessible.role: Accessible.Grouping
                Accessible.name: playbackTitle.text
                Title { id: playbackTitle; text: "Playback"; Accessible.ignored: true }
                RowLayout {
                    spacing: 12
                    Layout.fillWidth: true
                    Caption { text: "Output device:" }
                    ChoiceBox {
                        id: picker
                        objectName: "outputPicker"
                        label: "Output device"
                        model: root.vm.outputs
                        currentIndex: root.vm.outputIndex
                        enabled: root.page.outputEnabled
                        expandGlyph: root.vm.expandGlyph
                        onActivated: index => root.vm.pickOutput(index)
                        Keys.onReturnPressed: event => root.activateUnlessChoosing(event)
                        Keys.onEnterPressed: event => root.activateUnlessChoosing(event)
                        Layout.fillWidth: true
                        Layout.minimumWidth: 120
                        Accessible.description: root.tipOutput
                        Tip {
                            text: root.tipOutput
                            delay: 500
                            visible: picker.hovered || picker.visualFocus
                        }
                    }
                    PushButton {
                        id: refresh
                        objectName: "refreshButton"
                        text: "Refresh"
                        Accessible.name: "Refresh playback devices"
                        Accessible.description: root.tipRefresh
                        onClicked: root.vm.refresh()
                        Tip {
                            text: root.tipRefresh
                            delay: 500
                            visible: refresh.hovered || refresh.visualFocus
                        }
                    }
                }
                RowLayout {
                    spacing: 12
                    Layout.fillWidth: true
                    Caption { text: "Playback:" }
                    PushButton {
                        id: stop
                        objectName: "stopButton"
                        text: "Stop"
                        enabled: root.page.stopEnabled
                        // A button that disables itself loses the keyboard; pass it on, as a dialog does.
                        onActiveFocusChanged: if (!activeFocus && !enabled) levelMatch.forceActiveFocus(Qt.TabFocusReason)
                        Accessible.name: "Stop playback"
                        Accessible.description: root.tipStop
                        onClicked: root.vm.stop()
                        Tip {
                            text: root.tipStop
                            delay: 500
                            visible: stop.hovered || stop.visualFocus
                        }
                    }
                    Item { Layout.fillWidth: true }
                }
                RowLayout {
                    spacing: 12
                    Layout.fillWidth: true
                    Item { Layout.preferredWidth: root.captionWidth }
                    // The caption beside the switch wraps; a click on it flips the switch, as on a check box.
                    SwitchButton {
                        id: levelMatch
                        objectName: "levelMatchSwitch"
                        on: root.page.levelMatch
                        onRequested: root.vm.requestLevelMatch(checked)
                        Accessible.name: root.levelMatchText
                        Accessible.description: root.tipLevelMatch
                        Layout.alignment: Qt.AlignTop
                        Tip {
                            text: root.tipLevelMatch
                            delay: 500
                            visible: levelMatch.hovered || levelMatch.visualFocus || levelMatchHover.hovered
                        }
                    }
                    BodyText {
                        objectName: "levelMatchText"
                        text: root.levelMatchText
                        Accessible.ignored: true
                        HoverHandler { id: levelMatchHover }
                        TapHandler { onTapped: levelMatch.toggle() }
                    }
                }
            }

            ColumnLayout {
                spacing: 6
                Layout.fillWidth: true
                MutedText {
                    objectName: "progressText"
                    text: root.page.progressText
                    font.pixelSize: Look.fsCaption
                    wrapMode: Text.Wrap
                    Layout.fillWidth: true
                    Accessible.role: Accessible.StaticText
                    Accessible.name: text
                }
                ProgressTrack {
                    objectName: "progressBar"
                    fraction: root.page.progress / 100
                    Layout.fillWidth: true
                    Accessible.name: "Listening comparison progress bar"
                    Accessible.description: root.page.progress + " percent"
                }
            }

            Surface {
                objectName: "listenCard"
                Layout.fillWidth: true
                pad: 16
                Accessible.role: Accessible.Grouping
                Accessible.name: listenTitle.text
                Title { id: listenTitle; text: "Listen"; Accessible.ignored: true }
                Repeater {
                    model: root.vm.clips
                    delegate: RowLayout {
                        id: clipRow
                        required property var modelData
                        readonly property string key: modelData.key
                        spacing: 12
                        Layout.fillWidth: true
                        Caption { text: clipRow.modelData.caption }
                        PushButton {
                            objectName: clipRow.key + "PlayButton"
                            text: root.vm.playLabels[clipRow.key]
                            enabled: root.page.playEnabled
                            onClicked: root.vm.play(clipRow.key)
                        }
                        MutedText {
                            objectName: clipRow.key + "Detail"
                            text: root.vm.details[clipRow.key]
                            font.pixelSize: Look.fsCaption
                            wrapMode: Text.Wrap
                            Layout.fillWidth: true
                            Accessible.role: Accessible.StaticText
                            Accessible.name: text
                        }
                    }
                }
            }

            BodyText {
                objectName: "statusText"
                text: root.status.text
                color: Look.t["status_" + root.status.state + "_text"]
                font.pixelSize: Look.fsLead
                Accessible.role: Accessible.StaticText
                Accessible.name: text
            }

            Chip {
                objectName: "mutedNotice"
                text: root.vm.mutedNotice
                tone: "warn"
                Layout.fillWidth: true
            }
        }

        // The actions stay in view while the page above them scrolls.
        RowLayout {
            spacing: 8
            Layout.fillWidth: true
            Layout.margins: 12
            PushButton {
                objectName: "rejectButton"
                text: root.page.rejectText
                enabled: root.page.rejectEnabled
                Accessible.name: root.page.rejectName
                onClicked: root.vm.reject()
            }
            Item { Layout.fillWidth: true }
            PushButton {
                objectName: "keepButton"
                text: "Keep Proposed"
                visible: root.page.keepShown
                enabled: root.page.keepEnabled
                look: "primary"
                Accessible.name: "Keep proposed settings"
                onClicked: root.vm.keep()
            }
        }
    }
}
