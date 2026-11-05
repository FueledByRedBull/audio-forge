// Latency calibration dialog scene. The dialog (latency_calibration_dialog.py) owns the work;
// `vm` is its LatencyCalibrationView: what is shown here, and the commands sent back.
pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls.Basic
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

    component Body: Text {
        textFormat: Text.PlainText
        color: Look.t.text_primary
        font.pixelSize: Look.fsBody
        wrapMode: Text.Wrap
        Layout.fillWidth: true
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
        Body {
            text: reading.value
            Accessible.role: Accessible.StaticText
            Accessible.name: reading.label + ": " + text
        }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        Item {
            Layout.fillWidth: true
            Layout.fillHeight: true

            Flickable {
                id: scroll
                objectName: "contentScroll"
                anchors.fill: parent
                contentWidth: width
                contentHeight: column.implicitHeight + 32
                boundsBehavior: Flickable.StopAtBounds
                clip: true
                // A tab stop that scrolls as a scroll area does: arrows, Page Up, Page Down, Home, End.
                activeFocusOnTab: true
                Accessible.role: Accessible.Pane
                Accessible.name: "Latency calibration page"
                readonly property real lineStep: Look.fsBody * 3
                readonly property real maxY: Math.max(0, contentHeight - height)
                function scrollTo(y) {
                    contentY = Math.max(0, Math.min(maxY, y))
                }
                function targetFor(key) {
                    switch (key) {
                    case Qt.Key_Up: return contentY - lineStep
                    case Qt.Key_Down: return contentY + lineStep
                    case Qt.Key_PageUp: return contentY - height
                    case Qt.Key_PageDown: return contentY + height
                    case Qt.Key_Home: return 0
                    case Qt.Key_End: return maxY
                    }
                    return undefined
                }
                Keys.onPressed: event => {
                    const target = targetFor(event.key)
                    const shortcut = event.modifiers & (Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier)
                    if (target === undefined || shortcut)
                        return
                    scrollTo(target)
                    event.accepted = true
                }
                // Keep the control that has keyboard focus on screen.
                readonly property Item focusItem: Window.activeFocusItem
                onFocusItemChanged: {
                    const item = focusItem
                    let inside = false
                    for (let p = item; p && !inside; p = p.parent)
                        inside = p === column
                    if (!inside)
                        return
                    const top = item.mapToItem(contentItem, 0, 0).y
                    if (top < contentY + 12)
                        scrollTo(top - 12)
                    else if (top + item.height > contentY + height - 12)
                        scrollTo(top + item.height - height + 12)
                }
                ScrollBar.vertical: ScrollBar {
                    id: bar
                    Accessible.name: "Content scroll bar"
                    policy: scroll.contentHeight > scroll.height ? ScrollBar.AlwaysOn : ScrollBar.AlwaysOff
                    contentItem: Rectangle {
                        implicitWidth: 6
                        radius: 3
                        color: bar.pressed || bar.hovered ? Look.t.text_muted : Look.t.control_edge
                    }
                }
                ColumnLayout {
                    id: column
                    objectName: "contentColumn"
                    x: 16
                    y: 16
                    width: scroll.width - 40
                    spacing: 16

                    Body {
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
                        Rectangle {
                            objectName: "progressBar"
                            implicitHeight: 14
                            radius: Look.t.radiusControl
                            color: Look.t.control_surface
                            border.color: Look.t.control_edge
                            Layout.fillWidth: true
                            Accessible.role: Accessible.ProgressBar
                            Accessible.name: "Latency calibration progress"
                            Accessible.description: root.vm.progress + " percent"
                            Rectangle {
                                x: 1
                                y: 1
                                width: (parent.width - 2) * root.vm.progress / 100
                                height: parent.height - 2
                                radius: Look.t.radiusControl
                                color: Look.t.accent
                            }
                        }
                        MutedText {
                            text: root.vm.progress + "%"
                            Accessible.ignored: true
                        }
                    }

                    Body {
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
            }

            FocusRing {
                objectName: "scrollFocusRing"
                visible: scroll.activeFocus
                outset: -2
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
