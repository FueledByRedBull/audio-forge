// Compact call window scene. The window (compact_window.py) is `host`; everything else reads the
// main window's own model, so the two windows show one state.
pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Layouts
import QtQuick.Templates as T
import AudioForge

Rectangle {
    id: root

    required property var ui
    required property var host
    readonly property var m: ui.model
    readonly property var t: Look.t
    readonly property var preset: m.presets.status.data
    readonly property real labelWidth: Math.ceil(Math.max(clipMetrics.width, outMetrics.width)) + 8
    // The smallest size at which every row is whole; the window takes it as its minimum.
    readonly property int minimumWidth: Math.ceil(content.implicitWidth) + 2 * content.anchors.margins
    readonly property int minimumHeight: Math.ceil(content.implicitHeight) + 2 * content.anchors.margins

    color: t.app_surface

    TextMetrics { id: clipMetrics; text: "CLIP"; font.pixelSize: Look.fsCaption; font.weight: Font.DemiBold }
    TextMetrics { id: outMetrics; text: "OUT"; font.pixelSize: Look.fsCaption; font.weight: Font.DemiBold }

    // One shared tooltip, kept inside this small window.
    Tip {
        id: tip
        objectName: "compactTooltip"
        delay: 500
        margins: 8
        width: Math.min(implicitWidth, root.width - 2 * margins)
    }

    function showTip(item, text) {
        if (text === "")
            return
        tip.parent = item
        tip.text = text
        tip.open()
    }

    function hideTip(item) {
        if (tip.parent === item)
            tip.close()
    }

    // Hovering the parent or moving keyboard focus onto it shows the help, when there is any.
    component Hint: HoverHandler {
        property string text: ""
        property bool shown: true
        property bool focused: (parent as T.Control)?.visualFocus === true
        onHoveredChanged: sync()
        onFocusedChanged: sync()
        onShownChanged: sync()
        function sync() {
            if ((hovered || focused) && shown)
                root.showTip(parent, text)
            else
                root.hideTip(parent)
        }
    }

    // Menus that belong to a button open under it, also from the keyboard.
    function press(item, proxy) {
        const below = item.mapToGlobal(0, item.height + 4)
        proxy.clickAt(below.x, below.y)
    }

    // A PushButton that acts through a shared command.
    component Btn: PushButton {
        property var proxy
        text: proxy ? proxy.text : ""
        enabled: proxy ? proxy.enabled : false
        onClicked: if (proxy) root.press(this, proxy)
        Accessible.name: proxy && proxy.name !== "" ? proxy.name : text
        Accessible.description: proxy ? proxy.toolTip : ""
    }

    // A level meter: its name, or CLIP on a filled tag, and a thin bar.
    component Readout: RowLayout {
        id: readout
        property var meter
        property string label
        spacing: 6
        Rectangle {
            implicitWidth: root.labelWidth
            implicitHeight: tag.implicitHeight + 2
            radius: 3
            color: readout.meter.clipTagShown ? root.t.meter_clip : "transparent"
            Text {
                id: tag
                anchors.centerIn: parent
                text: readout.meter.clipTagShown ? "CLIP" : readout.label
                textFormat: Text.PlainText
                color: readout.meter.clipTagShown ? root.t.text_on_mark : root.t.text_muted
                font.pixelSize: Look.fsCaption
                font.weight: Font.DemiBold
                Accessible.ignored: true
            }
        }
        QuickMeterItem {
            objectName: readout.objectName + "Bar"
            source: readout.meter
            horizontal: true
            implicitHeight: 6
            Layout.fillWidth: true
            Layout.minimumWidth: 60
            Layout.preferredWidth: 60
            Accessible.role: Accessible.ProgressBar
            Accessible.name: readout.meter.name
            Accessible.description: readout.meter.accessibleText
        }
    }

    ColumnLayout {
        id: content
        anchors.fill: parent
        anchors.margins: 12
        spacing: 6

        // The status in the Overview's words, beside a dot whose colour only repeats them.
        RowLayout {
            spacing: 8
            Layout.fillWidth: true
            Rectangle {
                objectName: "compactToneDot"
                implicitWidth: 10
                implicitHeight: 10
                radius: 5
                color: ({
                    ok: root.t.status_ok_text,
                    warn: root.t.status_warn_text,
                    bad: root.t.status_bad_text,
                    mute: root.t.mute,
                    callers: root.t.callers_affected
                })[root.m.overview.tone] ?? root.t.control_edge
                Behavior on color { ColorAnimation { duration: root.t.motionStatus } }
                Accessible.ignored: true
            }
            Text {
                objectName: "compactHeadline"
                text: root.m.overview.headline
                textFormat: Text.PlainText
                color: root.m.overview.callersAffected ? root.t.callers_affected_text : root.t.text_primary
                font.pixelSize: Look.fsBody
                font.weight: Font.DemiBold
                elide: Text.ElideRight
                Layout.fillWidth: true
                Layout.preferredWidth: 0
                Accessible.role: Accessible.StaticText
                Accessible.name: text
            }
        }

        Rectangle {
            id: health
            objectName: "compactHealth"
            readonly property var proxy: root.m.status.health
            readonly property string tone: proxy.state !== "" ? proxy.state : "idle"
            implicitHeight: healthText.implicitHeight + 10
            radius: root.t.radiusControl
            color: root.t["status_" + tone + "_surface"] ?? root.t.status_idle_surface
            border.color: root.t["status_" + tone + "_border"] ?? root.t.status_idle_border
            Layout.fillWidth: true
            // The full text, when it does not fit.
            Hint { text: health.proxy.text; shown: healthText.truncated }
            Text {
                id: healthText
                anchors.fill: parent
                anchors.leftMargin: 10
                anchors.rightMargin: 10
                text: health.proxy.text
                textFormat: Text.PlainText
                color: root.t["status_" + health.tone + "_text"] ?? root.t.status_idle_text
                font.pixelSize: Look.fsCaption
                elide: Text.ElideRight
                verticalAlignment: Text.AlignVCenter
                Accessible.role: Accessible.StaticText
                Accessible.name: health.proxy.name !== "" ? health.proxy.name + ": " + text : text
                Accessible.description: health.proxy.toolTip
            }
        }

        RowLayout {
            spacing: 12
            Layout.fillWidth: true
            ColumnLayout {
                spacing: 4
                Layout.fillWidth: true
                Readout { objectName: "compactInput"; meter: root.m.meters[0]; label: "IN"; Layout.fillWidth: true }
                Readout { objectName: "compactOutput"; meter: root.m.meters[1]; label: "OUT"; Layout.fillWidth: true }
            }
            Column {
                objectName: "compactLatency"
                Text {
                    id: latencyFigure
                    text: root.m.top.latency.text
                    textFormat: Text.PlainText
                    color: root.t.text_primary
                    font.pixelSize: Look.fsBody
                    font.weight: Font.DemiBold
                    font.features: { "tnum": 1 }
                    Accessible.role: Accessible.StaticText
                    Accessible.name: (root.m.top.latency.data.measured ? "Latency" : "Latency, not measured") + ": " + text
                    Accessible.description: root.m.top.latency.toolTip
                }
                Text {
                    text: root.m.top.latency.data.measured ? "Latency" : "Not measured"
                    textFormat: Text.PlainText
                    color: root.t.text_muted
                    font.pixelSize: Look.fsCaption
                    Accessible.ignored: true
                }
            }
        }

        RowLayout {
            spacing: 8
            Layout.fillWidth: true
            MuteButton {
                id: compactMute
                objectName: "compactMute"
                proxy: root.m.top.mute
                Hint { text: compactMute.proxy.toolTip }
            }
            Btn {
                id: compactPresets
                objectName: "compactPresets"
                proxy: root.m.presets.menu
                // The button shows the preset in use and opens the Presets menu.
                text: root.preset.name
                Accessible.description: root.m.presets.status.text
                // The full preset name when the button cuts it short, then what the button does.
                Hint {
                    text: (compactPresets.contentItem as Text)?.truncated
                        ? compactPresets.text + "\n" + compactPresets.proxy.toolTip
                        : compactPresets.proxy.toolTip
                }
                Layout.fillWidth: true
                Layout.minimumWidth: 60
                Layout.preferredWidth: 60
            }
            PushButton {
                objectName: "compactExpand"
                text: "Expand"
                onClicked: root.host.expand()
                Accessible.description: "Show the main window and close this one"
            }
        }

        SwitchButton {
            objectName: "compactKeepOnTop"
            text: "Keep on top"
            on: root.host.keepOnTop
            onRequested: root.host.setKeepOnTop(!root.host.keepOnTop)
            Accessible.name: text
            Accessible.description: "Keep this window above other windows"
        }
    }
}
