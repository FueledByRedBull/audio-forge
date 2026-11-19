// The head of a dialog scene: its glyph, its title and where it is, the key that closes it and
// a cross that does the same. `closeRequested` goes to the dialog's own cancel command.
pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Layouts

RowLayout {
    id: header
    property string title
    property string subtitle
    // A code point of the system icon font; the tile is left out without that font.
    property string glyph
    property string escHint: "Esc"
    // False when another item already reads the subtitle, such as a progress bar or a step strip.
    property bool subtitleAnnounced: true
    property string subtitleName: "dialogSubtitle"
    // A subtitle that is a whole instruction wraps instead of being cut off at the edge.
    property bool subtitleWraps: false
    // The name of the cross, for a dialog whose own Close button already has the default.
    property string closeName: "closeButton"
    signal closeRequested()
    spacing: 12
    // The narrowest the header can be without cutting off its title: the tile, the title, the
    // key chip and the cross, with the gaps between them.
    readonly property int minimumWidth: {
        titleMetrics.font
        tickMetrics.font
        return 30 + Math.ceil(titleMetrics.advanceWidth(header.title)) + 2
               + Math.ceil(tickMetrics.advanceWidth(header.escHint)) + 12 + 30 + 3 * spacing
    }

    FontMetrics {
        id: titleMetrics
        font.pixelSize: Look.fsTitle
        font.weight: Font.DemiBold
    }
    FontMetrics {
        id: tickMetrics
        font.pixelSize: Look.fsTick
    }

    Rectangle {
        visible: Look.hasIcons && header.glyph !== ""
        implicitWidth: 30
        implicitHeight: 30
        radius: Look.t.radiusControl
        color: Look.t.accent_fill
        border.color: Look.t.accent_edge
        Text {
            anchors.centerIn: parent
            text: header.glyph
            textFormat: Text.PlainText
            font.family: Look.t.iconFont
            font.pixelSize: Look.fsLead
            color: Look.t.accent_bright
            Accessible.ignored: true
        }
    }
    ColumnLayout {
        spacing: 2
        Layout.fillWidth: true
        Text {
            objectName: "dialogTitle"
            text: header.title
            textFormat: Text.PlainText
            color: Look.t.text_primary
            font.pixelSize: Look.fsTitle
            font.weight: Font.DemiBold
            elide: Text.ElideRight
            Layout.fillWidth: true
            Accessible.role: Accessible.Heading
            Accessible.name: text
        }
        MutedText {
            objectName: header.subtitleName
            text: header.subtitle
            visible: text !== ""
            wrapMode: header.subtitleWraps ? Text.Wrap : Text.NoWrap
            elide: header.subtitleWraps ? Text.ElideNone : Text.ElideRight
            Layout.fillWidth: true
            Accessible.ignored: !header.subtitleAnnounced
            Accessible.role: Accessible.StaticText
            Accessible.name: text
        }
    }
    EscChip {
        text: header.escHint
        visible: header.escHint !== ""
    }
    // Esc and the Cancel button reach the same command from the keyboard, so the cross stays
    // out of the Tab order.
    KeyButton {
        id: closer
        objectName: header.closeName
        focusPolicy: Qt.NoFocus
        implicitWidth: 30
        implicitHeight: 30
        background: Rectangle {
            visible: closer.hovered
            radius: Look.t.radiusControl
            color: Look.t.control_hover
        }
        // Two strokes, so the cross needs no icon font.
        contentItem: Item {
            Repeater {
                model: [45, -45]
                delegate: Rectangle {
                    required property int modelData
                    anchors.centerIn: parent
                    width: 14
                    height: 2
                    radius: 1
                    rotation: modelData
                    color: closer.hovered ? Look.t.hover_text : Look.t.text_muted
                }
            }
        }
        Accessible.role: Accessible.Button
        Accessible.name: "Close"
        Accessible.description: header.escHint
        onClicked: header.closeRequested()
    }
}
