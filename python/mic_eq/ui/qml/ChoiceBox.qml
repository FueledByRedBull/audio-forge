// A drop-down list of text choices. `label` names it for assistive technology; the shown choice
// is added to the name, because a combo box has no value channel in QML and a name change is
// spoken where a description change often is not. The visible text stays in the name.
pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls.Basic

ComboBox {
    id: picker
    property string label
    // Set for a choice that changes what the sound does, so it does not read as the default.
    // Colour is not the only cue (a contrast theme has one text colour): the edge is thicker
    // and the text bolder.
    property bool alert: false
    // With nothing chosen the box shows `prompt`, set apart from a real choice.
    property bool prompting: false
    property string prompt: ""
    property string expandGlyph: ""
    readonly property alias shown: shownText
    readonly property real windowHeight: Window.height
    opacity: enabled ? 1 : 0.45
    implicitHeight: 32
    implicitWidth: 170
    leftPadding: 10
    rightPadding: 26
    Accessible.name: label + (shownText.text !== "" ? ": " + shownText.text : "")
    contentItem: Text {
        id: shownText
        text: picker.prompting ? picker.prompt : picker.displayText
        textFormat: Text.PlainText
        color: picker.hovered || picker.down ? Look.t.hover_text
             : picker.prompting ? Look.t.text_muted
             : picker.alert ? Look.t.status_warn_text : Look.t.text_primary
        font.pixelSize: Look.fsBody
        font.italic: picker.prompting
        font.weight: picker.alert ? Font.DemiBold : Font.Normal
        elide: Text.ElideRight
        verticalAlignment: Text.AlignVCenter
    }
    indicator: Text {
        x: picker.width - width - 11
        y: (picker.height - height) / 2
        text: picker.expandGlyph
        textFormat: Text.PlainText
        font.family: Look.t.iconFont
        font.pixelSize: Look.fsCaption
        color: picker.hovered || picker.down ? Look.t.hover_text : Look.t.text_muted
    }
    background: Rectangle {
        radius: Look.t.radiusControl
        color: picker.hovered || picker.down ? Look.t.control_hover : Look.t.control_surface
        border.width: picker.alert ? 2 : 1
        border.color: picker.prompting ? Look.t.status_warn_border
                    : picker.alert ? Look.t.status_warn_text : Look.t.control_edge
        FocusRing { control: picker }
    }
    delegate: ItemDelegate {
        id: option
        required property int index
        required property var modelData
        width: ListView.view ? ListView.view.width : 0
        height: 30
        highlighted: picker.highlightedIndex === index
        contentItem: Text {
            text: option.modelData
            textFormat: Text.PlainText
            color: option.highlighted ? Look.t.hover_text : Look.t.text_primary
            font.pixelSize: Look.fsBody
            elide: Text.ElideRight
            verticalAlignment: Text.AlignVCenter
        }
        background: Rectangle {
            radius: 4
            color: option.highlighted ? Look.t.control_hover : "transparent"
        }
    }
    popup: Popup {
        y: picker.height + 4
        width: Math.max(picker.width, 240)
        padding: 4
        // Kept inside the window, as the stock combo box does.
        topMargin: 6
        bottomMargin: 6
        implicitHeight: Math.min(contentItem.implicitHeight + 8, 320, picker.windowHeight - 12)
        contentItem: ListView {
            clip: true
            implicitHeight: contentHeight
            model: picker.delegateModel
            currentIndex: picker.highlightedIndex
            footer: Text {
                width: ListView.view ? ListView.view.width : 0
                height: picker.count === 0 ? 30 : 0
                visible: picker.count === 0
                text: "No choices available"
                textFormat: Text.PlainText
                color: Look.t.text_muted
                font.pixelSize: Look.fsBody
                leftPadding: 8
                verticalAlignment: Text.AlignVCenter
            }
        }
        background: Rectangle {
            color: Look.t.control_surface
            border.color: Look.t.border_strong
            radius: Look.t.radiusControl
        }
    }
}
