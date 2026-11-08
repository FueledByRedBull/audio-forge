// A number field with step buttons, typed or stepped from the keyboard (Up and Down, Page Up and
// Page Down). `label` names it for assistive technology; the shown value is added to the name,
// because a spin box has no value channel in QML that reads in real units. The value is a real
// number between `low` and `high`, shown with `decimals` places and followed by `suffix`;
// `modified` carries the number the person chose.
pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls.Basic

SpinBox {
    id: box
    property string label
    property real low: 0
    property real high: 100
    property int decimals: 2
    property string suffix
    property real number: 0
    signal modified(real number)
    readonly property real factor: Math.pow(10, decimals)
    from: Math.round(low * factor)
    to: Math.round(high * factor)
    stepSize: factor
    editable: true
    opacity: enabled ? 1 : 0.45
    implicitHeight: 32
    implicitWidth: 170
    leftPadding: 30
    rightPadding: 30
    onNumberChanged: value = Math.round(number * factor)
    Component.onCompleted: value = Math.round(number * factor)
    onValueModified: modified(value / factor)
    textFromValue: (v, locale) => (v / factor).toFixed(decimals)
    valueFromText: (text, locale) => {
        const parsed = Number(text.replace(",", "."))
        return isNaN(parsed) ? value : Math.round(parsed * factor)
    }
    validator: DoubleValidator {
        bottom: box.low
        top: box.high
        decimals: box.decimals
        notation: DoubleValidator.StandardNotation
        locale: "C"
    }
    Accessible.name: label + ": " + field.text + suffix
    contentItem: TextInput {
        id: field
        objectName: box.objectName + "Input"
        text: box.textFromValue(box.value, box.locale)
        font.pixelSize: Look.fsBody
        color: Look.t.text_primary
        selectionColor: Look.t.accent
        selectedTextColor: Look.t.text_on_accent
        verticalAlignment: Text.AlignVCenter
        readOnly: !box.editable
        validator: box.validator
        inputMethodHints: Qt.ImhFormattedNumbersOnly
        // The box takes the typed number when the field loses focus; Enter, which goes on to the
        // scene's main button, must not wait for that.
        onAccepted: {
            const before = box.value
            box.value = box.valueFromText(text, box.locale)
            if (box.value !== before)
                box.valueModified()
        }
        Accessible.ignored: true
        Text {
            x: field.contentWidth + 4
            height: parent.height
            text: box.suffix
            textFormat: Text.PlainText
            font.pixelSize: Look.fsBody
            color: Look.t.text_muted
            verticalAlignment: Text.AlignVCenter
            Accessible.ignored: true
        }
    }
    up.indicator: Rectangle {
        x: box.width - width
        height: box.height
        implicitWidth: 28
        radius: Look.t.radiusControl
        color: box.up.pressed || box.up.hovered ? Look.t.control_hover : "transparent"
        Text {
            anchors.centerIn: parent
            text: "+"
            textFormat: Text.PlainText
            font.pixelSize: Look.fsLead
            color: box.up.hovered ? Look.t.hover_text : Look.t.text_muted
        }
    }
    down.indicator: Rectangle {
        height: box.height
        implicitWidth: 28
        radius: Look.t.radiusControl
        color: box.down.pressed || box.down.hovered ? Look.t.control_hover : "transparent"
        Text {
            anchors.centerIn: parent
            text: "−"
            textFormat: Text.PlainText
            font.pixelSize: Look.fsLead
            color: box.down.hovered ? Look.t.hover_text : Look.t.text_muted
        }
    }
    background: Rectangle {
        radius: Look.t.radiusControl
        color: Look.t.control_surface
        border.color: Look.t.control_edge
        FocusRing { control: box }
    }
}
