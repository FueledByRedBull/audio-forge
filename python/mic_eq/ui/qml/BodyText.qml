// Running text of a dialog scene: wraps to the width the layout gives it.
import QtQuick
import QtQuick.Layouts

Text {
    textFormat: Text.PlainText
    color: Look.t.text_primary
    font.pixelSize: Look.fsBody
    wrapMode: Text.Wrap
    Layout.fillWidth: true
}
