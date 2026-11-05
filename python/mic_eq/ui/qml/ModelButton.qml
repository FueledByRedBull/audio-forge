// A button whose check state belongs to the model. Every way of switching it (pointer, keys,
// assistive technology) changes `checked`, so requested() follows that; the model decides,
// and the button goes back to the model's state `on`.
import QtQuick

KeyButton {
    property bool on
    signal requested()
    // The real check state, so assistive technology also hears it before the first change.
    checkable: true
    checked: on
    onCheckedChanged: if (checked !== on) {
        if (enabled) requested()
        checked = Qt.binding(() => on)
    }
    Accessible.onPressAction: click()
}
