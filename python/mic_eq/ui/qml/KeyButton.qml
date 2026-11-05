// Return and Enter press a button as Space does; a held key presses it once.
import QtQuick
import QtQuick.Controls.Basic

AbstractButton {
    Keys.onReturnPressed: event => { if (!event.isAutoRepeat) click() }
    Keys.onEnterPressed: event => { if (!event.isAutoRepeat) click() }
}
