// A Flickable that scrolls from the keyboard and keeps the control with the keyboard focus in
// view; the page of a window and the page of a dialog share it. Keys the focused control did not
// take scroll it: the arrows by a few lines, Page Up and Page Down by a screen, Home and End to
// the ends. Ctrl, Alt and Meta leave a key to shortcuts, and a text field keeps the arrows and
// Home and End for its cursor.
import QtQuick
import QtQuick.Controls.Basic

Flickable {
    id: area
    // The item whose descendants are kept in view when one of them takes the focus.
    property Item followed
    // Off while something else, such as a pointer press, is placing the page.
    property bool following: true
    // What assistive technology calls the scroll bar.
    property string scrollName
    ScrollBar.vertical: ScrollBar {
        id: bar
        Accessible.name: area.scrollName
        policy: area.contentHeight > area.height ? ScrollBar.AlwaysOn : ScrollBar.AlwaysOff
        contentItem: Rectangle {
            implicitWidth: 6
            radius: 3
            color: bar.pressed || bar.hovered ? Look.t.text_muted : Look.t.control_edge
        }
    }
    readonly property real lineStep: Look.fsBody * 3
    readonly property real maxY: Math.max(0, contentHeight - height)
    boundsBehavior: Flickable.StopAtBounds
    clip: true
    function scrollTo(y) {
        contentY = Math.max(0, Math.min(maxY, y))
    }
    function targetFor(key, typing) {
        switch (key) {
        case Qt.Key_PageUp: return contentY - height
        case Qt.Key_PageDown: return contentY + height
        }
        if (typing)
            return undefined
        switch (key) {
        case Qt.Key_Up: return contentY - lineStep
        case Qt.Key_Down: return contentY + lineStep
        case Qt.Key_Home: return 0
        case Qt.Key_End: return maxY
        }
        return undefined
    }
    Keys.onPressed: event => {
        const typing = (focusItem as TextInput) !== null || (focusItem as TextEdit) !== null
        const target = targetFor(event.key, typing)
        const shortcut = event.modifiers & (Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier)
        if (target === undefined || shortcut)
            return
        scrollTo(target)
        event.accepted = true
    }
    readonly property Item focusItem: Window.activeFocusItem
    onFocusItemChanged: {
        const item = focusItem
        let inside = false
        for (let p = item; p && !inside; p = p.parent)
            inside = p === followed
        if (!inside || !visible || !following)
            return
        const top = item.mapToItem(contentItem, 0, 0).y
        if (top < contentY + 12)
            scrollTo(top - 12)
        else if (top + item.height > contentY + height - 12)
            scrollTo(top + item.height - height + 12)
    }
}
