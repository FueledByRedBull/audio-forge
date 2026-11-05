// The page of a dialog scene: a column that scrolls only vertically, and from the keyboard.
// Children are laid out in the column; `label` is what assistive technology calls the page.
import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts

Item {
    id: page
    property string label
    default property alias content: column.data

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
        Accessible.name: page.label
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
        }
    }

    FocusRing {
        objectName: "scrollFocusRing"
        visible: scroll.activeFocus
        outset: -2
    }
}
