// The page of a dialog scene: a column that scrolls only vertically, and from the keyboard.
// Children are laid out in the column; `label` is what assistive technology calls the page.
import QtQuick
import QtQuick.Layouts

Item {
    id: page
    property string label
    default property alias content: column.data

    // Give the page the keyboard, for a control that is about to hide or disable itself.
    function focusPage() {
        scroll.forceActiveFocus(Qt.TabFocusReason)
    }

    ScrollArea {
        id: scroll
        objectName: "contentScroll"
        anchors.fill: parent
        contentWidth: width
        contentHeight: column.implicitHeight + 32
        followed: column
        // A tab stop that scrolls as a scroll area does: arrows, Page Up, Page Down, Home, End.
        activeFocusOnTab: true
        Accessible.role: Accessible.Pane
        Accessible.name: page.label
        scrollName: "Content scroll bar"
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
