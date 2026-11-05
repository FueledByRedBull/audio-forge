// What every scene of the app shares: the theme tokens the host passes in as the `theme`
// context property (quick_scene.load_scene), and the text sizes that follow the app font.
pragma Singleton
import QtQuick

QtObject {
    readonly property var t: theme
    readonly property bool hasIcons: t.iconFont !== ""
    // The sizes of text, in pixels, follow the app font (9 pt is 12 px); nothing is smaller than 12.
    readonly property real zoom: Application.font.pointSize > 0 ? Application.font.pointSize / 9 : 1
    readonly property int fsCaption: Math.round(12 * zoom)
    readonly property int fsBody: Math.round(13 * zoom)
    readonly property int fsLead: Math.round(15 * zoom)
    readonly property int fsHeading: Math.round(18 * zoom)
    readonly property int fsTitle: Math.round(20 * zoom)
}
