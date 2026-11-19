// What every scene of the app shares: the theme tokens the host passes in as the `theme`
// context property (quick_scene.load_scene), and the text sizes that follow the app font.
pragma Singleton
import QtQuick

QtObject {
    readonly property var t: theme
    readonly property bool hasIcons: t.iconFont !== ""
    readonly property bool reducedMotion: t.reducedMotion
    // Under reduced motion nothing moves; only fades run, at the theme's own fade length.
    function movement(ms: int): int {
        return reducedMotion ? 0 : ms
    }
    // The sizes of text, in pixels, follow the app font (9 pt is the scale as written).
    // Text that carries information never drops below the floor; only ticks and keycaps
    // use the smaller tick size.
    readonly property real zoom: Application.font.pointSize > 0 ? Application.font.pointSize / 9 : 1
    function scaled(px: real, floor: int): int {
        return Math.max(floor, Math.round(px * zoom))
    }
    readonly property int fsTick: scaled(t.typeTick, t.typeTick)
    readonly property int fsCaption: scaled(t.typeScale.caption, t.typeFloor)
    readonly property int fsBody: scaled(t.typeScale.body, t.typeFloor)
    readonly property int fsLead: scaled(t.typeScale.lead, t.typeFloor)
    readonly property int fsHeading: scaled(t.typeScale.heading, t.typeFloor)
    readonly property int fsTitle: scaled(t.typeScale.title, t.typeFloor)
    readonly property int fsDisplay: scaled(t.typeScale.display, t.typeFloor)
}
