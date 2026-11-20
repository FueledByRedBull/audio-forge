// What every scene of the app shares: the theme tokens the host passes in as the `theme`
// context property (quick_scene.load_scene), and the text sizes that follow the app font.
pragma Singleton
import QtQuick

QtObject {
    id: look
    readonly property var t: theme
    readonly property bool hasIcons: t.iconFont !== ""
    readonly property bool reducedMotion: t.reducedMotion
    // The fill of a raised control in the top bar, lit from above; one object every such
    // control shares, so a control that is not raised makes none.
    readonly property Gradient raisedFill: Gradient {
        GradientStop { position: 0; color: look.t.toolbar_control_top }
        GradientStop { position: 1; color: look.t.toolbar_control_bottom }
    }
    // Under reduced motion nothing moves; only fades run, at the theme's own fade length.
    function movement(ms: int): int {
        return reducedMotion ? 0 : ms
    }
    // The state a status proxy reports; one that reports none, or no proxy, reads as idle.
    function statusState(proxy): string {
        return proxy && proxy.state !== "" ? proxy.state : "idle"
    }
    // The colours of a state; idle's for a state the theme lacks.
    function statusSurface(state: string): color {
        return t["status_" + state + "_surface"] ?? t.status_idle_surface
    }
    function statusBorder(state: string): color {
        return t["status_" + state + "_border"] ?? t.status_idle_border
    }
    function statusText(state: string): color {
        return t["status_" + state + "_text"] ?? t.status_idle_text
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
