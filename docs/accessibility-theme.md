# Accessibility and semantic theme contract

AudioForge's current light interface uses semantic visual tokens from
`python/mic_eq/ui/theme.py`. Widget and custom-painter code consumes roles such
as action, status, data-surface, meter, and text colors; it does not own literal
colors. This is infrastructure hardening, not a dark-theme redesign.

## Retention gates

| Gate | Requirement |
|---|---|
| Text contrast | Every registered foreground/background pair is at least WCAG AA 4.5:1. |
| Names and labels | Every button, combo, spin box, slider, progress bar, and read-only workflow passage has an accessible name; form labels are explicit buddies where applicable. |
| Keyboard flow | Device selection follows input, output, channel mode, cleanup, and refresh; workflow dialogs and processing actions have explicit local tab order. |
| Scaling | The top-level content and dense EQ area scroll instead of clipping; the window remains operable at 900 x 640 logical pixels. Fonts use points or the platform default, never fixed pixel sizes. |
| Reduced motion | Audio meters remain live data, but their nonessential refresh rate drops from about 60 Hz to 10 Hz when Windows or `AUDIOFORGE_REDUCED_MOTION` requests reduced motion. No decorative animation is used. |
| Regression | A source audit rejects literal colors or pixel font sizes outside the semantic token module. Widget-tree tests instantiate the real main window and all calibration dialogs. |

The deterministic checks live in
`python/tests/test_ui_accessibility_theme.py`. Disabled-control contrast is not
treated as readable body text because disabled controls are intentionally
non-interactive; all active text pairs are registered and tested.

## Remaining boundaries

- Screen-reader behavior still depends on Qt's Windows accessibility bridge and
  should receive a manual Narrator smoke check before a release claiming formal
  accessibility conformance.
- The custom meters expose concise names but not a continuously announced value;
  announcing 10-60 updates per second would be disruptive. Adjacent textual
  health/status labels carry decisions instead.
- A future dark theme must define and independently contrast-test a complete
  token set. It must not infer colors by inverting this light palette.
- The 900 x 640 gate uses Qt logical pixels. At extreme text enlargement, scroll
  access is the fallback rather than shrinking text or controls.
