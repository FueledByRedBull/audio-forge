# Constrained EQ graph editing

The EQ response graph is an editor backed by the same typed band schema and
native Rust response renderer as the numeric controls. It does not maintain a
second filter implementation.

## Interaction contract

- Drag a handle horizontally to set center/cutoff frequency on the same
  20-20,000 Hz logarithmic range as the spinbox.
- Drag vertically to set gain from -12 to +12 dB for bell and shelf filters.
- Notch and high/low-pass handles move horizontally only because their native
  gain field is ignored.
- Graph values snap to the controls' 1 Hz and 0.1 dB precision and are sent to
  native DSP through one validated typed-band batch.
- `[` and `]` select a handle without a mouse. Arrow keys edit the selected
  handle; Shift uses a coarser step. The existing spinboxes remain the exact
  keyboard alternative for frequency, gain, Q, type, slope, and bypass.

The graph rate-limits processor updates to about 30 Hz while painting the
pointer position immediately. Release flushes the exact final value. One drag
or key gesture is one bounded-history transaction; a long drag cannot fill the
undo stack with intermediate points.

## Visual and scaling behavior

Hit testing and coordinate conversion use Qt logical pixels, so Windows DPI
scaling does not require device-pixel compensation. Handles are rendered for
all ten bands, with selected, enabled, and bypassed states visually distinct.
Coordinate functions clamp before conversion and remain invertible within the
1 Hz graph precision after resize.

## Retention checks

Focused tests cover mouse drag, horizontal-only pass filters, native/control
synchronization, range clamping, keyboard editing, resize mapping, signal
boundaries, one-entry history integration, undo, and type changes. The feature
adds no audio-thread allocation, algorithmic latency, or background work; only
the existing off-thread UI/native control path runs during an edit.

Q remains a numeric control rather than a wheel/secondary drag gesture. That
keeps the interaction discoverable and avoids an inaccessible hidden mode.
