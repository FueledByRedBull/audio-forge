# Repository screenshot contract

The README screenshots are generated from in-memory, sanitized application
state. The capture does not read user configuration, enumerate hardware, load
custom presets, or start an audio stream.

Regenerate them from the repository root with:

```powershell
./.venv/Scripts/python.exe python/tools/capture_repository_screenshots.py
```

The tool fixes Qt to offscreen Fusion rendering at 96 logical DPI, a scale
factor of 1, and Segoe UI 9 pt. It writes optimized PNGs to `docs/images/` and a
hash/dimension/provenance report to `evaluation/ui-screenshot-report.json`.

Refresh the screenshots whenever any of these change:

- a shipped control, label, workflow step, default window geometry, or tab;
- routing, EQ, dynamics, or Auto Voice Setup presentation;
- semantic theme tokens, typography, accessibility layout, or Qt version;
- the major/minor release whose UI the README describes.

After regeneration, run the focused screenshot tests and visually inspect all
three images for clipping, stale labels, personal data, illegible text, and
unexpected scroll positions. Hash changes alone are not approval.

The offscreen images intentionally omit Windows title-bar chrome and do not
claim to demonstrate live hardware activity. Their purpose is a stable product
tour with reproducible, non-personal content.
