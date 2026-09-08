# AudioForge 1.12.2

AudioForge 1.12.2 delivers the reliability, packaging, and runtime updates
below. See [the release workflow](../RELEASING.md) for the candidate validation
and publication gates.

## Reliability

- Voice Setup uses the applied limiter settings throughout analysis and
  second-passage verification, which runs in a cancellable background worker.
  Temporary calibration evidence survives rollback and Undo only within the
  same capture context.
- First-run setup lets you select devices directly. Minimal Processing disables
  compression, and preset, bypass, and latency labels describe their behavior.
- Analysis cancellation retains worker ownership and rejects stale results.
- Offline native processing releases the Python GIL and rejects invalid inputs.
- Configuration recovery preserves damaged files and incomplete migrations;
  preset creation refuses accidental replacement after filename sanitization.
- EQ enable/disable transitions run through the production processing path.
- Noise-model discovery no longer panics when an inherited stderr pipe is closed.
- DeepFilter remains operational through sustained digital silence.
- Windows device lists include full friendly names, such as the microphone model.
- VAD Only preserves quiet detected speech independently of the disabled level
  threshold, while retaining level fallback if VAD is unavailable.

## Installation and distribution

Windows distribution provides a portable archive and a per-user MSI installer.
Original AudioForge source remains MIT licensed. The combined PyQt6 application
is distributed under GPLv3; dependency terms and notices accompany the bundle.

Source builds use CPython 3.13.15 x64 and Rust 1.94.0. VAD uses the pinned
CPU-only ONNX Runtime; DeepFilter uses a verified source build.
Published checksum, manifest, metadata, and qualification files describe the
exact downloadable artifacts.
The release workflow validates a candidate once and promotes those same bytes.

Windows 10/11 remain compatibility targets. For v1.12.2, release qualification
covers Windows 11 x64 with a USB microphone and virtual routes at the observed
48 kHz, an automated baseline of at least 1,800 seconds, and model
switching/restoration. Windows 10, analog or built-in input, 44.1 kHz, and
physical reconnect, default-device change, and sleep/resume cases remain
unqualified for this release. Hosted software checks do not replace
the exact-artifact hardware evidence, and the qualification records do not
imply every device was tested.
