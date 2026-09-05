# AudioForge 2.0.0

Release candidate: software and packaging validation is in progress. Hardware
microphone qualification is pending. See [the readiness register](../docs/v2-readiness.md)
for evidence and remaining publication gates.

## Reliability

- Analysis cancellation retains worker ownership and rejects stale results.
- Offline native processing releases the Python GIL and rejects invalid inputs.
- Configuration recovery preserves damaged files and incomplete migrations;
  preset creation refuses accidental replacement after filename sanitization.
- EQ enable/disable transitions run through the production processing path.

## Installation and distribution

Windows distribution provides a portable archive and a per-user MSI installer.
Original AudioForge source remains MIT licensed. The combined PyQt6 application
is distributed under GPLv3; dependency terms and notices accompany the bundle.
Corresponding-source arrangements must be complete before final publication.

Source builds use CPython 3.13.15 x64 and Rust 1.94.0. VAD uses the pinned
CPU-only ONNX Runtime; DeepFilter uses a verified source build.
Published checksum, manifest,
metadata, and qualification files describe the exact downloadable artifacts.
The release workflow validates a candidate once and promotes those same bytes.

Windows 10/11 are the intended product targets. Hosted software checks do not
replace physical-device, reconnect, sleep/resume, or sustained audio qualification.
