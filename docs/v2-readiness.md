# AudioForge v2.0 readiness review

This is a pre-package audit snapshot. Baseline: `885a5e98c3ccbb3f455b8da1a075d4f5ef56c22f`,
v1.11.4, initially reviewed September 5, 2026 and rechecked on September 6,
2026. The software CI evidence cited here is fixed run `33993979463` for
revision `083b908`. The final candidate's exact build, source receipt, and
qualification evidence belongs in the generated release sidecars and PR #62,
so this register does not require a routine post-build documentation commit.
The pre-existing [correctness audit](audit-findings.md) is preserved as
historical evidence. Its findings and the supplied AF-01 through AF-20 readiness
audit are checked against the implementation, runnable reproductions, and
release history.

## Release decision

A hardening release is justified; the evidence does not support a wholesale DSP
or UI rewrite. The adopted v2 runtime path is CPython 3.13.15 with the pinned
CPU-only ONNX Runtime 1.23.2 package. Its isolated software probe produced exact
posterior parity for 498 captures and 20,908 frames. The release gate still
requires the 3.13 portable and MSI artifacts to be built, smoke-tested, and
qualified; the prior CPython 3.12 package and MSI lifecycle checks are baseline
evidence only.

Microphone and device qualification remains pending by decision. The
corresponding-source graph verifies 342 hydrated archives with zero blockers.
The release bundle must generate a project Git source receipt for the reviewed
release commit. Tagged-candidate package, publication, hardware, and promotion
are separate release gates; their exact evidence belongs in generated sidecars
and PR #62.

At audit time, local evidence included 660 passing Python tests, 321 Rust unit
tests plus one stress test and one doc test, six hardware tests explicitly
ignored, passing formatting, Clippy, RustSec, Ruff, and Pyright checks, and
green Python/Rust CI run `33993979463` for revision `083b908`. These results
support targeted hardening and the final release gates; the generated candidate
sidecars determine v2.0 qualification.

## Audit register

| Item | Review and required disposition |
|---|---|
| AF-01: failure propagation | **Implemented locally.** `build_exe.ps1`, `build_msi.ps1`, and the release workflows check each native command's exit code; `python/tools/check_workflows.py` parses the executable steps. The local workflow check and Python/Rust CI run `33993979463` passed for revision `083b908`; the candidate package gate must preserve that failure propagation. |
| AF-02: licensing | **Implemented for the candidate; publication pending.** GPLv3 distribution and MIT original-source terms are recorded in `licenses/THIRD_PARTY_NOTICES.md`; `license_inventory.py` and the source manifest retain versioned notices for CPython 3.13.15, Qt/PyQt, the CPU ONNX Runtime graph, DeepFilter, and the locked native dependencies. The source graph has zero blockers; the final receipt and published-artifact inventory are release sidecar checks. |
| AF-03 / F01 / F13: analysis lifetime | **Implemented locally.** Calibration and Voice Setup stop delayed capture timers, cancel workers cooperatively, retain active workers, and reject obsolete generations. `test_ui_sample_rate_and_diagnostics.py` covers delayed-capture cancellation, stale generations, and worker cancellation. |
| AF-04 / F09: GIL | **Implemented and measured.** Native offline work uses `py.detach` in `rust-core/src/audio/processor/python_api.rs`; `test_native_regressions.py` verifies a concurrent Python heartbeat. The measured local gaps are recorded above; this is not a hardware claim. |
| AF-05: allocation instrumentation | **Implemented.** `rust-core/src/lib.rs` uses thread-local nested allocation scopes, with concurrent and nested-scope regressions; steady-state DSP allocation tests remain in `rust-core/src/audio/processor/tests.rs`. |
| AF-06: configuration recovery | **Implemented.** Invalid and future-schema configuration is preserved, save is blocked when recovery is unavailable, and migration state is exposed. `test_config_v17.py` covers preservation, collision-safe backups, and blocked saves. |
| AF-07: supported tools | **Implemented as a support boundary.** `requirements/*.txt`, `rust-toolchain.toml`, `check_versions.py`, and the source recipe pin CPython 3.13.15, NumPy/SciPy, and Rust 1.94.0. The supported release path does not claim broader Python or Rust compatibility. |
| AF-08: asset origins | **Implemented for the software candidate.** Models and Silero are pinned, the inherited `df.dll` was replaced by the verified DeepFilter source recipe, lock, patch record, and attestation, and the previous restricted runtime path was removed in favor of the pinned CPU-only ONNX Runtime archive. Final packaged-artifact qualification is still pending. |
| AF-09: hardware coverage | **Pending by decision.** The maintainer deferred microphone testing and no hardware runner is registered. Six hardware cases remain ignored; hosted software tests cannot close device lifecycle or route qualification. |
| AF-10 / F11: qualification origin | **Implemented as validators; candidate evidence required.** `release_provenance.py` and the qualification workflows bind reports to repository, revision, workflow, event, attempt, and typed status. CI run `33993979463` passed for revision `083b908`; tagged-candidate package and hardware reports must be bound by the same validators. |
| AF-11 / F12: publication | **Workflow implemented; publication pending.** Promotion now handles drafts, digest-checked retries, exact asset sets, final-state checks, and publication last in `.github/workflows/release-promote.yml`. No release was published in this review. |
| AF-12: durable evidence | **Implemented in the release path.** The package and promotion workflows produce privacy-safe provenance, source, MSI, and qualification sidecars for release assets. Those generated sidecars are the authoritative candidate evidence to review with PR #62. |
| AF-13: prereleases | **Implemented.** `python/tools/release_version.py` and package/promotion workflows validate canonical final/RC tags, Cargo versions, artifact names, and MSI versions; release-version tests pass. Immutable tag rules remain the remote control. |
| AF-14: native freshness | **Implemented locally.** `build_exe.ps1` rebuilds the Rust extension with `maturin develop --release --locked` before PyInstaller. The adopted CPython 3.13/CPU ORT native path has parity evidence and a source-built DeepFilter attestation; the release gate must attach final portable/MSI build and candidate execution evidence. |
| AF-15: reduced build flag | **Implemented.** The unsupported reduced packaging path is absent; full asset verification is required by `build_exe.ps1` and the package smoke checks. |
| AF-16: workflow checks | **Implemented locally.** `python/tools/check_workflows.py` inspects parsed executable steps and failure propagation, and the workflow check passes. Run `33993979463` is the successful Python/Rust software check for revision `083b908`; it does not replace tagged-candidate, hardware, source-receipt, or promotion evidence. |
| AF-17: native fallback cause | **Implemented.** Native headroom and DeepFilter paths distinguish unavailable, invalid, and runtime failures; Python fallback output is labeled advisory while Rust output is authoritative. Regression tests cover these labels and safety gates. |
| AF-18: ownership refactor | **Retained as targeted maintenance.** The changes isolate demonstrated ownership boundaries in configuration, analysis cancellation, UI lifecycle, and native DSP; the evidence does not justify a broad rewrite. |
| AF-19: CI time | **Implemented and measured.** RustSec uses a pinned, cached `cargo-audit` executable and advisory database in the CI workflows, and software CI run `33993979463` passed. Release package qualification is tracked under AF-10 through AF-12. |
| AF-20: repository hygiene | **Implemented and reviewed.** Contributor/security guidance, issue/PR templates, version checks, and `test_repository_hygiene.py` are present. Revision `083b908` has both CI jobs green; the final candidate must be reviewed, committed, and pushed before tag qualification. |
| F02: preset collision | **Implemented.** Default preset creation refuses an occupied sanitized path, explicit overwrite is separate, and temporary-file plus exclusive-link publication is atomic in `config_parts/presets.py`; config tests cover it. |
| F03: stopped recovery | **Implemented locally.** Stopped diagnostics continue pending native recovery and reconcile controls; `test_stopped_diagnostics_continue_native_recovery_and_reconcile_controls` passes. |
| F04: malformed presets | **Implemented.** Parser, type, and recursion failures normalize at the preset boundary to `PresetValidationError`; malformed files are covered by configuration tests. |
| F05: partial migration | **Implemented.** Migration copies into a staging directory, records pending/staged state, publishes atomically, and retries after an injected copy failure; `test_config_v17.py` covers the failure path. |
| F06: route keys | **Implemented.** Structured JSON route keys are parsed before the legacy `||` form; delimiter-containing names round-trip in `test_config_v17.py`. |
| F07: refresh signals | **Implemented.** Refresh restores every affected control's prior signal-block state; `test_refresh_devices_restores_all_control_signal_states` passes. |
| F08: endpoint identity | **Implemented in software; lifecycle pending.** CPAL 0.17.3 and stable endpoint-ID lookup are used for opening and recovery, while persisted raw WASAPI IDs remain supported. Four endpoint-ID regressions pass; reconnect qualification still requires hardware. |
| F10: offline input | **Implemented.** Native offline APIs reject nonfinite audio/settings and unsupported rates before DSP construction; `test_native_regressions.py` covers the former 40 Hz panic and NaN cases. |
| F14: EQ transitions | **Implemented with native regression coverage.** The live bypass path participates in the EQ transition, and `rust-core/src/dsp/eq.rs` tests stale-state crossfading with bounded, allocation-free processing. Hardware listening evidence is outside this review. |
| Additional: MSI | **Baseline implemented and tested locally; v2 candidate gate required.** Pinned WiX 7.0.0 passed administrative extraction and exact payload comparison, baseline installation, major upgrade, logged downgrade rejection, shortcut checks, configuration preservation, and uninstall cleanup for the prior CPython 3.12 payload. The CPython 3.13/CPU ORT MSI must pass the same lifecycle before publication. |
| Additional: pywin32 | **Implemented.** pywin32 is explicitly pinned in the runtime lock and included in source/inventory packaging; it is no longer obtained only as a Semgrep side effect. |
| Additional: dataset paths | **Implemented.** Evaluation path resolution rejects drive, UNC, and traversal escapes before I/O; six path regressions pass. |
| Additional: CPU runtime transition | **Implemented in source and software qualification; final packaging pending.** The pinned CPU-only ONNX Runtime 1.23.2 archive, source graph, full notices, extracted asset hashes, and 498-capture/20,908-frame parity report are present. Final portable/MSI packaging and tagged-artifact qualification remain open. |

## Repository and security evidence

- The baseline refs contained 275 commits and 3,655 objects. Four
  historical 7z blobs total 397,191,316 bytes; none is in the repository tree. A
  history rewrite would disrupt published tags and is not part of this work.
- A narrow, value-redacted scan of 1,717 source/document blobs found no matches
  for private-key headers, GitHub tokens, or AWS access-key IDs. It excludes
  likely-secret files and is not a complete secret-scan certification.
- The refreshed Rust lock passed `cargo audit --deny warnings`. The developer
  lock still carries the three explicitly documented advisories for Semgrep's
  optional MCP server; those are outside the runtime lock and remain a tooling
  decision before publication.
- The adopted CPython 3.13.15 runtime's `pip-audit` scan reported zero known
  vulnerabilities. The application lock currently resolves 141 Rust packages;
  its `cargo audit --deny warnings` result and the release-mode contention
  checks are green. These are local software checks and do not replace the
  final packaged-artifact or remote-candidate gates.
- The refreshed Semgrep SARIF has four warning-level URL-download results and
  twelve notes, with no error-level results. The configured scan ran 269 rules
  over 272 targets; its reported parse coverage was 99.2%. Download entry points use trusted
  HTTPS hosts; FFI/COM notes require source review rather than counting every
  `unsafe` block as a vulnerability. The dataset path issue above was found
  during that review. A newer developer-tool/Semgrep pin and any additional
  finding exceptions remain pending explicit approval; they are not runtime
  release waivers.
- The CPU runtime transition removed the old unused ONNX Runtime/`ort-sys`
  download chain from the application lock; the current lock contains no
  `openssl`, `native-tls`, or `ureq` package entry. The earlier GitHub alerts
  for that unused OpenSSL 0.10.75 path therefore describe the superseded lock,
  while the developer-only MCP advisories remain separately documented. A fresh
  remote advisory scan is still part of candidate promotion.
- GitHub vulnerability alerts and automatic security updates were disabled
  despite the documentation. They were enabled and the resulting settings
  verified. Secret scanning and push protection were already enabled.
- `master` already required `python` and `rust` status checks, strict up-to-date
  checks, and prohibited force pushes/deletion. Administrative enforcement was
  enabled without replacing those settings. Active ruleset `22336761` now
  prohibits updating or deleting `v*` tags with no bypass actors. Administrators
  can still change repository policy; this is not an immutable external ledger.

## Embedded runtime security cross-check

The following table is historical evidence from the prior CPython 3.12.10
portable bundle. It is retained to identify the baseline artifact and must not
be read as the final v2.0 payload:

| Component | Bundle path | Product identity | SHA-256 |
|---|---|---|---|
| CPython | `_internal/python312.dll` | 3.12.10 | `9a0e3435aaa680d868150f87ab3e388ad2eebc22f87e036155c7b4eda8cd2120` |
| OpenSSL libssl | `_internal/libssl-3.dll` | 3.0.16 | `007142039f04d04e0ed607bda53de095e5bc6a8a10d26ecedde94ea7d2d7eefe` |
| OpenSSL libcrypto | `_internal/libcrypto-3.dll` | 3.0.16 | `ccfffddcd3defb8d899026298af9af43bc186130f8483d77e97c93233d5f27d7` |
| Qt Core | `_internal/PyQt6/Qt6/bin/Qt6Core.dll` | 6.11.1.0 | `fae4778a42e93adc82b831c879c886a05147e9cc26760808d21116be5547259b` |
| Qt GUI | `_internal/PyQt6/Qt6/bin/Qt6Gui.dll` | 6.11.1.0 | `8fceee959a670372aaa5763287c2ef7924cd9ecdbe2c29cf4b6c12a63079c503` |
| Qt Network | `_internal/PyQt6/Qt6/bin/Qt6Network.dll` | 6.11.1.0 | `0232076731e386b6cc353bdf743d5fb17e95a3493c2d02afac984db87ed47b96` |
| Qt Widgets | `_internal/PyQt6/Qt6/bin/Qt6Widgets.dll` | 6.11.1.0 | `4d603bff620ae0830d15c25a787f9f10ed3f972d998f1f17c9c4f83937280399` |

The adopted runtime evidence is separate from that baseline. The official
CPython 3.13.15 NuGet runtime exposes OpenSSL 3.0.21 DLLs, and the
corresponding-source graph retains the CPython 3.13.15 source-deps archive and
its OpenSSL 3.0.21 input. The packaged OpenSSL identity and final dependency
inventory remain release checks rather than claims made from the baseline table.

The current `launcher.py` and `python/mic_eq` source path uses local audio,
configuration, and Qt Core/GUI/Widgets APIs. It has no demonstrated
`PyQt6.QtNetwork`, TLS, CMS, or HTTP call path. `Qt6Network.dll` and Python's
`_ssl.pyd` were present in the historical bundle, while the new packaging path
must validate the exact 3.13 payload. This bounded review did not trace every
importer to a complete call graph, so it does not claim that every bundled
Python module is unreachable. No vulnerable-function execution or exploit was
demonstrated in the offline UI. The source-distribution tool reads verified
archive bytes in memory, the license inventory uses digest-checked archive
members, and corpus extraction uses controlled member names; no UI extraction
path was found.

Qt 6.11.1 remains the pinned Qt version. The reviewed advisories do not
demonstrate exposure in the loaded modules: Qt5Compat and QtXml are absent from
the bundle, the older Unix-only Qt OpenSSL backend does not describe this
Windows path, and SVG is pruned from the bundle. Qt 6.11.2 may be evaluated in a
later maintenance refresh, but its availability does not create a current v2.0
release blocker on the evidence reviewed here.

## Measured native responsiveness

The rebuilt release extension processed ten seconds at 48 kHz in
282.05–301.68 ms across five runs, comparable to the prior audit's 299 ms.
Concurrent 5 ms Python heartbeat gaps were 13.09–40.47 ms, compared with the
prior 301 ms stall. Numerical kernels remain native; releasing the GIL restores
Python callback scheduling rather than claiming an unmeasured DSP speedup.
These measurements are workstation observations, not hardware audio-route
qualification. The native regression also includes call-boundary gaps so the
end of a blocking native call cannot evade the check.

The CPU-only ONNX Runtime transition was measured separately against the prior
software backend: 498 captures and 20,908 frames produced zero maximum
absolute delta, zero RMSE, and unchanged threshold/hysteresis decisions. The
probe also measured whole-clip throughput, but it did not measure per-frame
jitter, microphone timing, output-route behavior, or a final packaged artifact.

## DeepFilter source-build investigation

The maintainer approved replacing the inherited DLL with a verified source
build. The original DLL's source revision cannot be recovered from repository
history or PE version metadata. A minimal build from DeepFilterNet commit
`d375b2d8309e0935d165700c91da9de862a99c31` works with installed Rust/MSVC;
cargo-c, cbindgen, Perl, and a nightly toolchain are unnecessary for the DLL.

The first isolated source build matched the inherited DLL exactly on eight
deterministic real-corpus comparisons across both pinned models. Frame p99 was
at most 1.603 ms against the predefined 10 ms gate. This is bounded offline
parity evidence, not proof of the inherited binary's origin.

That probe's upstream dependency lock failed its security audit: bytes,
crossbeam-channel, crossbeam-epoch, tar (two advisories), time, and tract-nnef
accounted for seven vulnerabilities, with additional maintenance/yank warnings.
The subsequent source build uses tract 0.21.17 and compatible security updates.
Its 227-package lock reports zero vulnerabilities. Two unmaintained-package
warnings remain documented: the compile-time `paste` macro and an inactive
optional dependency on `proc-macro-error`. A small, hash-checked patch removes
tract-linalg's unused restrictive `time` build dependency.

The checked-in recipe's output repeated all eight model/corpus comparisons with
zero maximum absolute error and zero RMSE. Low-latency and standard-model
latencies remain 480 and 1,440 samples respectively; all 18 measured recipe runs
produced finite output and frame p99 below 10 ms. These bounded comparisons do
not establish hardware qualification or binary bit-reproducibility. The recipe
records its output digest, compiler, source revision, patches, and dependency
lock separately from the inherited binary.

The adopted DLL is 13,767,680 bytes with SHA-256
`5f666380bd0f69b6def04d7aeb6d38c8c1c5780e2cb3c3f9602f05564aa948a5`.
The final-output validation runs stayed below 1.8 ms for frame p99 and 0.11 for
real-time factor; the checked-in evaluator records the exact measured results.
Subsequent builds carry their own output identity. Recipe hashes normalize text line endings so
Windows checkouts and Git source archives validate consistently. The final
manifest was regenerated against this recipe; 342 hydrated archives verify
with zero blockers. The project source receipt is intentionally deferred until
the reviewed release commit is available.

## Packaged startup

A real `--smoke-test` on the prior CPython 3.12 baseline found a packaged-only
Qt import failure. PyInstaller had collected ICU 78 from an unrelated Poppler
installation on the build process's PATH, shadowing the Windows ICU API used by
Qt. Removing those two generated files restored successful startup with the
normal environment. The supported builder now isolates PyInstaller's search
path to Python and Windows locations; package validation rejects app-local copies
of Windows ICU. The baseline package and exact EXE smoke checks passed. The
release gate must run the same check against the CPython 3.13/CPU ORT portable
artifact and its tagged remote candidate. Windows provides
the ICU C API in its system libraries ([Microsoft documentation](https://learn.microsoft.com/en-us/windows/win32/intl/international-components-for-unicode--icu-)).

## Release maintenance

The standing dependency source in `release-assets.json` remains pinned until
the asset set changes. The promotion path now generates the sizes, hashes,
inventory, and qualification sidecars as release assets, so documentation can
refer to the same candidate instead of requiring a post-release fallback commit.
Build, qualification, and promotion remain separately verifiable gates. Python/Rust
CI run `33993979463` passed for revision `083b908`; candidate runtime/package
changes must be reviewed and committed before tagged-candidate qualification.
Hardware qualification, source receipt publication, and final artifact
promotion remain explicit release gates.

This register reflects the implemented candidate and its remaining release
gaps; it is closed only after those gates produce evidence for the tagged
release.
