# AudioForge v2.0 readiness review

Baseline: `885a5e98c3ccbb3f455b8da1a075d4f5ef56c22f`, v1.11.4, initially
reviewed September 5, 2026 and rechecked against the current tree on September
6, 2026. The pre-existing [correctness audit](audit-findings.md) is preserved
as historical evidence. Its findings and the supplied AF-01 through AF-20
readiness audit are checked against the current implementation, runnable
reproductions, and actual release history.

## Release decision

A hardening release is justified; the evidence does not support a wholesale DSP
or UI rewrite. The current candidate has passed the local software gates and a
fresh portable build and MSI lifecycle validation, but it is not release
complete. Microphone and device qualification remains pending, corresponding
source is incomplete because of the restricted DirectML runtime, and no remote
CI or publication run has been independently verified.

The latest local evidence is 626 Python tests, 320 Rust unit tests plus one
stress test and one doc test, six hardware tests explicitly ignored, passing
formatting, Clippy, and cargo-audit checks, and a fresh `build_exe.ps1 -Clean`
build whose package and exact EXE smoke tests passed. The corresponding-source
tool verified 344 hydrated archives with `--allow-incomplete`; its only current
blocker is DirectML licensing. These results support targeted hardening and
promotion gates, not a completed v2.0 qualification claim.

## Audit register

| Item | Review and required disposition |
|---|---|
| AF-01: failure propagation | **Implemented locally.** `build_exe.ps1`, `build_msi.ps1`, and the release workflows check each native command's exit code; `python/tools/check_workflows.py` parses the executable steps. The local workflow check passes; a remote run is still required for release evidence. |
| AF-02: licensing | **Partly implemented; publication pending.** GPLv3 distribution and MIT original-source terms are recorded in `licenses/THIRD_PARTY_NOTICES.md`; `license_inventory.py` and the source manifest retain versioned notices. The source manifest remains incomplete for restricted DirectML, so this is not a release approval. |
| AF-03 / F01 / F13: analysis lifetime | **Implemented locally.** Calibration and Voice Setup stop delayed capture timers, cancel workers cooperatively, retain active workers, and reject obsolete generations. `test_ui_sample_rate_and_diagnostics.py` covers delayed-capture cancellation, stale generations, and worker cancellation. |
| AF-04 / F09: GIL | **Implemented and measured.** Native offline work uses `py.detach` in `rust-core/src/audio/processor/python_api.rs`; `test_native_regressions.py` verifies a concurrent Python heartbeat. The measured local gaps are recorded above; this is not a hardware claim. |
| AF-05: allocation instrumentation | **Implemented.** `rust-core/src/lib.rs` uses thread-local nested allocation scopes, with concurrent and nested-scope regressions; steady-state DSP allocation tests remain in `rust-core/src/audio/processor/tests.rs`. |
| AF-06: configuration recovery | **Implemented.** Invalid and future-schema configuration is preserved, save is blocked when recovery is unavailable, and migration state is exposed. `test_config_v17.py` covers preservation, collision-safe backups, and blocked saves. |
| AF-07: supported tools | **Implemented as a support boundary.** `requirements/*.txt`, `rust-toolchain.toml`, `check_versions.py`, and the source recipe pin CPython 3.12.10, NumPy/SciPy, and Rust 1.94.0. No broader Python/Rust support is claimed. |
| AF-08: asset origins | **Partly implemented; DirectML pending.** Models and Silero are pinned, and the inherited `df.dll` was replaced by the verified DeepFilter source recipe, lock, patch record, and attestation. DirectML is correctly classified as Microsoft Software License Terms; a CPU-only replacement and qualification are still pending. |
| AF-09: hardware coverage | **Pending by decision.** The maintainer deferred microphone testing and no hardware runner is registered. Six hardware cases remain ignored; hosted software tests cannot close device lifecycle or route qualification. |
| AF-10 / F11: qualification origin | **Implemented as validators; evidence pending.** `release_provenance.py` and the qualification workflows bind reports to repository, revision, workflow, event, attempt, and typed status. No successful remote qualification run has been verified for this candidate. |
| AF-11 / F12: publication | **Workflow implemented; publication pending.** Promotion now handles drafts, digest-checked retries, exact asset sets, final-state checks, and publication last in `.github/workflows/release-promote.yml`. No release was published in this review. |
| AF-12: durable evidence | **Implemented in the release path; candidate evidence pending.** The package and promotion workflows produce privacy-safe provenance, source, MSI, and qualification sidecars for release assets. They have not yet been reconciled against a published candidate. |
| AF-13: prereleases | **Implemented.** `python/tools/release_version.py` and package/promotion workflows validate canonical final/RC tags, Cargo versions, artifact names, and MSI versions; release-version tests pass. Immutable tag rules remain the remote control. |
| AF-14: native freshness | **Implemented locally.** `build_exe.ps1` rebuilds the Rust extension with `maturin develop --release --locked` before PyInstaller. The fresh clean build and exact EXE smoke passed; remote candidate execution is pending. |
| AF-15: reduced build flag | **Implemented.** The unsupported reduced packaging path is absent; full asset verification is required by `build_exe.ps1` and the package smoke checks. |
| AF-16: workflow checks | **Implemented locally.** `python/tools/check_workflows.py` inspects parsed executable steps and failure propagation, and the current workflow check passes. It does not replace an actual remote run. |
| AF-17: native fallback cause | **Implemented.** Native headroom and DeepFilter paths distinguish unavailable, invalid, and runtime failures; Python fallback output is labeled advisory while Rust output is authoritative. Regression tests cover these labels and safety gates. |
| AF-18: ownership refactor | **Retained as targeted maintenance.** The changes isolate demonstrated ownership boundaries in configuration, analysis cancellation, UI lifecycle, and native DSP; the evidence does not justify a broad rewrite. |
| AF-19: CI time | **Implemented in configuration; remote timing pending.** RustSec uses a pinned, cached `cargo-audit` executable and advisory database in the CI workflows. The local audit passes; hosted timing must be observed after the next run. |
| AF-20: repository hygiene | **Implemented locally.** Contributor/security guidance, issue/PR templates, version checks, and `test_repository_hygiene.py` are present. The release branch still needs the normal reviewed commit and remote checks. |
| F02: preset collision | **Implemented.** Default preset creation refuses an occupied sanitized path, explicit overwrite is separate, and temporary-file plus exclusive-link publication is atomic in `config_parts/presets.py`; config tests cover it. |
| F03: stopped recovery | **Implemented locally.** Stopped diagnostics continue pending native recovery and reconcile controls; `test_stopped_diagnostics_continue_native_recovery_and_reconcile_controls` passes. |
| F04: malformed presets | **Implemented.** Parser, type, and recursion failures normalize at the preset boundary to `PresetValidationError`; malformed files are covered by configuration tests. |
| F05: partial migration | **Implemented.** Migration copies into a staging directory, records pending/staged state, publishes atomically, and retries after an injected copy failure; `test_config_v17.py` covers the failure path. |
| F06: route keys | **Implemented.** Structured JSON route keys are parsed before the legacy `||` form; delimiter-containing names round-trip in `test_config_v17.py`. |
| F07: refresh signals | **Implemented.** Refresh restores every affected control's prior signal-block state; `test_refresh_devices_restores_all_control_signal_states` passes. |
| F08: endpoint identity | **Implemented in software; lifecycle pending.** CPAL 0.17.3 and stable endpoint-ID lookup are used for opening and recovery, while persisted raw WASAPI IDs remain supported. Four endpoint-ID regressions pass; reconnect qualification still requires hardware. |
| F10: offline input | **Implemented.** Native offline APIs reject nonfinite audio/settings and unsupported rates before DSP construction; `test_native_regressions.py` covers the former 40 Hz panic and NaN cases. |
| F14: EQ transitions | **Implemented with native regression coverage.** The live bypass path participates in the EQ transition, and `rust-core/src/dsp/eq.rs` tests stale-state crossfading with bounded, allocation-free processing. Hardware listening evidence is outside this review. |
| Additional: MSI | **Implemented and tested locally.** Pinned WiX 7.0.0 built the fresh portable payload. `msi_smoke.py` passed administrative extraction and exact payload comparison, baseline installation, major upgrade, logged downgrade rejection, shortcut checks, configuration preservation, and uninstall cleanup. Remote exact-artifact qualification remains pending. |
| Additional: pywin32 | **Implemented.** pywin32 is explicitly pinned in the runtime lock and included in source/inventory packaging; it is no longer obtained only as a Semgrep side effect. |
| Additional: dataset paths | **Implemented.** Evaluation path resolution rejects drive, UNC, and traversal escapes before I/O; six path regressions pass. |
| Additional: DirectML license | **Pending release decision.** The exact Microsoft.AI.DirectML 1.15.4 package is recorded with its Microsoft terms and remains the sole source-manifest blocker. The CPU-only ONNX Runtime replacement and qualification have not been adopted. |

## Repository and security evidence

- All local refs contain 275 commits and 3,655 objects at the baseline. Four
  historical 7z blobs total 397,191,316 bytes; none is in the current tree. A
  history rewrite would disrupt published tags and is not part of this work.
- A narrow, value-redacted scan of 1,717 source/document blobs found no matches
  for private-key headers, GitHub tokens, or AWS access-key IDs. It excludes
  likely-secret files and is not a complete secret-scan certification.
- The initial runtime Python lock and the Rust lock returned no known
  vulnerabilities. The developer lock passed with the three existing explicit
  advisories for Semgrep's optional MCP server; those are not runtime ignores.
- The refreshed Semgrep SARIF has four warning-level URL-download results and
  twelve notes, with no error-level results. The configured scan ran 269 rules
  over 265 targets; its reported parse coverage was 98.9%. Download entry points use trusted
  HTTPS hosts; FFI/COM notes require source review rather than counting every
  `unsafe` block as a vulnerability. The dataset path issue above was found
  during that review.
- After the CPAL upgrade, `cargo audit --deny warnings` passed against the
  refreshed RustSec database for the 203-package application lock.
- GitHub's advisory feed separately reports eight alerts for OpenSSL 0.10.75
  still present in the application lock and three for the developer-only MCP
  pin. `cargo tree --target x86_64-pc-windows-msvc -i openssl` resolves no
  package; the all-target graph places it under `native-tls` / `ureq` in
  `ort-sys` build dependencies. It is not compiled into the Windows runtime,
  but the lockfile alerts remain open. The proposed CPU runtime integration
  can remove that unused download dependency chain. The clean RustSec result
  alone does not close GitHub's alerts.
- GitHub vulnerability alerts and automatic security updates were disabled
  despite the documentation. They were enabled and the resulting settings
  verified. Secret scanning and push protection were already enabled.
- `master` already required `python` and `rust` status checks, strict up-to-date
  checks, and prohibited force pushes/deletion. Administrative enforcement was
  enabled without replacing those settings. Active ruleset `22336761` now
  prohibits updating or deleting `v*` tags with no bypass actors. Administrators
  can still change repository policy; this is not an immutable external ledger.

## Measured native responsiveness

The rebuilt release extension processed ten seconds at 48 kHz in
282.05–301.68 ms across five runs, comparable to the prior audit's 299 ms.
Concurrent 5 ms Python heartbeat gaps were 13.09–40.47 ms, compared with the
prior 301 ms stall. Numerical kernels remain native; releasing the GIL restores
Python callback scheduling rather than claiming an unmeasured DSP speedup.
These measurements are workstation observations, not hardware audio-route
qualification. The native regression also includes call-boundary gaps so the
end of a blocking native call cannot evade the check.

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
real-time factor; the checked-in evaluator records the exact latest results.
Subsequent builds carry their own output identity. Recipe hashes normalize text line endings so
Windows checkouts and Git source archives validate consistently. The final
manifest was regenerated against this recipe; 344 hydrated archives verify
with `--allow-incomplete`, with the DirectML terms blocker retained.

## Packaged startup

A real `--smoke-test` found a packaged-only Qt import failure. PyInstaller had
collected ICU 78 from an unrelated Poppler installation on the build process's
PATH, shadowing the Windows ICU API used by Qt. Removing those two generated
files restored successful startup with the normal environment. The supported
builder now isolates PyInstaller's search path to Python and Windows locations;
package validation rejects app-local copies of Windows ICU. The fresh clean
build passed the same package and exact EXE smoke checks; the tagged remote
candidate must still be run through that gate. Windows provides
the ICU C API in its system libraries ([Microsoft documentation](https://learn.microsoft.com/en-us/windows/win32/intl/international-components-for-unicode--icu-)).

## Release maintenance

The standing dependency source in `release-assets.json` remains pinned until
the asset set changes. The promotion path now generates the sizes, hashes,
inventory, and qualification sidecars as release assets, so documentation can
refer to the same candidate instead of requiring a post-release fallback commit.
Build, qualification, and promotion remain separately verifiable gates; remote
CI, hardware qualification, source publication, and final artifact promotion
are still open in this candidate.

This register reflects the implemented candidate and its remaining release
gaps; it is closed only after those gates produce evidence for the tagged
release.
