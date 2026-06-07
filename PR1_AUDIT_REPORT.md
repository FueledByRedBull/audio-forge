# PR #1 Audit Report - strict RT migration

Target: https://github.com/FueledByRedBull/audio-forge/pull/1

Audited checkout:

- Repository: `C:/Users/ancha/Documents/Projects/audio-forge/main_proj`
- Base: `09baeac` (`origin/master`)
- PR head before local fixes: `5abf78a` (`origin/codex/strict-rt-migration`)
- Working tree includes local audit/workflow fixes not yet committed or pushed
- `tauri_rewrite`: intentionally not inspected or touched

## Verdict

Ready from local validation, after committing and pushing the current local fixes.

The remote PR head alone is not the final merge candidate because the fixes below are still local working-tree changes. Once those files are committed and pushed, I do not see a remaining merge blocker. The new release workflow also has one operational precondition: the configured asset-source GitHub Release must contain either the raw runtime assets or an existing verified `AudioForge-*-win64-ultra.7z` archive the workflow can extract them from.

## Fixed Findings

### P1 - RT control snapshot fallback could apply a torn snapshot

`stable_control_snapshot` now returns `None` after bounded retries instead of taking an unguarded final read. RT dirty-flag consumers re-arm their dirty flags when a snapshot is unstable, so control updates are deferred rather than applied partially.

Files:

- `rust-core/src/audio/processor/control.rs`
- `rust-core/src/audio/processor.rs`
- `rust-core/src/audio/processor/tests.rs`

### P2 - DeepFilter runtime failures allocated error strings on the RT path and lost detail

DeepFilter runtime processing errors are now represented by a small copy enum with static messages. The RT path marks the backend failed without formatting or allocating and the backend error remains observable through diagnostics.

Files:

- `rust-core/src/dsp/deepfilter_ffi.rs`
- `rust-core/src/dsp/noise_suppressor.rs`
- `rust-core/src/audio/processor/tests.rs`

### P2 - New RT diagnostics were exposed through PyO3 but not surfaced in UI health chips

The UI dropped/health chip now includes RT buffer overflow, input callback error, output callback error, and active RT error name diagnostics. Tests cover the labels and warning state.

Files:

- `python/mic_eq/ui/main_window.py`
- `python/tests/test_ui_sample_rate_and_diagnostics.py`

### Release workflow packaging

Added a Windows release packaging workflow that downloads raw runtime assets from a GitHub Release, or extracts them from an existing `AudioForge-*-win64-ultra.7z` fallback archive, verifies them against `release-assets.json`, builds the Rust extension, builds the PyInstaller app, runs package smoke, creates the `.7z` archive plus `.sha256`, uploads both as workflow artifacts, and uploads both to the GitHub Release on tag pushes or when manual dispatch enables release upload.

Files:

- `.github/workflows/release-package.yml`
- `RELEASING.md`
- `python/tools/package_smoke.py`

## Security And Risk Review

Threat model source: repository `AGENTS.md`.

Primary assets and invariants reviewed:

- Realtime CPAL input/output callbacks and DSP loop must not block, allocate in steady state, log, format strings, or grow buffers.
- Native DeepFilter/VAD model and DLL discovery must prefer trusted bundled assets unless external overrides are explicit.
- Packaged release assets must be repository-relative and complete.
- PyO3 diagnostics must expose native runtime failures accurately enough for UI and support workflows.
- Audio output must remain finite and bounded.

Diff areas reviewed:

- Rust realtime helpers, fixed buffers/rings, dirty-flag control state, RT diagnostics, input/output callbacks, DSP chain, suppressor model handoff.
- DeepFilter/RNNoise/VAD integration paths directly touched by the PR.
- Python bootstrap, launcher runtime environment behavior, release asset verifier, package smoke checks.
- Python UI diagnostics for newly exposed RT counters.
- GitHub Actions release workflow asset acquisition, package build, artifact upload, and release upload flow.

Result: no remaining security finding or RT-safety merge blocker found after the local fixes.

Additional static-analysis remediation:

- Installed `actionlint` `1.7.12` under the user-local Codex tools directory and ran it against both workflows.
- Installed Semgrep `1.165.0` in the ignored project venv and ran OSS Semgrep with `--metrics=off`.
- Semgrep initially found a high-confidence GitHub Actions command-injection issue in `.github/workflows/release-package.yml`; the workflow now passes GitHub/input context through `env` before PowerShell reads it.
- A follow-up Semgrep registry scan after that fix passed with `0 findings`.
- After adding archive fallback logic, the Semgrep registry became unreachable due DNS failures; an offline local Semgrep rule for direct `github.*` interpolation inside `run` blocks passed with `0 findings`, and `actionlint` remained clean.

## Validation

Passed locally:

- `git diff --check origin/master...HEAD`
- `git diff --check`
- `cargo fmt --check`
- `cargo test -p mic_eq_core`
- `cargo clippy -p mic_eq_core --all-targets -- -D warnings`
- `cargo test -p mic_eq_core --release`
- `.\\.venv\\Scripts\\python.exe -m ruff check python/mic_eq python/tests python/tools`
- `.\\.venv\\Scripts\\python.exe -m pyright`
- `.\\.venv\\Scripts\\python.exe -m maturin develop --release`
- `.\\.venv\\Scripts\\python.exe -m pytest python/tests -q --basetemp .\\.codex_tmp\\pytest\\basetemp -p no:cacheprovider`
- `.\\.venv\\Scripts\\python.exe python\\tools\\check_versions.py`
- `.\\.venv\\Scripts\\python.exe python\\tools\\verify_release_assets.py`
- `powershell -ExecutionPolicy Bypass -File .\\build_exe.ps1`
- `.\\.venv\\Scripts\\python.exe python\\tools\\package_smoke.py --source-only`
- `.\\.venv\\Scripts\\python.exe python\\tools\\package_smoke.py`
- `.\\.venv\\Scripts\\python.exe python\\tools\\self_test.py`
- `.\\.venv\\Scripts\\python.exe -m ruff check python/tools/package_smoke.py python/tests/test_package_tools.py`
- `.\\.venv\\Scripts\\python.exe -m pytest python/tests/test_package_tools.py -q --basetemp .\\.codex_tmp\\pytest\\basetemp -p no:cacheprovider`
- `%LOCALAPPDATA%\\CodexTools\\actionlint\\actionlint.exe .github/workflows/ci.yml .github/workflows/release-package.yml`
- `.\\.venv\\Scripts\\semgrep.exe scan --metrics=off --disable-version-check --config p/ci --config p/security-audit --config p/secrets --config p/python --config p/rust --config p/github-actions --severity WARNING --severity ERROR ...`
- `.\\.venv\\Scripts\\semgrep.exe scan --metrics=off --disable-version-check --config .\\.codex_tmp\\semgrep\\local-github-actions.yml .github/workflows`

Observed validation notes:

- Full Python tests passed: `87 passed`.
- Rust debug tests passed: `164` unit tests, `2` stress tests, `1` doctest.
- Rust release tests passed: `164` unit tests, `2` stress tests, `1` doctest.
- Pyright passed with `0 errors, 0 warnings, 0 informations`.
- Release assets verified.
- PyInstaller build produced `dist\\AudioForge\\AudioForge.exe`.
- Package smoke check passed for both source and built bundle.
- Workflow source expectations are covered by `package_smoke.py --source-only` and `test_package_tools.py`.
- `actionlint` passed on `.github/workflows/ci.yml` and `.github/workflows/release-package.yml`.
- Semgrep OSS registry scan passed with `0 findings` after fixing the workflow shell-injection finding; the later offline local Actions rule also passed with `0 findings`.
- Local archive-fallback validation extracted `df.dll`, `DirectML.dll`, both DeepFilter model archives, and `silero_vad.onnx` from `AudioForge-v1.8.0-win64-ultra.7z`; all sizes and SHA256 hashes matched `release-assets.json`.
- Self-test passed using `Microphone (Realtek USB Audio) -> Speakers (Realtek USB Audio)`, backend `rnnoise`, RT `124.15ms`, confidence `0.457`, dropped input `0`, underruns `1`, restarts `0`, non-finite `0`.

Not run:

- Semgrep Pro. The CLI is installed, but Pro requires `semgrep login` or `SEMGREP_APP_TOKEN`.
- The GitHub release workflow itself. It requires the workflow to be committed on GitHub and uses GitHub release upload APIs; the local equivalent packaging path and archive fallback checks passed.
- Subagent review. Two requested read-only explorer agents failed immediately due account usage limits, so this report is based on local single-agent audit plus local tooling.

## Hygiene

- `git diff --check` is clean apart from Git CRLF normalization warnings in working-copy notices.
- Changed-file marker scan found only intentional existing concepts: DeepFilter/RNNoise passthrough fallback, legacy latency-profile migration, and a Cargo feature compatibility note.
- No temporary validation folder remains.
