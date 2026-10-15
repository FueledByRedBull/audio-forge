# Releasing AudioForge

A release is built once by the `Release package` workflow, then promoted
without rebuilding. The public release has exactly five files: the portable
archive, the per-user MSI, the corresponding-source archive, an evidence
archive, and one `SHA256SUMS.txt`.

## Automated workflow

1. **Prepare one release commit.** Update the version, `CHANGELOG.md`, the
   upcoming `release-notes/release-notes-v<version>.md`, and documentation.
   Run `python/tools/check_versions.py`. Published notes live on
   [GitHub Releases](https://github.com/FueledByRedBull/audio-forge/releases)
   and in their tagged source, so only the upcoming notes stay in
   `release-notes/`; `check_versions.py` fails if any other file is there.
   Don't write archive sizes, hashes, or file counts into
   docs; the generated checksum and metadata sidecars own them.
2. **Build a candidate.** Dispatch `Release package` on the branch with
   `release_tag` blank. It hydrates and verifies the runtime assets and
   corresponding source, builds the portable app and MSI, and validates the
   exact archive. The jobs split the work:
   - `static` (version and evaluation hygiene checks, Ruff, Pyright, workflow
     check, source smoke, Semgrep, dependency and RustSec audits,
     `cargo fmt`) needs no built file and runs beside the packaging job.
   - `package-windows` builds DeepFilter from the pinned source in every run,
     builds the app, and uploads the candidate.
   - When it finishes, the pytest shards (`.github/pytest-shards.json`),
     the Rust tests with Clippy, and the release-mode stress test run on the
     files it produced: the packaged native module, `df.dll`, the models, and
     the CPU ONNX Runtime pass between jobs as a same-run artifact, never the
     Actions cache. Each job verifies them before testing, and checks the
     module's SHA-256.
   - `validate-candidate` runs beside those jobs, extracts the archive, and
     checks that its native module has the same SHA-256 the tests ran on.

   Every job must pass: promotion refuses a run that is not successful. Fix
   failures on the branch without tagging or changing the version, and
   dispatch again. A new dispatch cancels a candidate still running on the same
   branch (`cancel-in-progress: true`), so let a run you want to keep finish
   first. Promotion also binds the run attempt, so re-running only failed jobs
   doesn't produce a promotable candidate. Set `asset_source_tag`
   only to override where model assets come from (see
   [Runtime assets](#runtime-assets)). Supply an existing tag in
   `release_tag` only when rebuilding that tag.
3. **Record the candidate:** workflow run ID, source commit, and archive
   SHA-256. The candidate artifact expires after three days; an expired
   candidate must be rebuilt and revalidated.
4. **Tag the candidate commit** with an annotated `v<version>` tag; promotion
   refuses a lightweight one. A tag push doesn't rebuild anything. Never move a
   released tag to fix documentation; reconcile documentation in a later commit.
5. **Review hardware evidence (optional).** Hardware qualification workflows
   can add measurements when a suitable runner exists; they don't block
   publication, unlike the required
   [manual release checks](#manual-release-checks). `self_test.py` and `health_check.py` are the hardware checks
   that `release-hardware-qualify.yml` runs through
   `evaluate_hardware_validation.py`. They need a real audio route, so no
   hosted job can run them. Keep each measurement bound to its candidate
   revision and digest, never attribute an earlier candidate's results to the
   final binary, and name untested configurations in the release notes.
6. **Promote.** Run `Promote release candidate` with `release_tag`,
   `candidate_run_id`, and `expected_archive_sha256`. It downloads the same
   candidate bytes, verifies sidecars, evidence, and corresponding source
   against that digest, assembles a draft, and publishes only when all five
   files are present and verified. Release notes come from the workflow
   revision; the binary stays bound to the tag. Reruns compare GitHub's
   SHA-256 digests in preflight, and the final check downloads and hashes
   every published file.

GitHub release immutability is enabled: publication locks the assets and tag
and creates GitHub's
[release attestation](https://docs.github.com/en/code-security/how-tos/secure-your-supply-chain/secure-your-dependencies/verify-release-integrity).
Verify every asset on the draft before publishing. A self-hosted runner isn't
required.

## Versioning

Final tags are `vMAJOR.MINOR.PATCH`; release candidates are
`vMAJOR.MINOR.PATCH-rc.N`. The source version may use PEP 440 (`1.12.0rc1`);
`python/tools/release_version.py` maps it to Cargo's `1.12.0-rc.1`, the tag,
and the artifact name. MSI has only three numeric fields, so each 100-value
patch block reserves its last value for the final release: `1.12.0rc1` is MSI
`1.12.1` and final `1.12.0` is MSI `1.12.99`. RC numbers run 1–98 and patch
components up to 654. The MSI number is an installer identity only; the app
and release metadata keep the public version. RC promotion creates or verifies
a GitHub prerelease; final promotion rejects one.

## Runtime assets

Runtime binaries and models are never committed. `release-assets.json` owns
their paths, sizes, hashes, origins, and licenses, and
`python/tools/fetch_release_assets.py` hydrates them:

- `df.dll` is built from the pinned DeepFilter recipe
  (`build-support/deepfilter/`, `build_deepfilter.ps1`). The recipe is not an
  attestation; keep the per-build `target/deepfilter/df.dll.provenance.json`
  with its DLL. Matching source and settings don't promise byte-identical
  output across machines. The release workflow builds it in every run; no
  cache supplies the DLL or its attestation.
- CPU-only ONNX Runtime and Silero v6.2.1 come from their recorded upstream
  identities.
- Both DeepFilter model archives come from the standing asset-source release:
  the workflow's `asset_source_tag` input, else the repository
  `AUDIOFORGE_ASSET_SOURCE_TAG` variable, else `fallback_release_tag` in
  `release-assets.json`. The fallback can supply raw assets or extract the
  exact models from an existing portable archive.

Every downloaded or extracted asset is verified against `release-assets.json`.
Leave `fallback_release_tag` unchanged while the dependency assets are
unchanged; advance it only for a verified new asset source.

## Local build

From a clean clone with the hashed development environment:

```powershell
.\.venv\Scripts\python.exe python/tools/fetch_release_assets.py
.\.venv\Scripts\python.exe -m maturin develop --release --locked
.\.venv\Scripts\python.exe python\tools\verify_release_assets.py
```

Stale files under `dist/` are never valid packaging inputs. For a release
candidate, hydrate the corresponding source for the exact commit first (see
[licenses/SOURCE_DISTRIBUTION.md](licenses/SOURCE_DISTRIBUTION.md)):

```powershell
$env:AUDIOFORGE_SOURCE_DIR = "build/source-distribution"
$env:AUDIOFORGE_SOURCE_REVISION = (git rev-parse HEAD).Trim()
.\.venv\Scripts\python.exe python\tools\source_distribution.py bundle `
  --manifest licenses/source-manifest.json `
  --output $env:AUDIOFORGE_SOURCE_DIR `
  --revision $env:AUDIOFORGE_SOURCE_REVISION `
  --include-runtime-assets
```

Then build the portable tree from `AudioForge.spec` and the MSI from it:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_exe.ps1
powershell -ExecutionPolicy Bypass -File .\build_msi.ps1
```

`build_exe.ps1` verifies runtime assets, rebuilds the native extension with
locked Cargo resolution, collects dependency notices, and reuses PyInstaller's
analysis cache (`-Clean` forces a cold PyInstaller build). It fails before
PyInstaller on a missing or mismatched asset or a failed native rebuild.
Builds without models aren't a supported edition. The MSI payload must match
the portable tree; `python/tools/msi_smoke.py` checks extraction and per-user
install and uninstall. Archive creation and provenance commands belong to the
workflow; its solid LZMA2 settings come from
`evaluation/archive-format-benchmark.json`.

## Release checks

Run the [development checks](CONTRIBUTING.md), including the release-mode
stress test, then:

```powershell
.\.venv\Scripts\python.exe -m pip_audit --require-hashes -r requirements/runtime.txt --disable-pip
.\.venv\Scripts\python.exe -m pip_audit --require-hashes -r requirements/dev.txt --disable-pip
.\.venv\Scripts\python.exe python\tools\run_semgrep.py --sarif semgrep-results.sarif
.\.venv\Scripts\python.exe python\tools\check_versions.py
.\.venv\Scripts\python.exe python\tools\check_workflows.py
.\.venv\Scripts\python.exe python\tools\package_smoke.py --source-only
cargo audit
.\.venv\Scripts\python.exe python\tools\package_smoke.py
```

- **Semgrep:** `run_semgrep.py` fails on any ERROR or WARNING that isn't
  suppressed in source with `nosemgrep` and a reason. Review the notes by
  hand in the SARIF. The rule packs in `semgrep-rulesets.txt` aren't pinned,
  so a new upstream rule can fail CI without a code change: fix the code or
  suppress the finding with its reason.
- **RustSec:** `cargo audit` must be clean for the application lockfile; never
  add an ignore just to release. The separate
  `build-support/deepfilter/Cargo.lock` graph ignores only three reviewed
  unmaintained-crate notices (`RUSTSEC-2024-0436`, `RUSTSEC-2024-0370`,
  `RUSTSEC-2026-0319`); vulnerabilities there still block. CI runs the same
  audit, so a new notice shows up on a pull request, not at release time.
- **Python:** install `requirements/dev.txt` with `--require-hashes`; never
  release from an environment resolved from open `pyproject.toml` ranges.
  Neither the runtime nor the development audit has ignores; Semgrep 1.179.0
  allows a patched PyJWT, so the former PyJWT exception is gone.
- **Dependabot:** GitHub Actions version updates stay on, weekly and grouped
  into one pull request, to keep the pinned actions current. Python and Cargo
  version updates are off (`open-pull-requests-limit: 0`); security updates stay
  on. Scope each dependency refresh on its own and pass the build and package
  gates.

## Manual release checks

These checks run the packaged app on a real microphone, which no automated gate
does. They are required before tagging (step 4); a failure means fixing the
cause and building a new candidate. The hardware-qualification workflows in
step 5 stay optional and don't replace them. Run them on the candidate from
step 2 (the extracted portable archive, plus the MSI for item 13), not on source
or an earlier candidate. The criteria are fixed before the run. Keep the record
(candidate run ID and archive digest, date, machine, route, readings, what
wasn't covered) with the other unpublished evidence, as in step 5.

### 60-minute soak

The soak shows that a started AudioForge keeps memory, threads and handles flat
and neither drops audio nor restarts or recovers a stream for an hour. Its
criteria come from a one-hour pilot on 2026-10-15.

1. From PowerShell in the extracted folder, with sleep and screen lock off,
   start the app with factory settings in a scratch `APPDATA` that skips the
   first-run Guided Setup dialog:

   ```powershell
   $dir = "$env:USERPROFILE\soak"          # outside the repository
   $env:APPDATA = "$dir\appdata"
   New-Item -ItemType Directory -Force "$env:APPDATA\AudioForge" | Out-Null
   '{"first_run_setup_state": "completed"}' |
     Set-Content "$env:APPDATA\AudioForge\config.json"
   Start-Process .\AudioForge.exe
   ```

2. Choose the real microphone as input and an output endpoint nobody listens
   to, and turn the top-bar **Mute** toggle on. Change nothing else, and don't
   press **Reset drop counter**. Press **Start Processing** (minute 0), then
   run this in a second PowerShell window:

   ```powershell
   $dir = "$env:USERPROFILE\soak"
   $p = Get-Process AudioForge | Sort-Object StartTime | Select-Object -Last 1
   "time,private_bytes,cpu_seconds,threads,handles" | Set-Content "$dir\soak.csv"
   1..60 | ForEach-Object {
     Start-Sleep 60; $p.Refresh()
     "{0:s},{1},{2},{3},{4}" -f (Get-Date), $p.PrivateMemorySize64,
       $p.TotalProcessorTime.TotalSeconds, $p.Threads.Count, $p.HandleCount |
       Add-Content "$dir\soak.csv"
   }
   ```

   Row 5 is the 5-minute baseline and row 60 the end.
3. At minute 5 and minute 60, note the Health indicator and export diagnostics
   (**Health > Technical details > Export diagnostics**) into the same folder.

It passes only if all of these hold:

- Private memory in row 60 is within 5 % of row 5.
- In the minute-60 export's `runtime` section, `output_underrun_total`,
  `input_dropped_samples`, `input_backlog_dropped_samples`,
  `jitter_dropped_samples`, `output_short_write_dropped_samples`,
  `stream_restart_count`, `output_recovery_count` and
  `output_recovery_event_count` are 0.
- No engine fault: `input_callback_error_count`, `output_callback_error_count`,
  `rt_buffer_overflow_count`, `lock_contention_count` and
  `suppressor_non_finite_count` are 0, `noise_backend_failed` is false, and at
  both readings the Health indicator names no fault other than gate chatter
  (`gate_chatter_event_count` may be above 0).
- Median CPU is under 15 seconds a minute (a quarter of one core), from the
  differences between consecutive `cpu_seconds` values.

Threads and handles are recorded, not gated; a steady climb is a finding to
explain before tagging. Not covered: other machines and microphone routes,
non-factory settings, an audible output (a muted Output row doesn't count
toward the verdict), device changes, sleep or resume, other audio applications.
The fault criterion can fail: it would have failed 1.16.0, whose Drops row
called gate chatter an engine fault (see CHANGELOG).

### Hands-on checklist

Run it on the same extracted candidate, with the real microphone and a silent
output endpoint, after `--smoke-test` passes (hidden, 60 second limit, exit code
0, as [CONTRIBUTING](CONTRIBUTING.md) requires):

```powershell
$p = Start-Process .\AudioForge.exe -ArgumentList '--smoke-test' -WindowStyle Hidden -PassThru
if ($p.WaitForExit(60000)) { $p.ExitCode } else { $p.Kill(); 'timed out' }
```

Use a fresh `APPDATA` with no config so item 1 happens. An item passes only if
every part of its criterion holds; record "not run" for any skipped.

1. **First launch.** Start with no saved settings. Pass: the window draws with
   no error screen and **Guided Setup** opens over it. Finish or close it; the
   window then responds and no stream is running.
2. **Pages.** Open **Mic**, **Health** and **Settings** from the rail. Pass:
   each shows all its controls, no dialog opens, processing is stopped, and the
   About version (**Settings > Help**) is the candidate's.
3. **Start and Stop.** Press **Start Processing**, speak, press **Stop
   Processing**, then Start again. Pass: the input meter follows your voice,
   the status dot shows running, and the Health page's Drops row stays clear for
   30 seconds after Start. After Stop the window stays open, the meters fall to
   idle and the dot shows stopped.
4. **Mute.** While running, turn the top-bar **Mute** toggle on, then off. Pass:
   the OUT meter reads MUTE, the Output row says muted, the status dot shows
   muted, and the tray tooltip says "Running / Muted"; unmuting returns all
   three. The tray's **Mute Output** item follows the same state.
5. **Edit, Undo and Redo.** Drag one EQ band, then change the gate threshold.
   Press **Undo** twice and **Redo** twice, then repeat with **Ctrl+Z** and
   **Ctrl+Shift+Z**. Pass: each press moves exactly one edit and both values
   return exactly. After item 9, the first **Ctrl+Z** undoes your last edit, not
   one you never made.
6. **Preset load.** While running, load a built-in preset from the **Presets**
   menu, then press **Undo**. Pass: the settings change without stopping audio
   or flagging a dropout, the preset's name shows, and Undo restores the
   previous settings.
7. **Health indicator and status bar.** While running, mute the microphone in
   Windows sound settings for ten seconds. Pass: the indicator names the quiet
   input and Health gives advice for it; unmuting clears both. After **Stop
   Processing**, the advice and tooltips are gone and the dot shows stopped.
8. **Tray and quit.** In **Settings > Tray and background**, turn on
   close-to-tray, start processing and close the window. Pass: audio keeps
   running, the tray tooltip reads "AudioForge: Running", **Show AudioForge**
   restores the same window, and a second launch shows that window instead of
   starting another stream. **Ctrl+Q** and the tray **Quit AudioForge** each
   leave no `AudioForge.exe` in Task Manager within five seconds. With
   close-to-tray off, closing the window stops audio and quits; with unsaved
   changes it asks Save, Discard or Cancel, and Cancel keeps audio running.
9. **Device change while running.** Pick another input in the top bar, disable
   that input in Windows sound settings, then enable it and use **Refresh**.
   Pass: the meter follows the new input; when a device disappears the app
   stays responsive, says so, and picks no other device; after Refresh the
   device is listed and Start works.
10. **Guided Setup.** Open **Run Guided Setup** (**Settings > Options**), go to
    the first step that lists devices, and close it; open it again and close it
    during a level check. Pass: it closes cleanly both times, no stream is left
    running, and the settings are as before.
11. **Export diagnostics.** In **Health > Technical details**, press **Export
    diagnostics**. Pass: a file is written, shows the candidate's version and
    the soak's counters, and contains no Windows user name, device identifier
    or audio.
12. **Minimum size.** Drag the window to its smallest size (900 by 640), open
    each page, load a preset with a long name, and toggle **Mute**. Pass: no
    control overlaps another, no text is cut off without a tooltip, and the
    Health indicator and status message are both readable in the status bar.
13. **Login startup.** On the portable copy and on the MSI install (with the
    previous release installed first, for the upgrade step). Pass for each:
    - **Configure login shortcut for this copy** creates one **AudioForge
      Login** shortcut in the Startup folder, switching it off removes it, and
      nothing is created without the switch.
    - Disable the shortcut in Windows Settings, then upgrade: the app reports
      it as configured, and it is still disabled.
    - Sign in with the saved input unplugged and plug it in within 60 seconds:
      the app waits in the tray, starts with output mute as it was, and uses no
      other device. If it stays unplugged, audio stays stopped with the reason
      in the tray and the log. With no tray, the process logs the failure and
      exits with no dialog.
    - An ordinary uninstall removes the shortcut; an upgrade doesn't. A shortcut
      reassigned to a portable copy survives uninstalling the MSI.

The checklist can fail: item 3's Drops criterion fails 1.16.0, which counted
three or four dropouts on every Start, and item 5's last sentence fails a
build that records a phantom edit after a device change.

### What is automated

Source tests run offscreen with no real device, so they don't show that the
drawn window, the tray or Windows' Startup behavior work. They cover only the
logic behind items 4-8, 11 and 13 (`test_quick_undo.py`, `test_quick_shell.py`,
`test_runtime_health_ui.py`, `test_login_startup*.py`, `test_single_instance.py`,
`test_desktop_controls.py`, `test_diagnostics_export.py`).

- `--smoke-test` (packaged exe, no audio) checks the window becomes visible with
  processing stopped, the Qt Quick view loaded, and the shortcut round trip and
  Auto-EQ solvers run: part of item 2 only. `package_smoke.py` never launches
  the app.
- `health_check.py --duration 1800` is the engine-only counterpart of the soak:
  the same kinds of counters, `--bundle-root` to load an extracted bundle, a
  real route. It reads no memory, thread or handle figure and doesn't run the
  app. `self_test.py` (also `--bundle-root`) checks a probe through a route.
- Everything else is manual.

## Packaging invariants

- `AudioForge.spec` is the package definition. `package_smoke.py` checks the
  exact bundled DLLs, models, native extension, and license notices, rejects
  duplicate top-level native-extension payloads, and rejects a stale
  bundle-version manifest.
- `evaluation/release-bundle-path-baseline.json` gates reviewed bundle path
  additions and removals. Its binary hashes are provenance, not
  reproducible-build expectations.
- `prune_bundle.py` keeps every dependency `.dist-info` directory, removes a
  duplicate native-extension payload only when the canonical
  `_internal/mic_eq/mic_eq_core*.pyd` exists, and removes app-local
  `ucrtbase.dll` and `api-ms-win-*.dll`: Windows 10 and 11 always use the
  [system UCRT](https://learn.microsoft.com/en-us/cpp/windows/universal-crt-deployment).
  Package smoke fails if they return.
- Keep the NumPy BLAS DLL (`libscipy_openblas64_*`, named for the upstream
  build), `opengl32sw.dll`, `Qt6OpenGL.dll` and `QtOpenGL.pyd` (Qt Quick links
  the first; libshiboken imports the second, and the frozen app does not start
  without it), all models, the CPU-only ONNX Runtime, and `df.dll`. SciPy is a
  development-only dependency: nothing under `python/mic_eq` imports it, and
  `_internal/scipy` and `_internal/scipy.libs` must not appear in the bundle.
  The prune step removes unused Qt SVG payloads and plugins. Package smoke
  fails if a SciPy import or any removed payload returns. The release profile
  strips native symbols without changing optimization.
- Bundled DeepFilter and Silero assets take precedence over the working
  directory and user paths. External DeepFilter paths require
  `AUDIOFORGE_ALLOW_EXTERNAL_DF=1`. Runtime-asset and bundle paths stay
  repository-relative without `..`.
- Release validation covers fixed-buffer overflow and drop diagnostics and
  model-discovery smoke checks.
- `licenses/source-manifest.json` is generated by
  `source_distribution.py manifest`; regenerate it when its inputs change and
  review the source closure instead of hand-editing entries.
- Original AudioForge source stays MIT; new candidate builds use PySide6 and
  dynamically replaceable LGPLv3 Qt libraries. Earlier PyQt6 packages retain
  their GPLv3 terms. Complete the corresponding-source and library-replacement
  arrangements and component review
  in [licenses/THIRD_PARTY_NOTICES.md](licenses/THIRD_PARTY_NOTICES.md) before
  publishing; a generated license inventory alone doesn't satisfy them.

## Login startup lifecycle

Login startup is off by default. The packaged app's **Settings > Tray and
background > Configure login shortcut for this copy** explicitly creates one per-user
`AudioForge Login.lnk` in the Windows Startup known folder. Its target is the
current executable, with a separate `--login-startup` argument and ownership
description. Configuration is represented by that shortcut, not another setting.
Source launches do not offer registration.

The page reports **configured**, not **enabled**: Windows Settings or Task Manager
can disable a Startup-folder app. AudioForge neither reads undocumented
`StartupApproved` values nor rewrites a configured shortcut on launch, repair, or
upgrade. Windows documents these controls in
[Configure startup applications](https://support.microsoft.com/en-gb/windows/experience/startup-boot/configure-startup-applications-in-windows).
Remove the shortcut explicitly before moving a portable copy, then configure it
from the new location. An unreadable, unrecognized, or other-copy shortcut is
preserved; remove that stale entry in the Startup folder before reassignment.

A login launch does not focus an existing session. A new session opens in the
tray and waits at most 60 seconds for that tray and the exact persisted input and
output endpoint IDs. It starts only after settings and any bound route preset
restore successfully, preserving output mute. Stop, route edits, or removing the
registration cancel pending startup. Missing devices never select replacements
automatically. Failure leaves audio stopped with tray/status/log evidence; if no
tray becomes available, the process logs the failure and exits without a dialog.

The MSI invokes the installed executable's `--remove-login-startup` helper before
`RemoveFiles` on ordinary uninstall, excluding major upgrade removal. It removes
only a link whose target, arguments, and description all match that executable.
The helper constructs no Qt application or audio processor. Unrecognized or
unreadable links remain; helper failure is recorded by MSI and does not prevent
uninstall. A failed uninstall can therefore leave the app installed with its
login shortcut already removed. Re-enabling remains an explicit user action.

Source tests cover shortcut ownership with fake COM and temporary folders, quiet
duplicate acquisition, endpoint retry/cancellation, and mute/preset guards. Before
shipping, verify the exact portable/MSI payload on Windows: explicit opt-in/out,
Windows-disabled registration surviving upgrade, delayed endpoint enumeration,
tray recovery/absence, ordinary uninstall cleanup, and preservation of a shortcut
reassigned to a portable copy. Source tests do not establish those installer or
Windows-shell lifecycle results; item 12 of the
[hands-on checklist](#hands-on-checklist) lists the checks.

## Local build cleanup

Check exact output paths before deleting. Completed `dist/` packages and Rust
build outputs are disposable only after required candidates and unpublished
evidence are kept elsewhere; prefer each tool's own clean command. Don't
delete `target/` or `build/` wholesale: they can hold the project Python, the
hydrated runtime DLLs, source receipts, and unpublished measurements. Keep
virtual environments, `models/`, and corpora.
