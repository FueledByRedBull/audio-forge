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
   - When it finishes, the three pytest shards (`.github/pytest-shards.json`),
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
   publication. `self_test.py` and `health_check.py` are the hardware checks
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

The automated gates never run the packaged app against real audio hardware.
These checks do, on the candidate from step 2 of the workflow: the extracted
portable archive, and for the login-startup items the MSI as well. A result on
source, on an older build, or on an earlier candidate doesn't count. Run them
before step 4 (tagging); a failure means a new candidate. The criteria below
are fixed before the run, so don't reinterpret one afterwards. Keep the record
(candidate run ID and archive digest, date, machine, route, each reading, and
what wasn't covered) with the other unpublished evidence, as in step 5.

### 60-minute soak

The soak shows that a started AudioForge keeps its memory, threads and handles
flat, and doesn't drop audio, restart a stream or recover an output, for an
hour on a real microphone.

1. Extract the candidate archive into a clean folder. For factory settings
   without touching your own, start from a shell where `APPDATA` points at an
   empty folder. Turn off sleep and screen lock for the hour.
2. Launch `AudioForge.exe`. Choose the real microphone as input. Choose an
   output endpoint nobody listens to and turn **Mute Output** on. Change
   nothing else. The soak needs no test signal; leave the microphone in the
   room as it normally is.
3. Press **Start Processing** and note the time. This is minute 0.
4. From a second PowerShell window, write one line a minute for 60 minutes
   to a file outside the repository:

   ```powershell
   $p = Get-Process AudioForge
   1..60 | ForEach-Object {
     Start-Sleep 60; $p.Refresh()
     "{0:s},{1},{2},{3},{4}" -f (Get-Date), $p.PrivateMemorySize64,
       $p.TotalProcessorTime.TotalSeconds, $p.Threads.Count, $p.HandleCount |
       Add-Content soak.csv
   }
   ```

5. Each minute, also write down from the **Health** page the dropouts, samples
   dropped, restarts and output recoveries (**Technical details**; hover
   **Drops** and **Recovery** for the raw counters), and the Health verdict in
   the status bar. The counters are cumulative since Start, so the minute
   readings say when a count appeared and the 60-minute value decides.
6. At minute 60, before stopping, export diagnostics from **Technical
   details** and keep the file with the record.

The soak passes only if all of these hold:

- Private memory at 60 minutes is within 5 % of its value at 5 minutes.
- Dropouts, samples dropped, restarts and output recoveries are all zero.
- Health reports no engine fault other than gate chatter. The factory gate can
  chatter on room noise; that is reported as the gate, not as a fault.
- Median CPU time is under 15 seconds a minute (a quarter of one core),
  taken from the differences between consecutive readings.

Threads and handles are recorded, not gated; a steady climb across the hour is
a finding to explain before tagging, so read the series, not the endpoints.
For scale, the one-hour pilot on
2026-10-15 read private memory 754 MB at start, 800 MB after 5 minutes and
800-817 MB after that; median CPU 13.3 seconds a minute; 51-53 threads;
2116-2140 handles; and no dropouts, restarts or recoveries.

What the soak doesn't cover: one machine, one microphone route, factory
settings only, a muted output (the muted Output row doesn't count toward the
verdict, and nothing is heard on the output side), and no device changes,
sleep or resume, or other audio applications. Name those limits in the record
and the release notes.

### Hands-on checklist

Run it on the same extracted candidate, with a fresh `APPDATA`, the real
microphone and a silent output endpoint, after `AudioForge.exe --smoke-test`
(hidden, 60 second limit, exit code 0) has passed. An item passes only if
every part of its criterion holds. Write the result of each item in the
record, including "not run".

1. **Pages.** Open **Mic**, **Health** and **Settings** from the rail. Pass:
   the window draws with no error screen or blank area, every page shows all
   its controls, nothing opens a dialog, processing is stopped on open, and
   the About version is the candidate's.
2. **Start and Stop.** Press **Start Processing**, speak, then press **Stop
   Processing**, then Start again. Pass: the button toggles, the input meter
   follows your voice, the status dot shows running, and the dropout count
   stays at zero for 30 seconds after Start. After Stop the window stays open,
   the meters fall to idle, the dot shows stopped, and the second Start works.
3. **Mute.** While running, turn **Mute Output** on, then off. Pass: the OUT
   meter reads MUTE, the Output row says it is muted, the status dot shows
   muted, and the tray tooltip says "Running / Muted"; unmuting returns all
   four.
4. **Edit, Undo and Redo.** Drag one EQ band, then change the gate threshold.
   Use the **Undo** button twice, then **Redo** twice, then repeat with
   **Ctrl+Z** and **Ctrl+Shift+Z**. Pass: each press moves exactly one edit,
   both values return exactly, and Redo disables once nothing is left to
   redo. After item 8, the first **Ctrl+Z** undoes your last edit, not an
   edit you never made.
5. **Preset load.** While running, load a built-in preset from **Presets**,
   then press Undo. Pass: the settings change without stopping audio or
   adding a dropout, the loaded preset's name shows, and Undo returns the
   previous settings.
6. **Health indicator and status bar.** While running, read the Health
   indicator, then mute the microphone in Windows sound settings for ten
   seconds. Pass: with the microphone muted the indicator names the quiet
   input and Health gives advice for it; unmuting clears both. After **Stop Processing**, the advice and tooltips
   are gone, the dot shows stopped, and the status message doesn't keep
   describing a running stream.
7. **Tray and quit.** In **Settings > Tray and background**, turn on
   close-to-tray, start processing and close the window. Pass: audio keeps
   running, the tray icon shows "AudioForge: Running", **Show AudioForge**
   brings back the same window with its state, and the tray **Mute** toggles
   the output. Starting a second copy shows the running window and doesn't
   start another stream. **Ctrl+Q** and the tray **Quit AudioForge** each
   leave no `AudioForge.exe` in Task Manager within five seconds and release
   the audio devices. With close-to-tray off, closing the window stops audio
   and quits; with unsaved changes, it asks Save, Discard or Cancel, and
   Cancel keeps audio running.
8. **Device change while running.** With processing started, pick another
   input in the top bar, then disable that input in Windows sound settings,
   then enable it again and use **Refresh**. Pass: the meter follows the new
   input; when the device disappears the app stays responsive, says so, and
   doesn't pick another device by itself; after Refresh the device is listed
   and Start works.
9. **Guided Setup.** Open **Run Guided Setup** from **Settings**, go to the
   first step that lists devices, and close it. Open it again and close it
   during a level check. Pass: it opens and closes cleanly both times, no
   stream is left running, and the settings are as they were before.
10. **Export diagnostics.** In **Health > Technical details**, press **Export
    diagnostics**. Pass: a file is written with no error; it shows the
    candidate's version and the counters used in the soak; and it contains
    no Windows user name, device identifier or audio. Read it before it
    leaves the machine.
11. **Minimum size.** Drag the window to its smallest size (900 by 640). Open
    each page, load a preset with a long name, and turn mute on and off.
    Pass: the window can't be made smaller; no control overlaps another; no
    text is cut off without a tooltip; and the Health indicator and the
    status message are both readable in the status bar.
12. **Login startup.** Run on the portable copy and on the MSI install, with
    the previous release installed first for the upgrade step. Pass for each:
    - Opting in creates one **AudioForge Login** shortcut in the Startup
      folder and opting out removes it; nothing is created without the
      switch.
    - Disable the shortcut in Windows Settings, then upgrade: the app reports
      it as configured, and the shortcut is still disabled.
    - Sign in with the saved input unplugged, then plug it in within 60
      seconds: the app waits in the tray, starts with output mute as it was,
      and uses no other device. If it stays unplugged, audio stays stopped
      with the reason in the tray and the log.
    - With no tray available, the process logs the failure and exits with no
      dialog.
    - An ordinary uninstall removes the shortcut; an upgrade doesn't. A
      shortcut reassigned to a portable copy is still there after uninstalling
      the MSI.

### What is automated

Source tests and `--smoke-test` don't replace the items above. They run on
source or on the bundle without audio, and say so in their record.

- `--smoke-test` (the packaged exe, no audio) checks only that the window
  becomes visible, processing is stopped, the Qt Quick view loaded, and the
  Windows shortcut round trip and Auto-EQ solvers run. It covers the "open"
  part of item 1 and nothing else. `package_smoke.py` checks the bundle's
  contents and never launches it; `msi_smoke.py` checks extraction and
  per-user install and uninstall, not the login shortcut's lifecycle.
- Source tests (`test_quick_undo.py`, `test_quick_shell.py`,
  `test_runtime_health_ui.py`, `test_login_startup.py`,
  `test_login_startup_ui.py`, `test_single_instance.py`,
  `test_desktop_controls.py`, `test_diagnostics_export.py`) cover the logic
  behind items 3-7, 10 and 12: history, mute state, Health wording, tray and
  close-to-tray state, shortcut ownership, retry and cancellation, single
  instance, and the export. They run offscreen on source
  with no real device, so they don't show that the drawn window, the tray or
  Windows' Startup behavior work.
- `health_check.py --duration 1800` is the engine-only counterpart of the
  soak: it fails on the same kind of counters (dropped samples, restarts,
  recoveries, callback errors), takes `--bundle-root` to load the native
  module and models from an extracted bundle, and needs a real route. It
  reads no memory, thread or handle figure and doesn't run the app, so it
  doesn't replace the soak. `self_test.py` plays a probe through a route and
  checks the correlation, also with `--bundle-root`.
- Manual: the soak, items 1-12 apart from the points above, and the login
  startup lifecycle on Windows. The soak's per-minute readings could be
  scripted (the snippet reads the process; the counters are in the diagnostics
  export), but no script does it yet.

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
