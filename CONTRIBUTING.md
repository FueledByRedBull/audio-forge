# Contributing to AudioForge

Use Windows x64, CPython 3.13.15 and the Rust compiler in `rust-toolchain.toml`.
The supported source-build contract uses the checked-in dependency locks; newer
Python or Rust releases are not implicitly qualified. See the [README](README.md)
for setup and [RELEASING](RELEASING.md) for packaging and publication.

Use `dev.ps1` from PowerShell 7+. Its default environment is `.venv`; add
`-VenvPath .venv313` for a checkout using that existing environment. Bootstrap
requires the prerequisites listed in the README and an existing pinned Python
executable when creating an environment. It installs `requirements/dev.txt` with
`--require-hashes`, hydrates and verifies the runtime assets, and runs
`maturin develop --release --locked`; no separate uv lock or automatic tool
installation is used.

```powershell
.\dev.ps1 bootstrap -PythonPath (py -3.13 -c "import sys; print(sys.executable)")
.\dev.ps1 doctor
.\dev.ps1 run
.\dev.ps1 test -VenvPath .venv313 -TestPath python/tests/test_voice_setup.py
```

`test` always rebuilds the native extension with
`maturin develop --release --locked`. With `-TestPath`, it then runs only that
Python test scope; without it, it runs the Rust tests and full Python suite.
The script scopes its interpreter, DLL search paths and runtime environment to
the command and restores them afterward. `run` enables verified bundled
DeepFilter assets. `-DryRun` prints steps without executing them.

`doctor` makes no changes or network requests. It checks tool availability,
interpreter/toolchain pins, runtime assets, `pip check` and native import.
Dependency consistency is not exact lock parity, and doctor does not replace
compilation, hardware checks or release interpreter/source provenance checks.
An incomplete environment is preserved; repair it explicitly or choose a new
`-VenvPath` before retrying bootstrap.

Run the closest regression first. For the remaining checks, use the project
interpreter explicitly. Prepare the shell for the native checks (substitute
`.venv313` if needed):

```powershell
$env:VIRTUAL_ENV = (Resolve-Path .venv).Path
$env:PYO3_PYTHON = Join-Path $env:VIRTUAL_ENV "Scripts/python.exe"
$basePython = & $env:PYO3_PYTHON -c "import sys; print(sys.base_prefix)"
$env:PATH = "$env:VIRTUAL_ENV\Scripts;$basePython;$env:PATH"

.\dev.ps1 test
.\.venv\Scripts\python.exe -m ruff check python/mic_eq python/tests python/tools
.\.venv\Scripts\python.exe -m pyright
cargo fmt --check
cargo test --release --locked -p mic_eq_core --test stress_tests seeded_control_and_dsp_loops_remain_finite_under_contention
cargo clippy --locked -p mic_eq_core --all-targets -- -D warnings
```

Keep bug fixes, ownership refactors, and
release preparation independently reviewable. Include a reproduction and a test
that fails before a nontrivial fix.

Changes to audible DSP behavior come in three kinds. A transition fix removes an
artifact at a change: a click, step or restart when a stage toggles, a filter
retunes, a parameter changes or the output recovers from a dropout. It needs a
deterministic test on a fixed input (synthetic or committed, with no open random
seed) showing that steady-state output still matches the pre-fix output within a
tolerance the test states, and that the artifact (peak step, click energy,
restart gap) is measurably smaller. A bug fix that makes the output match its
documented or tested intent, such as a wrong gain or a broken filter, names the
document or test that states that intent and needs the failing-before test
above. Any other change to steady-state sound is a tuning change, whatever its
motive (a default, threshold, model or Voice Setup decision).

A tuning pull request carries the bench report on the fixed bench clip set,
before and after, in its description. The numbers are information, not a
pass/fail gate. The clips are in `python/tests/fixtures/bench/`; run
`python/tools/bench_report.py --json-output before.json` on the base, then the
same command with `--compare before.json` on the change, which also prints the
deltas. The full set takes about ten minutes. The invariant tests must be
green: output finite and under the limiter ceiling, no new clipping, latency in
budget, silent, clipped and too-short takes refused, and a refused Voice Setup
leaving settings untouched.
The maintainer decides, and may treat a claimed bug fix as tuning. A large
change can ship behind a setting, or off by default, for one release first.
Listening is never a gate.

Change a test's expected output (a golden file or expected values in the test)
only in a pull request meant to change that output, and say why. The offline
chain's expected values are in `python/tests/fixtures/golden/`; regenerate them
with `AUDIOFORGE_UPDATE_GOLDEN=1 python -m pytest python/tests/test_golden_chain.py`.
A transition fix keeps the steady-state blocks and may only shrink the artifact
numbers, and it regenerates the transition cases in its own pull request; the
regenerated file differs from the old one in those cases only.

Preserve strict realtime constraints in CPAL callbacks and the DSP loop after
startup: no allocation, vector growth or Vec-returning suppressor helpers,
formatting/logging, I/O, blocking locks, or `try_lock`. Keep parameter updates in
atomic snapshots or bounded queues, and model loading and suppressor construction
outside those regions. Live DeepFilter inference runs on a worker the DSP loop
never waits on; the loop only polls it, and the RT scan forbids `.unpark(`.
Changes within `RT_REGION_*` markers must pass the RT
source-scan tests. Do not enable `extension-module` in Rust test builds.

A preset schema change needs a migration and round-trip tests. Migrate a stored
value only when its provenance is `migration_default`; a legacy value without
provenance counts as explicit. A new field in `config.json` raises
`CONFIG_SCHEMA_VERSION`. An older build saves only the keys it knows, so it would
drop a field added under the same version; a newer version makes it keep the
file and not save over it.

Two benchmarks cover callback cost. They are ignored release-mode tests that
nothing runs automatically. They are recommended, not required, for changes on
the audio path:

```powershell
cargo test --release --locked -p mic_eq_core audio::input::tests::benchmark_phase_safe_mono_callback_cost -- --ignored --nocapture
cargo test --release --locked -p mic_eq_core dsp::biquad::tests::benchmark_biquad_morph_cost -- --ignored --nocapture
```

A change to packaging, workflows, `AudioForge.spec` or `release-assets.json`
needs `python/tools/package_smoke.py --source-only` and
`python/tools/check_workflows.py`, then a full `package_smoke.py` run after
building. That script never launches the app, so after a local bundle build also
run `dist\AudioForge\AudioForge.exe --smoke-test` hidden, with a 60 second
limit, and require exit code 0. Without that run, a startup failure shows only
as a hidden dialog and a process that never exits. CI runs the same build and
smoke steps (`package-check.yml`) on pushes to `master` and on pull requests that
touch packaging inputs. Its Rust job builds `df.dll` from the pinned source and
fails when a DeepFilter or Silero test reports that it skipped itself.

`ci.yml` restores `df.dll` and its attestation from the Actions cache, keyed on
`build_deepfilter.ps1`, `build-support/deepfilter/`, `release-assets.json` and
`rust-toolchain.toml`, so a change to any of them rebuilds the DLL. The step after
the restore still runs `verify_release_assets.py` on the files. Package check and
the release workflow always build it. Closing a pull request runs
`cache-cleanup.yml`, which deletes the caches that pull request created, and
`weekly-checks.yml` re-runs the source, asset and advisory checks with nothing
cached. After you edit a workflow, run `zizmor .github/workflows` as well as
`check_workflows.py`; CI runs both.

After a release is published, download the assets, recompute their SHA-256, and
reconcile the README and notes in a later commit. Never move a tag.

Do not commit environments, generated binaries, release archives, recordings,
device identifiers, credentials, or local planning files. Dependency changes must
update both `pyproject.toml` and the hashed requirements when applicable. Preserve
upstream copyright and license notices. Original contributions remain under MIT;
bundled applications also follow the distribution policy in
[licenses/THIRD_PARTY_NOTICES.md](licenses/THIRD_PARTY_NOTICES.md).

Bug reports should include the version, Windows version, reproduction steps, and
expected/actual behavior. `Settings > Help > Export Diagnostics` produces a bounded,
pseudonymized report; inspect it before sharing. Do not attach microphone audio
or unredacted logs by default.

## What makes a check valid

A check that gates a release or a product decision, such as an invariant test,
a qualification run, or a hardware or smoke check, counts only if it passes four
tests:

1. It runs on what ships: the packaged build or the exact artifact. If it runs
   on something else, such as source or a simulator, its record says so.
2. Its pass criterion is written down before it runs. An invariant test states
   its own. A bench report has none, because it is information, not a gate.
3. It can fail. Before relying on a new check, run it once on a deliberately
   broken input and see it fail.
4. Its record says what it did not cover.

Other unit tests and linters are not held to these four.

## Adding a neural model

A newer or higher published score is not a reason to bundle a model. Before one
ships, record its license and redistribution terms, and put the SHA-256 of the
exact file in `release-assets.json`. Document its sample rate, state tensors,
pre- and post-processing, how it resets, and how it fails. Measure CPU, latency
and memory, and run it for 30 minutes to check stability.

Swapping or adding a model is a tuning change. Put the bench report against the
current model in the pull request, with the invariant tests green; the
maintainer decides. State which hardware was covered and which was not. A model
that ships also needs an entry in the explicit model list in `AudioForge.spec`,
in `package_smoke.py`'s required files, and in the reviewed bundle paths,
`evaluation/release-bundle-path-baseline.json`.

## Runtime assets and configuration

[release-assets.json](release-assets.json) owns runtime asset paths, origins,
sizes, and hashes. Follow the [source setup](README.md#quick-start-from-source)
to hydrate them, then verify before running or packaging:

```powershell
.\.venv\Scripts\python.exe python/tools/verify_release_assets.py
```

Bundled assets take precedence over the working directory. Source runs use
`AUDIOFORGE_ENABLE_DEEPFILTER=1` to enable DeepFilterNet. External
`DEEPFILTER_LIB_PATH` and `DEEPFILTER_MODEL_PATH` overrides require
`AUDIOFORGE_ALLOW_EXTERNAL_DF=1`; an unavailable external path uses the registered
bundled asset. `VAD_MODEL_PATH` selects the Silero model for source runs.

`AUDIOFORGE_FIXED_INPUT_BUFFER_FRAMES` and `AUDIOFORGE_FIXED_OUTPUT_BUFFER_FRAMES`
are optional callback-size diagnostics. Values must be 16–8192 frames and fit the
endpoint's advertised range; an unsupported request keeps the driver default.

## Runtime checks and screenshots

For route changes, use `python/tools/health_check.py --duration 1800` and
`python/tools/self_test.py` through the project interpreter. Both need a real
audio route, the native module and the ONNX Runtime, so no hosted CI job runs
them. Collect optional route measurements with explicit endpoints:

```powershell
.\.venv\Scripts\python.exe python/tools/evaluate_hardware_validation.py `
  --health-input "<microphone>" --health-output "<virtual output>" `
  --correlation-input "<loopback input>" --correlation-output "<loopback output>"
```

Keep measured revisions and coverage limits with the evidence described in
[evaluation/README.md](evaluation/README.md).

Regenerate sanitized README screenshots with
`python/tools/capture_repository_screenshots.py`. The screenshot tests check
dimensions, hashes, alt text, and the privacy boundary.
