# Contributing to AudioForge

Use Windows x64, CPython 3.13.15 and the Rust compiler in `rust-toolchain.toml`.
The supported source-build contract uses the checked-in dependency locks; newer
Python or Rust releases are not implicitly qualified. See the [README](README.md)
for setup and [RELEASING](RELEASING.md) for packaging and publication.

Use `dev.ps1` from PowerShell 7+. Its default environment is `.venv`; add
`-VenvPath .venv313` for a checkout using that existing environment. Bootstrap
requires the prerequisites listed in the README and an existing pinned Python
executable when creating an environment. It reuses `requirements/dev.txt` with
`--require-hashes`; no separate uv lock or automatic tool installation is used.

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
that fails before a nontrivial fix. Changes to audible DSP behavior need the
objective, held-out evidence described in [evaluation/README.md](evaluation/README.md).
Preserve strict realtime constraints in CPAL callbacks and the DSP loop after
startup: no allocation, vector growth or Vec-returning suppressor helpers,
formatting/logging, I/O, blocking locks, or `try_lock`. Keep parameter updates in
atomic snapshots or bounded queues, and model loading and suppressor construction
outside those regions. Changes within `RT_REGION_*` markers must pass the RT
source-scan tests. Do not enable `extension-module` in Rust test builds.

For changes affecting callback cost, run the focused release benchmarks:

```powershell
cargo test --release --locked -p mic_eq_core audio::input::tests::benchmark_phase_safe_mono_callback_cost -- --ignored --nocapture
cargo test --release --locked -p mic_eq_core dsp::biquad::tests::benchmark_biquad_morph_cost -- --ignored --nocapture
```

Do not commit environments, generated binaries, release archives, recordings,
device identifiers, credentials, or local planning files. Dependency changes must
update both `pyproject.toml` and the hashed requirements when applicable. Preserve
upstream copyright and license notices. Original contributions remain under MIT;
bundled applications also follow the distribution policy in
[licenses/THIRD_PARTY_NOTICES.md](licenses/THIRD_PARTY_NOTICES.md).

Bug reports should include the version, Windows version, reproduction steps, and
expected/actual behavior. `Help > Export Diagnostics...` produces a bounded,
pseudonymized report; inspect it before sharing. Do not attach microphone audio
or unredacted logs by default.

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
`python/tools/self_test.py` through the project interpreter. Collect optional
route measurements with explicit endpoints:

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
