# Contributing to AudioForge

Use Windows x64, CPython 3.13.15 and the Rust compiler in `rust-toolchain.toml`.
The supported source-build contract uses the checked-in dependency locks; newer
Python or Rust releases are not implicitly qualified. See the [README](README.md)
for setup and [RELEASING](RELEASING.md) for packaging and publication.

Use the project virtual environment explicitly. After any Rust change, rebuild
the extension before testing Python behavior:

```powershell
.\.venv\Scripts\python.exe -m maturin develop --release --locked
.\.venv\Scripts\python.exe -m pytest python/tests -q
.\.venv\Scripts\python.exe -m ruff check python/mic_eq python/tests python/tools
.\.venv\Scripts\python.exe -m pyright
cargo fmt --check
cargo test --locked -p mic_eq_core
cargo test --release --locked -p mic_eq_core --test stress_tests seeded_control_and_dsp_loops_remain_finite_under_contention
cargo clippy --locked -p mic_eq_core --all-targets -- -D warnings
```

Run the closest regression first. Keep bug fixes, ownership refactors, and
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
