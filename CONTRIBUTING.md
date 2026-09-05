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
cargo clippy --locked -p mic_eq_core --all-targets -- -D warnings
```

Run the closest regression first. Keep bug fixes, ownership refactors, and
release preparation independently reviewable. Include a reproduction and a test
that fails before a nontrivial fix. Changes to audible DSP behavior need the
objective, held-out evidence described in [evaluation/README.md](evaluation/README.md).
Preserve real-time constraints: no allocation, logging, I/O, or blocking locks in
callbacks or the DSP loop after startup. Do not enable `extension-module` in Rust
test builds.

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
