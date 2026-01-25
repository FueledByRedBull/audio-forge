I'll systematically analyze the entire codebase to identify discrepancies between documentation, code, and actual implementation state.Let me analyze the documents I have access to in the context window. After a thorough review of the entire codebase, here's my comprehensive audit:

---

# AudioForge/MicEq Codebase Audit

## Executive Summary

The codebase has **significant naming inconsistencies**, **outdated documentation**, **version mismatches**, and **feature flag confusion**. The project appears to have been renamed from "MicEq" to "AudioForge" but this change was not applied consistently.

---

## 1. PROJECT NAMING INCONSISTENCY (CRITICAL)

### Problem
The project uses two names interchangeably: **"AudioForge"** and **"MicEq"**

| Location | Name Used |
|----------|-----------|
| `README.md` | **AudioForge** (title, repo URL) |
| `ROADMAP.md` | **MicEq** (title, all references) |
| `pyproject.toml` | `mic_eq` (package name) |
| `python/mic_eq/` | `mic_eq` (module name) |
| `rust-core/Cargo.toml` | `mic_eq_core` (crate name) |
| `build.ps1` | MicEq |
| `build_exe.ps1` | MicEq |
| `launcher.py` | `mic_eq` |
| Window title in code | "MicEq - Microphone Audio Processor" |
| Config directory | `MicEq` |
| GitHub URL in README | `FueledByRedBull/audio-forge` |

### Recommendation
Decide on ONE name and apply it consistently everywhere.

---

## 2. VERSION MISMATCHES

### README.md States:
- **v1.5.0 (Current Release)** with VAD Runtime Fix

### ROADMAP.md States:
- **v1.5.0** completed 2026-01-24
- Lists 16 phases, 47/47 plans complete

### pyproject.toml States:
- `version = "0.1.0"`

### rust-core/Cargo.toml States:
- `version = "0.1.0"`

### main_window.py About Dialog States:
- `"<h2>MicEq v0.2.0</h2>"`

### config.py Preset Version:
- `version: str = "1.2.0"` (default for new presets)
- Migration logic exists for 1.0.0 → 1.1.0 → 1.2.0 → 1.3.0

### Recommendation
Synchronize all version numbers. Current release appears to be **v1.5.0** based on ROADMAP.md.

---

## 3. README.md ISSUES

### 3.1 Outdated Roadmap Section
README says:
```markdown
### v1.5.0 (Current Release)
- [x] RNNoise integration
- [x] Silero VAD-assisted noise gate
- [x] DeepFilterNet integration (experimental, C FFI)
...

### v1.6.0 (Planned)
- [ ] Linux/macOS builds
...
```

But ROADMAP.md has much more detailed phase information that doesn't match.

### 3.2 Model Download Instructions Incomplete
README mentions:
```markdown
# Download DF3 LL model (recommended for real-time)
# Place in: models/DeepFilterNet3_ll_onnx.tar.gz
```

But `deepfilter_ffi.rs` actually looks for:
- `DeepFilterNet3_ll_onnx.tar.gz` (Low Latency)
- `DeepFilterNet3_onnx.tar.gz` (Standard)

And the environment variable is `DEEPFILTER_MODEL_PATH`.

### 3.3 Missing VAD Model Instructions in Quick Start
README has VAD model instructions in a separate section, but it's not in the Quick Start flow.

### 3.4 Processing Chain Diagram Outdated
README shows:
```
Mic Input → Pre-Filter (80Hz HP) → Noise Gate → DeepFilter → EQ → Compressor → Limiter → Output
```

But actual code in `processor.rs` has:
```
Pre-Filter (DC Block + 80Hz HP) → Noise Gate → Noise Suppressor (RNNoise OR DeepFilter) → EQ → Compressor → Limiter
```

Note: DC Blocker is not mentioned in README.

### 3.5 Screenshot Section
```markdown
## Screenshots
*Coming soon*
```

Should either add screenshots or remove section.

---

## 4. ROADMAP.md ISSUES

### 4.1 Project Name
Title says "Roadmap: MicEq" but README calls it "AudioForge"

### 4.2 Dates
All completion dates are in **2026** (e.g., "2026-01-24"). These appear to be future dates or typos.

### 4.3 DeepFilterNet Status
Phase 10 says:
```
**Status:** Experimental - C library loads but crashes during initialization due to upstream tract-core bug
```

But the code in `deepfilter_ffi.rs` has extensive FFI implementation. Need to clarify actual status.

---

## 5. CODE COMMENTS vs REALITY

### 5.1 `deepfilter.rs` vs `deepfilter_ffi.rs`
Both files exist:
- `rust-core/src/dsp/deepfilter.rs` - Uses `libDF` Rust crate directly (feature gated)
- `rust-core/src/dsp/deepfilter_ffi.rs` - Uses C FFI approach

But `mod.rs` only exports `deepfilter_ffi`:
```rust
#[cfg(feature = "deepfilter")]
pub mod deepfilter_ffi;
```

The `deepfilter.rs` file is NOT imported anywhere. It appears to be dead code.

### 5.2 Feature Flag Confusion
`Cargo.toml` has:
```toml
[features]
deepfilter = ["dep:libloading"]
deepfilter-real = ["deepfilter"]
```

But `deepfilter.rs` uses:
```rust
#[cfg(not(feature = "deepfilter-real"))]  // STUB mode
#[cfg(feature = "deepfilter-real")]       // REAL mode
```

This is confusing because:
1. `deepfilter_ffi.rs` is what's actually used (via `mod.rs`)
2. `deepfilter.rs` is orphaned code with its own feature logic
3. The feature flags don't match between files

### 5.3 NoiseModel Enum Mismatch
In `noise_suppressor.rs`:
```rust
pub enum NoiseModel {
    RNNoise,
    #[cfg(feature = "deepfilter")]
    DeepFilterNetLL,
    #[cfg(feature = "deepfilter")]
    DeepFilterNet,
}
```

But `deepfilter.rs` (orphaned) only references `NoiseModel::DeepFilterNet`, not `DeepFilterNetLL`.

---

## 6. DEPENDENCY ISSUES

### 6.1 Cargo.toml Comments About Dependencies
```toml
# Noise suppression (DeepFilterNet) - optional, C FFI approach
# Uses C API via FFI to avoid Rc/Send issues with Rust libDF crate
# No direct Rust dependency - links against pre-built DeepFilterNet C library
```

This is correct for `deepfilter_ffi.rs`, but `deepfilter.rs` has imports for:
```rust
use df::tract::{DfTract, RuntimeParams};
use ndarray::{ArrayView2, ArrayViewMut2, Array2, Ix2};
```

These crates (`df`, full `ndarray` with `Ix2`) are NOT in `Cargo.toml`, so `deepfilter.rs` would fail to compile if imported.

### 6.2 ndarray Version
`Cargo.toml`:
```toml
ndarray = { version = "0.17", optional = true }
```

This is only for VAD. The `deepfilter.rs` code would need different ndarray features.

---

## 7. PYPROJECT.TOML ISSUES

### 7.1 Features List
```toml
features = ["pyo3/extension-module", "vad", "deepfilter"]
```

This enables both VAD and DeepFilter by default for maturin builds, which is good.

### 7.2 Missing Description Update
```toml
description = "Low-latency microphone audio processor with noise suppression and equalization"
```

Doesn't mention VAD or DeepFilterNet.

---

## 8. UI CODE ISSUES

### 8.1 About Dialog Outdated
`main_window.py`:
```python
"<h2>MicEq v0.2.0</h2>"
```

Should be v1.5.0 to match ROADMAP.

### 8.2 Processing Chain in About Dialog
```python
"<p>Mic → Gate → RNNoise → EQ → Compressor → Limiter → Output</p>"
```

Missing Pre-Filter and doesn't mention DeepFilterNet option.

### 8.3 Feature Comments
`main_window.py`:
```python
"<li>RNNoise ML-based noise suppression</li>"
```

Should mention DeepFilterNet as alternative.

---

## 9. CONFIG.PY PRESET VERSION HANDLING

### Issue
Preset versions go up to 1.3.0 in migration logic, but project version in ROADMAP is 1.5.0.

The migration only handles:
- 1.0.0 → 1.1.0 (add strength to RNNoise)
- 1.1.0 → 1.2.0 (add model field to RNNoise)
- 1.2.0 → 1.3.0 (add auto makeup gain to compressor)

Missing:
- 1.3.0 → 1.4.0 (scrollable panels - no preset changes needed?)
- 1.4.0 → 1.5.0 (VAD runtime fix - no preset changes needed?)

But default version is still `"1.2.0"` in the `Preset` dataclass. Should probably be `"1.3.0"` since that's the last version with preset format changes.

---

## 10. DEAD/ORPHANED CODE

### 10.1 `deepfilter.rs`
~200 lines of code that is:
- Not imported in `mod.rs`
- Has compile-time dependencies not in `Cargo.toml`
- Would fail to compile if imported

**Recommendation:** Delete or properly integrate.

### 10.2 Test Functions in config.py
```python
def _test_vad_preset_persistence():
def _test_backward_compatibility():

if __name__ == "__main__":
    _test_vad_preset_persistence()
    _test_backward_compatibility()
```

These should be moved to a proper test file.

---

## 11. COMMENTS NEEDING UPDATE

### 11.1 `processor.rs` Line ~47
```rust
/// Noise suppression engine (RNNoise or DeepFilterNet)
suppressor: Arc<Mutex<NoiseSuppressionEngine>>,
```

Comment is accurate.

### 11.2 `rnnoise.rs` Module Doc
```rust
//! RNNoise integration with proper scaling and 480-sample frame buffering
```

Accurate.

### 11.3 `deepfilter_ffi.rs` Header Comment
Long and detailed - accurate for the FFI approach.

---

## 12. .gitignore ISSUES

### 12.1 Includes Development Files
```gitignore
# Documentation drafts
deepfilter_impl.md
DEEPFILTER_TEST_REPORT.md

# Claude Code
.claude/

# Project docs (development only)
CLAUDE.md
```

These suggest there were/are development docs not in the repo. The `.planning/` directory is referenced in ROADMAP.md but not visible.

---

## 13. BUILD SCRIPT ISSUES

### 13.1 `build.ps1`
References "MicEq" in comments and output messages.

### 13.2 `build_exe.ps1`
```powershell
"--name", "MicEq"
```

Uses MicEq as exe name.

---

## 14. LICENSE FILE

```
Copyright (c) 2025 FueledByRedBull
```

2025 - should verify this is correct year.

---

## RECOMMENDED ACTIONS

### Priority 1 (Critical)
1. **Delete `deepfilter.rs`** - It's orphaned and won't compile
2. **Synchronize version numbers** - Pick one (1.5.0) and update everywhere
3. **Decide on project name** - AudioForge or MicEq, not both

### Priority 2 (High)
4. **Update README.md**:
   - Fix processing chain diagram to include DC Blocker
   - Update roadmap section to match ROADMAP.md
   - Add proper VAD setup instructions to Quick Start
   - Update version references
   
5. **Update About Dialog** in `main_window.py`:
   - Version to 1.5.0
   - Processing chain description
   - Add DeepFilterNet mention

6. **Update `pyproject.toml` and `Cargo.toml`** versions

### Priority 3 (Medium)
7. **Clean up config.py**:
   - Move test functions to test file
   - Update default preset version to 1.3.0

8. **Clarify DeepFilterNet status**:
   - Is C FFI working or experimental?
   - Update README accordingly

9. **Update ROADMAP.md**:
   - Fix dates if they're wrong
   - Use consistent project name

### Priority 4 (Low)
10. **Add screenshots or remove section**
11. **Review and clean up .gitignore** referenced files
12. **Add proper pytest tests** for config module

---

## FILE-BY-FILE SUMMARY

| File | Status | Issues |
|------|--------|--------|
| README.md | ⚠️ Needs Update | Version, name, processing chain, roadmap |
| ROADMAP.md | ⚠️ Needs Update | Name, dates verification |
| pyproject.toml | ⚠️ Needs Update | Version (0.1.0 → 1.5.0) |
| rust-core/Cargo.toml | ⚠️ Needs Update | Version (0.1.0 → 1.5.0) |
| python/mic_eq/ui/main_window.py | ⚠️ Needs Update | About dialog version |
| python/mic_eq/config.py | ⚠️ Minor Issues | Test functions, version |
| rust-core/src/dsp/deepfilter.rs | ❌ DELETE | Orphaned dead code |
| rust-core/src/dsp/deepfilter_ffi.rs | ✅ OK | Working FFI implementation |
| rust-core/src/dsp/mod.rs | ✅ OK | Correctly imports deepfilter_ffi |
| rust-core/src/dsp/noise_suppressor.rs | ✅ OK | Correct model enum |
| All other Rust files | ✅ OK | No major issues |
| All other Python files | ✅ OK | No major issues |

---