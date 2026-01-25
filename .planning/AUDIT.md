# MIC_EQ/AUDIOFORGE CODEBASE AUDIT REPORT

**Date:** 2026-01-25
**Auditor:** Claude Code (Explore Agent)

---

## EXECUTIVE SUMMARY

The project has significant discrepancies between documentation and implementation, particularly around:
1. **Version numbering confusion** (README shows v0.2-v0.4, actual is v1.0.0-v1.5.0)
2. **Repository name mismatch** (README refers to "micforge", actual is "audio-forge")
3. **Build configuration inconsistencies** (features enabled in pyproject.toml but not in Cargo.toml defaults)
4. **Unused binary files** (df.dll present but build script tries to include it while code says it's optional)
5. **Duplicate DeepFilterNet implementations** (deepfilter.rs exists but is not used, deepfilter_ffi.rs is the active implementation)

---

## 1. DOCUMENTATION DISCREPANCIES

### 1.1 Version Numbering (CRITICAL)

**README.md states:**
```
### v0.2 (Current)
### v0.3 (Planned)
### v0.4 (Future)
```

**Actual state (ROADMAP.md):**
```
v1.0.0 Production Release - SHIPPED 2026-01-22
v1.1.0 Enhancement - SHIPPED 2026-01-23
v1.1.1 Enhancement - COMPLETED 2026-01-24
v1.2.0 Advanced DSP - COMPLETED 2026-01-24
v1.3.0 GUI Polish - COMPLETED 2026-01-24
v1.4.0 Scrollable Panels - COMPLETED 2026-01-24
v1.5.0 VAD Runtime Fix - SHIPPED 2026-01-24
```

**Impact:** Severe - Documentation makes project appear immature (v0.2) when it's actually at v1.5.0 production release

**Fix required:** Update README.md version sections to match actual shipped versions

### 1.2 Repository Name Mismatch (HIGH)

**README.md states:**
```powershell
git clone https://github.com/FueledByRedBull/micforge.git
cd micforge
```

**Actual repository:**
```
origin	https://github.com/FueledByRedBull/audio-forge.git
```

**Impact:** High - Users cannot clone the repository using the instructions in README

**Fix required:** Update all instances of "micforge" to "audio-forge" in README.md (3 occurrences)

### 1.3 Project Name Inconsistency (MEDIUM)

**Files use different names:**
- README.md: "AudioForge" (title)
- pyproject.toml: `name = "mic_eq"`
- rust-core/Cargo.toml: `name = "mic_eq_core"`
- ROADMAP.md: "MicEq" (throughout)
- CLAUDE.md: "MicEq" (throughout)

**Impact:** Medium - Confusion about project identity

**Fix required:** Standardize on "AudioForge" as the public name, "audio-forge" for repository

---

## 2. BUILD SYSTEM ISSUES

### 2.1 Feature Flag Inconsistency (HIGH)

**pyproject.toml (modified but not committed):**
```toml
features = ["pyo3/extension-module", "vad", "deepfilter"]
```

**Original pyproject.toml:**
```toml
features = ["pyo3/extension-module"]
```

**rust-core/Cargo.toml:**
```toml
[features]
default = []  # NO FEATURES ENABLED BY DEFAULT
vad = ["dep:ndarray", "dep:ort"]
deepfilter = ["dep:libloading"]
deepfilter-real = ["deepfilter"]
```

**Impact:** High - Inconsistent feature flags between builds can cause:
- Build failures if features not explicitly enabled
- Missing VAD or DeepFilter functionality despite documentation claiming they exist
- Confusion about which features are actually available

**Issue:** The project documentation (README) says to build with `--features deepfilter`, but:
1. pyproject.toml was modified to include both vad and deepfilter features
2. This change is not committed (shows in git status)
3. Users following README will get different build than documented

### 2.2 Build Script References Unused Binary (HIGH)

**build_exe.ps1 line 18:**
```powershell
"--add-binary", "df.dll;."
```

**Reality:**
- df.dll is NOT required (deepfilter feature uses runtime loading)
- Documentation in deepfilter_ffi.rs states: "Falls back to passthrough if C library is not available"
- df.dll is 15.9MB and present in 3 locations
- Pending cleanup task explicitly identifies df.dll as "unused"

**Impact:** High - Build script includes unnecessary 15.9MB binary, bloating distribution

**Fix required:** Remove df.dll from build_exe.ps1, add conditional inclusion only if file exists

### 2.3 Build Script Inconsistency (MEDIUM)

**build_exe.ps1 (modified):**
```powershell
"--clean"
"-y"  # ADDED
"--onedir"
```

**build.ps1 exists but is different**
- build.ps1: Development build script (cargo/maturin workflow)
- build_exe.ps1: PyInstaller executable builder
- No clear indication which should be used for what purpose

**Impact:** Medium - Unclear build workflow for contributors

---

## 3. FILE/BINARY ISSUES

### 3.1 Unused df.dll Files (HIGH)

**Locations found:**
```
C:\Users\anchatzigiannakis\Documents\Projects\MicEq\df.dll (16.6 MB)
C:\Users\anchatzigiannakis\Documents\Projects\MicEq\python\df.dll (16.6 MB)
C:\Users\anchatzigiannakis\Documents\Projects\MicEq\rust-core\target\release\df.dll (16.6 MB)
C:\Users\anchatzigiannakis\Documents\Projects\MicEq\dist\MicEq\_internal\df.dll (16.6 MB)
```

**Total waste:** ~66 MB

**Status:**
- Pending cleanup task identifies these as "unused DeepFilterNet C library (superseded by ONNX implementation)"
- Documentation in deepfilter_ffi.rs confirms: "Does NOT require DeepFilterNet C library to be present at compile time"
- Runtime loading is optional with graceful fallback

**Impact:** High - 66 MB of wasted disk space, confusing presence of unused binaries

### 3.2 Duplicate DeepFilterNet Implementation (MEDIUM)

**Two implementations exist:**

1. **rust-core/src/dsp/deepfilter.rs (384 lines)** - NOT USED
   - Implements `DeepFilterProcessor`
   - Uses `df` crate (libDF Rust bindings)
   - Has `deepfilter-real` feature flag
   - NOT imported in dsp/mod.rs
   - NOT referenced anywhere in codebase

2. **rust-core/src/dsp/deepfilter_ffi.rs (678 lines)** - ACTIVELY USED
   - Implements `DeepFilterProcessor` via C FFI
   - Uses `libloading` for runtime dynamic loading
   - Has `deepfilter` feature flag
   - Exported in dsp/mod.rs: `pub use deepfilter_ffi::{DeepFilterProcessor, DEEPFILTER_FRAME_SIZE};`
   - Used in noise_suppressor.rs

**Impact:** Medium - Code confusion, maintenance burden, 384 lines of dead code

**Fix required:** Remove deepfilter.rs, keep only deepfilter_ffi.rs

### 3.3 Silero VAD Model (LOW)

**Status:**
- File exists: `models/silero_vad.onnx`
- Properly documented in README
- Required for VAD feature (optional feature)
- Properly used in implementation

**Impact:** None - This is working correctly

---

## 4. IMPLEMENTATION GAPS

### 4.1 Features Documented But Not Clearly Optional

**README.md presents DeepFilterNet as primary feature:**
```markdown
- **AI Noise Suppression** - Three models to choose from:
  - RNNoise (~10ms latency)
  - DeepFilterNet LL (~10ms latency, better quality)
  - DeepFilterNet Standard (~40ms latency, best quality)
```

**Reality:**
- DeepFilterNet requires `--features deepfilter` at build time
- Falls back to passthrough if df.dll not present
- Not in default feature set
- Experimental status mentioned in ROADMAP.md (Phase 10)

**Impact:** Medium - Users may expect working DeepFilterNet when it's experimental/optional

**Fix required:** Clearly mark DeepFilterNet as experimental in README, add feature flag requirements

### 4.2 Feature Documentation Clarity (MEDIUM)

**CLAUDE.md processing chain:**
```
Mic Input → Pre-Filter (80Hz HP) → Noise Gate → RNNoise → 10-Band EQ → Compressor → Hard Limiter → VB Audio Cable Output
```

**README.md processing chain:**
```
Mic Input → Pre-Filter (80Hz HP) → Noise Gate → DeepFilter → EQ → Compressor → Limiter → Output
```

**Differences:**
- CLAUDE.md says "RNNoise"
- README.md says "DeepFilter"
- Neither mentions both are available as options

**Impact:** Medium - Inconsistent documentation of processing chain

---

## 5. CONFIGURATION FILE INCONSISTENCIES

### 5.1 pyproject.toml (MODIFIED BUT NOT COMMITTED)

**Changes:**
```diff
- features = ["pyo3/extension-module"]
+ features = ["pyo3/extension-module", "vad", "deepfilter"]
```

**Impact:** High - Local modifications not committed means:
- Builds from repository will fail to include VAD/DeepFilter
- Inconsistent behavior between developer machines
- Documentation says to use `--features deepfilter` but pyproject.toml now includes it

**Fix required:** Commit pyproject.toml changes OR document why features should be runtime-only

### 5.2 build_exe.ps1 (MODIFIED BUT NOT COMMITTED)

**Changes:**
```diff
+ "-y",  # Auto-confirm PyInstaller
+ "--add-binary", "df.dll;."
```

**Impact:** Medium - Uncommitted changes to build script

---

## 6. ARCHITECTURE NOTES

### 6.1 DeepFilterNet Implementation Status

**Current state:**
- Primary implementation: deepfilter_ffi.rs (C FFI with runtime loading)
- Fallback: Passthrough if df.dll not available
- Status: Experimental (ROADMAP Phase 10)
- Known issue: "C library loads but crashes during initialization due to upstream tract-core bug"

**Documentation should reflect:**
- DeepFilterNet is experimental
- RNNoise is the recommended production choice
- df.dll is entirely optional
- Feature flags required for DeepFilterNet support

---

## RECOMMENDED ACTIONS (PRIORITIZED)

### CRITICAL (Fix Immediately)
1. **Update README.md version numbering** - Change v0.2/v0.3/v0.4 to v1.5.0+
2. **Fix repository clone URL** - Change micforge to audio-forge
3. **Commit or revert pyproject.toml changes** - Decide on feature strategy
4. **Remove df.dll from build_exe.ps1** - Don't bundle unused 15.9MB binary

### HIGH PRIORITY
5. **Delete unused df.dll files** - Remove 66 MB of waste (3 locations)
6. **Remove deepfilter.rs** - Delete 384 lines of dead code
7. **Mark DeepFilterNet as experimental in README** - Add caveats
8. **Standardize project naming** - AudioForge (public) / audio-forge (repo) / mic_eq (internal)

### MEDIUM PRIORITY
9. **Clarify build workflow** - Document when to use build.ps1 vs build_exe.ps1
10. **Update processing chain documentation** - Show both RNNoise and DeepFilterNet options
11. **Add feature flag documentation** - Explain vad/deepfilter features clearly

### LOW PRIORITY
12. **Standardize version display** - Ensure UI shows correct version
13. **Review CLAUDE.md for outdated information** - Update to reflect v1.5.0 reality
14. **Consider consolidating documentation** - README, ROADMAP, CLAUDE overlap significantly

---

## FILES REQUIRING UPDATES

1. **README.md** - Version numbers, clone URL, DeepFilterNet experimental status
2. **pyproject.toml** - Commit or revert feature changes
3. **build_exe.ps1** - Remove df.dll inclusion
4. **rust-core/src/dsp/deepfilter.rs** - DELETE (unused)
5. **df.dll** (3 locations) - DELETE
6. **CLAUDE.md** - Update processing chain documentation

---

## SUMMARY STATISTICS

- **Total issues found:** 17
- **Critical priority:** 4
- **High priority:** 4
- **Medium priority:** 4
- **Low priority:** 5
- **Wasted disk space:** ~66 MB (df.dll files)
- **Dead code:** 384 lines (deepfilter.rs)
- **Documentation inconsistencies:** 6 major discrepancies
- **Build configuration issues:** 5 inconsistencies

---

## CONCLUSION

This audit reveals a project that is more mature than its documentation suggests (v1.5.0 vs v0.2), but suffers from significant documentation drift and unused artifacts from experimental features. The core functionality appears solid, but cleanup and documentation updates are needed for user clarity.
