# AudioForge Codebase Cleanup & Consistency Plan

**Date:** 2026-01-25
**Based on:** Two independent audits (.planning/AUDIT.md + Audit.md) + Additional user recommendations
**Status:** Ready to execute

---

## SUMMARY

Two independent audits + additional review identified **consistent issues** across versioning, naming, and documentation. This plan consolidates all findings and provides a single source of truth for cleanup.

---

## COMPLETED (already fixed)

- [x] README.md version: v0.2 → v1.5.0
- [x] Clone URL: micforge → audio-forge
- [x] pyproject.toml: enabled vad+deepfilter features
- [x] build_exe.ps1: removed df.dll inclusion
- [x] pyproject.toml version: 0.1.0 → 1.5.0
- [x] rust-core/Cargo.toml version: 0.1.0 → 1.5.0

---

## REMAINING TASKS (in priority order)

### PRIORITY 1: Version & Migration Completeness

| File | Current | Target | Action |
|------|---------|--------|--------|
| python/mic_eq/ui/main_window.py | v0.2.0 | v1.5.0 | Update About dialog |
| python/mic_eq/config.py | default: "1.2.0" | "1.5.0" | Update Preset dataclass |
| python/mic_eq/config.py | migration to 1.3.0 | migration to 1.5.0 | Add migration stubs |

**Action - main_window.py:**
```python
# Line ~848
- "<h2>MicEq v0.2.0</h2>"
+ "<h2>AudioForge v1.5.0</h2>"

# Line ~852 - processing chain
- "<p>Mic → Gate → RNNoise → EQ → Compressor → Limiter → Output</p>"
+ "<p>Mic → Pre-Filter → Gate → AI Noise (RNNoise/DF) → EQ → Comp → Limiter → Output</p>"

# Features list - add DeepFilterNet mention
+ "<li>AI Noise Suppression: RNNoise or DeepFilterNet (experimental)</li>"
```

**Action - config.py (preset version):**
```python
# Line ~20
- version: str = "1.2.0"
+ version: str = "1.5.0"
```

**Action - config.py (migration stubs):**
```python
# After existing 1.3.0 migration block, add:

# Migrate v1.3 presets → v1.4 (no format changes, version bump only)
if version < '1.4.0':
    data['version'] = '1.4.0'
    version = '1.4.0'

# Migrate v1.4 presets → v1.5 (no format changes, version bump only)
if version < '1.5.0':
    data['version'] = '1.5.0'
    version = '1.5.0'
```

---

### PRIORITY 2: Project Name Consistency

**Decision:** Keep internal names as-is, update public-facing to "AudioForge"

| Context | Name | Status |
|---------|------|--------|
| Public name (README title) | AudioForge | ✅ Already correct |
| Repository name | audio-forge | ✅ Already correct |
| Package name (pyproject) | mic_eq | ✅ Keep (Python convention) |
| Crate name (Cargo) | mic_eq_core | ✅ Keep (Rust convention) |
| Window title | MicEq → AudioForge | ⚠️ Update |
| About dialog | MicEq → AudioForge | ⚠️ Update |
| Exe name (build_exe.ps1) | MicEq → AudioForge | ⚠️ Update |
| Config directory | MicEq | ✅ Keep for backward compat |
| ROADMAP.md title | MicEq → AudioForge | ⚠️ Update |
| launcher.py docstring | MicEq | ⚠️ Update |
| __init__.py docstring | MicEq | ⚠️ Update |

**Action - main_window.py:**
```python
# Window title (line ~30)
- "MicEq - Microphone Audio Processor"
+ "AudioForge - Microphone Audio Processor"

# About dialog (line ~847)
- "About MicEq"
+ "About AudioForge"
```

**Action - build_exe.ps1:**
```powershell
# Line ~16
- "--name", "MicEq"
+ "--name", "AudioForge"
```

**Action - ROADMAP.md:**
```markdown
- # Roadmap: MicEq
+ # Roadmap: AudioForge
```

**Action - launcher.py:**
```python
"""
- MicEq launcher script for PyInstaller
+ AudioForge launcher script for PyInstaller
"""
```

**Action - python/mic_eq/__init__.py:**
```python
"""
- MicEq - Low-latency microphone audio processor
+ AudioForge - Low-latency microphone audio processor
"""
```

---

### PRIORITY 3: Dead Code Removal

**File to delete:** `rust-core/src/dsp/deepfilter.rs`

**Reason:**
- NOT imported in dsp/mod.rs
- Uses `df` crate not in Cargo.toml dependencies
- Would fail to compile if imported
- 384 lines of dead code
- Functionality replaced by deepfilter_ffi.rs

**Action:**
```bash
rm rust-core/src/dsp/deepfilter.rs
```

---

### PRIORITY 4: File Cleanup

**Files to delete:**
- `df.dll` (root, 16.6 MB)
- `python/df.dll` (16.6 MB)
- `rust-core/target/release/df.dll` (16.6 MB) - ignored by git but cleanup local
- `dist/MicEq/_internal/df.dll` (16.6 MB) - ignored by git but cleanup local

**Action:**
```powershell
del df.dll
del python\df.dll
# Others are in ignored directories, will disappear on clean build
```

**.gitignore cleanup:**
Remove obsolete entries if files no longer exist:
```gitignore
- # Documentation drafts
- deepfilter_impl.md
- DEEPFILTER_TEST_REPORT.md
```

---

### PRIORITY 5: Icon File Rename (OPTIONAL)

If renaming to AudioForge, consider:
```
mic_eq.ico → audioforge.ico
```

Update references in:
- `build_exe.ps1`
- `main_window.py` (icon loading code around line 900)

**Note:** This is optional - keeping existing icon name is fine for backward compat.

---

### PRIORITY 6: Documentation Updates

**README.md - Add processing chain nuance:**
```markdown
## Architecture

**Processing Chain:**
```
Mic Input → Pre-Filter (DC Block + 80Hz HP) → Noise Gate → AI Noise (RNNoise/DeepFilter) → 10-Band EQ → Compressor → Limiter → Output
```

**Note:** DeepFilterNet is experimental. RNNoise is recommended for production.
```

**README.md - Quick Start simplification:**
```markdown
### Quick Start

...

> **Note:** VAD and DeepFilterNet features are enabled by default but require model files.
> See [Models](#models-optional) section for download instructions.
```

**CLAUDE.md - Update processing chain:**
```markdown
Mic Input → Pre-Filter (80Hz HP) → Noise Gate → AI Suppressor (RNNoise or DeepFilter) → 10-Band EQ → Compressor → Hard Limiter → VB Audio Cable Output
```

---

### PRIORITY 7: Code Quality & Polish

**Error message fix (main_window.py ~683):**
```python
# DeepFilterNet error message - misleading since features now enabled by default
- "3. Rebuilding with --features deepfilter if using custom build"
+ "3. DeepFilterNet requires df.dll - falls back to passthrough if not found"
```

**Add VAD showcase preset (config.py BUILTIN_PRESETS):**
```python
'vad_smart': Preset(
    name="VAD Smart Gate",
    description="AI-powered gate using voice activity detection",
    version="1.5.0",
    gate=GateSettings(
        enabled=True,
        threshold_db=-45.0,
        attack_ms=5.0,
        release_ms=100.0,
        gate_mode=1,  # VadAssisted
        vad_threshold=0.5,
        vad_hold_time_ms=200.0
    ),
    rnnoise=RNNoiseSettings(enabled=True, strength=0.7),
    # ... rest of default settings
),
```

**Test code organization (optional):**
Move test functions from `config.py` to `tests/test_config.py`:
- `_test_vad_preset_persistence()`
- `_test_backward_compatibility()`

---

## EXECUTION CHECKLIST

### Phase 1: Versions & Migration (HIGH)
- [ ] Update main_window.py About dialog version to v1.5.0
- [ ] Update main_window.py window title to "AudioForge"
- [ ] Update main_window.py processing chain description
- [ ] Add DeepFilterNet mention to About dialog features
- [ ] Update config.py default preset version to "1.5.0"
- [ ] Add migration stubs for 1.4.0 and 1.5.0 in config.py
- [ ] Commit: "chore: sync version to 1.5.0 across all files"

### Phase 2: Naming Consistency (MEDIUM)
- [ ] Update ROADMAP.md title from "MicEq" to "AudioForge"
- [ ] Update build_exe.ps1 exe name to "AudioForge"
- [ ] Update launcher.py docstring
- [ ] Update __init__.py docstring
- [ ] Update CLAUDE.md processing chain
- [ ] Commit: "docs: standardize public name to AudioForge"

### Phase 3: Dead Code Removal (HIGH)
- [ ] Delete rust-core/src/dsp/deepfilter.rs
- [ ] Delete df.dll files (2 in repo)
- [ ] Clean up obsolete .gitignore entries
- [ ] Commit: "refactor: remove dead deepfilter.rs code and unused df.dll"

### Phase 4: Documentation (MEDIUM)
- [ ] Update README.md processing chain with DC Blocker note
- [ ] Mark DeepFilterNet as experimental in README
- [ ] Add Quick Start note about model files
- [ ] Fix error message about deepfilter feature
- [ ] Commit: "docs: update README with accurate processing chain"

### Phase 5: Polish (LOW - OPTIONAL)
- [ ] Add VAD showcase preset to BUILTIN_PRESETS
- [ ] Consider icon rename (mic_eq.ico → audioforge.ico)
- [ ] Move test functions to tests/test_config.py
- [ ] Commit: "feat: add VAD preset and test organization"

---

## FILES TO MODIFY SUMMARY

| File | Changes |
|------|---------|
| python/mic_eq/ui/main_window.py | Version, title, About dialog, processing chain, error msg |
| python/mic_eq/config.py | Version, migration stubs, VAD preset (optional) |
| ROADMAP.md | Title: MicEq → AudioForge |
| CLAUDE.md | Processing chain mention |
| README.md | Processing chain, DeepFilterNet experimental note, Quick Start note |
| build_exe.ps1 | Exe name: MicEq → AudioForge |
| launcher.py | Docstring update |
| python/mic_eq/__init__.py | Docstring update |
| rust-core/src/dsp/deepfilter.rs | DELETE |
| df.dll (x2) | DELETE |
| .gitignore | Remove obsolete entries |

---

## VERSION POLICY GOING FORWARD

**Single Source of Truth:** ROADMAP.md milestone version

**When releasing vX.Y.Z:**
1. Update ROADMAP.md milestone status
2. Update pyproject.toml version
3. Update rust-core/Cargo.toml version
4. Update main_window.py About dialog version
5. Update config.py default preset version if format changed
6. Add migration stub if preset format changed
7. Tag commit: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`

---

## NOTES

- **Config directory naming:** Keep as "MicEq" for backward compatibility with existing user presets
- **Internal package names:** Keep `mic_eq` and `mic_eq_core` (Python/Rust conventions)
- **Public name:** "AudioForge" for user-facing displays only
- **DeepFilterNet status:** Experimental - document clearly, RNNoise is primary
- **Icon file:** Optional rename - keeping `mic_eq.ico` is acceptable
- **Test functions:** Moving to proper test file is best practice but not urgent

---

## PRIORITY MATRIX

| Phase | Priority | Effort | Impact |
|-------|----------|--------|--------|
| Phase 1: Versions | HIGH | 15 min | Critical - version consistency |
| Phase 2: Naming | MEDIUM | 10 min | High - brand consistency |
| Phase 3: Dead Code | HIGH | 5 min | High - removes confusion |
| Phase 4: Documentation | MEDIUM | 15 min | Medium - user clarity |
| Phase 5: Polish | LOW | 20 min | Low - nice to have |

