//! DeepFilterNet integration via C FFI (bypasses Rc/Send issues)
//!
//! This module uses DeepFilterNet's C API directly via FFI, which:
//! - Avoids the Rc/Send issues with the Rust libDF crate
//! - Provides a stable ABI through C interface
//! - Uses runtime dynamic loading (no compile-time library dependency)
//! - Falls back to passthrough if library is not available
//!
//! C API FUNCTIONS (from libDF/src/capi.rs):
//! - df_create(path, atten_lim, log_level) -> *mut DFState
//! - df_free(*mut DFState)
//! - df_get_frame_length(*mut DFState) -> usize
//! - df_process_frame(*mut DFState, *mut f32, *mut f32) -> f32
//! - df_set_atten_lim(*mut DFState, f32)
//! - df_set_post_filter_beta(*mut DFState, f32)
//!
//! RUNTIME REQUIREMENTS (Optional):
//! The application bootstrap passes canonical bundled library and model paths
//! directly to Rust. Ambient `DEEPFILTER_*` environment paths are ignored unless
//! `AUDIOFORGE_ALLOW_EXTERNAL_DF=1` explicitly opts into external assets.
//!
//! MODEL FILES (Optional):
//! DeepFilterNet requires a model tar.gz file supplied by the application
//! bootstrap, or by `DEEPFILTER_MODEL_PATH` under the external-assets opt-in.
//!
//! If the library or model is not found, the processor will use passthrough mode.
//!
//! Expected latency: ~10ms with LL variant (no lookahead)

use crate::audio::rt::FixedAudioRing;
use crate::dsp::noise_suppressor::{NoiseModel, NoiseSuppressor};
use std::env;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::sync::{Mutex, OnceLock};
use std::vec::Vec;

/// DeepFilterNet frame size (same as RNNoise: 10ms at 48kHz)
pub const DEEPFILTER_FRAME_SIZE: usize = 480;
const DEEPFILTER_BUFFER_CAPACITY: usize = 8192 + DEEPFILTER_FRAME_SIZE;

fn deepfilter_runtime_enabled() -> bool {
    env::var("AUDIOFORGE_ENABLE_DEEPFILTER")
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            normalized == "1" || normalized == "true" || normalized == "yes"
        })
        .unwrap_or(false)
}

fn external_deepfilter_paths_allowed() -> bool {
    env::var("AUDIOFORGE_ALLOW_EXTERNAL_DF")
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on"
        })
        .unwrap_or(false)
}

#[derive(Clone, Debug, Default)]
struct AppOwnedDeepFilterPaths {
    library: Option<PathBuf>,
    model: Option<PathBuf>,
}

fn app_owned_paths() -> &'static Mutex<AppOwnedDeepFilterPaths> {
    static PATHS: OnceLock<Mutex<AppOwnedDeepFilterPaths>> = OnceLock::new();
    PATHS.get_or_init(|| Mutex::new(AppOwnedDeepFilterPaths::default()))
}

fn canonical_app_owned_path(path: Option<&str>, kind: &str) -> Result<Option<PathBuf>, String> {
    let Some(path) = path else {
        return Ok(None);
    };
    let canonical = Path::new(path)
        .canonicalize()
        .map_err(|error| format!("Invalid app-owned DeepFilter {kind} path: {error}"))?;
    if !canonical.exists() {
        return Err(format!(
            "App-owned DeepFilter {kind} path does not exist: {}",
            canonical.display()
        ));
    }
    Ok(Some(canonical))
}

/// Configure bundled DeepFilter assets discovered by the application bootstrap.
///
/// These paths are intentionally separate from the external override environment
/// variables. Ambient `DEEPFILTER_*` paths are ignored unless the caller also sets
/// `AUDIOFORGE_ALLOW_EXTERNAL_DF=1`.
pub fn configure_app_owned_paths(
    library_path: Option<&str>,
    model_path: Option<&str>,
) -> Result<(), String> {
    let configured = AppOwnedDeepFilterPaths {
        library: canonical_app_owned_path(library_path, "library")?,
        model: canonical_app_owned_path(model_path, "model")?,
    };
    let mut paths = app_owned_paths()
        .lock()
        .map_err(|_| "App-owned DeepFilter path state is unavailable".to_string())?;
    *paths = configured;
    Ok(())
}

fn configured_app_owned_paths() -> AppOwnedDeepFilterPaths {
    app_owned_paths()
        .lock()
        .map(|paths| paths.clone())
        .unwrap_or_default()
}

// ============================================================================
// C FFI BINDINGS (Runtime dynamic loading)
// ============================================================================

/// Opaque C struct representing DeepFilterNet state
#[repr(C)]
pub struct DFState {
    _private: [u8; 0], // Zero-sized opaque type
}

// FFI function pointers (stored as raw pointers for Send safety)
type DfCreateFn =
    unsafe extern "C" fn(path: *const i8, atten_lim: f32, log_level: *const i8) -> *mut DFState;

type DfFreeFn = unsafe extern "C" fn(*mut DFState);

type DfGetFrameLengthFn = unsafe extern "C" fn(*mut DFState) -> usize;

type DfProcessFrameFn =
    unsafe extern "C" fn(st: *mut DFState, input: *mut f32, output: *mut f32) -> f32;

type DfSetAttenLimFn = unsafe extern "C" fn(*mut DFState, f32);

type DfSetPostFilterBetaFn = unsafe extern "C" fn(*mut DFState, f32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeepFilterProcessError {
    InputSizeMismatch,
    OutputSizeMismatch,
    NonFiniteSnr,
    NonFiniteOutput,
}

impl DeepFilterProcessError {
    fn as_str(self) -> &'static str {
        match self {
            Self::InputSizeMismatch => "DeepFilterNet input buffer size mismatch",
            Self::OutputSizeMismatch => "DeepFilterNet output buffer size mismatch",
            Self::NonFiniteSnr => "DeepFilterNet processing failed with non-finite SNR",
            Self::NonFiniteOutput => "DeepFilterNet processing produced non-finite output",
        }
    }
}

/// Dynamically loaded DeepFilterNet library symbols
struct DeepFilterLib {
    _library: libloading::Library,
    df_create: DfCreateFn,
    df_free: DfFreeFn,
    df_get_frame_length: DfGetFrameLengthFn,
    df_process_frame: DfProcessFrameFn,
    df_set_atten_lim: DfSetAttenLimFn,
    df_set_post_filter_beta: DfSetPostFilterBetaFn,
}

/// DeepFilterNet model variant
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepFilterModel {
    /// Low Latency variant (~10ms, no lookahead)
    LowLatency,
    /// Standard variant (~40ms, 2-frame lookahead, best quality)
    Standard,
}

impl DeepFilterModel {
    /// Get model filename for this variant
    pub fn filename(&self) -> &'static str {
        match self {
            DeepFilterModel::LowLatency => "DeepFilterNet3_ll_onnx.tar.gz",
            DeepFilterModel::Standard => "DeepFilterNet3_onnx.tar.gz",
        }
    }
}

/// Resolve a DeepFilterNet model from app-owned paths or an opted-in override.
fn find_model_path(
    model: DeepFilterModel,
    app_paths: &AppOwnedDeepFilterPaths,
    allow_external: bool,
) -> Option<PathBuf> {
    let filename = model.filename();

    // 1. Honor an explicit external override only after the separate opt-in.
    if allow_external {
        if let Ok(path) = env::var("DEEPFILTER_MODEL_PATH") {
            let path_buf = PathBuf::from(&path);
            if path_buf.is_file() {
                return Some(path_buf);
            }
            let candidate = path_buf.join(filename);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }

    // 2. Otherwise prefer the app-owned bundled model supplied by bootstrap.
    if let Some(path) = app_paths.model.as_ref() {
        if path.is_file() {
            return Some(path.clone());
        }
        let candidate = path.join(filename);
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    // 3. Dev convenience is anchored to the compiled manifest, never the CWD.
    #[cfg(debug_assertions)]
    if let Some(root) = Path::new(env!("CARGO_MANIFEST_DIR")).parent() {
        let candidate = root.join("models").join(filename);
        if candidate.is_file() {
            return candidate.canonicalize().ok();
        }
    }

    // 4. User data is an external location and therefore requires opt-in.
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        if allow_external {
            if let Ok(home) = env::var("HOME") {
                let user_model =
                    PathBuf::from(format!("{}/.local/share/deepfilter/{}", home, filename));
                if user_model.exists() {
                    return Some(user_model);
                }
            }
        }
    }

    None
}

impl DeepFilterLib {
    fn add_canonical_file_path(paths: &mut Vec<PathBuf>, path: PathBuf) {
        if let Ok(canonical) = path.canonicalize() {
            if canonical.is_file() && !paths.iter().any(|p| p == &canonical) {
                paths.push(canonical);
            }
        }
    }

    fn trusted_library_candidates(
        _lib_name: &str,
        app_paths: &AppOwnedDeepFilterPaths,
        allow_external: bool,
    ) -> Vec<PathBuf> {
        let mut candidates = Vec::new();

        // External override is lower trust and requires an explicit opt-in.
        if allow_external {
            if let Ok(lib_path) = env::var("DEEPFILTER_LIB_PATH") {
                Self::add_canonical_file_path(&mut candidates, PathBuf::from(lib_path));
            }
        }

        // App-owned bundled path supplied by the bootstrap is the safe default.
        if let Some(lib_path) = app_paths.library.as_ref() {
            Self::add_canonical_file_path(&mut candidates, lib_path.clone());
        }

        // Dev convenience in debug builds: repo-local copy adjacent to rust-core.
        #[cfg(debug_assertions)]
        {
            let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent();
            if let Some(root) = repo_root {
                Self::add_canonical_file_path(&mut candidates, root.join(_lib_name));
            }
        }

        candidates
    }

    unsafe fn load_symbols_from_path(path: &Path) -> Result<Self, String> {
        let library =
            libloading::Library::new(path).map_err(|e| format!("{} ({})", path.display(), e))?;

        // Load all required symbols and extract raw pointers.
        let df_create: libloading::Symbol<DfCreateFn> = library
            .get(b"df_create")
            .map_err(|e| format!("{} missing symbol df_create ({})", path.display(), e))?;
        let df_free: libloading::Symbol<DfFreeFn> = library
            .get(b"df_free")
            .map_err(|e| format!("{} missing symbol df_free ({})", path.display(), e))?;
        let df_get_frame_length: libloading::Symbol<DfGetFrameLengthFn> =
            library.get(b"df_get_frame_length").map_err(|e| {
                format!(
                    "{} missing symbol df_get_frame_length ({})",
                    path.display(),
                    e
                )
            })?;
        let df_process_frame: libloading::Symbol<DfProcessFrameFn> = library
            .get(b"df_process_frame")
            .map_err(|e| format!("{} missing symbol df_process_frame ({})", path.display(), e))?;
        let df_set_atten_lim: libloading::Symbol<DfSetAttenLimFn> = library
            .get(b"df_set_atten_lim")
            .map_err(|e| format!("{} missing symbol df_set_atten_lim ({})", path.display(), e))?;
        let df_set_post_filter_beta: libloading::Symbol<DfSetPostFilterBetaFn> =
            library.get(b"df_set_post_filter_beta").map_err(|e| {
                format!(
                    "{} missing symbol df_set_post_filter_beta ({})",
                    path.display(),
                    e
                )
            })?;

        // Convert to raw function pointers.
        let df_create = *df_create.into_raw();
        let df_free = *df_free.into_raw();
        let df_get_frame_length = *df_get_frame_length.into_raw();
        let df_process_frame = *df_process_frame.into_raw();
        let df_set_atten_lim = *df_set_atten_lim.into_raw();
        let df_set_post_filter_beta = *df_set_post_filter_beta.into_raw();

        Ok(DeepFilterLib {
            _library: library,
            df_create,
            df_free,
            df_get_frame_length,
            df_process_frame,
            df_set_atten_lim,
            df_set_post_filter_beta,
        })
    }

    /// Try to load the DeepFilterNet library from trusted explicit paths.
    fn try_load(app_paths: &AppOwnedDeepFilterPaths, allow_external: bool) -> Result<Self, String> {
        // Library name varies by platform
        #[cfg(target_os = "windows")]
        let lib_name = "df.dll";

        #[cfg(target_os = "linux")]
        let lib_name = "libdf.so";

        #[cfg(target_os = "macos")]
        let lib_name = "libdf.dylib";

        let candidates = Self::trusted_library_candidates(lib_name, app_paths, allow_external);
        if candidates.is_empty() {
            return Err(format!(
                "No trusted DeepFilter library file found. Bundle {} with the app, or set \
AUDIOFORGE_ALLOW_EXTERNAL_DF=1 together with DEEPFILTER_LIB_PATH.",
                lib_name
            ));
        }

        let mut errors = Vec::new();
        for path in candidates {
            // Canonical file candidates come only from bootstrap registration,
            // opted-in external paths, or the compile-time dev root.
            // nosemgrep: rust.lang.security.unsafe-usage.unsafe-usage
            unsafe {
                match Self::load_symbols_from_path(&path) {
                    Ok(lib) => return Ok(lib),
                    Err(e) => errors.push(e),
                }
            }
        }

        Err(format!(
            "Failed to load DeepFilter library from trusted paths: {}",
            errors.join("; ")
        ))
    }
}

// ============================================================================
// FFI WRAPPER (Send-safe)
// ============================================================================

/// Send-safe wrapper for DeepFilterNet C API
///
/// # Safety
/// This struct is safe to send across threads because:
/// 1. The underlying C library (DeepFilterNet) uses thread-local state only
/// 2. We ensure exclusive access through &mut self references
/// 3. The C pointer is opaque and managed through create/free
///
/// # Thread Safety
/// While DeepFilterNet's C API itself may not be fully thread-safe,
/// our wrapper ensures that only one thread can access the instance at a time
/// through Rust's borrowing rules (&mut self).
pub struct DeepFilterFFI {
    _lib: Arc<DeepFilterLib>, // Keep library loaded
    state: NonNull<DFState>,
    frame_size: usize,
}

// SAFETY: Access to the opaque C state remains serialized through `&mut self`,
// and the owning library handle is kept alive for the lifetime of the state.
unsafe impl Send for DeepFilterFFI {}

impl DeepFilterFFI {
    #[inline]
    fn state_ptr(&self) -> *mut DFState {
        self.state.as_ptr()
    }

    /// Create a new DeepFilterNet instance using model from file system
    ///
    /// # Safety
    /// This function calls unsafe FFI functions to create the DeepFilterNet instance.
    /// It validates the returned pointer and returns an error if creation failed.
    fn new(
        lib: Arc<DeepFilterLib>,
        model: DeepFilterModel,
        app_paths: &AppOwnedDeepFilterPaths,
        allow_external: bool,
    ) -> Result<Self, String> {
        // Find model path
        let model_path = find_model_path(model, app_paths, allow_external)
            .ok_or_else(|| {
                format!("DeepFilterNet model file not found. Bundle {} with the app, or opt in to an external DEEPFILTER_MODEL_PATH", model.filename())
            })?;

        // Convert path to C string
        let model_path_cstr = std::ffi::CString::new(
            model_path
                .to_str()
                .ok_or("Model path contains invalid UTF-8")?,
        )
        .map_err(|e| format!("Failed to create model path CString: {}", e))?;

        // Default attenuation limit: -80 dB (max suppression)
        let atten_lim = -80.0f32;

        // The symbol signature was checked during loading; the model path is a
        // live CString, and a null state is rejected before any further call.
        // nosemgrep: rust.lang.security.unsafe-usage.unsafe-usage
        unsafe {
            let df_create = lib.df_create;
            let ptr = df_create(model_path_cstr.as_ptr(), atten_lim, std::ptr::null());
            let state = NonNull::new(ptr).ok_or_else(|| {
                format!(
                    "Failed to create DeepFilterNet instance with model: {}",
                    model_path.display()
                )
            })?;

            let df_get_frame_length = lib.df_get_frame_length;
            let frame_size = df_get_frame_length(state.as_ptr());

            if frame_size != DEEPFILTER_FRAME_SIZE {
                let df_free = lib.df_free;
                df_free(state.as_ptr());
                return Err(format!(
                    "DeepFilterNet frame size mismatch: expected {}, got {}",
                    DEEPFILTER_FRAME_SIZE, frame_size
                ));
            }

            Ok(Self {
                _lib: lib,
                state,
                frame_size,
            })
        }
    }

    /// Get frame size
    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    /// Process a frame of audio samples
    ///
    /// # Arguments
    /// * `input` - Input slice (must be exactly frame_size samples)
    ///
    /// # Returns
    /// * Local SNR estimate in dB (for quality metering)
    ///
    /// # Safety
    /// This function calls unsafe FFI functions.
    /// It ensures buffers are properly sized and aligned before passing to C.
    fn process_into(
        &mut self,
        input: &mut [f32],
        output: &mut [f32],
    ) -> Result<f32, DeepFilterProcessError> {
        if input.len() != self.frame_size {
            return Err(DeepFilterProcessError::InputSizeMismatch);
        }
        if output.len() != self.frame_size {
            return Err(DeepFilterProcessError::OutputSizeMismatch);
        }

        // SAFETY: Buffer sizes match the backend frame size, the state is valid
        // and exclusively borrowed, and the library remains alive in `_lib`.
        // nosemgrep: rust.lang.security.unsafe-usage.unsafe-usage
        unsafe {
            let input_ptr = input.as_mut_ptr();
            let output_ptr = output.as_mut_ptr();
            let state_ptr = self.state_ptr();

            let df_process_frame = self._lib.df_process_frame;
            let lsnr = df_process_frame(state_ptr, input_ptr, output_ptr);

            if !lsnr.is_finite() {
                return Err(DeepFilterProcessError::NonFiniteSnr);
            }

            if output.iter().any(|sample| !sample.is_finite()) {
                return Err(DeepFilterProcessError::NonFiniteOutput);
            }

            Ok(lsnr)
        }
    }

    /// Set attenuation limit (dB)
    pub fn set_atten_lim(&mut self, lim_db: f32) {
        // SAFETY: `self.state` is non-null, exclusively borrowed, and `_lib`
        // keeps the validated function pointer's library loaded.
        // nosemgrep: rust.lang.security.unsafe-usage.unsafe-usage
        unsafe {
            let df_set_atten_lim = self._lib.df_set_atten_lim;
            df_set_atten_lim(self.state_ptr(), lim_db);
        }
    }

    /// Set post filter beta
    pub fn set_post_filter_beta(&mut self, beta: f32) {
        // SAFETY: The state and function pointer have the same lifetime and the
        // mutable borrow serializes access to the opaque C state.
        // nosemgrep: rust.lang.security.unsafe-usage.unsafe-usage
        unsafe {
            let df_set_post_filter_beta = self._lib.df_set_post_filter_beta;
            df_set_post_filter_beta(self.state_ptr(), beta);
        }
    }
}

impl Drop for DeepFilterFFI {
    fn drop(&mut self) {
        // SAFETY: The non-null state was returned by `df_create`, is freed once,
        // and the owning library handle is still alive during field drop.
        // nosemgrep: rust.lang.security.unsafe-usage.unsafe-usage
        unsafe {
            let df_free = self._lib.df_free;
            df_free(self.state.as_ptr());
        }
    }
}

// ============================================================================
// PROCESSOR IMPLEMENTATION
// ============================================================================

pub struct DeepFilterProcessor {
    df: Option<DeepFilterFFI>, // Option for graceful fallback if FFI fails
    _lib: Option<Arc<DeepFilterLib>>, // Keep library loaded
    input_buffer: FixedAudioRing<f32, DEEPFILTER_BUFFER_CAPACITY>,
    output_buffer: FixedAudioRing<f32, DEEPFILTER_BUFFER_CAPACITY>,
    enabled: bool,
    strength: Arc<AtomicU32>,
    smoothed_strength: f32,
    dry_frame: [f32; DEEPFILTER_FRAME_SIZE],
    frame_scratch: [f32; DEEPFILTER_FRAME_SIZE],
    output_frame: [f32; DEEPFILTER_FRAME_SIZE],
    load_error: Option<String>, // Store load error for reporting
    model: DeepFilterModel,     // Track which model variant we're using
    backend_failed: bool,
    runtime_error: Option<DeepFilterProcessError>,
}

impl DeepFilterProcessor {
    pub fn new(strength: Arc<AtomicU32>, model: DeepFilterModel) -> Self {
        if !deepfilter_runtime_enabled() {
            return Self {
                df: None,
                _lib: None,
                input_buffer: FixedAudioRing::new(),
                output_buffer: FixedAudioRing::new(),
                enabled: true,
                strength,
                smoothed_strength: 1.0,
                dry_frame: [0.0; DEEPFILTER_FRAME_SIZE],
                frame_scratch: [0.0; DEEPFILTER_FRAME_SIZE],
                output_frame: [0.0; DEEPFILTER_FRAME_SIZE],
                load_error: Some(
                    "DeepFilterNet disabled; set AUDIOFORGE_ENABLE_DEEPFILTER=1 to enable"
                        .to_string(),
                ),
                model,
                backend_failed: false,
                runtime_error: None,
            };
        }

        // Snapshot bootstrap-owned paths before initialization. This happens
        // outside the realtime processing loop.
        let app_paths = configured_app_owned_paths();
        let allow_external = external_deepfilter_paths_allowed();

        // Try to load library and initialize FFI
        let (df, lib, load_error) = match DeepFilterLib::try_load(&app_paths, allow_external) {
            Ok(lib) => {
                let lib_arc = Arc::new(lib);
                match DeepFilterFFI::new(lib_arc.clone(), model, &app_paths, allow_external) {
                    Ok(df) => {
                        let latency_str = match model {
                            DeepFilterModel::LowLatency => "~10ms",
                            DeepFilterModel::Standard => "~40ms",
                        };
                        eprintln!(
                            "DeepFilterNet initialized (C FFI MODE - {} variant, {} latency)",
                            if matches!(model, DeepFilterModel::LowLatency) {
                                "Low Latency"
                            } else {
                                "Standard"
                            },
                            latency_str
                        );
                        (Some(df), Some(lib_arc), None)
                    }
                    Err(e) => {
                        eprintln!(
                            "DeepFilterNet C FFI initialization failed: {}. Using passthrough fallback.",
                            e
                        );
                        (
                            None,
                            Some(lib_arc),
                            Some(format!("DeepFilterNet init failed: {}", e)),
                        )
                    }
                }
            }
            Err(e) => {
                eprintln!(
                    "DeepFilterNet C library unavailable ({}). Using passthrough fallback.",
                    e
                );
                eprintln!("NOTE: To use DeepFilterNet, set DEEPFILTER_LIB_PATH to an explicit library file path,");
                eprintln!("      or let the application bootstrap register bundled assets.");
                eprintln!("NOTE: Also ensure DeepFilterNet3 model file is available:");
                eprintln!("  - Set DEEPFILTER_MODEL_PATH environment variable");
                eprintln!("  - Or place in ./models/DeepFilterNet3_ll_onnx.tar.gz or DeepFilterNet3_onnx.tar.gz");
                (
                    None,
                    None,
                    Some(format!("DeepFilterNet C library unavailable: {}", e)),
                )
            }
        };

        Self {
            df,
            _lib: lib,
            input_buffer: FixedAudioRing::new(),
            output_buffer: FixedAudioRing::new(),
            enabled: true,
            strength,
            smoothed_strength: 1.0,
            dry_frame: [0.0; DEEPFILTER_FRAME_SIZE],
            frame_scratch: [0.0; DEEPFILTER_FRAME_SIZE],
            output_frame: [0.0; DEEPFILTER_FRAME_SIZE],
            load_error,
            model,
            backend_failed: false,
            runtime_error: None,
        }
    }

    fn mark_backend_failed_rt(&mut self, error: DeepFilterProcessError) {
        self.backend_failed = true;
        self.runtime_error = Some(error);
    }

    /// Process frames through FFI or fallback
    pub fn process_frames_internal(&mut self) {
        // Update smoothed strength with 15ms EMA
        let target_strength = f32::from_bits(self.strength.load(Ordering::Relaxed));
        const TAU_MS: f32 = 15.0;
        const SAMPLE_RATE: f32 = 48000.0;
        let alpha =
            1.0 - (-1.0 / (TAU_MS / 1000.0 * SAMPLE_RATE / DEEPFILTER_FRAME_SIZE as f32)).exp();
        self.smoothed_strength += alpha * (target_strength - self.smoothed_strength);

        // Process complete frames (480 samples each)
        while self.input_buffer.len() >= DEEPFILTER_FRAME_SIZE
            && self.output_buffer.remaining() >= DEEPFILTER_FRAME_SIZE
        {
            let read = self.input_buffer.pop_into(&mut self.dry_frame);
            if read != DEEPFILTER_FRAME_SIZE {
                break;
            }
            self.frame_scratch.copy_from_slice(&self.dry_frame);

            // Process through FFI if available
            if self.enabled && !self.backend_failed {
                if let Some(ref mut df) = self.df {
                    match df.process_into(&mut self.frame_scratch, &mut self.output_frame) {
                        Ok(_lsnr) => {
                            // Apply wet/dry mix to FFI output
                            for i in 0..DEEPFILTER_FRAME_SIZE {
                                let wet = self.output_frame[i];
                                let dry = self.dry_frame[i];
                                self.output_frame[i] = wet * self.smoothed_strength
                                    + dry * (1.0 - self.smoothed_strength);
                            }
                            self.output_buffer.push_slice(&self.output_frame);
                            continue;
                        }
                        Err(error) => {
                            self.mark_backend_failed_rt(error);
                        }
                    }
                }
            }

            // Passthrough (fallback or disabled)
            for i in 0..DEEPFILTER_FRAME_SIZE {
                let wet = self.dry_frame[i];
                let dry = self.dry_frame[i];
                self.output_frame[i] =
                    wet * self.smoothed_strength + dry * (1.0 - self.smoothed_strength);
            }
            self.output_buffer.push_slice(&self.output_frame);
        }
    }

    /// Check if FFI is available (not in passthrough fallback mode)
    pub fn is_ffi_available(&self) -> bool {
        self.df.is_some() && !self.backend_failed
    }

    /// Get load/runtime error if the backend is unavailable.
    pub fn backend_error(&self) -> Option<&str> {
        self.load_error
            .as_deref()
            .or_else(|| self.runtime_error.map(DeepFilterProcessError::as_str))
    }

    pub fn backend_failed(&self) -> bool {
        self.backend_failed
    }
}

// ============================================================================
// TRAIT IMPLEMENTATION
// ============================================================================

impl NoiseSuppressor for DeepFilterProcessor {
    fn push_samples(&mut self, samples: &[f32]) -> usize {
        self.input_buffer.push_slice(samples)
    }

    fn process_frames(&mut self) {
        if self.enabled {
            self.process_frames_internal();
        } else {
            // Disabled: passthrough
            while self.input_buffer.len() >= DEEPFILTER_FRAME_SIZE
                && self.output_buffer.remaining() >= DEEPFILTER_FRAME_SIZE
            {
                let read = self.input_buffer.pop_into(&mut self.dry_frame);
                if read != DEEPFILTER_FRAME_SIZE {
                    break;
                }
                self.output_buffer.push_slice(&self.dry_frame);
            }
        }
    }

    fn available_samples(&self) -> usize {
        self.output_buffer.len()
    }

    fn pop_samples(&mut self, count: usize) -> Vec<f32> {
        let actual = count.min(self.available_samples());
        let mut out = vec![0.0; actual];
        self.output_buffer.pop_into(&mut out);
        out
    }

    fn pop_samples_into(&mut self, buffer: &mut [f32]) -> usize {
        let count = buffer.len().min(self.available_samples());
        self.output_buffer.pop_into(&mut buffer[..count])
    }

    fn pop_all_samples(&mut self) -> Vec<f32> {
        self.output_buffer.pop_all_vec()
    }

    fn set_strength(&self, value: f32) {
        let clamped = value.clamp(0.0, 1.0);
        let bits = clamped.to_bits();
        self.strength.store(bits, Ordering::Relaxed);
    }

    fn get_strength(&self) -> f32 {
        f32::from_bits(self.strength.load(Ordering::Relaxed))
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn soft_reset(&mut self) {
        // Clear all buffers without resetting the model/state
        self.input_buffer.clear();
        self.output_buffer.clear();
        self.dry_frame.fill(0.0);
        self.frame_scratch.fill(0.0);
        self.output_frame.fill(0.0);
        // Keep smoothed_strength to avoid zipper noise
    }

    fn pending_input(&self) -> usize {
        self.input_buffer.len()
    }

    fn drain_pending_input(&mut self) -> Vec<f32> {
        self.input_buffer.pop_all_vec()
    }

    fn model_type(&self) -> NoiseModel {
        match self.model {
            DeepFilterModel::LowLatency => NoiseModel::DeepFilterNetLL,
            DeepFilterModel::Standard => NoiseModel::DeepFilterNet,
        }
    }

    fn latency_samples(&self) -> usize {
        match self.model {
            // LL has no lookahead: just frame size
            DeepFilterModel::LowLatency => DEEPFILTER_FRAME_SIZE,
            // Standard has 2-frame lookahead: frame size + 2 * frame size = 3 * frame size
            DeepFilterModel::Standard => DEEPFILTER_FRAME_SIZE * 3,
        }
    }
}

impl Default for DeepFilterProcessor {
    fn default() -> Self {
        Self::new(
            Arc::new(AtomicU32::new(1.0_f32.to_bits())),
            DeepFilterModel::LowLatency,
        ) // 100% strength default, LL model
    }
}

// ============================================================================
// TESTS
// ============================================================================

include!("deepfilter_ffi/tests.rs");
