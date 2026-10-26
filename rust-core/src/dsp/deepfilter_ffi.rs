//! DeepFilterNet integration via C FFI (bypasses Rc/Send issues)
//!
//! This module uses DeepFilterNet's C API directly via FFI, which:
//! - Avoids the Rc/Send issues with the Rust libDF crate
//! - Provides a stable ABI through C interface
//! - Uses runtime dynamic loading (no compile-time library dependency)
//! - Falls back to passthrough, with the reason in `backend_error`, when the
//!   library or model is unavailable, the model is not the pinned one,
//!   `df_create` returns null or a frame fails.
//!
//! WHAT THE HOST CAN AND CANNOT CATCH:
//! A panic inside the library cannot unwind across the C ABI, so no
//! `catch_unwind` on this side can see it: it ends the process. The upstream
//! C API panics (`expect`/`unwrap`) when a model cannot be loaded or a frame
//! cannot be processed. The host therefore keeps such input from reaching it:
//! the model file must be the pinned one (`identity`), and the audio handed to
//! `df_process_frame` is finite. The library built from this repository
//! (`build-support/deepfilter/capi.rs`) catches its own panics and answers a
//! null state (`df_create`), a zero frame length or a NaN SNR
//! (`df_process_frame`) instead; each falls back to passthrough with the
//! reason in `backend_error`.
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
//! If the library or model is not found, the processor uses passthrough mode.
//!
//! Expected latency: ~10ms with LL variant (no lookahead)

mod identity;

use crate::audio::rt::FixedAudioRing;
use crate::dsp::noise_suppressor::{
    ControlledNoiseSuppressor, ControlledSample, GateCompensation, GateControl, NoiseModel,
    NoiseSuppressor, GATE_COMPENSATION_MAX_DELAY,
};
use crate::dsp::util::unit_or;
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
const DEEPFILTER_MAX_LATENCY_SAMPLES: usize = DEEPFILTER_FRAME_SIZE * 3;
const _: () = assert!(DEEPFILTER_MAX_LATENCY_SAMPLES <= GATE_COMPENSATION_MAX_DELAY);
pub const DEFAULT_DEEPFILTER_ATTENUATION_LIMIT_DB: f32 = 30.0;
pub const DEFAULT_DEEPFILTER_POST_FILTER_BETA: f32 = 0.0;

#[cfg(test)]
thread_local! {
    pub(crate) static TEST_FAIL_PROCESS_AFTER: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeepFilterRuntimeConfig {
    attenuation_limit_db: f32,
    post_filter_beta: f32,
}

impl DeepFilterRuntimeConfig {
    pub fn try_new(attenuation_limit_db: f32, post_filter_beta: f32) -> Result<Self, String> {
        if !attenuation_limit_db.is_finite() || !(0.01..=100.0).contains(&attenuation_limit_db) {
            return Err(
                "DeepFilter attenuation limit must be finite and between 0.01 and 100 dB"
                    .to_string(),
            );
        }
        if !post_filter_beta.is_finite() || !(0.0..=0.05).contains(&post_filter_beta) {
            return Err(
                "DeepFilter post-filter beta must be finite and between 0 and 0.05".to_string(),
            );
        }
        Ok(Self {
            attenuation_limit_db,
            post_filter_beta,
        })
    }

    pub fn attenuation_limit_db(self) -> f32 {
        self.attenuation_limit_db
    }

    pub fn post_filter_beta(self) -> f32 {
        self.post_filter_beta
    }
}

impl Default for DeepFilterRuntimeConfig {
    fn default() -> Self {
        Self {
            attenuation_limit_db: DEFAULT_DEEPFILTER_ATTENUATION_LIMIT_DB,
            post_filter_beta: DEFAULT_DEEPFILTER_POST_FILTER_BETA,
        }
    }
}

fn parse_external_opt_in(value: &str) -> bool {
    let normalized = value.trim().to_ascii_lowercase();
    normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on"
}

fn external_deepfilter_paths_allowed() -> bool {
    env::var("AUDIOFORGE_ALLOW_EXTERNAL_DF")
        .map(|value| parse_external_opt_in(&value))
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
    /// `None` only for the stand-in libraries the tests build.
    _library: Option<libloading::Library>,
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
    /// Standard variant (~30ms, 2-frame lookahead, best quality)
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

    /// Size and SHA-256 of the pinned model file (release-assets.json).
    fn identity(self) -> identity::ModelIdentity {
        match self {
            DeepFilterModel::LowLatency => identity::ModelIdentity {
                size: 36_359_660,
                sha256: "5998e58e8ba0e09bb76986ef97b84afa065a571ef282d4a1222f341e3251cf3a",
            },
            DeepFilterModel::Standard => identity::ModelIdentity {
                size: 7_983_136,
                sha256: "c94d91f70911001c946e0fabb4aa9adc37045f45a03b56008cb0c8244cb63616",
            },
        }
    }

    pub fn latency_samples(self) -> usize {
        match self {
            // The 960-point STFT with a 480-sample hop contributes one frame.
            DeepFilterModel::LowLatency => DEEPFILTER_FRAME_SIZE,
            // The standard model adds two frames of convolution/DF lookahead.
            DeepFilterModel::Standard => DEEPFILTER_FRAME_SIZE * 3,
        }
    }
}

/// Resolve a DeepFilterNet model from app-owned paths or an opted-in override.
///
/// `external_model` is the caller's `DEEPFILTER_MODEL_PATH`; it is passed in so
/// tests need not mutate the process environment.
fn find_model_path(
    model: DeepFilterModel,
    app_paths: &AppOwnedDeepFilterPaths,
    allow_external: bool,
    external_model: Option<&Path>,
) -> Option<PathBuf> {
    let filename = model.filename();

    // 1. Honor an explicit external override only after the separate opt-in.
    if let Some(path_buf) = external_model.filter(|_| allow_external) {
        if path_buf.is_file() {
            return Some(path_buf.to_path_buf());
        }
        let candidate = path_buf.join(filename);
        if candidate.is_file() {
            return Some(candidate);
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

/// Open the library without the default Windows DLL search order, which would
/// resolve the library's own imports from the process CWD and `PATH`. Only its
/// own directory and the application, System32 and registered user directories
/// are searched.
#[cfg(windows)]
unsafe fn open_library(path: &Path) -> Result<libloading::Library, libloading::Error> {
    use libloading::os::windows::{
        Library, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR,
    };
    Library::load_with_flags(
        path,
        LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR,
    )
    .map(Into::into)
}

#[cfg(not(windows))]
unsafe fn open_library(path: &Path) -> Result<libloading::Library, libloading::Error> {
    libloading::Library::new(path)
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
        external_library: Option<&Path>,
    ) -> Vec<PathBuf> {
        let mut candidates = Vec::new();

        // External override is lower trust and requires an explicit opt-in.
        if let Some(lib_path) = external_library.filter(|_| allow_external) {
            Self::add_canonical_file_path(&mut candidates, lib_path.to_path_buf());
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
        let library = open_library(path).map_err(|e| format!("{} ({})", path.display(), e))?;

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
            _library: Some(library),
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

        let external_library = env::var_os("DEEPFILTER_LIB_PATH").map(PathBuf::from);
        let candidates = Self::trusted_library_candidates(
            lib_name,
            app_paths,
            allow_external,
            external_library.as_deref(),
        );
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
/// 1. The C state is owned by exactly one wrapper and is only touched through
///    `&mut self`, so it is never accessed from two threads at once
/// 2. The state is created, used and freed as a whole, and moving it between
///    threads relies on the pinned library keeping no per-thread state inside
///    it (re-check this when the DeepFilterNet revision changes)
/// 3. The C pointer is opaque and managed through create/free
///
/// # Thread Safety
/// DeepFilterNet's C API is not documented as thread-safe, so the wrapper
/// allows one thread at a time through Rust's borrowing rules (&mut self).
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

    /// Find the model file and create a DeepFilterNet instance from it.
    ///
    /// A model chosen through the external override is the user's own
    /// responsibility; every other model must be the pinned one.
    fn new(
        lib: Arc<DeepFilterLib>,
        model: DeepFilterModel,
        config: DeepFilterRuntimeConfig,
        app_paths: &AppOwnedDeepFilterPaths,
        allow_external: bool,
        external_model: Option<&Path>,
    ) -> Result<Self, String> {
        let model_path = find_model_path(model, app_paths, allow_external, external_model)
            .ok_or_else(|| {
                format!("DeepFilterNet model file not found. Bundle {} with the app, or opt in to an external DEEPFILTER_MODEL_PATH", model.filename())
            })?;
        let chosen_externally = allow_external
            && external_model.is_some_and(|external| {
                model_path == external || model_path == external.join(model.filename())
            });
        let identity = (!chosen_externally).then(|| model.identity());
        Self::open(lib, config, &model_path, identity)
    }

    /// Create a new DeepFilterNet instance from the model at `model_path`,
    /// which must match `identity` when one is given.
    ///
    /// # Safety
    /// This function calls unsafe FFI functions to create the DeepFilterNet instance.
    /// It validates the returned pointer and returns an error if creation failed.
    fn open(
        lib: Arc<DeepFilterLib>,
        config: DeepFilterRuntimeConfig,
        model_path: &Path,
        identity: Option<identity::ModelIdentity>,
    ) -> Result<Self, String> {
        // The library aborts the process on a model it cannot load.
        if let Some(identity) = identity {
            identity::verify(model_path, &identity)?;
        }

        // Convert path to C string
        let model_path_cstr = std::ffi::CString::new(
            model_path
                .to_str()
                .ok_or("Model path contains invalid UTF-8")?,
        )
        .map_err(|e| format!("Failed to create model path CString: {}", e))?;

        // The symbol signature was checked during loading; the model path is a
        // live CString, and a null state is rejected before any further call.
        // nosemgrep: rust.lang.security.unsafe-usage.unsafe-usage
        unsafe {
            let df_create = lib.df_create;
            let ptr = df_create(
                model_path_cstr.as_ptr(),
                config.attenuation_limit_db(),
                std::ptr::null(),
            );
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

            let mut wrapper = Self {
                _lib: lib,
                state,
                frame_size,
            };
            // Apply both runtime controls explicitly so construction and future
            // reconfiguration use the same C API semantics.
            wrapper.set_atten_lim(config.attenuation_limit_db());
            wrapper.set_post_filter_beta(config.post_filter_beta());
            Ok(wrapper)
        }
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

        // The library aborts the process on a frame it cannot process.
        for sample in input.iter_mut().filter(|sample| !sample.is_finite()) {
            *sample = 0.0;
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
            #[cfg(test)]
            let lsnr = TEST_FAIL_PROCESS_AFTER.with(|remaining| match remaining.get() {
                Some(0) => {
                    remaining.set(None);
                    f32::NAN
                }
                Some(count) => {
                    remaining.set(Some(count - 1));
                    lsnr
                }
                None => lsnr,
            });

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

#[inline]
fn mix_wet_with_aligned_dry(
    wet: &mut [f32; DEEPFILTER_FRAME_SIZE],
    aligned_dry: &[f32; DEEPFILTER_FRAME_SIZE],
    strength: f32,
) {
    for (wet_sample, dry_sample) in wet.iter_mut().zip(aligned_dry.iter().copied()) {
        *wet_sample = *wet_sample * strength + dry_sample * (1.0 - strength);
    }
}

pub struct DeepFilterProcessor {
    df: Option<DeepFilterFFI>, // Option for graceful fallback if FFI fails
    input_buffer: FixedAudioRing<ControlledSample, DEEPFILTER_BUFFER_CAPACITY>,
    output_buffer: FixedAudioRing<f32, DEEPFILTER_BUFFER_CAPACITY>,
    input_frame: [ControlledSample; DEEPFILTER_FRAME_SIZE],
    gate_compensation: GateCompensation,
    enabled: bool,
    strength: Arc<AtomicU32>,
    smoothed_strength: f32,
    dry_frame: [f32; DEEPFILTER_FRAME_SIZE],
    frame_scratch: [f32; DEEPFILTER_FRAME_SIZE],
    output_frame: [f32; DEEPFILTER_FRAME_SIZE],
    aligned_dry_frame: [f32; DEEPFILTER_FRAME_SIZE],
    dry_delay: [f32; DEEPFILTER_MAX_LATENCY_SAMPLES],
    dry_delay_index: usize,
    invalid_wet_samples: usize,
    load_error: Option<String>, // Store load error for reporting
    model: DeepFilterModel,     // Track which model variant we're using
    backend_failed: bool,
    runtime_error: Option<DeepFilterProcessError>,
    successful_inference_frames: u64,
}

impl DeepFilterProcessor {
    pub fn new(strength: Arc<AtomicU32>, model: DeepFilterModel) -> Self {
        Self::new_with_config(strength, model, DeepFilterRuntimeConfig::default())
    }

    pub fn new_with_config(
        strength: Arc<AtomicU32>,
        model: DeepFilterModel,
        runtime_config: DeepFilterRuntimeConfig,
    ) -> Self {
        if !crate::dsp::noise_suppressor::deepfilter_experimental_enabled() {
            return Self::with_backend(
                strength,
                model,
                None,
                Some(
                    "DeepFilterNet disabled; set AUDIOFORGE_ENABLE_DEEPFILTER=1 to enable"
                        .to_string(),
                ),
            );
        }

        // Snapshot bootstrap-owned paths before initialization. This happens
        // outside the realtime processing loop.
        let app_paths = configured_app_owned_paths();
        let allow_external = external_deepfilter_paths_allowed();
        let external_model = env::var_os("DEEPFILTER_MODEL_PATH").map(PathBuf::from);
        Self::from_library(
            strength,
            model,
            runtime_config,
            DeepFilterLib::try_load(&app_paths, allow_external),
            &app_paths,
            allow_external,
            external_model.as_deref(),
        )
    }

    /// Build the processor around the result of loading the library. Whatever
    /// goes wrong while the model is loaded leaves a passthrough processor
    /// whose `backend_error` says why.
    fn from_library(
        strength: Arc<AtomicU32>,
        model: DeepFilterModel,
        runtime_config: DeepFilterRuntimeConfig,
        library: Result<DeepFilterLib, String>,
        app_paths: &AppOwnedDeepFilterPaths,
        allow_external: bool,
        external_model: Option<&Path>,
    ) -> Self {
        let backend = match library {
            Ok(library) => DeepFilterFFI::new(
                Arc::new(library),
                model,
                runtime_config,
                app_paths,
                allow_external,
                external_model,
            )
            .map_err(|e| format!("DeepFilterNet init failed: {}", e)),
            Err(e) => Err(format!("DeepFilterNet C library unavailable: {}", e)),
        };
        match backend {
            Ok(df) => Self::with_backend(strength, model, Some(df), None),
            Err(load_error) => Self::with_backend(strength, model, None, Some(load_error)),
        }
    }

    fn with_backend(
        strength: Arc<AtomicU32>,
        model: DeepFilterModel,
        df: Option<DeepFilterFFI>,
        load_error: Option<String>,
    ) -> Self {
        let initial_strength = unit_or(f32::from_bits(strength.load(Ordering::Relaxed)), 1.0);
        let compensation_delay = match model {
            DeepFilterModel::LowLatency => DEEPFILTER_FRAME_SIZE,
            DeepFilterModel::Standard => 0,
        };
        Self {
            df,
            input_buffer: FixedAudioRing::new(),
            output_buffer: FixedAudioRing::new(),
            input_frame: [ControlledSample::default(); DEEPFILTER_FRAME_SIZE],
            gate_compensation: GateCompensation::new(compensation_delay),
            enabled: true,
            strength,
            smoothed_strength: initial_strength,
            dry_frame: [0.0; DEEPFILTER_FRAME_SIZE],
            frame_scratch: [0.0; DEEPFILTER_FRAME_SIZE],
            output_frame: [0.0; DEEPFILTER_FRAME_SIZE],
            aligned_dry_frame: [0.0; DEEPFILTER_FRAME_SIZE],
            dry_delay: [0.0; DEEPFILTER_MAX_LATENCY_SAMPLES],
            dry_delay_index: 0,
            invalid_wet_samples: 0,
            load_error,
            model,
            backend_failed: false,
            runtime_error: None,
            successful_inference_frames: 0,
        }
    }

    fn mark_backend_failed_rt(&mut self, error: DeepFilterProcessError) {
        self.backend_failed = true;
        self.runtime_error = Some(error);
    }

    fn push_samples_with_controls(
        &mut self,
        samples: &[f32],
        controls: Option<&[GateControl]>,
    ) -> usize {
        if let Some(controls) = controls {
            assert_eq!(samples.len(), controls.len());
        }
        let accepted = samples.len().min(self.input_buffer.remaining());
        for (index, &sample) in samples[..accepted].iter().enumerate() {
            let gate = controls.map_or(GateControl::BYPASS, |controls| controls[index]);
            self.input_buffer.push(ControlledSample { sample, gate });
        }
        accepted
    }

    fn read_input_frame(&mut self) -> usize {
        let read = self.input_buffer.pop_into(&mut self.input_frame);
        for (dry, input) in self.dry_frame[..read].iter_mut().zip(&self.input_frame) {
            *dry = input.sample;
        }
        read
    }

    #[inline]
    fn delay_dry_frame(&mut self) {
        let delay_samples = self.model.latency_samples();
        debug_assert!(delay_samples <= self.dry_delay.len());
        for (input, delayed) in self
            .dry_frame
            .iter()
            .copied()
            .zip(self.aligned_dry_frame.iter_mut())
        {
            *delayed = self.dry_delay[self.dry_delay_index];
            self.dry_delay[self.dry_delay_index] = input;
            self.dry_delay_index += 1;
            if self.dry_delay_index == delay_samples {
                self.dry_delay_index = 0;
            }
        }
    }

    fn reset_dry_delay(&mut self) {
        self.dry_delay.fill(0.0);
        self.aligned_dry_frame.fill(0.0);
        self.dry_delay_index = 0;
        self.invalid_wet_samples = self.model.latency_samples();
        self.gate_compensation.reset();
    }

    /// Process frames through FFI or fallback
    pub fn process_frames_internal(&mut self) {
        // Advance the 15 ms strength EMA once per processed 10 ms frame so
        // backlog size and callback chunking cannot change the ramp duration.
        // A producer that bypasses `set_strength` must not poison the smoother.
        let target_strength = unit_or(
            f32::from_bits(self.strength.load(Ordering::Relaxed)),
            self.smoothed_strength,
        );
        const TAU_MS: f32 = 15.0;
        const SAMPLE_RATE: f32 = 48000.0;
        let alpha =
            1.0 - (-1.0 / (TAU_MS / 1000.0 * SAMPLE_RATE / DEEPFILTER_FRAME_SIZE as f32)).exp();

        // Process complete frames (480 samples each)
        while self.input_buffer.len() >= DEEPFILTER_FRAME_SIZE
            && self.output_buffer.remaining() >= DEEPFILTER_FRAME_SIZE
        {
            self.smoothed_strength += alpha * (target_strength - self.smoothed_strength);
            let read = self.read_input_frame();
            if read != DEEPFILTER_FRAME_SIZE {
                break;
            }
            self.frame_scratch.copy_from_slice(&self.dry_frame);
            let use_aligned_dry = self.enabled && self.df.is_some();
            if use_aligned_dry {
                self.delay_dry_frame();
            }

            // Process through FFI if available
            if self.enabled && !self.backend_failed {
                if let Some(ref mut df) = self.df {
                    match df.process_into(&mut self.frame_scratch, &mut self.output_frame) {
                        Ok(_lsnr) => {
                            self.successful_inference_frames =
                                self.successful_inference_frames.saturating_add(1);
                            // Preserved model overlap predates a soft reset. Only
                            // the wet prefix is invalid; aligned dry stays intact.
                            let invalid = self.invalid_wet_samples.min(DEEPFILTER_FRAME_SIZE);
                            self.output_frame[..invalid].fill(0.0);
                            self.invalid_wet_samples -= invalid;
                            for (wet, input) in self.output_frame.iter_mut().zip(&self.input_frame)
                            {
                                let ratio = self.gate_compensation.next_ratio(input.gate);
                                if ratio != 1.0 {
                                    *wet *= ratio;
                                }
                            }
                            mix_wet_with_aligned_dry(
                                &mut self.output_frame,
                                &self.aligned_dry_frame,
                                self.smoothed_strength,
                            );
                            self.output_buffer.push_slice(&self.output_frame);
                            continue;
                        }
                        Err(error) => {
                            self.mark_backend_failed_rt(error);
                        }
                    }
                }
            }

            // Passthrough. Preserve the selected backend's fixed latency after
            // a runtime failure, but do not delay an unavailable backend.
            if use_aligned_dry {
                self.output_frame.copy_from_slice(&self.aligned_dry_frame);
            } else {
                self.output_frame.copy_from_slice(&self.dry_frame);
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

    fn effective_latency_samples(&self) -> usize {
        if self.enabled && self.df.is_some() {
            self.model.latency_samples()
        } else {
            // Disabled/unavailable passthrough still waits for one common
            // 480-sample frame, but does not enter the model lookahead.
            DEEPFILTER_FRAME_SIZE
        }
    }
}

// ============================================================================
// TRAIT IMPLEMENTATION
// ============================================================================

impl ControlledNoiseSuppressor for DeepFilterProcessor {
    fn push_controlled_samples(&mut self, samples: &[f32], controls: &[GateControl]) -> usize {
        self.push_samples_with_controls(samples, Some(controls))
    }
}

impl NoiseSuppressor for DeepFilterProcessor {
    fn push_samples(&mut self, samples: &[f32]) -> usize {
        self.push_samples_with_controls(samples, None)
    }

    fn process_frames(&mut self) {
        if self.enabled {
            self.process_frames_internal();
        } else {
            // Disabled: passthrough
            while self.input_buffer.len() >= DEEPFILTER_FRAME_SIZE
                && self.output_buffer.remaining() >= DEEPFILTER_FRAME_SIZE
            {
                let read = self.read_input_frame();
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

    fn pop_samples_into(&mut self, buffer: &mut [f32]) -> usize {
        let count = buffer.len().min(self.available_samples());
        self.output_buffer.pop_into(&mut buffer[..count])
    }

    fn set_strength(&self, value: f32) {
        let clamped = unit_or(value, self.get_strength());
        let bits = clamped.to_bits();
        self.strength.store(bits, Ordering::Relaxed);
    }

    fn get_strength(&self) -> f32 {
        f32::from_bits(self.strength.load(Ordering::Relaxed))
    }

    fn set_enabled(&mut self, enabled: bool) {
        if self.enabled != enabled {
            self.reset_dry_delay();
        }
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
        self.reset_dry_delay();
        // Keep smoothed_strength to avoid zipper noise
    }

    fn pending_input(&self) -> usize {
        self.input_buffer.len()
    }

    fn model_type(&self) -> NoiseModel {
        match self.model {
            DeepFilterModel::LowLatency => NoiseModel::DeepFilterNetLL,
            DeepFilterModel::Standard => NoiseModel::DeepFilterNet,
        }
    }

    fn latency_samples(&self) -> usize {
        self.effective_latency_samples()
    }

    fn backend_available(&self) -> bool {
        self.is_ffi_available()
    }

    fn backend_error(&self) -> Option<&str> {
        DeepFilterProcessor::backend_error(self)
    }

    fn backend_failed(&self) -> bool {
        DeepFilterProcessor::backend_failed(self)
    }

    fn successful_inference_frames(&self) -> u64 {
        self.successful_inference_frames
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
