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
//! If DeepFilterNet C library is available, it will be loaded at runtime:
//! - Windows: df.dll in PATH or working directory
//! - Linux: libdf.so in LD_LIBRARY_PATH or system paths
//! - macOS: libdf.dylib in DYLD_LIBRARY_PATH or system paths
//!
//! MODEL FILES (Optional):
//! DeepFilterNet requires a model tar.gz file. The library will look for:
//! - Environment variable: DEEPFILTER_MODEL_PATH
//! - Default locations:
//!   - Windows: ./models/DeepFilterNet3_ll_onnx.tar.gz
//!   - Linux/macOS: ~/.local/share/deepfilter/DeepFilterNet3_ll_onnx.tar.gz
//!
//! If the library or model is not found, the processor will use passthrough mode.
//!
//! Expected latency: ~10ms with LL variant (no lookahead)

#![cfg(feature = "deepfilter")]

use crate::dsp::noise_suppressor::{NoiseModel, NoiseSuppressor};
use std::env;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::vec::Vec;

/// DeepFilterNet frame size (same as RNNoise: 10ms at 48kHz)
pub const DEEPFILTER_FRAME_SIZE: usize = 480;

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

// SAFETY: The library owns the function pointers and ensures they remain valid
// for the lifetime of the library. The wrapper only accesses them through &self.
// Function pointers are safe to share between threads (they're just addresses).
unsafe impl Send for DeepFilterLib {}
unsafe impl Sync for DeepFilterLib {}

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

/// Find DeepFilterNet model path
///
/// Search order:
/// 1. Environment variable DEEPFILTER_MODEL_PATH
/// 2. ./models/{model_filename}
/// 3. ../models/{model_filename}
/// 4. ~/.local/share/deepfilter/{model_filename} (Linux/macOS)
fn find_model_path(model: DeepFilterModel) -> Option<PathBuf> {
    let filename = model.filename();

    // 1. Check environment variable
    if let Ok(path) = env::var("DEEPFILTER_MODEL_PATH") {
        let path_buf = PathBuf::from(&path);
        if path_buf.exists() {
            return Some(path_buf);
        }
    }

    // 2. Check ./models/ directory
    let local_model = PathBuf::from(format!("models/{}", filename));
    if local_model.exists() {
        return Some(local_model);
    }

    // 3. Check ../models/ directory (for dev builds)
    let parent_model = PathBuf::from(format!("../models/{}", filename));
    if parent_model.exists() {
        return Some(parent_model);
    }

    // 4. Check user data directory (Linux/macOS style)
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        if let Ok(home) = env::var("HOME") {
            let user_model =
                PathBuf::from(format!("{}/.local/share/deepfilter/{}", home, filename));
            if user_model.exists() {
                return Some(user_model);
            }
        }
    }

    None
}

impl DeepFilterLib {
    /// Try to load the DeepFilterNet library from system paths
    fn try_load() -> Option<Self> {
        // Library name varies by platform
        #[cfg(target_os = "windows")]
        let lib_name = "df.dll";

        #[cfg(target_os = "linux")]
        let lib_name = "libdf.so";

        #[cfg(target_os = "macos")]
        let lib_name = "libdf.dylib";

        unsafe {
            let library = libloading::Library::new(lib_name).ok()?;

            // Load all required symbols and extract raw pointers
            let df_create: libloading::Symbol<DfCreateFn> = library.get(b"df_create").ok()?;
            let df_free: libloading::Symbol<DfFreeFn> = library.get(b"df_free").ok()?;
            let df_get_frame_length: libloading::Symbol<DfGetFrameLengthFn> =
                library.get(b"df_get_frame_length").ok()?;
            let df_process_frame: libloading::Symbol<DfProcessFrameFn> =
                library.get(b"df_process_frame").ok()?;
            let df_set_atten_lim: libloading::Symbol<DfSetAttenLimFn> =
                library.get(b"df_set_atten_lim").ok()?;
            let df_set_post_filter_beta: libloading::Symbol<DfSetPostFilterBetaFn> =
                library.get(b"df_set_post_filter_beta").ok()?;

            // Convert to raw function pointers
            let df_create = *df_create.into_raw();
            let df_free = *df_free.into_raw();
            let df_get_frame_length = *df_get_frame_length.into_raw();
            let df_process_frame = *df_process_frame.into_raw();
            let df_set_atten_lim = *df_set_atten_lim.into_raw();
            let df_set_post_filter_beta = *df_set_post_filter_beta.into_raw();

            Some(DeepFilterLib {
                _library: library,
                df_create,
                df_free,
                df_get_frame_length,
                df_process_frame,
                df_set_atten_lim,
                df_set_post_filter_beta,
            })
        }
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
    ptr: *mut DFState,
    frame_size: usize,
}

// SAFETY: DeepFilterNet's C API is thread-safe for single-threaded access
// We enforce this through Rust's &mut self requirement
unsafe impl Send for DeepFilterFFI {}

impl DeepFilterFFI {
    /// Create a new DeepFilterNet instance using model from file system
    ///
    /// # Safety
    /// This function calls unsafe FFI functions to create the DeepFilterNet instance.
    /// It validates the returned pointer and returns an error if creation failed.
    pub fn new(lib: Arc<DeepFilterLib>, model: DeepFilterModel) -> Result<Self, String> {
        // Find model path
        let model_path = find_model_path(model)
            .ok_or_else(|| {
                format!("DeepFilterNet model file not found. Place {} in ./models/ or set DEEPFILTER_MODEL_PATH", model.filename())
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

        unsafe {
            let df_create = lib.df_create;
            let ptr = df_create(model_path_cstr.as_ptr(), atten_lim, std::ptr::null());

            if ptr.is_null() {
                return Err(format!(
                    "Failed to create DeepFilterNet instance with model: {}",
                    model_path.display()
                ));
            }

            let df_get_frame_length = lib.df_get_frame_length;
            let frame_size = df_get_frame_length(ptr);

            if frame_size != DEEPFILTER_FRAME_SIZE {
                let df_free = lib.df_free;
                df_free(ptr);
                return Err(format!(
                    "DeepFilterNet frame size mismatch: expected {}, got {}",
                    DEEPFILTER_FRAME_SIZE, frame_size
                ));
            }

            Ok(Self {
                _lib: lib,
                ptr,
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
    /// * Vector of processed samples (frame_size length)
    /// * Local SNR estimate in dB (for quality metering)
    ///
    /// # Safety
    /// This function calls unsafe FFI functions.
    /// It ensures buffers are properly sized and aligned before passing to C.
    pub fn process(&mut self, input: &[f32]) -> Result<(Vec<f32>, f32), String> {
        if input.len() != self.frame_size {
            return Err(format!(
                "Input buffer size mismatch: expected {}, got {}",
                self.frame_size,
                input.len()
            ));
        }

        let mut output = vec![0.0f32; self.frame_size];

        // SAFETY: We've verified buffer sizes and the pointer is valid
        unsafe {
            let input_ptr = input.as_ptr() as *mut f32;
            let output_ptr = output.as_mut_ptr();

            let df_process_frame = self._lib.df_process_frame;
            let lsnr = df_process_frame(self.ptr, input_ptr, output_ptr);

            if lsnr.is_nan() {
                return Err("DeepFilterNet processing failed (returned NaN)".to_string());
            }

            Ok((output, lsnr))
        }
    }

    /// Set attenuation limit (dB)
    pub fn set_atten_lim(&mut self, lim_db: f32) {
        unsafe {
            let df_set_atten_lim = self._lib.df_set_atten_lim;
            df_set_atten_lim(self.ptr, lim_db);
        }
    }

    /// Set post filter beta
    pub fn set_post_filter_beta(&mut self, beta: f32) {
        unsafe {
            let df_set_post_filter_beta = self._lib.df_set_post_filter_beta;
            df_set_post_filter_beta(self.ptr, beta);
        }
    }
}

impl Drop for DeepFilterFFI {
    fn drop(&mut self) {
        unsafe {
            let df_free = self._lib.df_free;
            df_free(self.ptr);
        }
    }
}

// ============================================================================
// PROCESSOR IMPLEMENTATION
// ============================================================================

pub struct DeepFilterProcessor {
    df: Option<DeepFilterFFI>, // Option for graceful fallback if FFI fails
    lib: Option<Arc<DeepFilterLib>>, // Keep library loaded
    input_buffer: Vec<f32>,
    output_buffer: Vec<f32>,
    enabled: bool,
    strength: Arc<AtomicU32>,
    smoothed_strength: f32,
    dry_buffer: Vec<f32>,
    load_error: Option<String>, // Store load error for reporting
    model: DeepFilterModel,     // Track which model variant we're using
}

impl DeepFilterProcessor {
    pub fn new(strength: Arc<AtomicU32>, model: DeepFilterModel) -> Self {
        // Try to load library and initialize FFI
        let (df, lib, load_error) = match DeepFilterLib::try_load() {
            Some(lib) => {
                let lib_arc = Arc::new(lib);
                match DeepFilterFFI::new(lib_arc.clone(), model) {
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
            None => {
                eprintln!("DeepFilterNet C library not found. Using passthrough fallback.");
                eprintln!("NOTE: To use DeepFilterNet, build the C library and place it in:");
                eprintln!("  - Windows: df.dll in PATH or working directory");
                eprintln!("  - Linux: libdf.so in LD_LIBRARY_PATH or /usr/local/lib");
                eprintln!("  - macOS: libdf.dylib in DYLD_LIBRARY_PATH or /usr/local/lib");
                eprintln!("NOTE: Also ensure DeepFilterNet3 model file is available:");
                eprintln!("  - Set DEEPFILTER_MODEL_PATH environment variable");
                eprintln!("  - Or place in ./models/DeepFilterNet3_ll_onnx.tar.gz or DeepFilterNet3_onnx.tar.gz");
                (
                    None,
                    None,
                    Some("DeepFilterNet C library not found - using passthrough".to_string()),
                )
            }
        };

        Self {
            df,
            lib,
            input_buffer: Vec::with_capacity(DEEPFILTER_FRAME_SIZE * 4),
            output_buffer: Vec::with_capacity(DEEPFILTER_FRAME_SIZE * 4),
            enabled: true,
            strength,
            smoothed_strength: 1.0,
            dry_buffer: Vec::with_capacity(DEEPFILTER_FRAME_SIZE),
            load_error,
            model,
        }
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
        while self.input_buffer.len() >= DEEPFILTER_FRAME_SIZE {
            // Store dry samples for wet/dry mixing
            self.dry_buffer.clear();
            self.dry_buffer
                .extend_from_slice(&self.input_buffer[..DEEPFILTER_FRAME_SIZE]);

            // Extract frame
            let frame: Vec<f32> = self.input_buffer.drain(..DEEPFILTER_FRAME_SIZE).collect();

            // Process through FFI if available
            if self.enabled {
                if let Some(ref mut df) = self.df {
                    match df.process(&frame) {
                        Ok((enhanced, _lsnr)) => {
                            // Apply wet/dry mix to FFI output
                            for (i, &wet) in enhanced.iter().enumerate() {
                                let dry = self.dry_buffer[i];
                                let mixed = wet * self.smoothed_strength
                                    + dry * (1.0 - self.smoothed_strength);
                                self.output_buffer.push(mixed);
                            }
                            continue;
                        }
                        Err(e) => {
                            eprintln!(
                                "DeepFilterNet FFI processing error: {}, using passthrough",
                                e
                            );
                            // Fall through to passthrough
                        }
                    }
                }
            }

            // Passthrough (fallback or disabled)
            for (i, &wet) in frame.iter().enumerate() {
                let dry = self.dry_buffer[i];
                let mixed = wet * self.smoothed_strength + dry * (1.0 - self.smoothed_strength);
                self.output_buffer.push(mixed);
            }
        }
    }

    /// Check if FFI is available (not in passthrough fallback mode)
    pub fn is_ffi_available(&self) -> bool {
        self.df.is_some()
    }

    /// Get load error if FFI failed to initialize
    pub fn load_error(&self) -> Option<&str> {
        self.load_error.as_deref()
    }
}

// ============================================================================
// TRAIT IMPLEMENTATION
// ============================================================================

impl NoiseSuppressor for DeepFilterProcessor {
    fn push_samples(&mut self, samples: &[f32]) {
        self.input_buffer.extend_from_slice(samples);
    }

    fn process_frames(&mut self) {
        if self.enabled {
            self.process_frames_internal();
        } else {
            // Disabled: passthrough
            while self.input_buffer.len() >= DEEPFILTER_FRAME_SIZE {
                let frame: Vec<f32> = self.input_buffer.drain(..DEEPFILTER_FRAME_SIZE).collect();
                self.output_buffer.extend_from_slice(&frame);
            }
        }
    }

    fn available_samples(&self) -> usize {
        self.output_buffer.len()
    }

    fn pop_samples(&mut self, count: usize) -> Vec<f32> {
        let actual = count.min(self.output_buffer.len());
        self.output_buffer.drain(..actual).collect()
    }

    fn pop_all_samples(&mut self) -> Vec<f32> {
        self.output_buffer.drain(..).collect()
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
        self.dry_buffer.clear();
        // Keep smoothed_strength to avoid zipper noise
    }

    fn pending_input(&self) -> usize {
        self.input_buffer.len()
    }

    fn drain_pending_input(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.input_buffer)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_availability() {
        let processor = DeepFilterProcessor::default();
        // This test passes regardless of whether FFI is available
        // If not available, the processor will use passthrough
        if processor.is_ffi_available() {
            println!("DeepFilterNet C FFI is available");
        } else {
            println!("DeepFilterNet C FFI is not available (using passthrough)");
        }
    }

    #[test]
    fn test_frame_size() {
        assert_eq!(DEEPFILTER_FRAME_SIZE, 480);
    }

    #[test]
    fn test_strength_operations() {
        let processor = DeepFilterProcessor::default();
        processor.set_strength(0.5);
        assert_eq!(processor.get_strength(), 0.5);
    }

    #[test]
    fn test_buffering_logic() {
        let mut processor = DeepFilterProcessor::default();
        assert_eq!(processor.pending_input(), 0);

        processor.push_samples(&[0.1; 100]);
        assert_eq!(processor.pending_input(), 100);

        processor.push_samples(&[0.2; 400]);
        assert_eq!(processor.pending_input(), 500); // Accumulated

        processor.process_frames();
        assert_eq!(processor.pending_input(), 20); // 500 - 480 = 20 remaining
    }

    #[test]
    fn test_latency_samples() {
        let processor = DeepFilterProcessor::default();
        assert_eq!(processor.latency_samples(), 480);
    }

    #[test]
    fn test_enable_disable() {
        let mut processor = DeepFilterProcessor::default();
        assert!(processor.is_enabled());

        processor.set_enabled(false);
        assert!(!processor.is_enabled());

        processor.set_enabled(true);
        assert!(processor.is_enabled());
    }

    #[test]
    fn test_soft_reset() {
        let mut processor = DeepFilterProcessor::default();
        processor.push_samples(&[0.1; 500]);
        processor.process_frames();

        processor.soft_reset();
        assert_eq!(processor.pending_input(), 0);
        assert_eq!(processor.available_samples(), 0);
    }

    #[test]
    fn test_passthrough_processing() {
        let mut processor = DeepFilterProcessor::default();
        let input = vec![0.5; 1000];
        processor.push_samples(&input);
        processor.process_frames();

        let output = processor.pop_all_samples();
        assert!(!output.is_empty(), "Should produce output");

        // Output should contain complete frames (1000 samples = 2 frames of 480 + 40 remaining)
        assert_eq!(output.len(), 960); // Two complete frames
    }
}
