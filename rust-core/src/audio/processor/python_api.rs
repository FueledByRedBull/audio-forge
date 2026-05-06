/// Gate operating modes
#[cfg(feature = "vad")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[pyclass(eq, eq_int)]
pub enum PyGateMode {
    /// Traditional gate using only level threshold
    ThresholdOnly = 0,
    /// Hybrid: gate opens when level exceeded OR speech detected
    VadAssisted = 1,
    /// VAD-only: gate opens solely based on speech probability
    VadOnly = 2,
}

/// Python-exposed audio processor
#[pyclass(name = "AudioProcessor", unsendable)]
pub struct PyAudioProcessor {
    processor: AudioProcessor,
}

#[pymethods]
impl PyAudioProcessor {
    #[new]
    fn new() -> Self {
        Self {
            processor: AudioProcessor::new(),
        }
    }

    /// Start audio processing
    #[pyo3(signature = (input_device=None, output_device=None))]
    fn start(
        &mut self,
        input_device: Option<&str>,
        output_device: Option<&str>,
    ) -> PyResult<String> {
        self.processor
            .start(input_device, output_device)
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)
    }

    /// Stop audio processing
    fn stop(&mut self) {
        self.processor.stop();
    }

    /// Check if running
    fn is_running(&self) -> bool {
        self.processor.is_running()
    }

    /// Get active input device name for the running stream.
    fn get_active_input_device(&self) -> Option<String> {
        self.processor.active_input_device_name()
    }

    /// Get active output device name for the running stream.
    fn get_active_output_device(&self) -> Option<String> {
        self.processor.active_output_device_name()
    }

    /// Get sample rate
    fn sample_rate(&self) -> u32 {
        self.processor.sample_rate()
    }

    /// Set master bypass
    fn set_bypass(&self, bypass: bool) {
        self.processor.set_bypass(bypass);
    }

    /// Get bypass state
    fn is_bypass(&self) -> bool {
        self.processor.is_bypass()
    }

    fn set_raw_monitor_enabled(&self, enabled: bool) {
        self.processor.set_raw_monitor_enabled(enabled);
    }

    fn is_raw_monitor_enabled(&self) -> bool {
        self.processor.is_raw_monitor_enabled()
    }

    // === Noise Gate ===

    fn set_gate_enabled(&self, enabled: bool) {
        self.processor.set_gate_enabled(enabled);
    }

    fn is_gate_enabled(&self) -> bool {
        self.processor.is_gate_enabled()
    }

    fn set_gate_threshold(&self, threshold_db: f64) {
        self.processor.set_gate_threshold(threshold_db);
    }

    fn set_gate_attack(&self, attack_ms: f64) {
        self.processor.set_gate_attack(attack_ms);
    }

    fn set_gate_release(&self, release_ms: f64) {
        self.processor.set_gate_release(release_ms);
    }

    // === VAD Gate Controls ===

    /// Set gate mode (0 = ThresholdOnly, 1 = VadAssisted, 2 = VadOnly)
    #[cfg(feature = "vad")]
    #[pyo3(signature = (mode))]
    fn set_gate_mode(&self, mode: u8) -> PyResult<()> {
        self.processor
            .set_gate_mode(mode)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
    }

    /// Get VAD speech probability (0.0-1.0)
    #[cfg(feature = "vad")]
    fn get_vad_probability(&self) -> f32 {
        self.processor.get_vad_probability()
    }

    /// Check whether VAD backend is available (model/runtime loaded)
    #[cfg(feature = "vad")]
    fn is_vad_available(&self) -> bool {
        self.processor.is_vad_available()
    }

    /// Set VAD probability threshold (0.0-1.0)
    #[cfg(feature = "vad")]
    fn set_vad_threshold(&self, threshold: f32) {
        self.processor.set_vad_threshold(threshold);
    }

    /// Set VAD hold time in milliseconds
    #[cfg(feature = "vad")]
    fn set_vad_hold_time(&self, hold_ms: f32) {
        self.processor.set_vad_hold_time(hold_ms);
    }

    /// Set VAD pre-gain to boost weak signals for better speech detection
    /// Default is 1.0 (no gain). Values > 1.0 boost the signal.
    /// This helps with quiet microphones where VAD can't detect speech.
    #[cfg(feature = "vad")]
    fn set_vad_pre_gain(&self, gain: f32) {
        self.processor.set_vad_pre_gain(gain);
    }

    /// Enable/disable auto-threshold mode (automatically adjusts gate threshold based on noise floor)
    #[cfg(feature = "vad")]
    fn set_auto_threshold(&self, enabled: bool) {
        self.processor.set_auto_threshold(enabled);
    }

    /// Set margin above noise floor for auto-threshold (in dB)
    #[cfg(feature = "vad")]
    fn set_gate_margin(&self, margin_db: f32) {
        self.processor.set_gate_margin(margin_db);
    }

    /// Get current noise floor estimate (in dB)
    #[cfg(feature = "vad")]
    fn get_noise_floor(&self) -> f32 {
        self.processor.get_noise_floor()
    }

    /// Get current gate margin (in dB)
    #[cfg(feature = "vad")]
    fn gate_margin(&self) -> f32 {
        self.processor.gate_margin()
    }

    /// Check if auto-threshold is enabled
    #[cfg(feature = "vad")]
    fn auto_threshold_enabled(&self) -> bool {
        self.processor.auto_threshold_enabled()
    }

    /// Get current VAD pre-gain
    #[cfg(feature = "vad")]
    fn vad_pre_gain(&self) -> f32 {
        self.processor.vad_pre_gain()
    }

    // === RNNoise ===

    fn set_rnnoise_enabled(&self, enabled: bool) {
        self.processor.set_rnnoise_enabled(enabled);
    }

    fn is_rnnoise_enabled(&self) -> bool {
        self.processor.is_rnnoise_enabled()
    }

    /// Set RNNoise wet/dry mix strength (0.0 = fully dry, 1.0 = fully wet)
    fn set_rnnoise_strength(&self, strength: f64) {
        self.processor.set_rnnoise_strength(strength as f32);
    }

    /// Get current RNNoise strength
    fn get_rnnoise_strength(&self) -> f64 {
        self.processor.get_rnnoise_strength() as f64
    }

    /// Set noise suppression model by name ("rnnoise" or "deepfilter")
    fn set_noise_model(&self, model: &str) -> bool {
        match NoiseModel::from_id(model) {
            Some(m) => self.processor.set_noise_model(m),
            None => false,
        }
    }

    /// Get current noise model name
    fn get_noise_model(&self) -> String {
        self.processor.get_noise_model().id().to_string()
    }

    /// Get current noise model display name
    fn get_noise_model_display_name(&self) -> String {
        self.processor.get_noise_model().display_name().to_string()
    }

    /// List available noise models: [(id, display_name), ...]
    fn list_noise_models(&self) -> Vec<(String, String)> {
        self.processor.list_noise_models()
    }

    // === EQ ===

    fn set_eq_enabled(&self, enabled: bool) {
        self.processor.set_eq_enabled(enabled);
    }

    fn is_eq_enabled(&self) -> bool {
        self.processor.is_eq_enabled()
    }

    fn set_eq_band_gain(&self, band: usize, gain_db: f64) -> PyResult<()> {
        self.processor
            .set_eq_band_gain(band, gain_db)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
    }

    fn set_eq_band_frequency(&self, band: usize, frequency: f64) -> PyResult<()> {
        self.processor
            .set_eq_band_frequency(band, frequency)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
    }

    fn set_eq_band_q(&self, band: usize, q: f64) -> PyResult<()> {
        self.processor
            .set_eq_band_q(band, q)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
    }

    fn get_eq_band_params(&self, band: usize) -> Option<(f64, f64, f64)> {
        self.processor.get_eq_band_params(band)
    }

    /// Apply EQ settings for all 10 bands in a single atomic call
    ///
    /// Args:
    ///     bands: List of (frequency_hz, gain_db, q) tuples for each band (must be 10)
    ///
    /// Raises:
    ///     ValueError: If band count is not 10 or parameters are out of range
    fn apply_eq_settings(&self, bands: Vec<(f64, f64, f64)>) -> PyResult<()> {
        self.processor.apply_eq_settings(bands)
    }

    // === De-Esser ===

    fn set_deesser_enabled(&self, enabled: bool) {
        self.processor.set_deesser_enabled(enabled);
    }

    fn is_deesser_enabled(&self) -> bool {
        self.processor.is_deesser_enabled()
    }

    fn set_deesser_low_cut_hz(&self, hz: f64) {
        self.processor.set_deesser_low_cut_hz(hz);
    }

    fn set_deesser_high_cut_hz(&self, hz: f64) {
        self.processor.set_deesser_high_cut_hz(hz);
    }

    fn set_deesser_threshold_db(&self, threshold_db: f64) {
        self.processor.set_deesser_threshold_db(threshold_db);
    }

    fn set_deesser_ratio(&self, ratio: f64) {
        self.processor.set_deesser_ratio(ratio);
    }

    fn set_deesser_attack_ms(&self, attack_ms: f64) {
        self.processor.set_deesser_attack_ms(attack_ms);
    }

    fn set_deesser_release_ms(&self, release_ms: f64) {
        self.processor.set_deesser_release_ms(release_ms);
    }

    fn set_deesser_max_reduction_db(&self, max_reduction_db: f64) {
        self.processor
            .set_deesser_max_reduction_db(max_reduction_db);
    }

    fn set_deesser_auto_enabled(&self, auto_enabled: bool) {
        self.processor.set_deesser_auto_enabled(auto_enabled);
    }

    fn is_deesser_auto_enabled(&self) -> bool {
        self.processor.is_deesser_auto_enabled()
    }

    fn set_deesser_auto_amount(&self, amount: f64) {
        self.processor.set_deesser_auto_amount(amount);
    }

    fn get_deesser_low_cut_hz(&self) -> f64 {
        self.processor.get_deesser_low_cut_hz()
    }

    fn get_deesser_high_cut_hz(&self) -> f64 {
        self.processor.get_deesser_high_cut_hz()
    }

    fn get_deesser_threshold_db(&self) -> f64 {
        self.processor.get_deesser_threshold_db()
    }

    fn get_deesser_ratio(&self) -> f64 {
        self.processor.get_deesser_ratio()
    }

    fn get_deesser_max_reduction_db(&self) -> f64 {
        self.processor.get_deesser_max_reduction_db()
    }

    fn get_deesser_auto_amount(&self) -> f64 {
        self.processor.get_deesser_auto_amount()
    }

    fn get_deesser_gain_reduction_db(&self) -> f32 {
        self.processor.get_deesser_gain_reduction_db()
    }

    // === Compressor ===

    fn set_compressor_enabled(&self, enabled: bool) {
        self.processor.set_compressor_enabled(enabled);
    }

    fn is_compressor_enabled(&self) -> bool {
        self.processor.is_compressor_enabled()
    }

    fn set_compressor_threshold(&self, threshold_db: f64) {
        self.processor.set_compressor_threshold(threshold_db);
    }

    fn set_compressor_ratio(&self, ratio: f64) {
        self.processor.set_compressor_ratio(ratio);
    }

    fn set_compressor_attack(&self, attack_ms: f64) {
        self.processor.set_compressor_attack(attack_ms);
    }

    fn set_compressor_release(&self, release_ms: f64) {
        self.processor.set_compressor_release(release_ms);
    }

    /// Get compressor release time.
    ///
    /// Note: When adaptive release is enabled, this returns the base release time.
    /// Use get_compressor_current_release() for the actual adaptive release time.
    fn get_compressor_release(&self) -> f64 {
        self.processor.get_compressor_release()
    }

    fn set_compressor_makeup_gain(&self, makeup_gain_db: f64) {
        self.processor.set_compressor_makeup_gain(makeup_gain_db);
    }

    /// Set compressor adaptive release mode
    fn set_compressor_adaptive_release(&self, enabled: bool) {
        self.processor.set_compressor_adaptive_release(enabled);
    }

    /// Get compressor adaptive release mode
    fn get_compressor_adaptive_release(&self) -> bool {
        self.processor.get_compressor_adaptive_release()
    }

    /// Set compressor base release time (milliseconds)
    fn set_compressor_base_release(&self, release_ms: f64) {
        self.processor.set_compressor_base_release(release_ms);
    }

    /// Get compressor base release time (milliseconds)
    fn get_compressor_base_release(&self) -> f64 {
        self.processor.get_compressor_base_release()
    }

    /// Get current compressor release time (adaptive or base, in milliseconds)
    fn get_compressor_current_release(&self) -> f64 {
        let release_raw = self
            .processor
            .compressor_current_release_ms
            .load(Ordering::Relaxed);
        release_raw as f64 / 10.0 // Convert back from 0.1ms resolution
    }

    // === Auto Makeup Gain ===

    /// Set compressor auto makeup gain mode
    fn set_compressor_auto_makeup_enabled(&self, enabled: bool) {
        self.processor.set_compressor_auto_makeup_enabled(enabled);
    }

    /// Get compressor auto makeup gain mode
    fn get_compressor_auto_makeup_enabled(&self) -> bool {
        self.processor.get_compressor_auto_makeup_enabled()
    }

    /// Set compressor target LUFS
    fn set_compressor_target_lufs(&self, target_lufs: f64) {
        self.processor.set_compressor_target_lufs(target_lufs);
    }

    /// Get compressor target LUFS
    fn get_compressor_target_lufs(&self) -> f64 {
        self.processor.get_compressor_target_lufs()
    }

    /// Get compressor current LUFS
    fn get_compressor_current_lufs(&self) -> f64 {
        self.processor.get_compressor_current_lufs()
    }

    /// Get compressor current makeup gain
    fn get_compressor_current_makeup_gain(&self) -> f64 {
        self.processor.get_compressor_current_makeup_gain()
    }

    // === Limiter ===

    fn set_limiter_enabled(&self, enabled: bool) {
        self.processor.set_limiter_enabled(enabled);
    }

    fn is_limiter_enabled(&self) -> bool {
        self.processor.is_limiter_enabled()
    }

    fn set_limiter_ceiling(&self, ceiling_db: f64) {
        self.processor.set_limiter_ceiling(ceiling_db);
    }

    fn set_limiter_release(&self, release_ms: f64) {
        self.processor.set_limiter_release(release_ms);
    }

    // === Metering ===

    fn get_input_peak_db(&self) -> f32 {
        self.processor.get_input_peak_db()
    }

    fn get_input_rms_db(&self) -> f32 {
        self.processor.get_input_rms_db()
    }

    fn get_output_peak_db(&self) -> f32 {
        self.processor.get_output_peak_db()
    }

    fn get_output_rms_db(&self) -> f32 {
        self.processor.get_output_rms_db()
    }

    fn get_compressor_gain_reduction_db(&self) -> f32 {
        self.processor.get_compressor_gain_reduction_db()
    }

    fn get_latency_ms(&self) -> f32 {
        self.processor.get_latency_ms()
    }

    fn set_latency_compensation_ms(&self, compensation_ms: f32) {
        self.processor.set_latency_compensation_ms(compensation_ms);
    }

    fn get_latency_compensation_ms(&self) -> f32 {
        self.processor.get_latency_compensation_ms()
    }

    // === DSP Performance Metrics ===

    fn get_dsp_time_ms(&self) -> f32 {
        self.processor.get_dsp_time_ms()
    }

    fn get_input_buffer_samples(&self) -> u32 {
        self.processor.get_input_buffer_samples()
    }

    fn get_input_buffer_smoothed_samples(&self) -> u32 {
        self.processor.get_input_buffer_smoothed_samples()
    }

    fn get_output_buffer_samples(&self) -> u32 {
        self.processor.get_output_buffer_samples()
    }

    fn output_sample_rate(&self) -> u32 {
        self.processor.output_sample_rate()
    }

    fn get_rnnoise_buffer_samples(&self) -> u32 {
        self.processor.get_rnnoise_buffer_samples()
    }

    /// Get smoothed DSP processing time in milliseconds
    fn get_dsp_time_smoothed_ms(&self) -> f32 {
        let us = self.processor.dsp_time_smoothed_us.load(Ordering::Relaxed);
        us as f32 / 1000.0
    }

    /// Get smoothed suppressor buffer fill level in samples
    fn get_buffer_smoothed_samples(&self) -> u32 {
        self.processor.smoothed_buffer_len.load(Ordering::Relaxed)
    }

    // === Dropped Sample Tracking ===

    fn get_dropped_samples(&self) -> u64 {
        self.processor.get_dropped_samples()
    }

    fn reset_dropped_samples(&self) {
        self.processor.reset_dropped_samples();
    }

    fn get_lock_contention_count(&self) -> u64 {
        self.processor.get_lock_contention_count()
    }

    fn reset_lock_contention_count(&self) {
        self.processor.reset_lock_contention_count();
    }

    fn get_input_callback_age_ms(&self) -> u64 {
        self.processor.get_input_callback_age_ms()
    }

    fn get_output_callback_age_ms(&self) -> u64 {
        self.processor.get_output_callback_age_ms()
    }

    fn get_output_underrun_streak(&self) -> u32 {
        self.processor.get_output_underrun_streak()
    }

    fn get_output_underrun_total(&self) -> u64 {
        self.processor.get_output_underrun_total()
    }

    fn get_jitter_dropped_samples(&self) -> u64 {
        self.processor.get_jitter_dropped_samples()
    }

    fn get_output_recovery_count(&self) -> u64 {
        self.processor.get_output_recovery_count()
    }

    fn get_suppressor_non_finite_count(&self) -> u64 {
        self.processor.get_suppressor_non_finite_count()
    }

    fn is_noise_backend_available(&self) -> bool {
        self.processor.is_noise_backend_available()
    }

    fn noise_backend_failed(&self) -> bool {
        self.processor.noise_backend_failed()
    }

    fn noise_backend_error(&self) -> Option<String> {
        self.processor.noise_backend_error()
    }

    fn set_recovery_suppressed(&self, suppressed: bool) {
        self.processor.set_recovery_suppressed(suppressed);
    }

    fn is_recovery_suppressed(&self) -> bool {
        self.processor.is_recovery_suppressed()
    }

    fn get_runtime_diagnostics(&self, py: Python) -> PyResult<PyObject> {
        let diagnostics = pyo3::types::PyDict::new_bound(py);
        diagnostics.set_item("noise_model", self.processor.get_noise_model().id())?;
        diagnostics.set_item(
            "noise_backend_available",
            self.processor.is_noise_backend_available(),
        )?;
        diagnostics.set_item(
            "noise_backend_failed",
            self.processor.noise_backend_failed(),
        )?;
        diagnostics.set_item("noise_backend_error", self.processor.noise_backend_error())?;
        diagnostics.set_item(
            "input_dropped_samples",
            self.processor.get_dropped_samples(),
        )?;
        diagnostics.set_item(
            "input_backlog_recovery_count",
            self.processor
                .input_backlog_recovery_count
                .load(Ordering::Relaxed),
        )?;
        diagnostics.set_item(
            "input_backlog_dropped_samples",
            self.processor
                .input_backlog_dropped_samples
                .load(Ordering::Relaxed),
        )?;
        diagnostics.set_item(
            "lock_contention_count",
            self.processor.get_lock_contention_count(),
        )?;
        diagnostics.set_item(
            "output_underrun_total",
            self.processor.get_output_underrun_total(),
        )?;
        diagnostics.set_item(
            "jitter_dropped_samples",
            self.processor.get_jitter_dropped_samples(),
        )?;
        diagnostics.set_item(
            "output_recovery_count",
            self.processor.get_output_recovery_count(),
        )?;
        diagnostics.set_item(
            "output_short_write_dropped_samples",
            self.processor
                .output_short_write_dropped_samples
                .load(Ordering::Relaxed),
        )?;
        diagnostics.set_item(
            "stream_restart_count",
            self.processor.get_stream_restart_count(),
        )?;
        diagnostics.set_item(
            "last_restart_reason",
            self.processor.get_last_restart_reason(),
        )?;
        diagnostics.set_item("last_stream_error", self.processor.get_last_stream_error())?;
        diagnostics.set_item(
            "suppressor_non_finite_count",
            self.processor.get_suppressor_non_finite_count(),
        )?;
        diagnostics.set_item(
            "clip_event_count",
            self.processor.clip_event_count.load(Ordering::Relaxed),
        )?;
        diagnostics.set_item(
            "clip_peak_db",
            f32::from_bits(self.processor.clip_peak_db.load(Ordering::Relaxed)),
        )?;
        diagnostics.set_item(
            "input_resampler_active",
            self.processor
                .input_resampler_active
                .load(Ordering::Relaxed),
        )?;
        diagnostics.set_item(
            "output_resampler_active",
            self.processor
                .output_resampler_active
                .load(Ordering::Relaxed),
        )?;
        diagnostics.set_item("output_sample_rate", self.processor.output_sample_rate())?;
        diagnostics.set_item(
            "recovery_suppressed",
            self.processor.is_recovery_suppressed(),
        )?;
        diagnostics.set_item(
            "raw_monitor_enabled",
            self.processor.is_raw_monitor_enabled(),
        )?;
        Ok(diagnostics.into_any().unbind())
    }

    // === Stream Recovery Status ===

    /// Service pending recovery requests (returns None if no attempt).
    fn service_recovery(&mut self) -> Option<bool> {
        self.processor.service_recovery()
    }

    fn is_recovery_requested(&self) -> bool {
        self.processor.is_recovery_requested()
    }

    fn is_recovering(&self) -> bool {
        self.processor.is_recovering()
    }

    fn get_stream_restart_count(&self) -> u64 {
        self.processor.get_stream_restart_count()
    }

    fn get_last_stream_error(&self) -> Option<String> {
        self.processor.get_last_stream_error()
    }

    fn get_last_restart_reason(&self) -> Option<String> {
        self.processor.get_last_restart_reason()
    }

    // === RAW AUDIO RECORDING (for calibration) ===

    /// Start recording raw audio for calibration (10 seconds @ 48kHz)
    fn start_raw_recording(&mut self, duration_secs: f64) -> PyResult<()> {
        self.processor
            .start_raw_recording(duration_secs)
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)
    }

    /// Stop recording and return audio data as NumPy array
    fn stop_raw_recording(&mut self, py: Python) -> PyResult<PyObject> {
        if let Some(audio) = self.processor.stop_raw_recording() {
            // Zero-copy transfer to NumPy
            use numpy::PyArray1;
            let array = PyArray1::from_vec_bound(py, audio);
            Ok(array.into_any().unbind())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "No recording in progress",
            ))
        }
    }

    /// Check if recording is complete
    fn is_recording_complete(&mut self) -> bool {
        self.processor.is_recording_complete()
    }

    /// Get recording progress (0.0 to 1.0)
    fn recording_progress(&mut self) -> f32 {
        self.processor.recording_progress()
    }

    /// Get current recording level as RMS in dB (for level meter visualization)
    fn recording_level_db(&mut self) -> f32 {
        self.processor.recording_level_db()
    }

    /// Manually set output mute state (useful for calibration workflow)
    fn set_output_mute(&mut self, muted: bool) {
        self.processor.set_output_mute(muted);
    }
}
