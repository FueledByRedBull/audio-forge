//! Downward expander with optional VAD gating.
//!
//! Reduces low-level background noise during pauses while preserving natural
//! decay. VAD modes can force closure when speech is not detected.

use crate::dsp::util;
#[cfg(feature = "vad")]
use crate::dsp::vad::{GateMode, VadAutoGate};

/// Enable debug output for gate operations
const GATE_DEBUG: bool = false;
const MIN_LEVEL_LINEAR: f64 = 1e-10;
const EXPANDER_RATIO: f64 = 4.0;
const EXPANDER_RANGE_DB: f64 = 36.0;

#[cfg(debug_assertions)]
macro_rules! gate_debug_log {
    ($($arg:tt)*) => {
        if GATE_DEBUG {
            eprintln!($($arg)*);
        }
    };
}

#[cfg(not(debug_assertions))]
macro_rules! gate_debug_log {
    ($($arg:tt)*) => {};
}

/// Noise gate processor implemented as a downward expander.
pub struct NoiseGate {
    /// Threshold in dB (e.g., -40.0)
    threshold_db: f64,
    /// Attack time constant (exponential smoothing coefficient)
    attack_coeff: f64,
    /// Release time constant (exponential smoothing coefficient)
    release_coeff: f64,
    /// Log-domain peak envelope follower (dBFS).
    peak_envelope_db: f64,
    /// Current linear gain applied to the signal.
    current_gain: f64,
    /// Sample rate
    sample_rate: f64,
    /// Whether gate is currently open (threshold-only detector state).
    is_open: bool,
    /// Whether gate is enabled
    enabled: bool,
    /// Previous gate state (for change detection in debug)
    was_open: bool,
    /// Sample counter for periodic debug output
    debug_counter: usize,
    /// Peak level since last debug output
    peak_level: f64,
    /// Previous VAD gate state (for VAD-specific change detection)
    #[cfg(feature = "vad")]
    vad_was_open: bool,
    /// Gate operating mode (VAD feature only)
    #[cfg(feature = "vad")]
    gate_mode: GateMode,
    /// VAD auto-gate controller (VAD feature only)
    #[cfg(feature = "vad")]
    vad_auto_gate: Option<VadAutoGate>,
}

impl NoiseGate {
    /// Create a new expander-style noise gate.
    pub fn new(threshold_db: f64, attack_ms: f64, release_ms: f64, sample_rate: f64) -> Self {
        let attack_coeff = util::time_constant_to_coeff(attack_ms, sample_rate);
        let release_coeff = util::time_constant_to_coeff(release_ms, sample_rate);

        if GATE_DEBUG {
            gate_debug_log!(
                "[GATE] Initialized: threshold={}dB, attack={}ms, release={}ms",
                threshold_db,
                attack_ms,
                release_ms
            );
        }

        Self {
            threshold_db,
            attack_coeff,
            release_coeff,
            peak_envelope_db: -120.0,
            current_gain: 0.0,
            sample_rate,
            is_open: false,
            enabled: true,
            was_open: false,
            debug_counter: 0,
            peak_level: f64::MIN,
            #[cfg(feature = "vad")]
            vad_was_open: false,
            #[cfg(feature = "vad")]
            gate_mode: GateMode::ThresholdOnly,
            #[cfg(feature = "vad")]
            vad_auto_gate: None,
        }
    }

    /// Update gate parameters
    pub fn set_threshold(&mut self, threshold_db: f64) {
        self.threshold_db = threshold_db;
        #[cfg(feature = "vad")]
        if let Some(vad) = &mut self.vad_auto_gate {
            vad.set_manual_threshold(threshold_db as f32);
        }
    }

    /// Get current threshold in dB
    pub fn threshold_db(&self) -> f64 {
        self.threshold_db
    }

    /// Set attack time
    pub fn set_attack_time(&mut self, attack_ms: f64) {
        self.attack_coeff = util::time_constant_to_coeff(attack_ms, self.sample_rate);
    }

    /// Set release time
    pub fn set_release_time(&mut self, release_ms: f64) {
        self.release_coeff = util::time_constant_to_coeff(release_ms, self.sample_rate);
    }

    /// Enable or disable the gate
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if gate is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Check if gate is currently open based on detector threshold.
    pub fn is_open(&self) -> bool {
        self.is_open
    }

    #[inline]
    fn update_detector(&mut self, input: f64) {
        let input_db = util::linear_to_db(input.abs(), MIN_LEVEL_LINEAR);
        let coeff = if input_db > self.peak_envelope_db {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.peak_envelope_db = coeff * self.peak_envelope_db + (1.0 - coeff) * input_db;
        self.is_open = self.peak_envelope_db >= self.threshold_db;
        if self.peak_envelope_db > self.peak_level {
            self.peak_level = self.peak_envelope_db;
        }
    }

    #[inline]
    fn compute_target_gr_db(&self, force_close: bool) -> f64 {
        if force_close {
            return EXPANDER_RANGE_DB;
        }
        if self.peak_envelope_db >= self.threshold_db {
            return 0.0;
        }

        // `target_gr_db` is always non-negative and represents gain reduction
        // magnitude in dB (not a signed gain value).
        ((self.threshold_db - self.peak_envelope_db) * (1.0 - 1.0 / EXPANDER_RATIO))
            .clamp(0.0, EXPANDER_RANGE_DB)
    }

    #[inline]
    fn apply_gain(&mut self, input: f64, target_gr_db: f64) -> f32 {
        let target_gain = util::db_to_linear(-target_gr_db);
        let coeff = if target_gain > self.current_gain {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.current_gain = coeff * self.current_gain + (1.0 - coeff) * target_gain;
        (input * self.current_gain) as f32
    }

    /// Process a single sample in threshold-only mode.
    #[inline]
    pub fn process_sample(&mut self, input: f32) -> f32 {
        if !self.enabled {
            return input;
        }

        let input_f64 = input as f64;
        self.update_detector(input_f64);
        let target_gr_db = self.compute_target_gr_db(false);
        self.apply_gain(input_f64, target_gr_db)
    }

    /// Process a block of samples in-place.
    pub fn process_block_inplace(&mut self, buffer: &mut [f32]) {
        if !self.enabled {
            return;
        }

        #[cfg(feature = "vad")]
        {
            if self.gate_mode != GateMode::ThresholdOnly {
                if let Some(vad) = &mut self.vad_auto_gate {
                    if vad.is_enabled() {
                        let (vad_gate_open, _probability) = vad.process(buffer);

                        if GATE_DEBUG {
                            self.debug_counter += buffer.len();
                            if self.debug_counter >= 48_000 {
                                let _mode_str = match self.gate_mode {
                                    GateMode::VadAssisted => "VAD-Assisted",
                                    GateMode::VadOnly => "VAD-Only",
                                    GateMode::ThresholdOnly => "Threshold-Only",
                                };
                                gate_debug_log!(
                                    "[GATE] mode={}, prob={:.4}, vad_gate={}",
                                    _mode_str,
                                    _probability,
                                    if vad_gate_open { "OPEN" } else { "CLOSED" }
                                );
                                self.debug_counter = 0;
                            }
                        }

                        if GATE_DEBUG && vad_gate_open != self.vad_was_open {
                            gate_debug_log!(
                                "[GATE] VAD transition: {} prob={:.4}",
                                if vad_gate_open { "OPEN" } else { "CLOSED" },
                                _probability
                            );
                            self.vad_was_open = vad_gate_open;
                        }

                        for sample in buffer.iter_mut() {
                            let input_f64 = *sample as f64;
                            // Detector tracks continuously even when VAD blocks.
                            self.update_detector(input_f64);
                            let force_close = match self.gate_mode {
                                GateMode::ThresholdOnly => false,
                                GateMode::VadAssisted | GateMode::VadOnly => !vad_gate_open,
                            };
                            let target_gr_db = self.compute_target_gr_db(force_close);
                            *sample = self.apply_gain(input_f64, target_gr_db);
                        }
                        return;
                    }
                }
            }
        }

        for sample in buffer.iter_mut() {
            *sample = self.process_sample(*sample);
        }

        if GATE_DEBUG && self.is_open != self.was_open {
            gate_debug_log!(
                "[GATE] {} detector={:.2}dB threshold={:.2}dB",
                if self.is_open { "OPENED" } else { "CLOSED" },
                self.peak_envelope_db,
                self.threshold_db
            );
            self.was_open = self.is_open;
        }
    }

    /// Reset gate state
    pub fn reset(&mut self) {
        self.current_gain = 0.0;
        self.peak_envelope_db = -120.0;
        self.is_open = false;
        self.was_open = false;
        #[cfg(feature = "vad")]
        {
            self.vad_was_open = false;
        }
    }

    /// Get current envelope level (0.0 to 1.0)
    pub fn current_envelope(&self) -> f64 {
        self.current_gain
    }

    /// Get current gain applied by the gate.
    pub fn current_gain(&self) -> f32 {
        self.current_gain as f32
    }

    // === VAD Integration Methods ===

    #[cfg(feature = "vad")]
    /// Set gate mode
    pub fn set_gate_mode(&mut self, mode: GateMode) {
        self.gate_mode = mode;
        if let Some(vad) = &mut self.vad_auto_gate {
            vad.set_gate_mode(mode);
        }
    }

    #[cfg(feature = "vad")]
    /// Get current gate mode
    pub fn gate_mode(&self) -> GateMode {
        self.gate_mode
    }

    #[cfg(feature = "vad")]
    /// Set VAD auto-gate controller
    pub fn set_vad_auto_gate(&mut self, vad: Option<VadAutoGate>) {
        self.vad_auto_gate = vad;
        if let Some(vad) = &mut self.vad_auto_gate {
            vad.set_manual_threshold(self.threshold_db as f32);
        }
    }

    #[cfg(feature = "vad")]
    /// Check if VAD backend is available (model loaded and runtime initialized)
    pub fn is_vad_available(&self) -> bool {
        self.vad_auto_gate
            .as_ref()
            .map(|v| v.is_available())
            .unwrap_or(false)
    }

    #[cfg(feature = "vad")]
    /// Get VAD probability (for metering)
    pub fn get_vad_probability(&self) -> f32 {
        if let Some(vad) = &self.vad_auto_gate {
            vad.probability()
        } else {
            0.0
        }
    }

    #[cfg(feature = "vad")]
    /// Set VAD probability threshold
    pub fn set_vad_threshold(&mut self, threshold: f32) {
        if let Some(vad) = &mut self.vad_auto_gate {
            vad.set_vad_threshold(threshold);
        }
    }

    #[cfg(feature = "vad")]
    /// Set VAD hold time in milliseconds
    pub fn set_hold_time(&mut self, hold_ms: f32) {
        if let Some(vad) = &mut self.vad_auto_gate {
            vad.set_hold_time(hold_ms);
        }
    }

    #[cfg(feature = "vad")]
    /// Set VAD pre-gain to boost weak signals for better speech detection
    /// Default is 1.0 (no gain). Values > 1.0 boost the signal.
    pub fn set_vad_pre_gain(&mut self, gain: f32) {
        if let Some(vad) = &mut self.vad_auto_gate {
            vad.set_pre_gain(gain);
        }
    }

    #[cfg(feature = "vad")]
    /// Get current VAD pre-gain
    pub fn vad_pre_gain(&self) -> f32 {
        if let Some(vad) = &self.vad_auto_gate {
            vad.pre_gain()
        } else {
            1.0
        }
    }

    #[cfg(feature = "vad")]
    /// Enable/disable auto-threshold mode
    pub fn set_auto_threshold(&mut self, enabled: bool) {
        if let Some(vad) = &mut self.vad_auto_gate {
            vad.set_auto_threshold(enabled);
        }
    }

    #[cfg(feature = "vad")]
    /// Set margin above noise floor for auto-threshold (in dB)
    pub fn set_margin(&mut self, margin_db: f32) {
        if let Some(vad) = &mut self.vad_auto_gate {
            vad.set_margin(margin_db);
        }
    }

    #[cfg(feature = "vad")]
    /// Get current margin above noise floor (in dB)
    pub fn margin(&self) -> f32 {
        self.vad_auto_gate
            .as_ref()
            .map(|v| v.margin())
            .unwrap_or(6.0)
    }

    #[cfg(feature = "vad")]
    /// Check if auto-threshold is enabled
    pub fn auto_threshold_enabled(&self) -> bool {
        self.vad_auto_gate
            .as_ref()
            .map(|v| v.auto_threshold_enabled())
            .unwrap_or(false)
    }

    #[cfg(feature = "vad")]
    /// Get current noise floor estimate (for UI display)
    pub fn noise_floor(&self) -> f32 {
        self.vad_auto_gate
            .as_ref()
            .map(|v| v.noise_floor())
            .unwrap_or(-60.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_gate_opens_above_threshold() {
        let mut gate = NoiseGate::new(-40.0, 10.0, 100.0, 48_000.0);
        let strong_signal = vec![0.1f32; 3_000];
        for sample in &strong_signal {
            gate.process_sample(*sample);
        }
        assert!(gate.current_gain() > 0.8);
    }

    #[test]
    fn test_noise_gate_closes_below_threshold() {
        let mut gate = NoiseGate::new(-40.0, 10.0, 100.0, 48_000.0);
        for _ in 0..3_000 {
            gate.process_sample(0.1);
        }
        let open_gain = gate.current_gain();
        for _ in 0..10_000 {
            gate.process_sample(0.0001);
        }
        assert!(gate.current_gain() < open_gain * 0.7);
        assert!(gate.current_gain() < 0.5);
    }

    #[test]
    fn test_noise_gate_expander_monotonic_gain() {
        let mut gate = NoiseGate::new(-40.0, 1.0, 1.0, 48_000.0);
        for _ in 0..2_000 {
            gate.process_sample(0.1);
        }
        let high_gain = gate.current_gain();

        gate.reset();
        for _ in 0..2_000 {
            gate.process_sample(0.0005);
        }
        let low_gain = gate.current_gain();

        assert!(high_gain > low_gain);
    }

    #[test]
    fn test_noise_gate_range_cap() {
        let mut gate = NoiseGate::new(-40.0, 1.0, 1.0, 48_000.0);
        for _ in 0..4_000 {
            gate.process_sample(0.0);
        }
        let floor_gain = util::db_to_linear(-EXPANDER_RANGE_DB) as f32;
        assert!((gate.current_gain() - floor_gain).abs() < 0.02);
    }

    #[test]
    fn test_noise_gate_force_close_transitions_are_smoothed() {
        let mut gate = NoiseGate::new(-40.0, 10.0, 100.0, 48_000.0);
        gate.current_gain = 1.0;
        let floor_gain = util::db_to_linear(-EXPANDER_RANGE_DB);

        let _ = gate.apply_gain(0.5, EXPANDER_RANGE_DB);
        assert!(gate.current_gain > floor_gain + 1e-3);
    }

    #[test]
    fn test_noise_gate_disabled() {
        let mut gate = NoiseGate::new(-40.0, 10.0, 100.0, 48_000.0);
        gate.set_enabled(false);

        let input = 0.0001f32;
        let output = gate.process_sample(input);

        assert_eq!(output, input);
    }
}
