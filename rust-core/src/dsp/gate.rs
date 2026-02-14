//! Noise gate with IIR envelope detection
//!
//! Reduces gain when signal level falls below threshold, useful for
//! removing background noise during silent periods.
//!
//! Uses single-pole IIR filter for RMS approximation instead of sliding window,
//! reducing memory from ~2400 samples to just 2 state variables.
//!
//! Adapted from Spectral Workbench project.

#[cfg(feature = "vad")]
use crate::dsp::vad::{GateMode, VadAutoGate};

/// Enable debug output for gate operations
#[cfg(debug_assertions)]
const GATE_DEBUG: bool = true;

#[cfg(not(debug_assertions))]
const GATE_DEBUG: bool = false;

/// Noise gate processor with IIR envelope follower
pub struct NoiseGate {
    /// Threshold in dB (e.g., -40.0)
    threshold_db: f64,

    /// Attack time constant (exponential smoothing coefficient)
    attack_coeff: f64,

    /// Release time constant (exponential smoothing coefficient)
    release_coeff: f64,

    /// Current envelope level (linear amplitude, not dB)
    envelope: f64,

    /// IIR envelope squared (for RMS approximation)
    envelope_squared: f64,

    /// RMS smoothing coefficient (single-pole IIR, ~50ms time constant)
    rms_coeff: f64,

    /// Sample rate
    sample_rate: f64,

    /// Whether gate is currently open (for hysteresis)
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
    /// Create a new noise gate
    ///
    /// # Arguments
    /// * `threshold_db` - Threshold in dB below which gate closes (e.g., -40.0)
    /// * `attack_ms` - Attack time in milliseconds (e.g., 10.0)
    /// * `release_ms` - Release time in milliseconds (e.g., 100.0)
    /// * `sample_rate` - Sample rate in Hz
    pub fn new(threshold_db: f64, attack_ms: f64, release_ms: f64, sample_rate: f64) -> Self {
        // Calculate time constants for exponential smoothing
        // tau = time_ms / 1000, coeff = exp(-1 / (tau * sample_rate))
        let attack_coeff = Self::time_constant_to_coeff(attack_ms, sample_rate);
        let release_coeff = Self::time_constant_to_coeff(release_ms, sample_rate);

        // RMS smoothing: 50ms time constant for IIR envelope follower
        let rms_coeff = Self::time_constant_to_coeff(50.0, sample_rate);

        if GATE_DEBUG {
            eprintln!("[GATE] Initialized: threshold={}dB, attack={}ms, release={}ms",
                threshold_db, attack_ms, release_ms);
        }

        Self {
            threshold_db,
            attack_coeff,
            release_coeff,
            envelope: 0.0,
            envelope_squared: 0.0,
            rms_coeff,
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

    /// Convert time constant in ms to exponential smoothing coefficient
    fn time_constant_to_coeff(time_ms: f64, sample_rate: f64) -> f64 {
        let tau = time_ms / 1000.0; // Convert to seconds
        (-1.0 / (tau * sample_rate)).exp()
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
        self.attack_coeff = Self::time_constant_to_coeff(attack_ms, self.sample_rate);
    }

    /// Set release time
    pub fn set_release_time(&mut self, release_ms: f64) {
        self.release_coeff = Self::time_constant_to_coeff(release_ms, self.sample_rate);
    }

    /// Enable or disable the gate
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if gate is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Check if gate is currently open
    pub fn is_open(&self) -> bool {
        self.is_open
    }

    /// Process a single sample
    #[inline]
    pub fn process_sample(&mut self, input: f32) -> f32 {
        if !self.enabled {
            return input;
        }

        let input_f64 = input as f64;

        // IIR envelope follower (RMS approximation)
        // Much more efficient than sliding window: 2 state variables vs ~2400 samples
        let input_squared = input_f64 * input_f64;
        self.envelope_squared =
            self.rms_coeff * self.envelope_squared + (1.0 - self.rms_coeff) * input_squared;

        // Calculate RMS from smoothed squared envelope
        let rms = self.envelope_squared.sqrt();

        // Convert to dB (with small epsilon to avoid log(0))
        let level_db = 20.0 * (rms + 1e-10).log10();

        // Track peak level for debug output
        if level_db > self.peak_level {
            self.peak_level = level_db;
        }

        // Determine if gate should be open
        // Use hysteresis: different thresholds for opening and closing
        let hysteresis_db = 3.0; // 3 dB hysteresis

        if self.is_open {
            // Gate is open: close if level drops below (threshold - hysteresis)
            if level_db < self.threshold_db - hysteresis_db {
                self.is_open = false;
            }
        } else {
            // Gate is closed: open if level rises above threshold
            if level_db >= self.threshold_db {
                self.is_open = true;
            }
        }

        // Debug output on state change
        if GATE_DEBUG && self.is_open != self.was_open {
            if self.is_open {
                eprintln!("[GATE] OPENED: level={:.1}dB >= threshold={:.1}dB",
                    level_db, self.threshold_db);
            } else {
                eprintln!("[GATE] CLOSED: level={:.1}dB < (threshold-hysteresis)={:.1}dB",
                    level_db, self.threshold_db - hysteresis_db);
            }
            self.was_open = self.is_open;
        }

        // Calculate target gain (0.0 when closed, 1.0 when open)
        let target_gain = if self.is_open { 1.0 } else { 0.0 };

        // Smooth gain transitions with attack/release
        let coeff = if target_gain > self.envelope {
            self.attack_coeff // Opening: use attack time
        } else {
            self.release_coeff // Closing: use release time
        };

        // Exponential smoothing: envelope = coeff * envelope + (1 - coeff) * target
        self.envelope = coeff * self.envelope + (1.0 - coeff) * target_gain;

        // Apply gain
        (input_f64 * self.envelope) as f32
    }

    /// Process a block of samples in-place
    pub fn process_block_inplace(&mut self, buffer: &mut [f32]) {
        if !self.enabled {
            return;
        }

        #[cfg(feature = "vad")]
        {
            // Check if we should use VAD-based gate decision
            if self.gate_mode != GateMode::ThresholdOnly {
                if let Some(vad) = &mut self.vad_auto_gate {
                    if vad.is_enabled() {
                        // Get VAD gate decision and probability
                        let (vad_gate_open, probability) = vad.process(buffer);

                        // Debug VAD probability and decision
                        if GATE_DEBUG {
                            self.debug_counter += buffer.len();
                            // Print VAD status every ~1 second (48000 samples)
                            if self.debug_counter >= 48000 {
                                let mode_str = match self.gate_mode {
                                    GateMode::VadAssisted => "VAD-Assisted",
                                    GateMode::VadOnly => "VAD-Only",
                                    GateMode::ThresholdOnly => "Threshold-Only",
                                };
                                eprintln!("[GATE] VAD mode={}, probability={:.4}, gate_state={}, threshold={:.2}",
                                    mode_str, probability, if vad_gate_open { "OPEN" } else { "CLOSED" },
                                    vad.vad_threshold());
                                self.debug_counter = 0;
                            }
                        }

                        // Check for VAD state change
                        if GATE_DEBUG && vad_gate_open != self.vad_was_open {
                            if vad_gate_open {
                                // FIX: Don't claim prob >= threshold, as it might be RMS or Hold triggered
                                eprintln!("[GATE] OPENED: probability={:.4}, threshold={:.2} (Triggered by VAD, RMS, or Hold Timer)",
                                    probability, vad.vad_threshold());
                            } else {
                                eprintln!("[GATE] CLOSED: probability={:.4} < threshold={:.2}",
                                    probability, vad.vad_threshold());
                            }
                            self.vad_was_open = vad_gate_open;
                        }

                        // Override is_open based on VAD decision
                        self.is_open = vad_gate_open;

                        // Apply VAD gate state to entire buffer
                        let gain = if self.is_open { 1.0 } else { 0.0 };
                        for sample in buffer.iter_mut() {
                            *sample *= gain as f32;
                        }
                        return;
                    }
                }
            }
        }

        // Default: use level-based gate
        for sample in buffer.iter_mut() {
            *sample = self.process_sample(*sample);
        }
    }

    /// Reset gate state
    pub fn reset(&mut self) {
        self.envelope = 0.0;
        self.envelope_squared = 0.0;
        self.is_open = false;
        self.was_open = false;
        #[cfg(feature = "vad")]
        {
            self.vad_was_open = false;
        }
    }

    /// Get current envelope level (0.0 to 1.0)
    pub fn current_envelope(&self) -> f64 {
        self.envelope
    }

    // === VAD Integration Methods ===

    #[cfg(feature = "vad")]
    /// Set gate mode
    pub fn set_gate_mode(&mut self, mode: GateMode) {
        self.gate_mode = mode;
        // FIX: Propagate the mode change to the inner VAD controller
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
            // Keep VAD-assisted manual threshold aligned with gate threshold UI.
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
        self.vad_auto_gate.as_ref()
            .map(|v| v.margin())
            .unwrap_or(6.0)
    }

    #[cfg(feature = "vad")]
    /// Check if auto-threshold is enabled
    pub fn auto_threshold_enabled(&self) -> bool {
        self.vad_auto_gate.as_ref()
            .map(|v| v.auto_threshold_enabled())
            .unwrap_or(false)
    }

    #[cfg(feature = "vad")]
    /// Get current noise floor estimate (for UI display)
    pub fn noise_floor(&self) -> f32 {
        self.vad_auto_gate.as_ref()
            .map(|v| v.noise_floor())
            .unwrap_or(-60.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_gate_opens_above_threshold() {
        let mut gate = NoiseGate::new(-40.0, 10.0, 100.0, 48000.0);

        // Feed strong signal (should open gate)
        // With IIR envelope follower, need longer signal to build up envelope
        let strong_signal = vec![0.1f32; 5000]; // About -20 dB, longer for IIR settling
        for sample in &strong_signal {
            gate.process_sample(*sample);
        }

        // After processing, envelope should be high
        assert!(gate.current_envelope() > 0.5);
    }

    #[test]
    fn test_noise_gate_closes_below_threshold() {
        let mut gate = NoiseGate::new(-40.0, 10.0, 100.0, 48000.0);

        // First open the gate with strong signal (longer for IIR)
        let strong_signal = vec![0.1f32; 5000];
        for sample in &strong_signal {
            gate.process_sample(*sample);
        }

        // Then feed weak signal (should close gate)
        // With IIR, envelope decays exponentially - need more samples
        let weak_signal = vec![0.0001f32; 20000]; // About -80 dB
        for sample in &weak_signal {
            gate.process_sample(*sample);
        }

        // Envelope should be low after sufficient decay time
        assert!(
            gate.current_envelope() < 0.5,
            "Expected envelope < 0.5, got {}",
            gate.current_envelope()
        );
    }

    #[test]
    fn test_noise_gate_hysteresis() {
        let mut gate = NoiseGate::new(-40.0, 1.0, 1.0, 48000.0);

        // Gate starts closed
        assert!(!gate.is_open());

        // Signal well above threshold should open it (longer for IIR settling)
        let signal_above = vec![0.02f32; 3000]; // About -34 dB, clearly above threshold
        for sample in &signal_above {
            gate.process_sample(*sample);
        }
        assert!(gate.is_open(), "Gate should be open after strong signal");

        // Signal slightly below threshold shouldn't close it immediately (hysteresis)
        // Need to stay above (threshold - hysteresis) = -43 dB
        let signal_slightly_below = vec![0.008f32; 1000]; // About -42 dB
        for sample in &signal_slightly_below {
            gate.process_sample(*sample);
        }
        assert!(
            gate.is_open(),
            "Gate should still be open due to hysteresis"
        );
    }

    #[test]
    fn test_noise_gate_disabled() {
        let mut gate = NoiseGate::new(-40.0, 10.0, 100.0, 48000.0);
        gate.set_enabled(false);

        let input = 0.0001f32;
        let output = gate.process_sample(input);

        assert_eq!(output, input);
    }
}
