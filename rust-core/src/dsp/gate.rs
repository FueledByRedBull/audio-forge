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
                        let (vad_gate_open, _probability) = vad.process(buffer);

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
