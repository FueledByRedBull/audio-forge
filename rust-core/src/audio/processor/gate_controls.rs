impl AudioProcessor {
    // === Noise Gate Controls ===

    /// Enable/disable noise gate
    pub fn set_gate_enabled(&self, enabled: bool) {
        self.gate_enabled.store(enabled, Ordering::Release);
        if let Ok(mut control) = self.gate_control.lock() {
            control.enabled = enabled;
        }
        self.gate_rt_control.set_enabled(enabled);
        self.gate_dirty.store(true, Ordering::Release);
    }

    /// Set noise gate threshold
    pub fn set_gate_threshold(&self, threshold_db: f64) {
        let Some(threshold_db) =
            clamp_control_value(threshold_db, GATE_THRESHOLD_MIN_DB, GATE_THRESHOLD_MAX_DB)
        else {
            return;
        };
        if let Ok(mut control) = self.gate_control.lock() {
            control.threshold_db = threshold_db;
        }
        self.gate_rt_control.set_threshold_db(threshold_db);
        self.gate_dirty.store(true, Ordering::Release);
    }

    /// Set noise gate attack time
    pub fn set_gate_attack(&self, attack_ms: f64) {
        let Some(attack_ms) =
            clamp_control_value(attack_ms, GATE_ATTACK_MIN_MS, GATE_ATTACK_MAX_MS)
        else {
            return;
        };
        if let Ok(mut control) = self.gate_control.lock() {
            control.attack_ms = attack_ms;
        }
        self.gate_rt_control.set_attack_ms(attack_ms);
        self.gate_dirty.store(true, Ordering::Release);
    }

    /// Set noise gate release time
    pub fn set_gate_release(&self, release_ms: f64) {
        let Some(release_ms) =
            clamp_control_value(release_ms, GATE_RELEASE_MIN_MS, GATE_RELEASE_MAX_MS)
        else {
            return;
        };
        if let Ok(mut control) = self.gate_control.lock() {
            control.release_ms = release_ms;
        }
        self.gate_rt_control.set_release_ms(release_ms);
        self.gate_dirty.store(true, Ordering::Release);
    }

    /// Check if gate is enabled
    pub fn is_gate_enabled(&self) -> bool {
        self.gate_enabled.load(Ordering::Acquire)
    }

    // === VAD Gate Controls (VAD feature only) ===

    #[cfg(feature = "vad")]
    /// Set gate mode
    pub fn set_gate_mode(&self, mode: u8) -> Result<(), String> {
        let gate_mode = match mode {
            0 => GateMode::ThresholdOnly,
            1 => GateMode::VadAssisted,
            2 => GateMode::VadOnly,
            _ => return Err("Invalid gate mode".to_string()),
        };
        if let Ok(mut control) = self.gate_control.lock() {
            control.gate_mode = gate_mode;
        }
        self.gate_rt_control.set_gate_mode(gate_mode);
        self.gate_dirty.store(true, Ordering::Release);
        Ok(())
    }

    #[cfg(feature = "vad")]
    /// Get VAD speech probability (0.0-1.0)
    pub fn get_vad_probability(&self) -> f32 {
        f32::from_bits(self.vad_probability.load(Ordering::Relaxed))
    }

    #[cfg(feature = "vad")]
    /// Get fused gate open score (0.0-1.0)
    pub fn get_gate_fused_score(&self) -> f32 {
        f32::from_bits(self.gate_fused_score.load(Ordering::Relaxed))
    }

    #[cfg(feature = "vad")]
    /// Check whether VAD backend is available (model/runtime loaded)
    pub fn is_vad_available(&self) -> bool {
        self.vad_available.load(Ordering::Relaxed)
    }

    #[cfg(feature = "vad")]
    /// Set VAD probability threshold (0.0-1.0)
    pub fn set_vad_threshold(&self, threshold: f32) {
        let Some(threshold) =
            clamp_control_value_f32(threshold, VAD_THRESHOLD_MIN, VAD_THRESHOLD_MAX)
        else {
            return;
        };
        if let Ok(mut control) = self.gate_control.lock() {
            control.vad_threshold = threshold;
        }
        self.gate_rt_control.set_vad_threshold(threshold);
        self.gate_dirty.store(true, Ordering::Release);
    }

    #[cfg(feature = "vad")]
    /// Set VAD hold time in milliseconds
    pub fn set_vad_hold_time(&self, hold_ms: f32) {
        let Some(hold_ms) = clamp_control_value_f32(hold_ms, VAD_HOLD_MIN_MS, VAD_HOLD_MAX_MS)
        else {
            return;
        };
        if let Ok(mut control) = self.gate_control.lock() {
            control.hold_ms = hold_ms;
        }
        self.gate_rt_control.set_hold_ms(hold_ms);
        self.gate_dirty.store(true, Ordering::Release);
    }

    #[cfg(feature = "vad")]
    /// Set VAD pre-gain to boost weak signals for better speech detection
    /// Default is 1.0 (no gain). Values > 1.0 boost the signal.
    /// This helps with quiet microphones where VAD can't detect speech.
    pub fn set_vad_pre_gain(&self, gain: f32) {
        let Some(gain) = clamp_control_value_f32(gain, VAD_PRE_GAIN_MIN, VAD_PRE_GAIN_MAX) else {
            return;
        };
        if let Ok(mut control) = self.gate_control.lock() {
            control.pre_gain = gain;
        }
        self.gate_rt_control.set_pre_gain(gain);
        self.gate_dirty.store(true, Ordering::Release);
    }

    #[cfg(feature = "vad")]
    /// Get current VAD pre-gain
    pub fn vad_pre_gain(&self) -> f32 {
        self.gate_control
            .lock()
            .map(|control| control.pre_gain)
            .unwrap_or(1.0)
    }

    #[cfg(feature = "vad")]
    /// Enable/disable auto-threshold mode (automatically adjusts gate threshold based on noise floor)
    pub fn set_auto_threshold(&self, enabled: bool) {
        if let Ok(mut control) = self.gate_control.lock() {
            control.auto_threshold = enabled;
        }
        self.gate_rt_control.set_auto_threshold(enabled);
        self.gate_dirty.store(true, Ordering::Release);
    }

    #[cfg(feature = "vad")]
    /// Set margin above noise floor for auto-threshold (in dB)
    pub fn set_gate_margin(&self, margin_db: f32) {
        let Some(margin_db) =
            clamp_control_value_f32(margin_db, GATE_MARGIN_MIN_DB, GATE_MARGIN_MAX_DB)
        else {
            return;
        };
        if let Ok(mut control) = self.gate_control.lock() {
            control.margin_db = margin_db;
        }
        self.gate_rt_control.set_margin_db(margin_db);
        self.gate_dirty.store(true, Ordering::Release);
    }

    #[cfg(feature = "vad")]
    /// Get current noise floor estimate (in dB)
    pub fn get_noise_floor(&self) -> f32 {
        f32::from_bits(self.gate_noise_floor_db.load(Ordering::Relaxed))
    }

    #[cfg(feature = "vad")]
    /// Get current gate margin (in dB)
    pub fn gate_margin(&self) -> f32 {
        self.gate_control
            .lock()
            .map(|control| control.margin_db)
            .unwrap_or(10.0)
    }

    #[cfg(feature = "vad")]
    /// Check if auto-threshold is enabled
    pub fn auto_threshold_enabled(&self) -> bool {
        self.gate_control
            .lock()
            .map(|control| control.auto_threshold)
            .unwrap_or(false)
    }


}
