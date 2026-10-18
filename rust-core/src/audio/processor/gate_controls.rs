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
            self.clamp_live_control(threshold_db, GATE_THRESHOLD_MIN_DB, GATE_THRESHOLD_MAX_DB)
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
            self.clamp_live_control(attack_ms, GATE_ATTACK_MIN_MS, GATE_ATTACK_MAX_MS)
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
            self.clamp_live_control(release_ms, GATE_RELEASE_MIN_MS, GATE_RELEASE_MAX_MS)
        else {
            return;
        };
        if let Ok(mut control) = self.gate_control.lock() {
            control.release_ms = release_ms;
        }
        self.gate_rt_control.set_release_ms(release_ms);
        self.gate_dirty.store(true, Ordering::Release);
    }

    /// Number of rapid gate open/close chatter events since processing start.
    pub fn get_gate_chatter_event_count(&self) -> u64 {
        self.gate_chatter_event_count.load(Ordering::Relaxed)
    }

    /// Whether VAD chatter mitigation is temporarily relaxing the gate.
    pub fn is_gate_auto_relax_active(&self) -> bool {
        self.gate_auto_relax_active.load(Ordering::Relaxed)
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
        f32::from_bits(self.vad_meter_probability.load(Ordering::Relaxed))
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
            self.clamp_live_control_f32(threshold, VAD_THRESHOLD_MIN, VAD_THRESHOLD_MAX)
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
        let Some(hold_ms) = self.clamp_live_control_f32(hold_ms, VAD_HOLD_MIN_MS, VAD_HOLD_MAX_MS)
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
        let Some(gain) = self.clamp_live_control_f32(gain, VAD_PRE_GAIN_MIN, VAD_PRE_GAIN_MAX)
        else {
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
            self.clamp_live_control_f32(margin_db, GATE_MARGIN_MIN_DB, GATE_MARGIN_MAX_DB)
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
    /// Check if auto-threshold is enabled
    pub fn auto_threshold_enabled(&self) -> bool {
        self.gate_control
            .lock()
            .map(|control| control.auto_threshold)
            .unwrap_or(false)
    }
}

impl AtomicGateControlState {
    /// Publish every field in one seqlock update (see the de-esser version).
    fn publish(&self, state: &GateControlState) {
        self.update(|atomic| {
            atomic.enabled.store(state.enabled, Ordering::Relaxed);
            atomic
                .threshold_db_bits
                .store(state.threshold_db.to_bits(), Ordering::Relaxed);
            atomic
                .attack_ms_bits
                .store(state.attack_ms.to_bits(), Ordering::Relaxed);
            atomic
                .release_ms_bits
                .store(state.release_ms.to_bits(), Ordering::Relaxed);
            #[cfg(feature = "vad")]
            {
                atomic
                    .gate_mode
                    .store(state.gate_mode as u8, Ordering::Relaxed);
                atomic
                    .vad_threshold_bits
                    .store(state.vad_threshold.to_bits(), Ordering::Relaxed);
                atomic
                    .hold_ms_bits
                    .store(state.hold_ms.to_bits(), Ordering::Relaxed);
                atomic
                    .pre_gain_bits
                    .store(state.pre_gain.to_bits(), Ordering::Relaxed);
                atomic
                    .auto_threshold
                    .store(state.auto_threshold, Ordering::Relaxed);
                atomic
                    .margin_db_bits
                    .store(state.margin_db.to_bits(), Ordering::Relaxed);
            }
        });
    }
}

impl AudioProcessor {
    /// Apply a complete gate configuration as one published snapshot.
    ///
    /// Each value is clamped like its single setter and a non-finite value
    /// keeps the current one.
    fn apply_gate_settings(&self, requested: &GateControlState) {
        let mut state = self
            .gate_control
            .lock()
            .map(|control| control.clone())
            .unwrap_or_else(|_| GateControlState::new());
        state.enabled = requested.enabled;
        for (value, min, max, slot) in [
            (
                requested.threshold_db,
                GATE_THRESHOLD_MIN_DB,
                GATE_THRESHOLD_MAX_DB,
                &mut state.threshold_db,
            ),
            (
                requested.attack_ms,
                GATE_ATTACK_MIN_MS,
                GATE_ATTACK_MAX_MS,
                &mut state.attack_ms,
            ),
            (
                requested.release_ms,
                GATE_RELEASE_MIN_MS,
                GATE_RELEASE_MAX_MS,
                &mut state.release_ms,
            ),
        ] {
            if let Some(clamped) = self.clamp_live_control(value, min, max) {
                *slot = clamped;
            }
        }
        #[cfg(feature = "vad")]
        {
            state.gate_mode = requested.gate_mode;
            state.auto_threshold = requested.auto_threshold;
            for (value, min, max, slot) in [
                (
                    requested.vad_threshold,
                    VAD_THRESHOLD_MIN,
                    VAD_THRESHOLD_MAX,
                    &mut state.vad_threshold,
                ),
                (
                    requested.hold_ms,
                    VAD_HOLD_MIN_MS,
                    VAD_HOLD_MAX_MS,
                    &mut state.hold_ms,
                ),
                (
                    requested.pre_gain,
                    VAD_PRE_GAIN_MIN,
                    VAD_PRE_GAIN_MAX,
                    &mut state.pre_gain,
                ),
                (
                    requested.margin_db,
                    GATE_MARGIN_MIN_DB,
                    GATE_MARGIN_MAX_DB,
                    &mut state.margin_db,
                ),
            ] {
                if let Some(clamped) = self.clamp_live_control_f32(value, min, max) {
                    *slot = clamped;
                }
            }
        }

        self.gate_enabled.store(state.enabled, Ordering::Release);
        self.gate_rt_control.publish(&state);
        if let Ok(mut control) = self.gate_control.lock() {
            *control = state;
        }
        self.gate_dirty.store(true, Ordering::Release);
    }
}
