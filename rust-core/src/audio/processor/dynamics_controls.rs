impl AudioProcessor {
    /// `clamp_control_value` that counts the non-finite values it drops.
    fn clamp_live_control(&self, value: f64, min_value: f64, max_value: f64) -> Option<f64> {
        let clamped = clamp_control_value(value, min_value, max_value);
        if clamped.is_none() {
            self.rejected_control_value_count
                .fetch_add(1, Ordering::Relaxed);
        }
        clamped
    }

    /// `clamp_control_value_f32` that counts the non-finite values it drops.
    fn clamp_live_control_f32(&self, value: f32, min_value: f32, max_value: f32) -> Option<f32> {
        let clamped = clamp_control_value_f32(value, min_value, max_value);
        if clamped.is_none() {
            self.rejected_control_value_count
                .fetch_add(1, Ordering::Relaxed);
        }
        clamped
    }

    /// Clamp `value` into `slot`; a non-finite value keeps the slot (and is counted).
    fn clamp_into(&self, slot: &mut f64, value: f64, min_value: f64, max_value: f64) {
        if let Some(clamped) = self.clamp_live_control(value, min_value, max_value) {
            *slot = clamped;
        }
    }

    /// `clamp_into` for the f32 (VAD gate) controls.
    #[cfg(feature = "vad")]
    fn clamp_into_f32(&self, slot: &mut f32, value: f32, min_value: f32, max_value: f32) {
        if let Some(clamped) = self.clamp_live_control_f32(value, min_value, max_value) {
            *slot = clamped;
        }
    }

    /// Non-finite values that clamping live setters ignored since creation.
    pub fn get_rejected_control_value_count(&self) -> u64 {
        self.rejected_control_value_count.load(Ordering::Relaxed)
    }

    // === De-Esser Controls ===

    pub fn get_deesser_auto_amount(&self) -> f64 {
        if let Ok(d) = self.deesser.lock() {
            d.auto_amount()
        } else {
            0.5
        }
    }

    pub fn get_deesser_low_cut_hz(&self) -> f64 {
        if let Ok(d) = self.deesser.lock() {
            d.low_cut_hz()
        } else {
            4000.0
        }
    }

    pub fn get_deesser_high_cut_hz(&self) -> f64 {
        if let Ok(d) = self.deesser.lock() {
            d.high_cut_hz()
        } else {
            11_000.0
        }
    }

    pub fn get_deesser_threshold_db(&self) -> f64 {
        if let Ok(d) = self.deesser.lock() {
            d.threshold_db()
        } else {
            -28.0
        }
    }

    pub fn get_deesser_ratio(&self) -> f64 {
        if let Ok(d) = self.deesser.lock() {
            d.ratio()
        } else {
            4.0
        }
    }

    pub fn get_deesser_max_reduction_db(&self) -> f64 {
        if let Ok(d) = self.deesser.lock() {
            d.max_reduction_db()
        } else {
            6.0
        }
    }

    pub fn get_deesser_gain_reduction_db(&self) -> f32 {
        f32::from_bits(self.deesser_gain_reduction.load(Ordering::Relaxed))
    }

    pub fn get_deesser_detector_confidence(&self) -> f32 {
        f32::from_bits(self.deesser_detector_confidence.load(Ordering::Relaxed))
    }

    // === Compressor Controls ===

    /// Check if compressor is enabled
    pub fn is_compressor_enabled(&self) -> bool {
        self.compressor_enabled.load(Ordering::Acquire)
    }

    /// Set compressor threshold in dB
    pub fn set_compressor_threshold(&self, threshold_db: f64) {
        let Some(threshold_db) = self.clamp_live_control(
            threshold_db,
            COMPRESSOR_THRESHOLD_MIN_DB,
            COMPRESSOR_THRESHOLD_MAX_DB,
        ) else {
            return;
        };
        if let Ok(mut c) = self.compressor.lock() {
            c.set_threshold(threshold_db);
        }
        if let Ok(mut control) = self.compressor_control.lock() {
            control.threshold_db = threshold_db;
        }
        self.compressor_rt_control.set_threshold_db(threshold_db);
        self.compressor_dirty.store(true, Ordering::Release);
    }

    /// Get compressor release time.
    ///
    /// Note: When adaptive release is enabled, this returns the base release time.
    /// Use get_compressor_current_release() for the actual adaptive release time.
    pub fn get_compressor_release(&self) -> f64 {
        if let Ok(c) = self.compressor.lock() {
            if c.adaptive_release() {
                // Return base release when adaptive mode is active
                c.base_release_ms()
            } else {
                // Return current release when in manual mode
                c.current_release_time()
            }
        } else {
            200.0
        }
    }

    /// Check whether the detector sidechain high-pass is enabled.
    pub fn get_compressor_sidechain_highpass_enabled(&self) -> bool {
        if let Ok(c) = self.compressor.lock() {
            c.sidechain_highpass_enabled()
        } else {
            true
        }
    }

    // === Limiter Controls ===

    /// Enable/disable limiter
    pub fn set_limiter_enabled(&self, enabled: bool) {
        self.limiter_enabled.store(enabled, Ordering::Release);
        if let Ok(mut l) = self.limiter.lock() {
            l.set_enabled(enabled);
        }
        if let Ok(mut control) = self.limiter_control.lock() {
            control.enabled = enabled;
        }
        self.limiter_rt_control.set_enabled(enabled);
        self.limiter_dirty.store(true, Ordering::Release);
    }

    /// Check if limiter is enabled
    pub fn is_limiter_enabled(&self) -> bool {
        self.limiter_enabled.load(Ordering::Acquire)
    }

    /// Set limiter ceiling in dB
    pub fn set_limiter_ceiling(&self, ceiling_db: f64) {
        let Some(ceiling_db) =
            self.clamp_live_control(ceiling_db, LIMITER_CEILING_MIN_DB, LIMITER_CEILING_MAX_DB)
        else {
            return;
        };
        let careful_output_enabled = self.is_limiter_careful_output_enabled();
        let effective_ceiling_db = effective_limiter_ceiling_db(ceiling_db, careful_output_enabled);
        if let Ok(mut l) = self.limiter.lock() {
            l.set_ceiling(effective_ceiling_db);
        }
        if let Ok(mut control) = self.limiter_control.lock() {
            control.ceiling_db = ceiling_db;
        }
        self.limiter_rt_control.set_ceiling_db(ceiling_db);
        self.limiter_dirty.store(true, Ordering::Release);
    }

    /// Enable or disable conservative limiter headroom.
    pub fn set_limiter_careful_output_enabled(&self, enabled: bool) {
        let mut requested_ceiling_db = -0.5;
        if let Ok(mut control) = self.limiter_control.lock() {
            control.careful_output_enabled = enabled;
            requested_ceiling_db = control.ceiling_db;
        }
        let effective_ceiling_db = effective_limiter_ceiling_db(requested_ceiling_db, enabled);
        if let Ok(mut l) = self.limiter.lock() {
            l.set_ceiling(effective_ceiling_db);
        }
        self.limiter_rt_control.set_careful_output_enabled(enabled);
        self.limiter_dirty.store(true, Ordering::Release);
    }

    /// Check whether conservative limiter headroom is enabled.
    pub fn is_limiter_careful_output_enabled(&self) -> bool {
        self.limiter_rt_control
            .careful_output_enabled
            .load(Ordering::Acquire)
    }

    /// Effective runtime limiter ceiling after careful-output headroom.
    pub fn limiter_effective_ceiling_db(&self) -> f64 {
        if let Ok(control) = self.limiter_control.lock() {
            effective_limiter_ceiling_db(control.ceiling_db, control.careful_output_enabled)
        } else {
            effective_limiter_ceiling_db(-0.5, self.is_limiter_careful_output_enabled())
        }
    }

    /// Set limiter release time in ms
    pub fn set_limiter_release(&self, release_ms: f64) {
        let Some(release_ms) =
            self.clamp_live_control(release_ms, LIMITER_RELEASE_MIN_MS, LIMITER_RELEASE_MAX_MS)
        else {
            return;
        };
        if let Ok(mut l) = self.limiter.lock() {
            l.set_release_time(release_ms);
        }
        if let Ok(mut control) = self.limiter_control.lock() {
            control.release_ms = release_ms;
        }
        self.limiter_rt_control.set_release_ms(release_ms);
        self.limiter_dirty.store(true, Ordering::Release);
    }

    // === Auto Makeup Gain Controls ===

    /// Get compressor auto makeup gain mode
    pub fn get_compressor_auto_makeup_enabled(&self) -> bool {
        if let Ok(c) = self.compressor.lock() {
            c.auto_makeup_enabled()
        } else {
            false
        }
    }

    /// Get compressor current LUFS
    pub fn get_compressor_current_lufs(&self) -> f64 {
        f64::from_bits(self.compressor_current_lufs.load(Ordering::Relaxed))
    }

    /// Get compressor current makeup gain
    pub fn get_compressor_current_makeup_gain(&self) -> f64 {
        f64::from_bits(self.compressor_current_makeup_gain.load(Ordering::Relaxed))
    }
}

impl AtomicDeesserControlState {
    /// Publish every field in one seqlock update so the DSP thread never
    /// snapshots a half-applied combination such as a new low cut with an old
    /// high cut.
    fn publish(&self, state: &DeesserControlState) {
        self.update(|atomic| {
            atomic.enabled.store(state.enabled, Ordering::Relaxed);
            atomic
                .auto_enabled
                .store(state.auto_enabled, Ordering::Relaxed);
            atomic
                .auto_amount_bits
                .store(state.auto_amount.to_bits(), Ordering::Relaxed);
            atomic
                .low_cut_hz_bits
                .store(state.low_cut_hz.to_bits(), Ordering::Relaxed);
            atomic
                .high_cut_hz_bits
                .store(state.high_cut_hz.to_bits(), Ordering::Relaxed);
            atomic
                .threshold_db_bits
                .store(state.threshold_db.to_bits(), Ordering::Relaxed);
            atomic
                .ratio_bits
                .store(state.ratio.to_bits(), Ordering::Relaxed);
            atomic
                .attack_ms_bits
                .store(state.attack_ms.to_bits(), Ordering::Relaxed);
            atomic
                .release_ms_bits
                .store(state.release_ms.to_bits(), Ordering::Relaxed);
            atomic
                .max_reduction_db_bits
                .store(state.max_reduction_db.to_bits(), Ordering::Relaxed);
        });
    }
}

impl AtomicCompressorControlState {
    /// Publish every field in one seqlock update (see the de-esser version).
    fn publish(&self, state: &CompressorControlState) {
        self.update(|atomic| {
            atomic.enabled.store(state.enabled, Ordering::Relaxed);
            atomic
                .threshold_db_bits
                .store(state.threshold_db.to_bits(), Ordering::Relaxed);
            atomic
                .ratio_bits
                .store(state.ratio.to_bits(), Ordering::Relaxed);
            atomic
                .attack_ms_bits
                .store(state.attack_ms.to_bits(), Ordering::Relaxed);
            atomic
                .base_release_ms_bits
                .store(state.base_release_ms.to_bits(), Ordering::Relaxed);
            atomic
                .makeup_gain_db_bits
                .store(state.makeup_gain_db.to_bits(), Ordering::Relaxed);
            atomic
                .adaptive_release
                .store(state.adaptive_release, Ordering::Relaxed);
            atomic
                .auto_makeup_enabled
                .store(state.auto_makeup_enabled, Ordering::Relaxed);
            atomic
                .target_lufs_bits
                .store(state.target_lufs.to_bits(), Ordering::Relaxed);
            atomic.noise_reference_reliability_bits.store(
                state.noise_reference_reliability.to_bits(),
                Ordering::Relaxed,
            );
            atomic
                .sidechain_highpass_enabled
                .store(state.sidechain_highpass_enabled, Ordering::Relaxed);
        });
    }
}

impl AudioProcessor {
    /// Apply a complete de-esser configuration as one published snapshot.
    ///
    /// Each value is clamped like its single setter and a non-finite value
    /// keeps the current one. A cut pair that violates the minimum bandwidth
    /// moves the low cut, so the published pair is always valid.
    fn apply_deesser_settings(&self, requested: &DeesserControlState) {
        let mut state = self
            .deesser_control
            .lock()
            .map(|control| control.clone())
            .unwrap_or_else(|_| DeesserControlState::new());
        state.enabled = requested.enabled;
        state.auto_enabled = requested.auto_enabled;
        self.clamp_into(
            &mut state.auto_amount,
            requested.auto_amount,
            DEESSER_AUTO_AMOUNT_MIN,
            DEESSER_AUTO_AMOUNT_MAX,
        );
        self.clamp_into(
            &mut state.low_cut_hz,
            requested.low_cut_hz,
            DEESSER_LOW_CUT_MIN_HZ,
            DEESSER_LOW_CUT_MAX_HZ,
        );
        self.clamp_into(
            &mut state.high_cut_hz,
            requested.high_cut_hz,
            DEESSER_HIGH_CUT_MIN_HZ,
            DEESSER_HIGH_CUT_MAX_HZ,
        );
        self.clamp_into(
            &mut state.threshold_db,
            requested.threshold_db,
            DEESSER_THRESHOLD_MIN_DB,
            DEESSER_THRESHOLD_MAX_DB,
        );
        self.clamp_into(
            &mut state.ratio,
            requested.ratio,
            DEESSER_RATIO_MIN,
            DEESSER_RATIO_MAX,
        );
        self.clamp_into(
            &mut state.attack_ms,
            requested.attack_ms,
            DEESSER_ATTACK_MIN_MS,
            DEESSER_ATTACK_MAX_MS,
        );
        self.clamp_into(
            &mut state.release_ms,
            requested.release_ms,
            DEESSER_RELEASE_MIN_MS,
            DEESSER_RELEASE_MAX_MS,
        );
        self.clamp_into(
            &mut state.max_reduction_db,
            requested.max_reduction_db,
            DEESSER_MAX_REDUCTION_MIN_DB,
            DEESSER_MAX_REDUCTION_MAX_DB,
        );
        if state.high_cut_hz <= state.low_cut_hz + DEESSER_MIN_BANDWIDTH_HZ {
            state.low_cut_hz = (state.high_cut_hz - DEESSER_MIN_BANDWIDTH_HZ)
                .clamp(DEESSER_LOW_CUT_MIN_HZ, DEESSER_LOW_CUT_MAX_HZ);
        }

        if let Ok(mut d) = self.deesser.lock() {
            d.set_enabled(state.enabled);
            d.set_auto_enabled(state.auto_enabled);
            d.set_auto_amount(state.auto_amount);
            d.set_low_cut_hz(state.low_cut_hz);
            d.set_high_cut_hz(state.high_cut_hz);
            d.set_threshold_db(state.threshold_db);
            d.set_ratio(state.ratio);
            d.set_attack_ms(state.attack_ms);
            d.set_release_ms(state.release_ms);
            d.set_max_reduction_db(state.max_reduction_db);
        }
        self.deesser_rt_control.publish(&state);
        if let Ok(mut control) = self.deesser_control.lock() {
            *control = state;
        }
        self.deesser_dirty.store(true, Ordering::Release);
    }

    /// Apply a complete compressor configuration as one published snapshot.
    ///
    /// `base_release_ms` is the release for the requested mode: the adaptive
    /// base (20-200 ms) when adaptive release is on, the manual release
    /// (10-1000 ms) otherwise.
    fn apply_compressor_settings(&self, requested: &CompressorControlState) {
        let mut state = self
            .compressor_control
            .lock()
            .map(|control| control.clone())
            .unwrap_or_else(|_| CompressorControlState::new());
        state.enabled = requested.enabled;
        state.adaptive_release = requested.adaptive_release;
        state.auto_makeup_enabled = requested.auto_makeup_enabled;
        state.sidechain_highpass_enabled = requested.sidechain_highpass_enabled;
        let (release_min, release_max) = if state.adaptive_release {
            (
                COMPRESSOR_BASE_RELEASE_MIN_MS,
                COMPRESSOR_BASE_RELEASE_MAX_MS,
            )
        } else {
            (COMPRESSOR_RELEASE_MIN_MS, COMPRESSOR_RELEASE_MAX_MS)
        };
        self.clamp_into(
            &mut state.threshold_db,
            requested.threshold_db,
            COMPRESSOR_THRESHOLD_MIN_DB,
            COMPRESSOR_THRESHOLD_MAX_DB,
        );
        self.clamp_into(
            &mut state.ratio,
            requested.ratio,
            COMPRESSOR_RATIO_MIN,
            COMPRESSOR_RATIO_MAX,
        );
        self.clamp_into(
            &mut state.attack_ms,
            requested.attack_ms,
            COMPRESSOR_ATTACK_MIN_MS,
            COMPRESSOR_ATTACK_MAX_MS,
        );
        self.clamp_into(
            &mut state.base_release_ms,
            requested.base_release_ms,
            release_min,
            release_max,
        );
        self.clamp_into(
            &mut state.makeup_gain_db,
            requested.makeup_gain_db,
            COMPRESSOR_MAKEUP_MIN_DB,
            COMPRESSOR_MAKEUP_MAX_DB,
        );
        self.clamp_into(
            &mut state.target_lufs,
            requested.target_lufs,
            COMPRESSOR_TARGET_LUFS_MIN,
            COMPRESSOR_TARGET_LUFS_MAX,
        );
        self.clamp_into(
            &mut state.noise_reference_reliability,
            requested.noise_reference_reliability,
            0.0,
            1.0,
        );

        self.compressor_enabled
            .store(state.enabled, Ordering::Release);
        if let Ok(mut c) = self.compressor.lock() {
            c.set_enabled(state.enabled);
            c.set_threshold(state.threshold_db);
            c.set_ratio(state.ratio);
            c.set_attack_time(state.attack_ms);
            c.set_adaptive_release(state.adaptive_release);
            c.set_base_release_time(state.base_release_ms);
            if !state.adaptive_release {
                c.set_release_time(state.base_release_ms);
            }
            c.set_makeup_gain(state.makeup_gain_db);
            c.set_sidechain_highpass_enabled(state.sidechain_highpass_enabled);
            c.set_auto_makeup_enabled(state.auto_makeup_enabled);
            c.set_target_lufs(state.target_lufs);
            c.set_noise_reference_reliability(state.noise_reference_reliability);
        }
        self.compressor_current_makeup_gain
            .store(state.makeup_gain_db.to_bits(), Ordering::Relaxed);
        self.compressor_rt_control.publish(&state);
        if let Ok(mut control) = self.compressor_control.lock() {
            *control = state;
        }
        self.compressor_dirty.store(true, Ordering::Release);
    }
}
