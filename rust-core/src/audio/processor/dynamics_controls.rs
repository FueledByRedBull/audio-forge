impl AudioProcessor {
    // === De-Esser Controls ===

    /// Enable/disable de-esser.
    pub fn set_deesser_enabled(&self, enabled: bool) {
        self.deesser_enabled.store(enabled, Ordering::Release);
        if let Ok(mut d) = self.deesser.lock() {
            d.set_enabled(enabled);
        }
        if let Ok(mut control) = self.deesser_control.lock() {
            control.enabled = enabled;
        }
        self.deesser_rt_control.set_enabled(enabled);
        self.deesser_dirty.store(true, Ordering::Release);
    }

    /// Check if de-esser is enabled.
    pub fn is_deesser_enabled(&self) -> bool {
        self.deesser_enabled.load(Ordering::Acquire)
    }

    pub fn set_deesser_low_cut_hz(&self, hz: f64) {
        let Some(mut hz) = clamp_control_value(hz, DEESSER_LOW_CUT_MIN_HZ, DEESSER_LOW_CUT_MAX_HZ)
        else {
            return;
        };
        let high_cut_hz = self
            .deesser_control
            .lock()
            .map(|control| control.high_cut_hz)
            .unwrap_or(9000.0);
        if high_cut_hz <= hz + DEESSER_MIN_BANDWIDTH_HZ {
            hz = (high_cut_hz - DEESSER_MIN_BANDWIDTH_HZ)
                .clamp(DEESSER_LOW_CUT_MIN_HZ, DEESSER_LOW_CUT_MAX_HZ);
        }
        if let Ok(mut d) = self.deesser.lock() {
            d.set_low_cut_hz(hz);
        }
        if let Ok(mut control) = self.deesser_control.lock() {
            control.low_cut_hz = hz;
        }
        self.deesser_rt_control.set_low_cut_hz(hz);
        self.deesser_dirty.store(true, Ordering::Release);
    }

    pub fn set_deesser_high_cut_hz(&self, hz: f64) {
        let Some(mut hz) =
            clamp_control_value(hz, DEESSER_HIGH_CUT_MIN_HZ, DEESSER_HIGH_CUT_MAX_HZ)
        else {
            return;
        };
        let low_cut_hz = self
            .deesser_control
            .lock()
            .map(|control| control.low_cut_hz)
            .unwrap_or(4000.0);
        if hz <= low_cut_hz + DEESSER_MIN_BANDWIDTH_HZ {
            hz = (low_cut_hz + DEESSER_MIN_BANDWIDTH_HZ)
                .clamp(DEESSER_HIGH_CUT_MIN_HZ, DEESSER_HIGH_CUT_MAX_HZ);
        }
        if let Ok(mut d) = self.deesser.lock() {
            d.set_high_cut_hz(hz);
        }
        if let Ok(mut control) = self.deesser_control.lock() {
            control.high_cut_hz = hz;
        }
        self.deesser_rt_control.set_high_cut_hz(hz);
        self.deesser_dirty.store(true, Ordering::Release);
    }

    pub fn set_deesser_threshold_db(&self, threshold_db: f64) {
        let Some(threshold_db) = clamp_control_value(
            threshold_db,
            DEESSER_THRESHOLD_MIN_DB,
            DEESSER_THRESHOLD_MAX_DB,
        ) else {
            return;
        };
        if let Ok(mut d) = self.deesser.lock() {
            d.set_threshold_db(threshold_db);
        }
        if let Ok(mut control) = self.deesser_control.lock() {
            control.threshold_db = threshold_db;
        }
        self.deesser_rt_control.set_threshold_db(threshold_db);
        self.deesser_dirty.store(true, Ordering::Release);
    }

    pub fn set_deesser_ratio(&self, ratio: f64) {
        let Some(ratio) = clamp_control_value(ratio, DEESSER_RATIO_MIN, DEESSER_RATIO_MAX) else {
            return;
        };
        if let Ok(mut d) = self.deesser.lock() {
            d.set_ratio(ratio);
        }
        if let Ok(mut control) = self.deesser_control.lock() {
            control.ratio = ratio;
        }
        self.deesser_rt_control.set_ratio(ratio);
        self.deesser_dirty.store(true, Ordering::Release);
    }

    pub fn set_deesser_attack_ms(&self, attack_ms: f64) {
        let Some(attack_ms) =
            clamp_control_value(attack_ms, DEESSER_ATTACK_MIN_MS, DEESSER_ATTACK_MAX_MS)
        else {
            return;
        };
        if let Ok(mut d) = self.deesser.lock() {
            d.set_attack_ms(attack_ms);
        }
        if let Ok(mut control) = self.deesser_control.lock() {
            control.attack_ms = attack_ms;
        }
        self.deesser_rt_control.set_attack_ms(attack_ms);
        self.deesser_dirty.store(true, Ordering::Release);
    }

    pub fn set_deesser_release_ms(&self, release_ms: f64) {
        let Some(release_ms) =
            clamp_control_value(release_ms, DEESSER_RELEASE_MIN_MS, DEESSER_RELEASE_MAX_MS)
        else {
            return;
        };
        if let Ok(mut d) = self.deesser.lock() {
            d.set_release_ms(release_ms);
        }
        if let Ok(mut control) = self.deesser_control.lock() {
            control.release_ms = release_ms;
        }
        self.deesser_rt_control.set_release_ms(release_ms);
        self.deesser_dirty.store(true, Ordering::Release);
    }

    pub fn set_deesser_max_reduction_db(&self, max_reduction_db: f64) {
        let Some(max_reduction_db) = clamp_control_value(
            max_reduction_db,
            DEESSER_MAX_REDUCTION_MIN_DB,
            DEESSER_MAX_REDUCTION_MAX_DB,
        ) else {
            return;
        };
        if let Ok(mut d) = self.deesser.lock() {
            d.set_max_reduction_db(max_reduction_db);
        }
        if let Ok(mut control) = self.deesser_control.lock() {
            control.max_reduction_db = max_reduction_db;
        }
        self.deesser_rt_control
            .set_max_reduction_db(max_reduction_db);
        self.deesser_dirty.store(true, Ordering::Release);
    }

    pub fn set_deesser_auto_enabled(&self, auto_enabled: bool) {
        if let Ok(mut d) = self.deesser.lock() {
            d.set_auto_enabled(auto_enabled);
        }
        if let Ok(mut control) = self.deesser_control.lock() {
            control.auto_enabled = auto_enabled;
        }
        self.deesser_rt_control.set_auto_enabled(auto_enabled);
        self.deesser_dirty.store(true, Ordering::Release);
    }

    pub fn is_deesser_auto_enabled(&self) -> bool {
        if let Ok(d) = self.deesser.lock() {
            d.is_auto_enabled()
        } else {
            true
        }
    }

    pub fn set_deesser_auto_amount(&self, amount: f64) {
        let Some(amount) =
            clamp_control_value(amount, DEESSER_AUTO_AMOUNT_MIN, DEESSER_AUTO_AMOUNT_MAX)
        else {
            return;
        };
        if let Ok(mut d) = self.deesser.lock() {
            d.set_auto_amount(amount);
        }
        if let Ok(mut control) = self.deesser_control.lock() {
            control.auto_amount = amount;
        }
        self.deesser_rt_control.set_auto_amount(amount);
        self.deesser_dirty.store(true, Ordering::Release);
    }

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
            9000.0
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

    // === Compressor Controls ===

    /// Enable/disable compressor
    pub fn set_compressor_enabled(&self, enabled: bool) {
        self.compressor_enabled.store(enabled, Ordering::Release);
        if let Ok(mut c) = self.compressor.lock() {
            c.set_enabled(enabled);
        }
        if let Ok(mut control) = self.compressor_control.lock() {
            control.enabled = enabled;
        }
        self.compressor_rt_control.set_enabled(enabled);
        self.compressor_dirty.store(true, Ordering::Release);
    }

    /// Check if compressor is enabled
    pub fn is_compressor_enabled(&self) -> bool {
        self.compressor_enabled.load(Ordering::Acquire)
    }

    /// Set compressor threshold in dB
    pub fn set_compressor_threshold(&self, threshold_db: f64) {
        let Some(threshold_db) = clamp_control_value(
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

    /// Set compressor ratio
    pub fn set_compressor_ratio(&self, ratio: f64) {
        let Some(ratio) = clamp_control_value(ratio, COMPRESSOR_RATIO_MIN, COMPRESSOR_RATIO_MAX)
        else {
            return;
        };
        if let Ok(mut c) = self.compressor.lock() {
            c.set_ratio(ratio);
        }
        if let Ok(mut control) = self.compressor_control.lock() {
            control.ratio = ratio;
        }
        self.compressor_rt_control.set_ratio(ratio);
        self.compressor_dirty.store(true, Ordering::Release);
    }

    /// Set compressor attack time in ms
    pub fn set_compressor_attack(&self, attack_ms: f64) {
        let Some(attack_ms) = clamp_control_value(
            attack_ms,
            COMPRESSOR_ATTACK_MIN_MS,
            COMPRESSOR_ATTACK_MAX_MS,
        ) else {
            return;
        };
        if let Ok(mut c) = self.compressor.lock() {
            c.set_attack_time(attack_ms);
        }
        if let Ok(mut control) = self.compressor_control.lock() {
            control.attack_ms = attack_ms;
        }
        self.compressor_rt_control.set_attack_ms(attack_ms);
        self.compressor_dirty.store(true, Ordering::Release);
    }

    /// Set compressor release time in ms
    ///
    /// Note: When adaptive release is enabled, this updates the base release time.
    /// Use set_compressor_adaptive_release(false) to disable adaptive mode and
    /// use set_compressor_base_release() to set the base release time directly.
    pub fn set_compressor_release(&self, release_ms: f64) {
        let Some(release_ms) = clamp_control_value(
            release_ms,
            COMPRESSOR_RELEASE_MIN_MS,
            COMPRESSOR_RELEASE_MAX_MS,
        ) else {
            return;
        };
        if let Ok(mut c) = self.compressor.lock() {
            // Update base release time regardless of adaptive mode
            c.set_base_release_time(release_ms);
            // Also update current release when not in adaptive mode
            if !c.adaptive_release() {
                c.set_release_time(release_ms);
            }
        }
        if let Ok(mut control) = self.compressor_control.lock() {
            control.base_release_ms = release_ms;
        }
        self.compressor_rt_control.set_base_release_ms(release_ms);
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

    /// Set compressor makeup gain in dB
    pub fn set_compressor_makeup_gain(&self, makeup_gain_db: f64) {
        let Some(makeup_gain_db) = clamp_control_value(
            makeup_gain_db,
            COMPRESSOR_MAKEUP_MIN_DB,
            COMPRESSOR_MAKEUP_MAX_DB,
        ) else {
            return;
        };
        if let Ok(mut c) = self.compressor.lock() {
            c.set_makeup_gain(makeup_gain_db);
        }
        if let Ok(mut control) = self.compressor_control.lock() {
            control.makeup_gain_db = makeup_gain_db;
        }
        self.compressor_rt_control
            .set_makeup_gain_db(makeup_gain_db);
        self.compressor_current_makeup_gain
            .store(makeup_gain_db.to_bits(), Ordering::Relaxed);
        self.compressor_dirty.store(true, Ordering::Release);
    }

    pub fn set_compressor_adaptive_release(&self, enabled: bool) {
        if let Ok(mut c) = self.compressor.lock() {
            c.set_adaptive_release(enabled);
        }
        if let Ok(mut control) = self.compressor_control.lock() {
            control.adaptive_release = enabled;
        }
        self.compressor_rt_control.set_adaptive_release(enabled);
        self.compressor_dirty.store(true, Ordering::Release);
    }

    pub fn get_compressor_adaptive_release(&self) -> bool {
        if let Ok(c) = self.compressor.lock() {
            c.adaptive_release()
        } else {
            false
        }
    }

    pub fn set_compressor_base_release(&self, release_ms: f64) {
        let Some(release_ms) = clamp_control_value(
            release_ms,
            COMPRESSOR_BASE_RELEASE_MIN_MS,
            COMPRESSOR_BASE_RELEASE_MAX_MS,
        ) else {
            return;
        };
        if let Ok(mut c) = self.compressor.lock() {
            c.set_base_release_time(release_ms);
        }
        if let Ok(mut control) = self.compressor_control.lock() {
            control.base_release_ms = release_ms;
        }
        self.compressor_rt_control.set_base_release_ms(release_ms);
        self.compressor_dirty.store(true, Ordering::Release);
    }

    pub fn get_compressor_base_release(&self) -> f64 {
        if let Ok(c) = self.compressor.lock() {
            c.base_release_ms()
        } else {
            200.0
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
            clamp_control_value(ceiling_db, LIMITER_CEILING_MIN_DB, LIMITER_CEILING_MAX_DB)
        else {
            return;
        };
        if let Ok(mut l) = self.limiter.lock() {
            l.set_ceiling(ceiling_db);
        }
        if let Ok(mut control) = self.limiter_control.lock() {
            control.ceiling_db = ceiling_db;
        }
        self.limiter_rt_control.set_ceiling_db(ceiling_db);
        self.limiter_dirty.store(true, Ordering::Release);
    }

    /// Set limiter release time in ms
    pub fn set_limiter_release(&self, release_ms: f64) {
        let Some(release_ms) =
            clamp_control_value(release_ms, LIMITER_RELEASE_MIN_MS, LIMITER_RELEASE_MAX_MS)
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

    /// Set compressor auto makeup gain mode
    pub fn set_compressor_auto_makeup_enabled(&self, enabled: bool) {
        if let Ok(mut c) = self.compressor.lock() {
            c.set_auto_makeup_enabled(enabled);
        }
        if let Ok(mut control) = self.compressor_control.lock() {
            control.auto_makeup_enabled = enabled;
        }
        self.compressor_rt_control.set_auto_makeup_enabled(enabled);
        self.compressor_dirty.store(true, Ordering::Release);
    }

    /// Get compressor auto makeup gain mode
    pub fn get_compressor_auto_makeup_enabled(&self) -> bool {
        if let Ok(c) = self.compressor.lock() {
            c.auto_makeup_enabled()
        } else {
            false
        }
    }

    /// Set compressor target LUFS
    pub fn set_compressor_target_lufs(&self, target_lufs: f64) {
        let Some(target_lufs) = clamp_control_value(
            target_lufs,
            COMPRESSOR_TARGET_LUFS_MIN,
            COMPRESSOR_TARGET_LUFS_MAX,
        ) else {
            return;
        };
        if let Ok(mut c) = self.compressor.lock() {
            c.set_target_lufs(target_lufs);
        }
        if let Ok(mut control) = self.compressor_control.lock() {
            control.target_lufs = target_lufs;
        }
        self.compressor_rt_control.set_target_lufs(target_lufs);
        self.compressor_dirty.store(true, Ordering::Release);
    }

    /// Get compressor target LUFS
    pub fn get_compressor_target_lufs(&self) -> f64 {
        if let Ok(c) = self.compressor.lock() {
            c.target_lufs()
        } else {
            -18.0
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
