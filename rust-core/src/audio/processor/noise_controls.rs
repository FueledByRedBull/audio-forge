impl AudioProcessor {
    // === Noise Suppression Controls ===

    /// Enable/disable noise suppression
    pub fn set_rnnoise_enabled(&self, enabled: bool) {
        self.suppressor_enabled.store(enabled, Ordering::Release);
        if let Ok(mut control) = self.suppressor_control.lock() {
            control.enabled = enabled;
        }
        self.suppressor_rt_control.set_enabled(enabled);
        if !enabled {
            self.suppressor_reset_requested
                .store(true, Ordering::Release);
        }
        self.suppressor_dirty.store(true, Ordering::Release);
    }

    /// Check if noise suppression is enabled
    pub fn is_rnnoise_enabled(&self) -> bool {
        self.suppressor_enabled.load(Ordering::Acquire)
    }

    /// Set noise suppression wet/dry mix strength (0.0 = fully dry, 1.0 = fully wet)
    pub fn set_rnnoise_strength(&self, strength: f32) {
        let Some(clamped) =
            clamp_control_value_f32(strength, RNNOISE_STRENGTH_MIN, RNNOISE_STRENGTH_MAX)
        else {
            return;
        };
        let bits = clamped.to_bits();
        self.suppressor_strength.store(bits, Ordering::Relaxed);
    }

    /// Get current noise suppression strength
    pub fn get_rnnoise_strength(&self) -> f32 {
        f32::from_bits(self.suppressor_strength.load(Ordering::Relaxed))
    }

    /// Set noise suppression model
    /// Returns true if successful
    pub fn set_noise_model(&self, model: NoiseModel) -> bool {
        self.drain_retired_suppressors();

        if self.get_noise_model() == model {
            return true;
        }

        #[cfg(feature = "deepfilter")]
        {
            if matches!(
                model,
                NoiseModel::DeepFilterNetLL | NoiseModel::DeepFilterNet
            ) && !crate::dsp::noise_suppressor::deepfilter_experimental_enabled()
            {
                return false;
            }
        }

        // Create new suppressor engine with current strength
        let strength = Arc::clone(&self.suppressor_strength);
        let new_engine = new_noise_suppression_engine(model, strength);
        let backend_diagnostics = noise_backend_diagnostics(&new_engine);

        if model != NoiseModel::RNNoise && !backend_diagnostics.available {
            // A neural backend can construct a dry fallback when an asset or
            // runtime fails. Reject selection so the UI does not mistake that
            // fallback for active suppression.
            store_backend_diagnostics(
                &self.noise_backend_available,
                &self.noise_backend_failed,
                self.noise_backend_error.as_ref(),
                backend_diagnostics,
            );
            return false;
        }

        if self.running.load(Ordering::Acquire) {
            let queued = if let Ok(mut tx_guard) = self.pending_suppressor_tx.lock() {
                if let Some(tx) = tx_guard.as_mut() {
                    tx.push(new_engine).is_ok()
                } else {
                    false
                }
            } else {
                false
            };

            if !queued {
                return false;
            }
        }

        store_backend_diagnostics(
            &self.noise_backend_available,
            &self.noise_backend_failed,
            self.noise_backend_error.as_ref(),
            backend_diagnostics,
        );

        if let Ok(mut control) = self.suppressor_control.lock() {
            control.model = model;
        }
        self.suppressor_rt_control.set_model(model);
        self.current_model.store(model as u8, Ordering::Release);
        self.suppressor_reset_requested
            .store(true, Ordering::Release);
        self.suppressor_dirty.store(true, Ordering::Release);
        true
    }

    fn rebuild_noise_suppressor(&self) -> bool {
        self.drain_retired_suppressors();
        let model = self.get_noise_model();
        let engine = new_noise_suppression_engine(model, Arc::clone(&self.suppressor_strength));
        let diagnostics = noise_backend_diagnostics(&engine);
        if model != NoiseModel::RNNoise && !diagnostics.available {
            store_backend_diagnostics(
                &self.noise_backend_available,
                &self.noise_backend_failed,
                self.noise_backend_error.as_ref(),
                diagnostics,
            );
            return false;
        }

        let queued = self
            .pending_suppressor_tx
            .lock()
            .ok()
            .and_then(|mut tx| tx.as_mut().map(|tx| tx.push(engine).is_ok()))
            .unwrap_or(false);
        if !queued {
            return false;
        }

        store_backend_diagnostics(
            &self.noise_backend_available,
            &self.noise_backend_failed,
            self.noise_backend_error.as_ref(),
            diagnostics,
        );
        self.suppressor_dirty.store(true, Ordering::Release);
        true
    }

    /// Get current noise suppression model
    pub fn get_noise_model(&self) -> NoiseModel {
        let model_u8 = self.current_model.load(Ordering::Acquire);
        match model_u8 {
            0 => NoiseModel::RNNoise,
            #[cfg(feature = "deepfilter")]
            1 => NoiseModel::DeepFilterNetLL,
            #[cfg(feature = "deepfilter")]
            2 => NoiseModel::DeepFilterNet,
            _ => NoiseModel::RNNoise,
        }
    }

    /// Get list of available noise models
    pub fn list_noise_models(&self) -> Vec<(String, String)> {
        #[cfg_attr(not(feature = "deepfilter"), allow(unused_mut))]
        let mut models = vec![(
            NoiseModel::RNNoise.id().to_string(),
            NoiseModel::RNNoise.display_name().to_string(),
        )];

        #[cfg(feature = "deepfilter")]
        {
            if crate::dsp::noise_suppressor::deepfilter_experimental_enabled() {
                let strength = Arc::clone(&self.suppressor_strength);

                let ll =
                    new_noise_suppression_engine(NoiseModel::DeepFilterNetLL, Arc::clone(&strength));
                if ll.backend_available() {
                    models.push((
                        NoiseModel::DeepFilterNetLL.id().to_string(),
                        NoiseModel::DeepFilterNetLL.display_name().to_string(),
                    ));
                }

                let std = new_noise_suppression_engine(NoiseModel::DeepFilterNet, strength);
                if std.backend_available() {
                    models.push((
                        NoiseModel::DeepFilterNet.id().to_string(),
                        NoiseModel::DeepFilterNet.display_name().to_string(),
                    ));
                }
            }
        }

        models
    }
}
