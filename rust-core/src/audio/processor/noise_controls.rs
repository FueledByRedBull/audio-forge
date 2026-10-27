use super::diagnostics::{
    noise_backend_diagnostics, store_backend_diagnostics, update_backend_diagnostics,
};
use super::{AudioProcessor, RNNOISE_STRENGTH_MAX, RNNOISE_STRENGTH_MIN};
// Engines built only to ask whether a backend loads answer inline: they need
// no worker thread for that.
use crate::dsp::noise_suppressor::{
    new_controlled_noise_suppression_engine as new_noise_suppression_engine,
    new_realtime_noise_suppression_engine,
    ControlledNoiseSuppressionEngine as NoiseSuppressionEngine, NoiseModel,
};
use ringbuf::traits::Producer;
use std::sync::atomic::Ordering;
use std::sync::Arc;
#[cfg(not(windows))]
use thread_priority::{set_current_thread_priority, ThreadPriority};

impl AudioProcessor {
    // === Noise Suppression Controls ===

    /// Enable/disable noise suppression
    pub fn set_rnnoise_enabled(&self, enabled: bool) {
        self.shared
            .suppressor
            .enabled
            .store(enabled, Ordering::Release);
        if let Ok(mut control) = self.shared.suppressor.control.lock() {
            control.enabled = enabled;
        }
        self.shared.suppressor.rt_control.set_enabled(enabled);
        if !enabled {
            self.shared
                .suppressor
                .reset_requested
                .store(true, Ordering::Release);
        }
        self.shared.suppressor.dirty.store(true, Ordering::Release);
    }

    /// Check if noise suppression is enabled
    pub fn is_rnnoise_enabled(&self) -> bool {
        self.shared.suppressor.enabled.load(Ordering::Acquire)
    }

    /// Set noise suppression wet/dry mix strength (0.0 = fully dry, 1.0 = fully wet)
    pub fn set_rnnoise_strength(&self, strength: f32) {
        let Some(clamped) =
            self.clamp_live_control_f32(strength, RNNOISE_STRENGTH_MIN, RNNOISE_STRENGTH_MAX)
        else {
            return;
        };
        let bits = clamped.to_bits();
        self.shared
            .suppressor
            .strength
            .store(bits, Ordering::Relaxed);
    }

    /// Get current noise suppression strength
    pub fn get_rnnoise_strength(&self) -> f32 {
        f32::from_bits(self.shared.suppressor.strength.load(Ordering::Relaxed))
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

        self.select_noise_model(model, self.engine_builder())
    }

    /// Builds the engines of this processor: `build(model, true)` for the live
    /// loop (inference at a worker), `build(model, false)` for one that only
    /// shows whether the backend loads, which a stopped processor needs and
    /// must not give a worker thread.
    pub(super) fn engine_builder(
        &self,
    ) -> impl Fn(NoiseModel, bool) -> NoiseSuppressionEngine + '_ {
        move |model, realtime| {
            let strength = Arc::clone(&self.shared.suppressor.strength);
            if realtime {
                new_realtime_noise_suppression_engine(model, strength)
            } else {
                new_noise_suppression_engine(model, strength)
            }
        }
    }

    pub(super) fn select_noise_model(
        &self,
        model: NoiseModel,
        build: impl Fn(NoiseModel, bool) -> NoiseSuppressionEngine,
    ) -> bool {
        let running = self.shared.stream.running.load(Ordering::Acquire);
        let new_engine = build(model, running);
        let backend_diagnostics = noise_backend_diagnostics(&new_engine);

        if model != NoiseModel::RNNoise && !backend_diagnostics.available {
            // A neural backend can construct a dry fallback when an asset or
            // runtime fails. Reject selection so the UI does not mistake that
            // fallback for active suppression.
            store_backend_diagnostics(
                &self.shared.suppressor.backend_available,
                &self.shared.suppressor.backend_failed,
                &self.shared.suppressor.backend_error,
                backend_diagnostics,
            );
            return false;
        }

        if running {
            let queued = if let Ok(mut tx_guard) = self.shared.suppressor.pending_tx.lock() {
                if let Some(tx) = tx_guard.as_mut() {
                    tx.try_push(new_engine).is_ok()
                } else {
                    false
                }
            } else {
                false
            };

            if !queued {
                return false;
            }
            store_backend_diagnostics(
                &self.shared.suppressor.backend_available,
                &self.shared.suppressor.backend_failed,
                &self.shared.suppressor.backend_error,
                backend_diagnostics,
            );
        } else {
            update_backend_diagnostics(
                &self.shared.suppressor.backend_available,
                &self.shared.suppressor.backend_failed,
                &self.shared.suppressor.backend_error,
                &new_engine,
            );
        }

        self.record_selected_model(model);
        self.shared
            .suppressor
            .reset_requested
            .store(true, Ordering::Release);
        self.shared.suppressor.dirty.store(true, Ordering::Release);
        // The stream can have started while the engine loaded (a model load
        // takes a few hundred milliseconds) or while the selection was being
        // recorded. `start` does not wait for this call.
        if !running && self.shared.stream.running.load(Ordering::Acquire) {
            return self.settle_selection_with_started_stream(model, build);
        }
        true
    }

    pub(super) fn record_selected_model(&self, model: NoiseModel) {
        self.replace_selection(None, model);
    }

    /// Record `model` as the selection; with `only_over`, only while that is
    /// still the selection. The control lock is held through all three writes,
    /// so a conditional replace never sees half of another recording.
    fn replace_selection(&self, only_over: Option<NoiseModel>, model: NoiseModel) {
        let suppressor = &self.shared.suppressor;
        let mut control = suppressor.control.lock();
        let current = suppressor.current_model.load(Ordering::Acquire);
        if only_over.is_some_and(|over| over as u8 != current) {
            return;
        }
        if let Ok(control) = control.as_mut() {
            control.model = model;
        }
        suppressor.rt_control.set_model(model);
        suppressor
            .current_model
            .store(model as u8, Ordering::Release);
    }

    /// The stream began on the model it found then, and only a queued engine
    /// moves it to another. When it found `model` already, nothing is needed.
    /// When the engine cannot be built the selection is counted as dropped and
    /// undone, to the model the loop runs, so the control and the loop agree and
    /// nothing waits for an engine that will not come. A newer selection made
    /// while the engine was loading stays.
    pub(super) fn settle_selection_with_started_stream(
        &self,
        model: NoiseModel,
        build: impl Fn(NoiseModel, bool) -> NoiseSuppressionEngine,
    ) -> bool {
        let live = self.shared.suppressor.engine_model.load(Ordering::Acquire);
        if live == model as u8 || self.rebuild_with(model, build) {
            return true;
        }
        self.shared
            .suppressor
            .dropped_selection_count
            .fetch_add(1, Ordering::Relaxed);
        self.replace_selection(Some(model), model_from_code(live));
        false
    }

    pub(super) fn rebuild_noise_suppressor(&self) -> bool {
        self.rebuild_with(self.get_noise_model(), self.engine_builder())
    }

    fn rebuild_with(
        &self,
        model: NoiseModel,
        build: impl Fn(NoiseModel, bool) -> NoiseSuppressionEngine,
    ) -> bool {
        self.drain_retired_suppressors();
        let engine = build(model, true);
        let diagnostics = noise_backend_diagnostics(&engine);
        if model != NoiseModel::RNNoise && !diagnostics.available {
            store_backend_diagnostics(
                &self.shared.suppressor.backend_available,
                &self.shared.suppressor.backend_failed,
                &self.shared.suppressor.backend_error,
                diagnostics,
            );
            return false;
        }

        let queued = self
            .shared
            .suppressor
            .pending_tx
            .lock()
            .ok()
            .and_then(|mut tx| tx.as_mut().map(|tx| tx.try_push(engine).is_ok()))
            .unwrap_or(false);
        if !queued {
            return false;
        }

        store_backend_diagnostics(
            &self.shared.suppressor.backend_available,
            &self.shared.suppressor.backend_failed,
            &self.shared.suppressor.backend_error,
            diagnostics,
        );
        self.shared.suppressor.dirty.store(true, Ordering::Release);
        true
    }

    /// Get current noise suppression model
    pub fn get_noise_model(&self) -> NoiseModel {
        model_from_code(self.shared.suppressor.current_model.load(Ordering::Acquire))
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
                let strength = Arc::clone(&self.shared.suppressor.strength);

                let ll = new_noise_suppression_engine(
                    NoiseModel::DeepFilterNetLL,
                    Arc::clone(&strength),
                );
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

fn model_from_code(code: u8) -> NoiseModel {
    match code {
        0 => NoiseModel::RNNoise,
        #[cfg(feature = "deepfilter")]
        1 => NoiseModel::DeepFilterNetLL,
        #[cfg(feature = "deepfilter")]
        2 => NoiseModel::DeepFilterNet,
        _ => NoiseModel::RNNoise,
    }
}
