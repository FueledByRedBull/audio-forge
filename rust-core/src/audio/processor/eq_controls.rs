impl AudioProcessor {
    // === EQ Controls ===

    /// Enable/disable EQ
    pub fn set_eq_enabled(&self, enabled: bool) {
        self.eq_enabled.store(enabled, Ordering::Release);
        if let Ok(mut e) = self.eq.lock() {
            e.set_enabled(enabled);
        }
        if let Ok(mut e) = self.correction_eq.lock() {
            e.set_enabled(enabled);
        }
        self.eq_control.set_enabled(enabled);
        self.eq_dirty.store(true, Ordering::Release);
    }

    /// Check if EQ is enabled
    pub fn is_eq_enabled(&self) -> bool {
        self.eq_enabled.load(Ordering::Acquire)
    }

    /// Get one complete typed EQ band configuration.
    pub fn get_eq_band_config(&self, band: usize) -> Option<EqBandConfig> {
        if let Ok(e) = self.eq.lock() {
            e.get_band_config(band)
        } else {
            None
        }
    }

    /// Apply complete typed settings for all 10 bands atomically.
    pub fn apply_eq_settings_v2(&self, bands: Vec<EqBandConfig>) -> PyResult<()> {
        if bands.len() != NUM_BANDS {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Expected {} bands, got {}",
                NUM_BANDS,
                bands.len()
            )));
        }
        for (index, config) in bands.iter().copied().enumerate() {
            self.validate_eq_band_config(index, config)
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        }

        let snapshot_bands = std::array::from_fn(|index| bands[index]);
        // This complete typed API is the tone-layer update. Preserve measured
        // correction settings published by apply_eq_layers.
        let correction_bands = self
            .eq_control
            .snapshot()
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "EQ settings changed while reading the correction layer",
                )
            })?
            .correction_bands;
        if let Ok(mut eq) = self.eq.lock() {
            for (index, config) in snapshot_bands.iter().copied().enumerate() {
                eq.set_band_config(index, config);
            }
        }
        self.eq_control
            .set_layers(&correction_bands, &snapshot_bands);
        self.eq_dirty.store(true, Ordering::Release);
        Ok(())
    }

    /// Apply independent measured correction and user tone EQ stages.
    pub fn apply_eq_layers(
        &self,
        correction_bands: Vec<EqBandConfig>,
        tone_bands: Vec<EqBandConfig>,
    ) -> PyResult<()> {
        if correction_bands.len() != NUM_BANDS || tone_bands.len() != NUM_BANDS {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Expected {} correction and tone bands, got {} and {}",
                NUM_BANDS,
                correction_bands.len(),
                tone_bands.len()
            )));
        }
        for (index, config) in correction_bands.iter().copied().enumerate() {
            self.validate_eq_band_config(index, config)
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        }
        for (index, config) in tone_bands.iter().copied().enumerate() {
            self.validate_eq_band_config(index, config)
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        }

        let correction_snapshot = std::array::from_fn(|index| correction_bands[index]);
        let tone_snapshot = std::array::from_fn(|index| tone_bands[index]);
        if let Ok(mut correction_eq) = self.correction_eq.lock() {
            for (index, config) in correction_snapshot.iter().copied().enumerate() {
                correction_eq.set_band_config(index, config);
            }
        }
        if let Ok(mut eq) = self.eq.lock() {
            for (index, config) in tone_snapshot.iter().copied().enumerate() {
                eq.set_band_config(index, config);
            }
        }
        self.eq_control
            .set_layers(&correction_snapshot, &tone_snapshot);
        self.eq_dirty.store(true, Ordering::Release);
        Ok(())
    }
}
