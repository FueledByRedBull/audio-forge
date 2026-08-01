impl AudioProcessor {
    // === EQ Controls ===

    /// Enable/disable EQ
    pub fn set_eq_enabled(&self, enabled: bool) {
        self.eq_enabled.store(enabled, Ordering::Release);
        if let Ok(mut e) = self.eq.lock() {
            e.set_enabled(enabled);
        }
        self.eq_control.set_enabled(enabled);
        self.eq_dirty.store(true, Ordering::Release);
    }

    /// Check if EQ is enabled
    pub fn is_eq_enabled(&self) -> bool {
        self.eq_enabled.load(Ordering::Acquire)
    }

    /// Set EQ band gain
    pub fn set_eq_band_gain(&self, band: usize, gain_db: f64) -> Result<(), String> {
        self.validate_eq_band_index(band)?;
        self.validate_eq_gain(band, gain_db)?;
        if let Ok(mut e) = self.eq.lock() {
            e.set_band_gain(band, gain_db);
        }
        self.eq_control.set_band_gain(band, gain_db);
        self.eq_dirty.store(true, Ordering::Release);
        Ok(())
    }

    /// Set EQ band frequency
    pub fn set_eq_band_frequency(&self, band: usize, frequency: f64) -> Result<(), String> {
        self.validate_eq_band_index(band)?;
        self.validate_eq_frequency(band, frequency)?;
        if let Ok(mut e) = self.eq.lock() {
            e.set_band_frequency(band, frequency);
        }
        self.eq_control.set_band_frequency(band, frequency);
        self.eq_dirty.store(true, Ordering::Release);
        Ok(())
    }

    /// Set EQ band Q
    pub fn set_eq_band_q(&self, band: usize, q: f64) -> Result<(), String> {
        self.validate_eq_band_index(band)?;
        self.validate_eq_q(band, q)?;
        if let Ok(mut e) = self.eq.lock() {
            e.set_band_q(band, q);
        }
        self.eq_control.set_band_q(band, q);
        self.eq_dirty.store(true, Ordering::Release);
        Ok(())
    }

    /// Set EQ band filter type.
    pub fn set_eq_band_filter_type(
        &self,
        band: usize,
        filter_type: EqFilterType,
    ) -> Result<(), String> {
        self.validate_eq_band_index(band)?;
        if let Ok(mut e) = self.eq.lock() {
            e.set_band_filter_type(band, filter_type);
        }
        self.eq_control.set_band_filter_type(band, filter_type);
        self.eq_dirty.store(true, Ordering::Release);
        Ok(())
    }

    /// Set EQ band pass-filter slope.
    pub fn set_eq_band_slope(
        &self,
        band: usize,
        slope_db_per_octave: u8,
    ) -> Result<(), String> {
        self.validate_eq_band_index(band)?;
        self.validate_eq_slope(band, slope_db_per_octave)?;
        if let Ok(mut e) = self.eq.lock() {
            e.set_band_slope(band, slope_db_per_octave);
        }
        self.eq_control.set_band_slope(band, slope_db_per_octave);
        self.eq_dirty.store(true, Ordering::Release);
        Ok(())
    }

    /// Enable or disable one EQ band using a click-safe bypass transition.
    pub fn set_eq_band_enabled(&self, band: usize, enabled: bool) -> Result<(), String> {
        self.validate_eq_band_index(band)?;
        if let Ok(mut e) = self.eq.lock() {
            e.set_band_enabled(band, enabled);
        }
        self.eq_control.set_band_enabled(band, enabled);
        self.eq_dirty.store(true, Ordering::Release);
        Ok(())
    }

    /// Get EQ band parameters (frequency, gain_db, q)
    pub fn get_eq_band_params(&self, band: usize) -> Option<(f64, f64, f64)> {
        if let Ok(e) = self.eq.lock() {
            e.get_band_params(band)
        } else {
            None
        }
    }

    /// Get one complete typed EQ band configuration.
    pub fn get_eq_band_config(&self, band: usize) -> Option<EqBandConfig> {
        if let Ok(e) = self.eq.lock() {
            e.get_band_config(band)
        } else {
            None
        }
    }

    /// Apply EQ settings for all 10 bands in a single atomic call
    ///
    /// # Arguments
    /// * `bands` - Vector of (frequency_hz, gain_db, q) tuples for each band (must be 10)
    ///
    /// # Returns
    /// * PyResult<()> - Ok(()) on success, Err if validation fails
    ///
    /// # Validation
    /// * bands.len() must equal NUM_BANDS
    /// * frequency: 20.0 Hz to Nyquist minus a small margin
    /// * gain_db: -12.0 to +12.0 dB
    /// * q: 0.1 to 10.0
    pub fn apply_eq_settings(&self, bands: Vec<(f64, f64, f64)>) -> PyResult<()> {
        // Validate band count
        if bands.len() != NUM_BANDS {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Expected {} bands, got {}",
                NUM_BANDS,
                bands.len()
            )));
        }

        // Validate each band's parameters
        for (i, (freq, gain, q)) in bands.iter().enumerate() {
            self.validate_eq_frequency(i, *freq)
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
            self.validate_eq_gain(i, *gain)
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
            self.validate_eq_q(i, *q)
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        }

        let snapshot_bands = std::array::from_fn(|index| {
            let (frequency_hz, gain_db, q) = bands[index];
            EqBandConfig {
                frequency_hz,
                gain_db,
                q,
                ..EqBandConfig::default_for_index(index)
            }
        });

        // All validation passed - apply atomically
        if let Ok(mut eq) = self.eq.lock() {
            for (index, config) in snapshot_bands.iter().copied().enumerate() {
                eq.set_band_config(index, config);
            }
        }
        self.eq_control.set_bands(&snapshot_bands);
        self.eq_dirty.store(true, Ordering::Release);

        Ok(())
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
        if let Ok(mut eq) = self.eq.lock() {
            for (index, config) in snapshot_bands.iter().copied().enumerate() {
                eq.set_band_config(index, config);
            }
        }
        self.eq_control.set_bands(&snapshot_bands);
        self.eq_dirty.store(true, Ordering::Release);
        Ok(())
    }

}
