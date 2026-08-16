//! 10-Band Parametric Equalizer
//!
//! Band configuration:
//! - Band 0: Low shelf (80 Hz)
//! - Bands 1-8: Peaking (160 Hz - 12 kHz)
//! - Band 9: High shelf (16 kHz)

use super::biquad::{Biquad, BiquadType};
use std::f64::consts::PI;

/// Default EQ band frequencies (Hz)
pub const DEFAULT_FREQUENCIES: [f64; 10] = [
    80.0,    // Band 0: Low shelf
    160.0,   // Band 1: Peaking
    320.0,   // Band 2: Peaking
    640.0,   // Band 3: Peaking
    1280.0,  // Band 4: Peaking
    2500.0,  // Band 5: Peaking
    5000.0,  // Band 6: Peaking
    8000.0,  // Band 7: Peaking
    12000.0, // Band 8: Peaking
    16000.0, // Band 9: High shelf
];

/// Default Q factor for peaking bands (~1 octave bandwidth)
pub const DEFAULT_Q: f64 = 1.41;

/// Number of EQ bands
pub const NUM_BANDS: usize = 10;

/// Maximum number of second-order sections in one selectable pass filter.
pub const MAX_PASS_SECTIONS: usize = 4;

/// Supported pass-filter slopes. Each biquad contributes 12 dB/octave.
pub const SUPPORTED_PASS_SLOPES_DB_PER_OCTAVE: [u8; 4] = [12, 24, 36, 48];
pub const EQ_GAIN_MIN_DB: f64 = -12.0;
pub const EQ_GAIN_MAX_DB: f64 = 12.0;
pub const EQ_Q_MIN: f64 = 0.1;
pub const EQ_Q_MAX: f64 = 10.0;
pub const EQ_FREQ_MIN_HZ: f64 = 20.0;
pub const EQ_NYQUIST_MARGIN_HZ: f64 = 1.0;

/// Stable public filter identifiers shared with Python and persisted presets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum EqFilterType {
    LowShelf = 0,
    Bell = 1,
    HighShelf = 2,
    Notch = 3,
    HighPass = 4,
    LowPass = 5,
}

impl EqFilterType {
    /// Parse a stable public filter identifier.
    pub fn from_id(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::LowShelf),
            1 => Some(Self::Bell),
            2 => Some(Self::HighShelf),
            3 => Some(Self::Notch),
            4 => Some(Self::HighPass),
            5 => Some(Self::LowPass),
            _ => None,
        }
    }

    /// Parse the persisted schema name.
    pub fn from_name(value: &str) -> Option<Self> {
        match value {
            "low_shelf" => Some(Self::LowShelf),
            "bell" => Some(Self::Bell),
            "high_shelf" => Some(Self::HighShelf),
            "notch" => Some(Self::Notch),
            "high_pass" => Some(Self::HighPass),
            "low_pass" => Some(Self::LowPass),
            _ => None,
        }
    }

    /// Return the persisted schema name.
    pub fn name(self) -> &'static str {
        match self {
            Self::LowShelf => "low_shelf",
            Self::Bell => "bell",
            Self::HighShelf => "high_shelf",
            Self::Notch => "notch",
            Self::HighPass => "high_pass",
            Self::LowPass => "low_pass",
        }
    }

    /// Whether this type is an even-order Butterworth pass filter.
    pub fn is_pass(self) -> bool {
        matches!(self, Self::HighPass | Self::LowPass)
    }

    fn biquad_type(self) -> BiquadType {
        match self {
            Self::LowShelf => BiquadType::LowShelf,
            Self::Bell => BiquadType::Peaking,
            Self::HighShelf => BiquadType::HighShelf,
            Self::Notch => BiquadType::Notch,
            Self::HighPass => BiquadType::HighPass,
            Self::LowPass => BiquadType::LowPass,
        }
    }
}

/// Complete realtime configuration for one EQ band.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EqBandConfig {
    pub filter_type: EqFilterType,
    pub frequency_hz: f64,
    pub gain_db: f64,
    pub q: f64,
    pub slope_db_per_octave: u8,
    pub enabled: bool,
}

impl EqBandConfig {
    /// Default band configuration preserving the historical 10-band layout.
    pub fn default_for_index(index: usize) -> Self {
        let filter_type = match index {
            0 => EqFilterType::LowShelf,
            9 => EqFilterType::HighShelf,
            _ => EqFilterType::Bell,
        };
        Self {
            filter_type,
            frequency_hz: DEFAULT_FREQUENCIES[index],
            gain_db: 0.0,
            q: DEFAULT_Q,
            slope_db_per_octave: 12,
            enabled: true,
        }
    }

    /// Validate the complete public runtime contract for one band.
    pub fn validate(self, index: usize, sample_rate: f64) -> Result<(), String> {
        validate_eq_frequency_hz(self.frequency_hz, sample_rate)
            .map_err(|message| format!("Band {index}: {message}"))?;
        validate_eq_gain_db(self.gain_db).map_err(|message| format!("Band {index}: {message}"))?;
        validate_eq_q(self.q).map_err(|message| format!("Band {index}: {message}"))?;
        validate_eq_slope(self.slope_db_per_octave)
            .map_err(|message| format!("Band {index}: {message}"))
    }
}

pub fn eq_max_frequency_hz(sample_rate: f64) -> Result<f64, String> {
    if !sample_rate.is_finite() || sample_rate <= 2.0 * EQ_FREQ_MIN_HZ {
        return Err("sample rate must be finite and support the EQ frequency range".to_string());
    }
    Ok((sample_rate / 2.0 - EQ_NYQUIST_MARGIN_HZ).max(EQ_FREQ_MIN_HZ))
}

pub fn validate_eq_frequency_hz(frequency_hz: f64, sample_rate: f64) -> Result<(), String> {
    if !frequency_hz.is_finite() {
        return Err("frequency must be finite".to_string());
    }
    let max_frequency = eq_max_frequency_hz(sample_rate)?;
    if !(EQ_FREQ_MIN_HZ..=max_frequency).contains(&frequency_hz) {
        return Err(format!(
            "frequency {frequency_hz} Hz out of range [{EQ_FREQ_MIN_HZ}, {max_frequency}]"
        ));
    }
    Ok(())
}

pub fn validate_eq_gain_db(gain_db: f64) -> Result<(), String> {
    if !gain_db.is_finite() {
        return Err("gain must be finite".to_string());
    }
    if !(EQ_GAIN_MIN_DB..=EQ_GAIN_MAX_DB).contains(&gain_db) {
        return Err(format!(
            "gain {gain_db} dB out of range [{EQ_GAIN_MIN_DB}, {EQ_GAIN_MAX_DB}]"
        ));
    }
    Ok(())
}

pub fn validate_eq_q(q: f64) -> Result<(), String> {
    if !q.is_finite() {
        return Err("Q must be finite".to_string());
    }
    if !(EQ_Q_MIN..=EQ_Q_MAX).contains(&q) {
        return Err(format!("Q {q} out of range [{EQ_Q_MIN}, {EQ_Q_MAX}]"));
    }
    Ok(())
}

pub fn validate_eq_slope(slope_db_per_octave: u8) -> Result<(), String> {
    if !SUPPORTED_PASS_SLOPES_DB_PER_OCTAVE.contains(&slope_db_per_octave) {
        return Err(format!(
            "slope {slope_db_per_octave} dB/octave is unsupported; expected one of {:?}",
            SUPPORTED_PASS_SLOPES_DB_PER_OCTAVE
        ));
    }
    Ok(())
}

fn butterworth_section_q(section_index: usize, section_count: usize) -> f64 {
    let order = 2 * section_count;
    let angle = (2 * section_index + 1) as f64 * PI / (2 * order) as f64;
    1.0 / (2.0 * angle.cos())
}

fn pass_section_count(slope_db_per_octave: u8) -> Option<usize> {
    SUPPORTED_PASS_SLOPES_DB_PER_OCTAVE
        .contains(&slope_db_per_octave)
        .then_some(slope_db_per_octave as usize / 12)
}

struct EqBand {
    sections: [Biquad; MAX_PASS_SECTIONS],
    config: EqBandConfig,
    processing_sections: usize,
    target_sections: usize,
}

impl EqBand {
    fn new(config: EqBandConfig, sample_rate: f64) -> Self {
        let target_sections = Self::required_sections(config);
        let sections = std::array::from_fn(|section_index| {
            if section_index < target_sections {
                let (filter_type, gain_db, q) =
                    Self::section_parameters(config, section_index, target_sections);
                Biquad::new(filter_type, config.frequency_hz, gain_db, q, sample_rate)
            } else {
                Biquad::new(
                    BiquadType::Bypass,
                    config.frequency_hz,
                    0.0,
                    DEFAULT_Q,
                    sample_rate,
                )
            }
        });
        Self {
            sections,
            config,
            processing_sections: target_sections,
            target_sections,
        }
    }

    fn required_sections(config: EqBandConfig) -> usize {
        if !config.enabled {
            0
        } else if config.filter_type.is_pass() {
            pass_section_count(config.slope_db_per_octave).unwrap_or(1)
        } else {
            1
        }
    }

    fn section_parameters(
        config: EqBandConfig,
        section_index: usize,
        section_count: usize,
    ) -> (BiquadType, f64, f64) {
        if config.filter_type.is_pass() {
            (
                config.filter_type.biquad_type(),
                0.0,
                butterworth_section_q(section_index, section_count),
            )
        } else {
            let gain_db = if config.filter_type == EqFilterType::Notch {
                0.0
            } else {
                config.gain_db
            };
            (config.filter_type.biquad_type(), gain_db, config.q)
        }
    }

    fn set_config(&mut self, config: EqBandConfig) {
        self.config = config;
        let target_sections = Self::required_sections(config);
        let processing_sections = self.processing_sections.max(target_sections);
        for section_index in 0..processing_sections {
            let (filter_type, gain_db, q) = if section_index < target_sections {
                Self::section_parameters(config, section_index, target_sections)
            } else {
                (BiquadType::Bypass, 0.0, DEFAULT_Q)
            };
            self.sections[section_index].set_parameters(
                filter_type,
                config.frequency_hz,
                gain_db,
                q,
            );
        }
        self.processing_sections = processing_sections;
        self.target_sections = target_sections;
    }

    fn finish_retired_sections(&mut self) {
        while self.processing_sections > self.target_sections
            && !self.sections[self.processing_sections - 1].is_crossfading()
        {
            self.processing_sections -= 1;
        }
    }

    #[inline]
    fn process_sample(&mut self, mut sample: f32) -> f32 {
        for section in &mut self.sections[..self.processing_sections] {
            sample = section.process_sample(sample);
        }
        self.finish_retired_sections();
        sample
    }

    fn process_block_inplace(&mut self, buffer: &mut [f32]) {
        for section in &mut self.sections[..self.processing_sections] {
            section.process_block_inplace(buffer);
        }
        self.finish_retired_sections();
    }

    fn reset(&mut self) {
        let target_sections = Self::required_sections(self.config);
        for (section_index, section) in self.sections.iter_mut().enumerate() {
            let (filter_type, gain_db, q) = if section_index < target_sections {
                Self::section_parameters(self.config, section_index, target_sections)
            } else {
                (BiquadType::Bypass, 0.0, DEFAULT_Q)
            };
            section.set_parameters_immediate(filter_type, self.config.frequency_hz, gain_db, q);
        }
        self.processing_sections = target_sections;
        self.target_sections = target_sections;
    }

    fn target_magnitude_response_db(&self, frequency_hz: f64) -> f64 {
        self.sections[..self.target_sections]
            .iter()
            .map(|section| section.target_magnitude_response_db(frequency_hz))
            .sum()
    }
}

/// 10-Band Parametric Equalizer
///
/// Provides professional-quality equalization with configurable
/// frequency, gain, and Q for each band.
pub struct ParametricEQ {
    bands: [EqBand; NUM_BANDS],
    enabled: bool,
    sample_rate: f64,
}

impl ParametricEQ {
    /// Create a new 10-band parametric EQ
    pub fn new(sample_rate: f64) -> Self {
        let bands = std::array::from_fn(|index| {
            EqBand::new(EqBandConfig::default_for_index(index), sample_rate)
        });

        Self {
            bands,
            enabled: true,
            sample_rate,
        }
    }

    /// Process a block of samples in-place
    pub fn process_block_inplace(&mut self, buffer: &mut [f32]) {
        if !self.enabled {
            return;
        }

        for band in &mut self.bands {
            band.process_block_inplace(buffer);
        }
    }

    /// Process a single sample through all bands
    #[inline]
    pub fn process_sample(&mut self, mut sample: f32) -> f32 {
        if !self.enabled {
            return sample;
        }

        for band in &mut self.bands {
            sample = band.process_sample(sample);
        }
        sample
    }

    /// Reset all filter states
    pub fn reset(&mut self) {
        for band in &mut self.bands {
            band.reset();
        }
    }

    /// Set gain for a specific band (0-9)
    ///
    /// # Arguments
    /// * `band_index` - Band index (0-9)
    /// * `gain_db` - Gain in dB (typically -12 to +12)
    pub fn set_band_gain(&mut self, band_index: usize, gain_db: f64) {
        if band_index < NUM_BANDS {
            let mut config = self.bands[band_index].config;
            config.gain_db = gain_db;
            self.bands[band_index].set_config(config);
        }
    }

    /// Set frequency for a specific band
    ///
    /// # Arguments
    /// * `band_index` - Band index (0-9)
    /// * `frequency` - Center frequency in Hz
    pub fn set_band_frequency(&mut self, band_index: usize, frequency: f64) {
        if band_index < NUM_BANDS {
            let mut config = self.bands[band_index].config;
            config.frequency_hz = frequency;
            self.bands[band_index].set_config(config);
        }
    }

    /// Set Q factor for a specific band
    ///
    /// # Arguments
    /// * `band_index` - Band index (0-9)
    /// * `q` - Q factor (higher = narrower bandwidth)
    pub fn set_band_q(&mut self, band_index: usize, q: f64) {
        if band_index < NUM_BANDS {
            let mut config = self.bands[band_index].config;
            config.q = q;
            self.bands[band_index].set_config(config);
        }
    }

    /// Enable or disable a specific band
    pub fn set_band_enabled(&mut self, band_index: usize, enabled: bool) {
        if band_index < NUM_BANDS {
            let mut config = self.bands[band_index].config;
            config.enabled = enabled;
            self.bands[band_index].set_config(config);
        }
    }

    /// Set the public filter type for a specific band.
    pub fn set_band_filter_type(&mut self, band_index: usize, filter_type: EqFilterType) {
        if band_index < NUM_BANDS {
            let mut config = self.bands[band_index].config;
            config.filter_type = filter_type;
            self.bands[band_index].set_config(config);
        }
    }

    /// Set an even-order Butterworth pass-filter slope for a specific band.
    pub fn set_band_slope(&mut self, band_index: usize, slope_db_per_octave: u8) {
        if band_index < NUM_BANDS {
            let mut config = self.bands[band_index].config;
            config.slope_db_per_octave = slope_db_per_octave;
            self.bands[band_index].set_config(config);
        }
    }

    /// Apply one complete band configuration in one coefficient transition.
    pub fn set_band_config(&mut self, band_index: usize, config: EqBandConfig) {
        if band_index < NUM_BANDS {
            self.bands[band_index].set_config(config);
        }
    }

    /// Enable or disable the entire EQ
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if EQ is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get band parameters (frequency, gain_db, q)
    pub fn get_band_params(&self, band_index: usize) -> Option<(f64, f64, f64)> {
        if band_index < NUM_BANDS {
            let config = self.bands[band_index].config;
            Some((config.frequency_hz, config.gain_db, config.q))
        } else {
            None
        }
    }

    /// Get the complete configuration for one band.
    pub fn get_band_config(&self, band_index: usize) -> Option<EqBandConfig> {
        self.bands.get(band_index).map(|band| band.config)
    }

    /// Get the sample rate
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Get default frequency for a band
    pub fn default_frequency(band_index: usize) -> Option<f64> {
        if band_index < NUM_BANDS {
            Some(DEFAULT_FREQUENCIES[band_index])
        } else {
            None
        }
    }

    /// Calculate the exact cascaded response used by the runtime EQ.
    pub fn magnitude_response_db(&self, frequencies_hz: &[f64]) -> Vec<f64> {
        if !self.enabled {
            return vec![0.0; frequencies_hz.len()];
        }
        frequencies_hz
            .iter()
            .map(|frequency_hz| {
                self.bands
                    .iter()
                    .map(|band| band.target_magnitude_response_db(*frequency_hz))
                    .sum()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eq_flat_response() {
        let mut eq = ParametricEQ::new(48000.0);

        // With all gains at 0 dB, EQ should be nearly transparent
        let input = 0.5f32;
        let output = eq.process_sample(input);

        assert!((output - input).abs() < 1e-5);
    }

    fn sine_gain_db(eq: &mut ParametricEQ, freq_hz: f64) -> f64 {
        let sample_rate = 48_000.0;
        let samples = 16_384;
        let settle = 4096;
        let mut in_sum = 0.0_f64;
        let mut out_sum = 0.0_f64;
        for n in 0..samples {
            let t = n as f64 / sample_rate;
            let input = (2.0 * std::f64::consts::PI * freq_hz * t).sin() as f32 * 0.25;
            let output = eq.process_sample(input);
            if n >= settle {
                in_sum += (input as f64) * (input as f64);
                out_sum += (output as f64) * (output as f64);
            }
        }
        let input_rms = (in_sum / (samples - settle) as f64).sqrt();
        let output_rms = (out_sum / (samples - settle) as f64).sqrt();
        20.0 * (output_rms / input_rms).log10()
    }

    #[test]
    fn test_eq_peaking_band_reaches_center_gain() {
        let mut eq = ParametricEQ::new(48_000.0);
        eq.set_band_frequency(4, 1000.0);
        eq.set_band_q(4, 2.0);
        eq.set_band_gain(4, 6.0);

        let gain = sine_gain_db(&mut eq, 1000.0);

        assert!((gain - 6.0).abs() < 0.8, "center gain was {gain:.2} dB");
    }

    #[test]
    fn test_eq_shelves_affect_expected_probe_frequencies() {
        let mut low = ParametricEQ::new(48_000.0);
        low.set_band_gain(0, 6.0);
        let low_probe = sine_gain_db(&mut low, 80.0);
        let high_probe_after_low_shelf = sine_gain_db(&mut low, 5000.0);
        assert!(low_probe > high_probe_after_low_shelf + 2.0);

        let mut high = ParametricEQ::new(48_000.0);
        high.set_band_gain(9, 6.0);
        let high_probe = sine_gain_db(&mut high, 20_000.0);
        let low_probe_after_high_shelf = sine_gain_db(&mut high, 1000.0);
        assert!(
            high_probe > low_probe_after_high_shelf + 2.0,
            "high shelf probe was {high_probe:.2} dB vs low probe {low_probe_after_high_shelf:.2} dB"
        );
    }

    #[test]
    fn test_eq_extreme_valid_settings_remain_finite() {
        let mut eq = ParametricEQ::new(48_000.0);
        for band in 0..NUM_BANDS {
            eq.set_band_gain(band, if band % 2 == 0 { 12.0 } else { -12.0 });
            eq.set_band_q(band, if band % 2 == 0 { 0.1 } else { 10.0 });
        }

        for n in 0..4096 {
            let input = if n % 2 == 0 { 0.9 } else { -0.9 };
            let output = eq.process_sample(input);
            assert!(output.is_finite());
            assert!(output.abs() < 64.0);
        }
    }

    #[test]
    fn test_eq_disabled() {
        let mut eq = ParametricEQ::new(48000.0);
        eq.set_band_gain(0, 12.0); // Boost low shelf
        eq.set_enabled(false);

        let input = 0.5f32;
        let output = eq.process_sample(input);

        assert_eq!(output, input);
    }

    #[test]
    fn test_eq_band_params() {
        let mut eq = ParametricEQ::new(48000.0);

        // Set custom parameters
        eq.set_band_frequency(5, 3000.0);
        eq.set_band_gain(5, 6.0);
        eq.set_band_q(5, 2.0);

        let params = eq.get_band_params(5).unwrap();
        assert!((params.0 - 3000.0).abs() < 0.001);
        assert!((params.1 - 6.0).abs() < 0.001);
        assert!((params.2 - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_eq_default_frequencies() {
        assert_eq!(ParametricEQ::default_frequency(0), Some(80.0));
        assert_eq!(ParametricEQ::default_frequency(9), Some(16000.0));
        assert_eq!(ParametricEQ::default_frequency(10), None);
    }

    #[test]
    fn test_eq_magnitude_response_is_flat_at_zero_gain() {
        let eq = ParametricEQ::new(48_000.0);
        let response = eq.magnitude_response_db(&[20.0, 80.0, 1_000.0, 20_000.0]);
        assert!(response.iter().all(|value| value.abs() < 1.0e-10));
    }

    #[test]
    fn test_eq_magnitude_response_sums_cascaded_bands() {
        let mut eq = ParametricEQ::new(48_000.0);
        eq.set_band_frequency(4, 1_000.0);
        eq.set_band_q(4, 2.0);
        eq.set_band_gain(4, 6.0);
        eq.set_band_frequency(5, 1_000.0);
        eq.set_band_q(5, 2.0);
        eq.set_band_gain(5, -3.0);

        let response = eq.magnitude_response_db(&[1_000.0]);
        assert!((response[0] - 3.0).abs() < 1.0e-8);
    }

    #[test]
    fn test_public_filter_type_ids_and_names_are_stable() {
        let expected = [
            (0, EqFilterType::LowShelf, "low_shelf"),
            (1, EqFilterType::Bell, "bell"),
            (2, EqFilterType::HighShelf, "high_shelf"),
            (3, EqFilterType::Notch, "notch"),
            (4, EqFilterType::HighPass, "high_pass"),
            (5, EqFilterType::LowPass, "low_pass"),
        ];
        for (id, filter_type, name) in expected {
            assert_eq!(EqFilterType::from_id(id), Some(filter_type));
            assert_eq!(EqFilterType::from_name(name), Some(filter_type));
            assert_eq!(filter_type.name(), name);
        }
        assert_eq!(EqFilterType::from_id(6), None);
        assert_eq!(EqFilterType::from_name("bypass"), None);
    }

    #[test]
    fn test_notch_band_nulls_center_without_using_gain() {
        let mut eq = ParametricEQ::new(48_000.0);
        let mut config = eq.get_band_config(4).unwrap();
        config.filter_type = EqFilterType::Notch;
        config.frequency_hz = 1_000.0;
        config.gain_db = 12.0;
        config.q = 4.0;
        eq.set_band_config(4, config);

        assert!(eq.magnitude_response_db(&[1_000.0])[0] < -150.0);
    }

    #[test]
    fn test_even_order_butterworth_slopes_are_minus_three_db_at_cutoff() {
        for slope in SUPPORTED_PASS_SLOPES_DB_PER_OCTAVE {
            for filter_type in [EqFilterType::HighPass, EqFilterType::LowPass] {
                let mut eq = ParametricEQ::new(48_000.0);
                let mut config = eq.get_band_config(4).unwrap();
                config.filter_type = filter_type;
                config.frequency_hz = 2_000.0;
                config.slope_db_per_octave = slope;
                eq.set_band_config(4, config);
                let cutoff_db = eq.magnitude_response_db(&[2_000.0])[0];
                assert!(
                    (cutoff_db + 3.010_299_956_64).abs() < 1.0e-8,
                    "type={filter_type:?} slope={slope} cutoff={cutoff_db}"
                );
            }
        }
    }

    #[test]
    fn test_pass_filter_stopband_matches_selected_asymptotic_slope() {
        for slope in SUPPORTED_PASS_SLOPES_DB_PER_OCTAVE {
            let mut high_pass = ParametricEQ::new(48_000.0);
            let mut config = high_pass.get_band_config(4).unwrap();
            config.filter_type = EqFilterType::HighPass;
            config.frequency_hz = 2_000.0;
            config.slope_db_per_octave = slope;
            high_pass.set_band_config(4, config);
            let response = high_pass.magnitude_response_db(&[125.0, 250.0]);
            let measured_per_octave = response[1] - response[0];
            assert!(
                (measured_per_octave - f64::from(slope)).abs() < 0.35,
                "high-pass slope={slope} measured={measured_per_octave}"
            );

            // Keep the stopband probes far below Nyquist so this measures the
            // selected asymptotic slope rather than bilinear-transform warping.
            let mut low_pass = ParametricEQ::new(768_000.0);
            let mut config = low_pass.get_band_config(4).unwrap();
            config.filter_type = EqFilterType::LowPass;
            config.frequency_hz = 2_000.0;
            config.slope_db_per_octave = slope;
            low_pass.set_band_config(4, config);
            let response = low_pass.magnitude_response_db(&[16_000.0, 32_000.0]);
            let measured_per_octave = response[0] - response[1];
            assert!(
                (measured_per_octave - f64::from(slope)).abs() < 1.0,
                "low-pass slope={slope} measured={measured_per_octave}"
            );
        }
    }

    #[test]
    fn test_every_filter_type_and_extreme_setting_stays_finite() {
        for filter_type in [
            EqFilterType::LowShelf,
            EqFilterType::Bell,
            EqFilterType::HighShelf,
            EqFilterType::Notch,
            EqFilterType::HighPass,
            EqFilterType::LowPass,
        ] {
            for frequency_hz in [20.0, 23_900.0] {
                let mut eq = ParametricEQ::new(48_000.0);
                let mut config = eq.get_band_config(4).unwrap();
                config.filter_type = filter_type;
                config.frequency_hz = frequency_hz;
                config.gain_db = 12.0;
                config.q = 10.0;
                config.slope_db_per_octave = 48;
                eq.set_band_config(4, config);
                assert!(eq
                    .magnitude_response_db(&[20.0, 1_000.0, 20_000.0])
                    .iter()
                    .all(|value| value.is_finite()));
                for sample_index in 0..4096 {
                    let input = if sample_index & 1 == 0 { 0.5 } else { -0.5 };
                    assert!(eq.process_sample(input).is_finite());
                }
            }
        }
    }

    #[test]
    fn test_live_type_slope_and_enable_changes_are_click_bounded() {
        let sample_rate = 48_000.0;
        let mut eq = ParametricEQ::new(sample_rate);
        let mut previous = 0.0_f32;
        let mut max_step = 0.0_f32;
        for sample_index in 0..24_000 {
            if sample_index == 4_000 {
                let mut config = eq.get_band_config(4).unwrap();
                config.filter_type = EqFilterType::HighPass;
                config.frequency_hz = 600.0;
                config.slope_db_per_octave = 48;
                eq.set_band_config(4, config);
            } else if sample_index == 10_000 {
                let mut config = eq.get_band_config(4).unwrap();
                config.filter_type = EqFilterType::Notch;
                config.frequency_hz = 1_000.0;
                config.q = 8.0;
                eq.set_band_config(4, config);
            } else if sample_index == 16_000 {
                eq.set_band_enabled(4, false);
            } else if sample_index == 20_000 {
                eq.set_band_enabled(4, true);
            }
            let phase = 2.0 * PI * 1_000.0 * sample_index as f64 / sample_rate;
            let output = eq.process_sample((0.25 * phase.sin()) as f32);
            assert!(output.is_finite());
            max_step = max_step.max((output - previous).abs());
            previous = output;
        }
        assert!(max_step < 0.2, "live EQ edit max step was {max_step}");
    }

    #[test]
    fn test_reset_commits_target_configuration_without_old_stream_state() {
        let mut eq = ParametricEQ::new(48_000.0);
        let mut config = eq.get_band_config(4).unwrap();
        config.filter_type = EqFilterType::HighPass;
        config.frequency_hz = 1_000.0;
        config.slope_db_per_octave = 48;
        eq.set_band_config(4, config);
        eq.reset();

        assert_eq!(eq.bands[4].processing_sections, 4);
        assert_eq!(eq.bands[4].target_sections, 4);
        assert!(eq.bands[4]
            .sections
            .iter()
            .all(|section| !section.is_crossfading()));
    }

    #[test]
    #[ignore = "release-mode EQ cost measurement"]
    fn benchmark_parametric_eq_filter_types() {
        use std::hint::black_box;
        use std::time::Instant;

        const SAMPLES: usize = 2_000_000;
        fn elapsed(mut eq: ParametricEQ) -> std::time::Duration {
            let started = Instant::now();
            for index in 0..SAMPLES {
                black_box(eq.process_sample(black_box((index as f32 * 0.013).sin())));
            }
            started.elapsed()
        }

        let default_elapsed = elapsed(ParametricEQ::new(48_000.0));
        let mut steep = ParametricEQ::new(48_000.0);
        let mut config = steep.get_band_config(0).unwrap();
        config.filter_type = EqFilterType::HighPass;
        config.slope_db_per_octave = 48;
        steep.set_band_config(0, config);
        steep.reset();
        let steep_elapsed = elapsed(steep);
        println!(
            "eq default={:.2} ns/sample one-48dB-pass={:.2} ns/sample ratio={:.2}",
            default_elapsed.as_nanos() as f64 / SAMPLES as f64,
            steep_elapsed.as_nanos() as f64 / SAMPLES as f64,
            steep_elapsed.as_secs_f64() / default_elapsed.as_secs_f64()
        );
    }
}
