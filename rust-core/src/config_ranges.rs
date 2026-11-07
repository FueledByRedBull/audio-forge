//! The numeric ranges of the processing settings.
//!
//! The live setters and the offline simulators clamp to these constants, and
//! Python reads the same table through [`processing_ranges`] to validate saved
//! presets and to bound the controls, so a range exists in one place.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::dsp::eq::{
    EQ_FREQ_MAX_HZ, EQ_FREQ_MIN_HZ, EQ_GAIN_MAX_DB, EQ_GAIN_MIN_DB, EQ_Q_MAX, EQ_Q_MIN,
};
use crate::dsp::noise_suppressor::NOISE_MODEL_IDS;

pub(crate) const GATE_THRESHOLD_MIN_DB: f64 = -80.0;
pub(crate) const GATE_THRESHOLD_MAX_DB: f64 = -10.0;
pub(crate) const GATE_ATTACK_MIN_MS: f64 = 0.1;
pub(crate) const GATE_ATTACK_MAX_MS: f64 = 100.0;
pub(crate) const GATE_RELEASE_MIN_MS: f64 = 10.0;
pub(crate) const GATE_RELEASE_MAX_MS: f64 = 1000.0;
/// The wire values of `GateMode`: 0 threshold only, 1 VAD assisted, 2 VAD only.
pub(crate) const GATE_MODE_MIN: f64 = 0.0;
pub(crate) const GATE_MODE_MAX: f64 = 2.0;
pub(crate) const VAD_THRESHOLD_MIN: f64 = 0.3;
pub(crate) const VAD_THRESHOLD_MAX: f64 = 0.7;
pub(crate) const VAD_HOLD_MIN_MS: f64 = 0.0;
pub(crate) const VAD_HOLD_MAX_MS: f64 = 500.0;
pub(crate) const VAD_PRE_GAIN_MIN: f64 = 1.0;
pub(crate) const VAD_PRE_GAIN_MAX: f64 = 10.0;
pub(crate) const GATE_MARGIN_MIN_DB: f64 = 0.0;
pub(crate) const GATE_MARGIN_MAX_DB: f64 = 20.0;
pub(crate) const RNNOISE_STRENGTH_MIN: f64 = 0.0;
pub(crate) const RNNOISE_STRENGTH_MAX: f64 = 1.0;
pub(crate) const DEESSER_AUTO_AMOUNT_MIN: f64 = 0.0;
pub(crate) const DEESSER_AUTO_AMOUNT_MAX: f64 = 1.0;
pub(crate) const DEESSER_LOW_CUT_MIN_HZ: f64 = 2000.0;
pub(crate) const DEESSER_LOW_CUT_MAX_HZ: f64 = 12000.0;
pub(crate) const DEESSER_HIGH_CUT_MIN_HZ: f64 = 2200.0;
pub(crate) const DEESSER_HIGH_CUT_MAX_HZ: f64 = 16000.0;
pub(crate) const DEESSER_THRESHOLD_MIN_DB: f64 = -60.0;
pub(crate) const DEESSER_THRESHOLD_MAX_DB: f64 = -6.0;
pub(crate) const DEESSER_RATIO_MIN: f64 = 1.0;
pub(crate) const DEESSER_RATIO_MAX: f64 = 20.0;
pub(crate) const DEESSER_ATTACK_MIN_MS: f64 = 0.1;
pub(crate) const DEESSER_ATTACK_MAX_MS: f64 = 50.0;
pub(crate) const DEESSER_RELEASE_MIN_MS: f64 = 5.0;
pub(crate) const DEESSER_RELEASE_MAX_MS: f64 = 500.0;
pub(crate) const DEESSER_MAX_REDUCTION_MIN_DB: f64 = 0.0;
pub(crate) const DEESSER_MAX_REDUCTION_MAX_DB: f64 = 24.0;
pub(crate) const COMPRESSOR_THRESHOLD_MIN_DB: f64 = -60.0;
pub(crate) const COMPRESSOR_THRESHOLD_MAX_DB: f64 = 0.0;
pub(crate) const COMPRESSOR_RATIO_MIN: f64 = 1.0;
pub(crate) const COMPRESSOR_RATIO_MAX: f64 = 20.0;
pub(crate) const COMPRESSOR_ATTACK_MIN_MS: f64 = 0.1;
pub(crate) const COMPRESSOR_ATTACK_MAX_MS: f64 = 100.0;
pub(crate) const COMPRESSOR_RELEASE_MIN_MS: f64 = 10.0;
pub(crate) const COMPRESSOR_RELEASE_MAX_MS: f64 = 1000.0;
pub(crate) const COMPRESSOR_BASE_RELEASE_MIN_MS: f64 = 20.0;
pub(crate) const COMPRESSOR_BASE_RELEASE_MAX_MS: f64 = 200.0;
pub(crate) const COMPRESSOR_MAKEUP_MIN_DB: f64 = 0.0;
pub(crate) const COMPRESSOR_MAKEUP_MAX_DB: f64 = 24.0;
pub(crate) const COMPRESSOR_TARGET_LUFS_MIN: f64 = -24.0;
pub(crate) const COMPRESSOR_TARGET_LUFS_MAX: f64 = -12.0;
pub(crate) const LIMITER_CEILING_MIN_DB: f64 = -12.0;
pub(crate) const LIMITER_CEILING_MAX_DB: f64 = 0.0;
pub(crate) const LIMITER_RELEASE_MIN_MS: f64 = 10.0;
pub(crate) const LIMITER_RELEASE_MAX_MS: f64 = 500.0;

/// One section's settings as `(name, min, max)`, named as in a saved preset.
type SectionRanges = &'static [(&'static str, f64, f64)];

const PROCESSING_RANGES: &[(&str, SectionRanges)] = &[
    (
        "gate",
        &[
            ("threshold_db", GATE_THRESHOLD_MIN_DB, GATE_THRESHOLD_MAX_DB),
            ("attack_ms", GATE_ATTACK_MIN_MS, GATE_ATTACK_MAX_MS),
            ("release_ms", GATE_RELEASE_MIN_MS, GATE_RELEASE_MAX_MS),
            ("gate_mode", GATE_MODE_MIN, GATE_MODE_MAX),
            ("vad_threshold", VAD_THRESHOLD_MIN, VAD_THRESHOLD_MAX),
            ("vad_hold_time_ms", VAD_HOLD_MIN_MS, VAD_HOLD_MAX_MS),
            ("vad_pre_gain", VAD_PRE_GAIN_MIN, VAD_PRE_GAIN_MAX),
            ("gate_margin_db", GATE_MARGIN_MIN_DB, GATE_MARGIN_MAX_DB),
        ],
    ),
    (
        "eq",
        &[
            ("band_freq", EQ_FREQ_MIN_HZ, EQ_FREQ_MAX_HZ),
            ("band_gain", EQ_GAIN_MIN_DB, EQ_GAIN_MAX_DB),
            ("band_q", EQ_Q_MIN, EQ_Q_MAX),
        ],
    ),
    (
        "rnnoise",
        &[("strength", RNNOISE_STRENGTH_MIN, RNNOISE_STRENGTH_MAX)],
    ),
    (
        "deesser",
        &[
            (
                "auto_amount",
                DEESSER_AUTO_AMOUNT_MIN,
                DEESSER_AUTO_AMOUNT_MAX,
            ),
            ("low_cut_hz", DEESSER_LOW_CUT_MIN_HZ, DEESSER_LOW_CUT_MAX_HZ),
            (
                "high_cut_hz",
                DEESSER_HIGH_CUT_MIN_HZ,
                DEESSER_HIGH_CUT_MAX_HZ,
            ),
            (
                "threshold_db",
                DEESSER_THRESHOLD_MIN_DB,
                DEESSER_THRESHOLD_MAX_DB,
            ),
            ("ratio", DEESSER_RATIO_MIN, DEESSER_RATIO_MAX),
            ("attack_ms", DEESSER_ATTACK_MIN_MS, DEESSER_ATTACK_MAX_MS),
            ("release_ms", DEESSER_RELEASE_MIN_MS, DEESSER_RELEASE_MAX_MS),
            (
                "max_reduction_db",
                DEESSER_MAX_REDUCTION_MIN_DB,
                DEESSER_MAX_REDUCTION_MAX_DB,
            ),
        ],
    ),
    (
        "compressor",
        &[
            (
                "threshold_db",
                COMPRESSOR_THRESHOLD_MIN_DB,
                COMPRESSOR_THRESHOLD_MAX_DB,
            ),
            ("ratio", COMPRESSOR_RATIO_MIN, COMPRESSOR_RATIO_MAX),
            (
                "attack_ms",
                COMPRESSOR_ATTACK_MIN_MS,
                COMPRESSOR_ATTACK_MAX_MS,
            ),
            (
                "release_ms",
                COMPRESSOR_RELEASE_MIN_MS,
                COMPRESSOR_RELEASE_MAX_MS,
            ),
            (
                "makeup_gain_db",
                COMPRESSOR_MAKEUP_MIN_DB,
                COMPRESSOR_MAKEUP_MAX_DB,
            ),
            (
                "base_release_ms",
                COMPRESSOR_BASE_RELEASE_MIN_MS,
                COMPRESSOR_BASE_RELEASE_MAX_MS,
            ),
            (
                "target_lufs",
                COMPRESSOR_TARGET_LUFS_MIN,
                COMPRESSOR_TARGET_LUFS_MAX,
            ),
        ],
    ),
    (
        "limiter",
        &[
            ("ceiling_db", LIMITER_CEILING_MIN_DB, LIMITER_CEILING_MAX_DB),
            ("release_ms", LIMITER_RELEASE_MIN_MS, LIMITER_RELEASE_MAX_MS),
        ],
    ),
];

/// `{section: {setting: (min, max)}}` for every numeric processing setting.
#[pyfunction]
pub(crate) fn processing_ranges(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let sections = PyDict::new(py);
    for &(section, ranges) in PROCESSING_RANGES {
        let settings = PyDict::new(py);
        for &(name, min, max) in ranges {
            settings.set_item(name, (min, max))?;
        }
        sections.set_item(section, settings)?;
    }
    Ok(sections)
}

/// The noise model ids a saved preset may name, in the order the UI lists them.
#[pyfunction]
pub(crate) fn noise_model_ids() -> Vec<&'static str> {
    NOISE_MODEL_IDS.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn range(section: &str, name: &str) -> (f64, f64) {
        PROCESSING_RANGES
            .iter()
            .find(|(candidate, _)| *candidate == section)
            .and_then(|(_, ranges)| ranges.iter().find(|(candidate, ..)| *candidate == name))
            .map(|&(_, min, max)| (min, max))
            .unwrap_or_else(|| panic!("no range for {section}.{name}"))
    }

    #[test]
    fn every_range_is_finite_and_ordered() {
        for (section, ranges) in PROCESSING_RANGES {
            for (name, min, max) in *ranges {
                assert!(min.is_finite() && max.is_finite(), "{section}.{name}");
                assert!(min < max, "{section}.{name}");
            }
        }
    }

    #[test]
    fn setting_names_are_unique_within_a_section() {
        for (section, ranges) in PROCESSING_RANGES {
            let mut names: Vec<_> = ranges.iter().map(|(name, ..)| *name).collect();
            names.sort_unstable();
            names.dedup();
            assert_eq!(names.len(), ranges.len(), "{section}");
        }
    }

    #[test]
    fn the_single_precision_clamps_equal_their_literals() {
        // Values published in single precision must be the rounding of the
        // shipped decimal, or a clamp and the table would disagree.
        assert_eq!(VAD_THRESHOLD_MIN as f32, 0.3_f32);
        assert_eq!(VAD_THRESHOLD_MAX as f32, 0.7_f32);
        assert_eq!(VAD_HOLD_MAX_MS as f32, 500.0_f32);
        assert_eq!(VAD_PRE_GAIN_MAX as f32, 10.0_f32);
        assert_eq!(GATE_MARGIN_MAX_DB as f32, 20.0_f32);
        assert_eq!(RNNOISE_STRENGTH_MAX as f32, 1.0_f32);
    }

    #[test]
    fn the_equalizer_rows_are_the_dsp_limits() {
        assert_eq!(range("eq", "band_gain"), (EQ_GAIN_MIN_DB, EQ_GAIN_MAX_DB));
        assert_eq!(range("eq", "band_q"), (EQ_Q_MIN, EQ_Q_MAX));
        assert_eq!(range("eq", "band_freq"), (EQ_FREQ_MIN_HZ, EQ_FREQ_MAX_HZ));
    }

    #[cfg(feature = "vad")]
    #[test]
    fn the_gate_mode_range_is_the_modes_that_exist() {
        use crate::dsp::vad::GateMode;

        let (min, max) = range("gate", "gate_mode");
        for wire in 0..=u8::MAX {
            let in_range = (min..=max).contains(&f64::from(wire));
            assert_eq!(GateMode::from_u8(wire).is_some(), in_range, "mode {wire}");
        }
    }
}
