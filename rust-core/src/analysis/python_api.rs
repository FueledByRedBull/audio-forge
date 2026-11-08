//! PyO3 bindings for the offline analysis primitives.

use numpy::{
    Element, PyArray1, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods, PyUntypedArray,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::catch_panic;

/// `np.asarray(value, dtype=T)` flattened in C order, with its original shape.
fn flattened<T: Element + Copy>(value: &Bound<'_, PyAny>) -> PyResult<(Vec<usize>, Vec<T>)> {
    let py = value.py();
    let array = py
        .import("numpy")?
        .call_method1("asarray", (value, numpy::dtype::<T>(py)))?
        .cast_into::<PyArrayDyn<T>>()?;
    let shape = array.shape().to_vec();
    let values = array.readonly().as_array().iter().copied().collect();
    Ok((shape, values))
}

fn vector<T: Element + Copy>(value: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<T>> {
    let (shape, values) = flattened(value)?;
    if shape.len() != 1 {
        return Err(PyValueError::new_err(format!(
            "{what} must be one-dimensional"
        )));
    }
    Ok(values)
}

#[pyfunction]
pub(crate) fn clamp(value: f64, low: f64, high: f64) -> f64 {
    super::clamp(value, low, high)
}

#[pyfunction]
pub(crate) fn band_mean(
    freqs: &Bound<'_, PyAny>,
    values: &Bound<'_, PyAny>,
    low: f64,
    high: f64,
) -> PyResult<f64> {
    catch_panic(|| {
        // The removed Python's mask paired equal shapes elementwise; other
        // shapes raised IndexError there whenever a bin fell in the band.
        let (freqs_shape, freqs) = flattened::<f64>(freqs)?;
        let (values_shape, values) = flattened::<f64>(values)?;
        if freqs_shape != values_shape {
            return Err(PyIndexError::new_err(
                "freqs and values must have the same shape",
            ));
        }
        super::band_mean(&freqs, &values, low, high).map_err(PyValueError::new_err)
    })
}

/// float32 audio resamples in float32; every other real dtype in float64.
#[pyfunction]
pub(crate) fn resample_poly<'py>(
    py: Python<'py>,
    samples: &Bound<'py, PyAny>,
    up: i64,
    down: i64,
) -> PyResult<Bound<'py, PyAny>> {
    catch_panic(|| {
        let array = py
            .import("numpy")?
            .call_method1("asarray", (samples,))?
            .cast_into::<PyUntypedArray>()?;
        if array.ndim() != 1 {
            return Err(PyValueError::new_err(
                "audio to resample must be one-dimensional",
            ));
        }
        let (Ok(up), Ok(down)) = (usize::try_from(up), usize::try_from(down)) else {
            return Err(PyValueError::new_err(
                "up and down must be positive integers",
            ));
        };
        let dtype = array.dtype();
        if dtype.kind() == b'c' {
            return Err(PyTypeError::new_err("complex audio is not supported"));
        }
        if dtype.is_equiv_to(&numpy::dtype::<f32>(py)) {
            let values = vector::<f32>(array.as_any(), "audio to resample")?;
            let output = py
                .detach(|| super::resample_poly(&values, up, down))
                .map_err(PyValueError::new_err)?;
            return Ok(PyArray1::from_vec(py, output).into_any());
        }
        let values = vector::<f64>(array.as_any(), "audio to resample")?;
        let output = py
            .detach(|| super::resample_poly(&values, up, down))
            .map_err(PyValueError::new_err)?;
        Ok(PyArray1::from_vec(py, output).into_any())
    })
}

#[pyfunction]
#[pyo3(signature = (samples, *, distance, prominence))]
pub(crate) fn find_peaks_distance_prominence<'py>(
    py: Python<'py>,
    samples: &Bound<'py, PyAny>,
    distance: i64,
    prominence: f64,
) -> PyResult<Bound<'py, PyArray1<isize>>> {
    catch_panic(|| {
        let values = vector::<f64>(samples, "peak detection input")?;
        let peaks = super::find_peaks_distance_prominence(
            &values,
            usize::try_from(distance).unwrap_or(0),
            prominence,
        )
        .map_err(PyValueError::new_err)?;
        Ok(PyArray1::from_vec(
            py,
            peaks.into_iter().map(|peak| peak as isize).collect(),
        ))
    })
}

#[pyfunction]
pub(crate) fn map_causal_vad_probabilities<'py>(
    py: Python<'py>,
    probabilities: Option<&Bound<'py, PyAny>>,
    positions: &Bound<'py, PyAny>,
    sample_rate: i64,
) -> PyResult<Option<Bound<'py, PyArray1<f32>>>> {
    catch_panic(|| {
        let Some(probabilities) = probabilities else {
            return Ok(None);
        };
        if sample_rate <= 0 {
            return Ok(None);
        }
        let (_, probabilities) = flattened::<f64>(probabilities)?;
        let (_, positions) = flattened::<i64>(positions)?;
        Ok(
            super::map_causal_vad_probabilities(&probabilities, &positions, sample_rate)
                .map(|mapped| PyArray1::from_vec(py, mapped)),
        )
    })
}

/// A keyword argument the removed Python passed through `float()`.
#[cfg(feature = "vad")]
pub(crate) enum FloatArgument<'py> {
    Default(f64),
    Given(Bound<'py, PyAny>),
}

#[cfg(feature = "vad")]
fn float_argument<'py>(value: &Bound<'py, PyAny>) -> PyResult<FloatArgument<'py>> {
    Ok(FloatArgument::Given(value.clone()))
}

#[cfg(feature = "vad")]
impl FloatArgument<'_> {
    /// `float(value)`, or `None` where that raises `TypeError` or `ValueError`.
    fn to_float(&self) -> PyResult<Option<f64>> {
        let value = match self {
            Self::Default(value) => return Ok(Some(*value)),
            Self::Given(value) => value,
        };
        let py = value.py();
        match py.get_type::<pyo3::types::PyFloat>().call1((value,)) {
            Ok(number) => number.extract().map(Some),
            Err(error)
                if error.is_instance_of::<PyTypeError>(py)
                    || error.is_instance_of::<PyValueError>(py) =>
            {
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }
}

/// Native Silero posteriors as float64 and `"silero"`, or `(None,
/// "energy_fallback")` when the capture, gain, threshold or model rules
/// inference out; callers then keep their energy analysis.
///
/// Arguments are checked in the removed Python's order, so a gain `float()`
/// rejects falls back before the audio is converted.
#[cfg(feature = "vad")]
#[pyfunction]
#[pyo3(signature = (
    audio,
    sample_rate,
    *,
    threshold = FloatArgument::Default(0.48),
    pre_gain = FloatArgument::Default(1.0),
))]
pub(crate) fn analyze_offline_vad<'py>(
    py: Python<'py>,
    audio: &Bound<'py, PyAny>,
    sample_rate: &Bound<'py, PyAny>,
    #[pyo3(from_py_with = float_argument)] threshold: FloatArgument<'py>,
    #[pyo3(from_py_with = float_argument)] pre_gain: FloatArgument<'py>,
) -> PyResult<(Option<Bound<'py, PyArray1<f64>>>, &'static str)> {
    const FALLBACK: &str = "energy_fallback";
    catch_panic(|| {
        let Some(pre_gain) = pre_gain
            .to_float()?
            .filter(|gain| gain.is_finite() && *gain > 0.0)
        else {
            return Ok((None, FALLBACK));
        };
        let (_, samples) = flattened::<f32>(audio)?;
        if samples.is_empty() || sample_rate.le(0)? {
            return Ok((None, FALLBACK));
        }
        let sample_rate: i64 = py
            .get_type::<pyo3::types::PyInt>()
            .call1((sample_rate,))?
            .extract()?;
        let Some(threshold) = threshold.to_float()? else {
            return Ok((None, FALLBACK));
        };
        let threshold = threshold as f32;
        let probabilities = py.detach(move || {
            super::offline_vad_probabilities(&samples, sample_rate, pre_gain, |prepared, rate| {
                if !threshold.is_finite()
                    || !(0.0..=1.0).contains(&threshold)
                    || prepared.iter().any(|sample| !sample.is_finite())
                {
                    return Err(());
                }
                std::panic::catch_unwind(|| {
                    crate::audio::processor::silero_offline_probabilities(prepared, rate, threshold)
                })
                .map_err(drop)?
                .map_err(drop)
            })
        });
        Ok(match probabilities {
            Some(probabilities) => (Some(PyArray1::from_vec(py, probabilities)), "silero"),
            None => (None, FALLBACK),
        })
    })
}

#[pyfunction]
#[pyo3(signature = (sample_rate = 48_000, duration_ms = 80.0, amplitude = 0.8))]
pub(crate) fn generate_probe_signal<'py>(
    py: Python<'py>,
    sample_rate: i64,
    duration_ms: f64,
    amplitude: f64,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    catch_panic(|| {
        let probe = super::latency::generate_probe_signal(sample_rate, duration_ms, amplitude)
            .map_err(PyValueError::new_err)?;
        Ok(PyArray1::from_vec(py, probe))
    })
}

/// The fields of `LatencyCalibrationResult` as a dict.
#[pyfunction]
#[pyo3(signature = (
    reference_probe,
    recorded_signal,
    sample_rate = 48_000,
    min_search_ms = 5.0,
    max_search_ms = 500.0,
    expected_playback_start_ms = None,
    expected_playback_jitter_ms = None,
    expected_latency_min_ms = None,
    expected_latency_max_ms = None,
    route_kind = None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn analyze_latency<'py>(
    py: Python<'py>,
    reference_probe: Option<&Bound<'py, PyAny>>,
    recorded_signal: Option<&Bound<'py, PyAny>>,
    sample_rate: i64,
    min_search_ms: f64,
    max_search_ms: f64,
    expected_playback_start_ms: Option<f64>,
    expected_playback_jitter_ms: Option<f64>,
    expected_latency_min_ms: Option<f64>,
    expected_latency_max_ms: Option<f64>,
    route_kind: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    catch_panic(|| {
        // str(route_kind or "output_to_input").strip().lower()
        let route_kind = match route_kind {
            Some(kind) if kind.is_truthy()? => kind
                .str()?
                .call_method0("strip")?
                .call_method0("lower")?
                .extract::<String>()?,
            _ => super::latency::DEFAULT_ROUTE_KIND.to_string(),
        };
        let reference = reference_probe
            .map(|probe| flattened::<f64>(probe).map(|(_, values)| values))
            .transpose()?;
        let recording = recorded_signal
            .map(|signal| flattened::<f64>(signal).map(|(_, values)| values))
            .transpose()?;
        let window = super::latency::SearchWindow {
            min_search_ms,
            max_search_ms,
            expected_playback_start_ms,
            expected_playback_jitter_ms,
            expected_latency_min_ms,
            expected_latency_max_ms,
        };
        let analysis = py
            .detach(|| {
                super::latency::analyze_latency(
                    reference.as_deref(),
                    recording.as_deref(),
                    sample_rate,
                    window,
                    &route_kind,
                )
            })
            .map_err(PyValueError::new_err)?;

        let result = pyo3::types::PyDict::new(py);
        result.set_item("success", analysis.success)?;
        result.set_item("measured_round_trip_ms", analysis.measured_round_trip_ms)?;
        result.set_item("estimated_one_way_ms", 0.0)?;
        result.set_item("applied_compensation_ms", analysis.route_latency_ms)?;
        result.set_item("confidence", analysis.confidence)?;
        result.set_item("peak_sample_offset", analysis.peak_sample_offset)?;
        result.set_item("message", analysis.message)?;
        result.set_item("repetition_count", analysis.repetition_count)?;
        result.set_item("agreement_ms", analysis.agreement_ms)?;
        result.set_item("ambiguity_score", analysis.ambiguity_score)?;
        result.set_item("sub_sample_offset", analysis.sub_sample_offset)?;
        result.set_item("route_latency_ms", analysis.route_latency_ms)?;
        result.set_item("directional_latency_ms", py.None())?;
        result.set_item("route_kind", analysis.route_kind)?;
        result.set_item("compensation_basis", super::latency::COMPENSATION_BASIS)?;
        Ok(result)
    })
}
