//! PyO3 bindings for the offline analysis primitives.

use numpy::{
    Complex32, Complex64, Element, PyArray1, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods,
    PyUntypedArray, PyUntypedArrayMethods,
};
use pyo3::exceptions::{
    PyIndexError, PyOverflowError, PyRuntimeWarning, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyInt};

use super::latency::LatencyError;
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

/// SciPy `resample_poly`'s default filter for reduced rates, as the removed
/// Python designed it in NumPy: `np.kaiser`'s `exp` is NumPy's own SIMD code
/// on some CPUs, so only NumPy reproduces its bits.
fn numpy_resample_taps(py: Python<'_>, up: usize, down: usize) -> PyResult<Vec<f64>> {
    let np = py.import("numpy")?;
    let max_rate = up.max(down);
    let half_length = 10 * max_rate;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", np.getattr("float64")?)?;
    let positions = np
        .call_method("arange", (2 * half_length + 1,), Some(&kwargs))?
        .sub(half_length)?;
    let cutoff = 1.0 / max_rate as f64;
    let window = np.call_method1("kaiser", (positions.len()?, 5.0))?;
    let taps = np
        .call_method1("sinc", (positions.mul(cutoff)?,))?
        .mul(cutoff)?
        .mul(window)?;
    let taps = taps.div(taps.call_method0("sum")?)?.mul(up)?;
    let taps = taps.cast_into::<PyArray1<f64>>()?;
    let taps = taps.readonly().as_slice()?.to_vec();
    Ok(taps)
}

/// `operator.index(rate)` as a positive integer; the removed Python raised
/// ValueError for anything else.
fn resample_rate(rate: &Bound<'_, PyAny>) -> PyResult<usize> {
    let invalid = || PyValueError::new_err("up and down must be positive integers");
    let index = rate
        .py()
        .import("operator")?
        .call_method1("index", (rate,))
        .map_err(|error| {
            if error.is_instance_of::<PyTypeError>(rate.py()) {
                invalid()
            } else {
                error
            }
        })?;
    if index.lt(1)? {
        return Err(invalid());
    }
    index.extract()
}

/// Resamples `array` in `T` and returns it as `output_dtype`, the dtype
/// NumPy's matmul gave the removed Python's output.
fn resample_as<'py, T>(
    py: Python<'py>,
    array: &Bound<'py, PyUntypedArray>,
    up: usize,
    down: usize,
    output_dtype: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>>
where
    T: super::ResampleSample + Element + Send + Sync,
{
    let taps = numpy_resample_taps(py, up, down)?;
    let values = vector::<T>(array.as_any(), "audio to resample")?;
    let output = py
        .detach(|| super::resample_poly(&values, up, down, &taps))
        .map_err(PyValueError::new_err)?;
    PyArray1::from_vec(py, output).call_method1("view", (output_dtype,))
}

/// Resamples in the precision the removed Python's NumPy arithmetic used.
///
/// Audio whose dtype equals float32, float64, complex64 or complex128 kept
/// it (taps and output); real audio of any other dtype met float64 taps
/// and took NumPy's promotion with float64. Extended precision, objects
/// and byte-swapped complex audio, which NumPy computed its own way, raise
/// TypeError. An identity ratio
/// copies the array, and empty audio keeps a float or complex dtype.
#[pyfunction]
pub(crate) fn resample_poly<'py>(
    py: Python<'py>,
    samples: &Bound<'py, PyAny>,
    up: &Bound<'py, PyAny>,
    down: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    catch_panic(|| {
        let np = py.import("numpy")?;
        let array = np
            .call_method1("asarray", (samples,))?
            .cast_into::<PyUntypedArray>()?;
        if array.ndim() != 1 {
            return Err(PyValueError::new_err(
                "audio to resample must be one-dimensional",
            ));
        }
        let up = resample_rate(up)?;
        let down = resample_rate(down)?;
        let (up, down) = super::reduced_rates(up, down).map_err(PyValueError::new_err)?;
        let dtype = array.dtype();
        if up == 1 && down == 1 {
            return array.call_method0("copy");
        }
        if array.len() == 0 {
            let kind = dtype.kind();
            let empty_dtype = if kind == b'f' || kind == b'c' {
                dtype.into_any()
            } else {
                np.getattr("float64")?
            };
            return np.call_method1("empty", (0, empty_dtype));
        }

        let kept = [
            numpy::dtype::<f32>(py),
            numpy::dtype::<f64>(py),
            numpy::dtype::<Complex32>(py),
            numpy::dtype::<Complex64>(py),
        ]
        .into_iter()
        .map(|candidate| dtype.eq(candidate))
        .collect::<PyResult<Vec<bool>>>()?;
        // `dtype in (...)`, as the removed Python chose the taps' dtype.
        let kept = kept.contains(&true);
        let output_dtype = if kept {
            dtype.clone().into_any()
        } else {
            np.call_method1("result_type", (&dtype, np.getattr("float64")?))?
        };
        let output = output_dtype.cast::<numpy::PyArrayDescr>()?;
        // Byte-swapped complex audio met NumPy's own cast-and-multiply path,
        // whose sums the native loop does not reproduce.
        match (output.kind(), output.itemsize(), kept) {
            (b'f', 4, _) => resample_as::<f32>(py, &array, up, down, &output_dtype),
            (b'f', 8, _) => resample_as::<f64>(py, &array, up, down, &output_dtype),
            (b'c', 8, true) => resample_as::<Complex32>(py, &array, up, down, &output_dtype),
            (b'c', 16, true) => resample_as::<Complex64>(py, &array, up, down, &output_dtype),
            _ => Err(PyTypeError::new_err(format!(
                "audio to resample must be float32, float64, complex64 or complex128 \
                 in native byte order, or real numbers that convert to float64 exactly, \
                 not {}",
                dtype.str()?
            ))),
        }
    })
}

#[pyfunction]
#[pyo3(signature = (samples, *, distance, prominence))]
pub(crate) fn find_peaks_distance_prominence<'py>(
    py: Python<'py>,
    samples: &Bound<'py, PyAny>,
    distance: f64,
    prominence: f64,
) -> PyResult<Bound<'py, PyArray1<isize>>> {
    catch_panic(|| {
        let np = py.import("numpy")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("dtype", np.getattr("float64")?)?;
        let array = np.call_method("asarray", (samples,), Some(&kwargs))?;
        let values = vector::<f64>(&array, "peak detection input")?;
        let candidates = super::peak_candidates(&values);
        // np.argsort(values[peaks]) as the removed Python ranked peaks: its
        // order among equal heights is NumPy's sort's, SIMD on some CPUs.
        let order = if candidates.is_empty() {
            Vec::new()
        } else {
            let peaks =
                PyArray1::from_vec(py, candidates.iter().map(|&peak| peak as isize).collect());
            let order = np
                .call_method1("argsort", (array.get_item(peaks)?,))?
                .cast_into::<PyArray1<isize>>()?;
            let order = order.readonly();
            order
                .as_slice()?
                .iter()
                .map(|&index| index as usize)
                .collect()
        };
        let peaks = super::select_peaks(&values, &candidates, &order, distance, prominence)
            .map_err(PyValueError::new_err)?;
        Ok(PyArray1::from_vec(
            py,
            peaks.into_iter().map(|peak| peak as isize).collect(),
        ))
    })
}

/// The removed Python's `int(sample_rate)` for a positive rate, refused
/// where NumPy's int64 arithmetic with it overflowed.
fn whole_vad_rate(sample_rate: &Bound<'_, PyAny>) -> PyResult<i64> {
    let rate: i64 = sample_rate
        .py()
        .get_type::<PyInt>()
        .call1((sample_rate,))?
        .extract()?;
    if rate.checked_mul(super::SILERO_WINDOW_SAMPLES).is_none() {
        return Err(PyOverflowError::new_err(format!(
            "Python integer {} out of bounds for int64",
            i128::from(rate) * i128::from(super::SILERO_WINDOW_SAMPLES)
        )));
    }
    Ok(rate)
}

#[pyfunction]
pub(crate) fn map_causal_vad_probabilities<'py>(
    py: Python<'py>,
    probabilities: Option<&Bound<'py, PyAny>>,
    positions: &Bound<'py, PyAny>,
    sample_rate: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyArray1<f32>>>> {
    catch_panic(|| {
        let Some(probabilities) = probabilities else {
            return Ok(None);
        };
        if sample_rate.le(0)? {
            return Ok(None);
        }
        let (_, probabilities) = flattened::<f64>(probabilities)?;
        let (_, positions) = flattened::<i64>(positions)?;
        let mut whole_rate = None;
        let mapped = super::map_causal_vad_probabilities(&probabilities, &positions, || {
            let rate = whole_vad_rate(sample_rate)?;
            whole_rate = Some(rate);
            Ok::<_, PyErr>(rate)
        })?;
        if whole_rate == Some(0) {
            // NumPy warned for its integer division by zero.
            PyErr::warn(
                py,
                &py.get_type::<PyRuntimeWarning>(),
                c"divide by zero encountered in floor_divide",
                1,
            )?;
        }
        Ok(mapped.map(|mapped| PyArray1::from_vec(py, mapped)))
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
        let probabilities = py
            .detach(move || {
                super::offline_vad_probabilities(
                    &samples,
                    sample_rate,
                    pre_gain,
                    |up, down| Python::attach(|py| numpy_resample_taps(py, up, down)),
                    |prepared, rate| {
                        std::panic::catch_unwind(|| {
                            crate::audio::processor::silero_offline_probabilities(
                                prepared, rate, threshold,
                            )
                        })
                        .map_err(drop)?
                        .map_err(drop)
                    },
                )
            })
            .map_err(|error| match error {
                super::VadResampleError::Design(error) => error,
                super::VadResampleError::Filter(message) => PyValueError::new_err(message),
            })?;
        Ok(match probabilities {
            Some(probabilities) => (Some(PyArray1::from_vec(py, probabilities)), "silero"),
            None => (None, FALLBACK),
        })
    })
}

fn latency_error(error: LatencyError) -> PyErr {
    match error {
        LatencyError::Value(message) => PyValueError::new_err(message),
        LatencyError::Overflow(message) => PyOverflowError::new_err(message),
        LatencyError::Index(message) => PyIndexError::new_err(message),
        LatencyError::Caller(error) => match error.downcast::<PyErr>() {
            Ok(error) => *error,
            Err(error) => PyValueError::new_err(error.to_string()),
        },
    }
}

/// `np.hanning(size)`, computed by NumPy so the probe does not rest on Rust
/// and NumPy reaching the same `cos`.
fn numpy_hanning(size: usize) -> Result<Vec<f64>, LatencyError> {
    Python::attach(|py| {
        let window = py
            .import("numpy")?
            .call_method1("hanning", (size,))?
            .cast_into::<PyArray1<f64>>()?;
        let window = window.readonly().as_slice()?.to_vec();
        Ok(window)
    })
    .map_err(|error: PyErr| LatencyError::Caller(Box::new(error)))
}

/// A latency sample rate: the 48 kHz default or what the caller passed.
pub(crate) enum LatencyRate<'py> {
    Default,
    Given(Bound<'py, PyAny>),
}

fn latency_rate_argument<'py>(value: &Bound<'py, PyAny>) -> PyResult<LatencyRate<'py>> {
    Ok(LatencyRate::Given(value.clone()))
}

impl LatencyRate<'_> {
    /// The rate as the removed Python's float arithmetic used it.
    fn to_float(&self) -> PyResult<f64> {
        match self {
            Self::Default => Ok(48_000.0),
            Self::Given(rate) => rate.extract(),
        }
    }
}

#[pyfunction]
#[pyo3(signature = (sample_rate = LatencyRate::Default, duration_ms = 80.0, amplitude = 0.8))]
pub(crate) fn generate_probe_signal<'py>(
    py: Python<'py>,
    #[pyo3(from_py_with = latency_rate_argument)] sample_rate: LatencyRate<'py>,
    duration_ms: f64,
    amplitude: f64,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    catch_panic(|| {
        let probe = super::latency::generate_probe_signal(
            sample_rate.to_float()?,
            duration_ms,
            amplitude,
            &mut numpy_hanning,
        )
        .map_err(latency_error)?;
        Ok(PyArray1::from_vec(py, probe))
    })
}

/// The fields of `LatencyCalibrationResult` as a dict.
///
/// The route and missing-input refusals come before the audio is converted,
/// and the rate is converted where it is first used, both as in the removed
/// Python.
#[pyfunction]
#[pyo3(signature = (
    reference_probe,
    recorded_signal,
    sample_rate = LatencyRate::Default,
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
    #[pyo3(from_py_with = latency_rate_argument)] sample_rate: LatencyRate<'py>,
    min_search_ms: f64,
    max_search_ms: f64,
    expected_playback_start_ms: Option<f64>,
    expected_playback_jitter_ms: Option<f64>,
    expected_latency_min_ms: Option<f64>,
    expected_latency_max_ms: Option<f64>,
    route_kind: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
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
        let analyzed = route_kind == super::latency::DEFAULT_ROUTE_KIND
            && reference_probe.is_some()
            && recorded_signal.is_some();
        let audio = |signal: Option<&Bound<'py, PyAny>>| {
            signal
                .filter(|_| analyzed)
                .map(|signal| flattened::<f64>(signal).map(|(_, values)| values))
                .transpose()
        };
        let reference = audio(reference_probe)?;
        let recording = audio(recorded_signal)?;
        let sample_rate = sample_rate.to_float();
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
                    || sample_rate.map_err(|error| LatencyError::Caller(Box::new(error))),
                    window,
                    &route_kind,
                    &mut numpy_hanning,
                )
            })
            .map_err(latency_error)?;

        let result = PyDict::new(py);
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
