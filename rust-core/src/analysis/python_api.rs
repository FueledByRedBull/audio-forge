//! PyO3 bindings for the offline analysis primitives.

use numpy::{
    Element, PyArray1, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods, PyUntypedArray,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::catch_panic;

/// `np.asarray(value, dtype=T)` flattened in C order, with its original rank.
fn flattened<T: Element + Copy>(value: &Bound<'_, PyAny>) -> PyResult<(usize, Vec<T>)> {
    let py = value.py();
    let array = py
        .import("numpy")?
        .call_method1("asarray", (value, numpy::dtype::<T>(py)))?
        .cast_into::<PyArrayDyn<T>>()?;
    let rank = array.ndim();
    let values = array.readonly().as_array().iter().copied().collect();
    Ok((rank, values))
}

fn vector<T: Element + Copy>(value: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<T>> {
    let (rank, values) = flattened(value)?;
    if rank != 1 {
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
        let (_, freqs) = flattened::<f64>(freqs)?;
        let (_, values) = flattened::<f64>(values)?;
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

/// Native Silero posteriors as float64 and `"silero"`, or `(None,
/// "energy_fallback")` when the capture, gain, threshold or model rules
/// inference out; callers then keep their energy analysis.
#[cfg(feature = "vad")]
#[pyfunction]
#[pyo3(signature = (audio, sample_rate, *, threshold = 0.48, pre_gain = 1.0))]
pub(crate) fn analyze_offline_vad<'py>(
    py: Python<'py>,
    audio: &Bound<'py, PyAny>,
    sample_rate: i64,
    threshold: f64,
    pre_gain: f64,
) -> PyResult<(Option<Bound<'py, PyArray1<f64>>>, &'static str)> {
    catch_panic(|| {
        let (_, samples) = flattened::<f32>(audio)?;
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
            None => (None, "energy_fallback"),
        })
    })
}
