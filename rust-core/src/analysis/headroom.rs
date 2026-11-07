//! The headroom rule applied to a simulated render, and the chain-settings
//! parsing the simulator's callers share.
//!
//! These replaced Python helpers and read the same Python objects the same
//! way: whatever the helpers accepted, refused or defaulted, these do.

use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};

use crate::catch_panic;

pub const HEADROOM_TARGET_DB: f64 = 1.0;
pub const LIMITER_GAIN_REDUCTION_WARN_DB: f64 = 1.0;
pub const TRUE_PEAK_GAIN_REDUCTION_WARN_DB: f64 = 0.5;
pub const NATIVE_SAFETY_METRICS: [&str; 3] = [
    "pre_limiter_true_peak_headroom_db",
    "limiter_gain_reduction_db",
    "true_peak_limiter_gain_reduction_db",
];
const NUM_EQ_BANDS: usize = 10;

/// A band's frequency, gain and Q.
type Band = (f64, f64, f64);

/// How far a render passes the headroom rule, whichever term fails most; it
/// fits at <= 0. Python's `max`: on a tie the earlier term, so a signed zero
/// comes out as it did.
pub fn headroom_excess_db(
    pre_limiter_true_peak_headroom_db: f64,
    limiter_gain_reduction_db: f64,
    true_peak_limiter_gain_reduction_db: f64,
) -> f64 {
    let mut excess = HEADROOM_TARGET_DB - pre_limiter_true_peak_headroom_db;
    for term in [
        limiter_gain_reduction_db - LIMITER_GAIN_REDUCTION_WARN_DB,
        true_peak_limiter_gain_reduction_db - TRUE_PEAK_GAIN_REDUCTION_WARN_DB,
    ] {
        if term > excess {
            excess = term;
        }
    }
    excess
}

enum Reading {
    NotNumeric,
    Value(f64),
}

/// `isinstance(value, (int, float, np.number))` for anything that is not a bool.
fn is_real_number(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    if value.is_instance_of::<PyBool>() {
        return Ok(false);
    }
    if value.is_instance_of::<PyInt>() || value.is_instance_of::<PyFloat>() {
        return Ok(true);
    }
    let number = value.py().import("numpy")?.getattr("number")?;
    value.is_instance(&number)
}

/// Python's `float(value)`.
fn python_float(value: &Bound<'_, PyAny>) -> PyResult<f64> {
    value
        .py()
        .get_type::<PyFloat>()
        .call1((value,))?
        .extract::<f64>()
}

fn read_metric(simulation: &Bound<'_, PyAny>, key: &str) -> PyResult<Reading> {
    let value = simulation.call_method1("get", (key,))?;
    if !is_real_number(&value)? {
        return Ok(Reading::NotNumeric);
    }
    Ok(Reading::Value(python_float(&value)?))
}

fn simulation_excess_db(simulation: &Bound<'_, PyAny>) -> PyResult<f64> {
    let mut metrics = [0.0; 3];
    for (slot, key) in metrics.iter_mut().zip(NATIVE_SAFETY_METRICS) {
        match read_metric(simulation, key)? {
            Reading::Value(value) if value.is_finite() => *slot = value,
            _ => return Ok(f64::INFINITY),
        }
    }
    Ok(headroom_excess_db(metrics[0], metrics[1], metrics[2]))
}

/// How far a simulation passes the headroom rule; a reading that is missing
/// or not a finite number makes it infinite.
#[pyfunction(name = "headroom_excess_db")]
pub(crate) fn py_headroom_excess_db(simulation: &Bound<'_, PyAny>) -> PyResult<f64> {
    catch_panic(|| simulation_excess_db(simulation))
}

#[pyfunction]
pub(crate) fn is_headroom_safe(simulation: &Bound<'_, PyAny>) -> PyResult<bool> {
    catch_panic(|| Ok(simulation_excess_db(simulation)? <= 0.0))
}

/// What is wrong with the safety metrics of a simulator result, one message per metric.
#[pyfunction]
pub(crate) fn safety_metric_problems(simulation: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    catch_panic(|| {
        let mut problems = Vec::new();
        for key in NATIVE_SAFETY_METRICS {
            match read_metric(simulation, key)? {
                Reading::NotNumeric => problems.push(format!("{key} is missing or non-numeric")),
                Reading::Value(value) if !value.is_finite() => {
                    problems.push(format!("{key} is not finite"));
                }
                Reading::Value(_) => {}
            }
        }
        Ok(problems)
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("HEADROOM_TARGET_DB", HEADROOM_TARGET_DB)?;
    m.add(
        "LIMITER_GAIN_REDUCTION_WARN_DB",
        LIMITER_GAIN_REDUCTION_WARN_DB,
    )?;
    m.add(
        "TRUE_PEAK_GAIN_REDUCTION_WARN_DB",
        TRUE_PEAK_GAIN_REDUCTION_WARN_DB,
    )?;
    m.add(
        "NATIVE_SAFETY_METRICS",
        PyTuple::new(m.py(), NATIVE_SAFETY_METRICS)?,
    )?;
    m.add_function(wrap_pyfunction!(py_headroom_excess_db, m)?)?;
    m.add_function(wrap_pyfunction!(is_headroom_safe, m)?)?;
    m.add_function(wrap_pyfunction!(safety_metric_problems, m)?)?;
    m.add_function(wrap_pyfunction!(flatten_chain_settings, m)?)?;
    m.add_function(wrap_pyfunction!(legacy_eq_inputs, m)?)?;
    Ok(())
}

fn not_a(what: &str, value: &Bound<'_, PyAny>) -> PyResult<PyErr> {
    Ok(PyValueError::new_err(format!(
        "chain setting is not {what}: {}",
        value.repr()?
    )))
}

/// A refusal of `float(value)` or `int(value)` for this value; any other error is real.
fn is_conversion_refusal(error: &PyErr, py: Python<'_>) -> bool {
    error.is_instance_of::<PyTypeError>(py)
        || error.is_instance_of::<PyValueError>(py)
        || error.is_instance_of::<PyOverflowError>(py)
}

fn finite_float(value: &Bound<'_, PyAny>) -> PyResult<f64> {
    if !value.is_instance_of::<PyBool>() {
        match python_float(value) {
            Ok(parsed) if parsed.is_finite() => return Ok(parsed),
            Ok(_) => {}
            Err(error) if is_conversion_refusal(&error, value.py()) => {}
            Err(error) => return Err(error),
        }
    }
    Err(not_a("a finite number", value)?)
}

/// `default` for a setting that is absent; a value that is no finite number is an error.
fn as_float(value: &Bound<'_, PyAny>, default: f64) -> PyResult<f64> {
    if value.is_none() {
        Ok(default)
    } else {
        finite_float(value)
    }
}

fn as_bool(value: &Bound<'_, PyAny>, default: bool) -> PyResult<bool> {
    if value.is_none() {
        return Ok(default);
    }
    let numpy_bool = value.py().import("numpy")?.getattr("bool_")?;
    if value.is_instance_of::<PyBool>() || value.is_instance(&numpy_bool)? {
        return value.is_truthy();
    }
    Err(not_a("a boolean", value)?)
}

fn as_int<'py>(value: &Bound<'py, PyAny>, default: i64) -> PyResult<Bound<'py, PyAny>> {
    let py = value.py();
    if value.is_none() {
        return Ok(default.into_pyobject(py)?.into_any());
    }
    if !value.is_instance_of::<PyBool>() {
        match py.get_type::<PyInt>().call1((value,)) {
            Ok(parsed) => return Ok(parsed),
            Err(error) if is_conversion_refusal(&error, py) => {}
            Err(error) => return Err(error),
        }
    }
    Err(not_a("an integer", value)?)
}

fn get<'py>(section: &Bound<'py, PyAny>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    section.call_method1("get", (key,))
}

/// Python's `a.get(k1) or a.get(k2) or ... or {}`.
fn first_truthy<'py>(chain: &Bound<'py, PyAny>, keys: &[&str]) -> PyResult<Bound<'py, PyAny>> {
    for key in keys {
        let section = get(chain, key)?;
        if section.is_truthy()? {
            return Ok(section);
        }
    }
    Ok(PyDict::new(chain.py()).into_any())
}

/// The flat settings the chain simulator takes, from the nested ones the
/// presets and dialogs hold. An absent setting takes its default; one that is
/// no finite number, boolean or integer is an error rather than a default,
/// since the default would be simulated and certified as the candidate's.
/// Settings the simulator already takes flat override what the nested ones gave.
#[pyfunction]
#[pyo3(signature = (chain_settings=None))]
pub(crate) fn flatten_chain_settings<'py>(
    py: Python<'py>,
    chain_settings: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    catch_panic(|| {
        let chain = match chain_settings {
            Some(chain) if chain.is_truthy()? => chain.clone(),
            _ => PyDict::new(py).into_any(),
        };
        let deesser = first_truthy(&chain, &["deesser"])?;
        let compressor = first_truthy(&chain, &["compressor"])?;
        let limiter = first_truthy(&chain, &["limiter"])?;
        let gate = first_truthy(&chain, &["gate", "gate_settings"])?;
        let suppressor = first_truthy(
            &chain,
            &[
                "suppressor",
                "suppression",
                "rnnoise",
                "suppressor_settings",
            ],
        )?;

        let flattened = PyDict::new(py);
        flattened.set_item(
            "return_output_audio",
            as_bool(&get(&chain, "return_output_audio")?, false)?,
        )?;
        let section_bool = |section: &Bound<'py, PyAny>, key: &str, default: bool| {
            as_bool(&get(section, key)?, default)
        };
        let section_float = |section: &Bound<'py, PyAny>, key: &str, default: f64| {
            as_float(&get(section, key)?, default)
        };

        flattened.set_item("deesser_enabled", section_bool(&deesser, "enabled", false)?)?;
        flattened.set_item(
            "deesser_auto_enabled",
            section_bool(&deesser, "auto_enabled", true)?,
        )?;
        for (name, key, default) in [
            ("deesser_auto_amount", "auto_amount", 0.5),
            ("deesser_low_cut_hz", "low_cut_hz", 4000.0),
            ("deesser_high_cut_hz", "high_cut_hz", 11000.0),
            ("deesser_threshold_db", "threshold_db", -28.0),
            ("deesser_ratio", "ratio", 4.0),
            ("deesser_attack_ms", "attack_ms", 2.0),
            ("deesser_release_ms", "release_ms", 80.0),
            ("deesser_max_reduction_db", "max_reduction_db", 6.0),
        ] {
            flattened.set_item(name, section_float(&deesser, key, default)?)?;
        }

        flattened.set_item(
            "compressor_enabled",
            section_bool(&compressor, "enabled", true)?,
        )?;
        for (name, key, default) in [
            ("compressor_threshold_db", "threshold_db", -20.0),
            ("compressor_ratio", "ratio", 4.0),
            ("compressor_attack_ms", "attack_ms", 10.0),
            ("compressor_release_ms", "release_ms", 200.0),
            ("compressor_makeup_gain_db", "makeup_gain_db", 0.0),
        ] {
            flattened.set_item(name, section_float(&compressor, key, default)?)?;
        }
        flattened.set_item(
            "compressor_adaptive_release",
            section_bool(&compressor, "adaptive_release", false)?,
        )?;
        flattened.set_item(
            "compressor_base_release_ms",
            section_float(&compressor, "base_release_ms", 50.0)?,
        )?;
        flattened.set_item(
            "compressor_auto_makeup_enabled",
            section_bool(&compressor, "auto_makeup_enabled", false)?,
        )?;
        flattened.set_item(
            "compressor_target_lufs",
            section_float(&compressor, "target_lufs", -18.0)?,
        )?;
        flattened.set_item(
            "compressor_noise_reference_reliability",
            section_float(&compressor, "noise_reference_reliability", 0.0)?,
        )?;
        flattened.set_item(
            "compressor_sidechain_highpass_enabled",
            section_bool(&compressor, "sidechain_highpass_enabled", true)?,
        )?;

        flattened.set_item("limiter_enabled", section_bool(&limiter, "enabled", true)?)?;
        flattened.set_item(
            "limiter_ceiling_db",
            section_float(&limiter, "ceiling_db", -0.5)?,
        )?;
        flattened.set_item(
            "limiter_release_ms",
            section_float(&limiter, "release_ms", 50.0)?,
        )?;
        flattened.set_item(
            "limiter_careful_output_enabled",
            section_bool(&limiter, "careful_output_enabled", true)?,
        )?;

        let mapping = py.import("collections.abc")?.getattr("Mapping")?;
        if gate.is_instance(&mapping)? {
            flattened.set_item("gate_enabled", section_bool(&gate, "enabled", true)?)?;
            flattened.set_item(
                "gate_threshold_db",
                section_float(&gate, "threshold_db", -40.0)?,
            )?;
            flattened.set_item("gate_attack_ms", section_float(&gate, "attack_ms", 10.0)?)?;
            flattened.set_item(
                "gate_release_ms",
                section_float(&gate, "release_ms", 100.0)?,
            )?;
            flattened.set_item("gate_mode", as_int(&get(&gate, "gate_mode")?, 0)?)?;
            flattened.set_item(
                "gate_vad_threshold",
                section_float(&gate, "vad_threshold", 0.48)?,
            )?;
            flattened.set_item(
                "gate_vad_hold_time_ms",
                section_float(&gate, "vad_hold_time_ms", 200.0)?,
            )?;
            flattened.set_item(
                "gate_vad_pre_gain",
                section_float(&gate, "vad_pre_gain", 1.0)?,
            )?;
            flattened.set_item(
                "gate_auto_threshold_enabled",
                section_bool(&gate, "auto_threshold_enabled", true)?,
            )?;
            flattened.set_item(
                "gate_margin_db",
                section_float(&gate, "gate_margin_db", 10.0)?,
            )?;
        }
        if suppressor.is_instance(&mapping)? {
            flattened.set_item(
                "suppressor_enabled",
                section_bool(&suppressor, "enabled", true)?,
            )?;
            flattened.set_item(
                "suppressor_strength",
                section_float(&suppressor, "strength", 1.0)?,
            )?;
            flattened.set_item(
                "noise_model",
                suppressor
                    .call_method1("get", ("model", "rnnoise"))?
                    .str()?,
            )?;
        }

        // Explicit native-boundary values override defaults derived from missing
        // nested settings. Accept only the controls this adapter already knows.
        let deepcopy = py.import("copy")?.getattr("deepcopy")?;
        let known: Vec<Bound<'py, PyAny>> = flattened.keys().iter().collect();
        let passthrough = [
            "full_chain",
            "input_pre_filtered",
            "input_cleanup_mode",
            "processing_mode",
            "vad_probabilities",
            "vad_available",
            "auto_makeup_activity",
            "return_auto_makeup_activity",
        ]
        .map(|key| PyString::new(py, key).into_any());
        for key in known.into_iter().chain(passthrough) {
            if chain.contains(&key)? {
                flattened.set_item(&key, deepcopy.call1((chain.get_item(&key)?,))?)?;
            }
        }
        Ok(flattened)
    })
}

fn legacy_list<'py>(eq_settings: &Bound<'py, PyAny>, key: &str) -> PyResult<Bound<'py, PyList>> {
    let values = get(eq_settings, key)?;
    let values = if values.is_truthy()? {
        values
    } else {
        PyList::empty(eq_settings.py()).into_any()
    };
    Ok(eq_settings
        .py()
        .get_type::<PyList>()
        .call1((values,))?
        .cast_into::<PyList>()?)
}

/// The bands and the enabled flag of an Auto-EQ result that carries the
/// legacy `band_freqs`, `band_gains` and `band_qs` arrays.
#[pyfunction]
pub(crate) fn legacy_eq_inputs(eq_settings: &Bound<'_, PyAny>) -> PyResult<(Vec<Band>, bool)> {
    catch_panic(|| {
        let freqs = legacy_list(eq_settings, "band_freqs")?;
        let gains = legacy_list(eq_settings, "band_gains")?;
        let qs = legacy_list(eq_settings, "band_qs")?;
        if !(freqs.len() == gains.len() && gains.len() == qs.len() && qs.len() == NUM_EQ_BANDS) {
            return Err(PyValueError::new_err(
                "Auto-EQ settings must contain 10 frequencies, gains, and Q values",
            ));
        }
        let mut bands = Vec::with_capacity(NUM_EQ_BANDS);
        for ((freq, gain), q) in freqs.iter().zip(gains.iter()).zip(qs.iter()) {
            bands.push((
                finite_float(&freq)?,
                finite_float(&gain)?,
                finite_float(&q)?,
            ));
        }
        let fallback = get(eq_settings, "eq_enabled")?;
        let enabled = eq_settings.call_method1("get", ("enabled", fallback))?;
        Ok((bands, as_bool(&enabled, true)?))
    })
}

#[cfg(test)]
mod tests;
