#![cfg_attr(test, allow(clippy::float_cmp))]

//! AudioForge core - high-performance DSP for real-time audio processing
//!
//! Processing chain: input cleanup -> noise gate -> noise suppression ->
//! de-esser -> 10-band EQ -> compressor -> sample/true-peak limiter -> output.

use pyo3::prelude::*;
use pyo3::types::PyModule;
use std::time::Instant;

pub mod audio;
pub mod dsp;

#[cfg(test)]
pub(crate) mod test_alloc {
    // This measures Rust global allocations on the calling thread. Allocations
    // performed inside foreign libraries are outside this instrumentation.
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;
    use std::sync::Barrier;
    use std::thread;

    pub struct CountingAllocator;

    thread_local! {
        static COUNTING_ALLOCATIONS: Cell<bool> = const { Cell::new(false) };
        static ALLOCATION_COUNT: Cell<usize> = const { Cell::new(0) };
    }

    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            count_allocation();
            System.alloc(layout)
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            count_allocation();
            System.alloc_zeroed(layout)
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            count_allocation();
            System.realloc(ptr, layout, new_size)
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            System.dealloc(ptr, layout);
        }
    }

    #[global_allocator]
    static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

    fn count_allocation() {
        COUNTING_ALLOCATIONS.with(|enabled| {
            if enabled.get() {
                ALLOCATION_COUNT.with(|count| count.set(count.get().saturating_add(1)));
            }
        });
    }

    struct AllocationScope {
        previous: bool,
        previous_count: usize,
    }

    impl AllocationScope {
        fn enter() -> Self {
            let previous = COUNTING_ALLOCATIONS.with(|enabled| {
                let previous = enabled.get();
                enabled.set(true);
                previous
            });
            let previous_count = ALLOCATION_COUNT.with(|count| {
                let previous_count = count.get();
                count.set(0);
                previous_count
            });
            Self {
                previous,
                previous_count,
            }
        }
    }

    impl Drop for AllocationScope {
        fn drop(&mut self) {
            let nested_count = ALLOCATION_COUNT.with(Cell::get);
            COUNTING_ALLOCATIONS.with(|enabled| enabled.set(self.previous));
            ALLOCATION_COUNT.with(|count| {
                count.set(self.previous_count.saturating_add(nested_count));
            });
        }
    }

    pub fn allocation_count_during(function: impl FnOnce()) -> usize {
        let _scope = AllocationScope::enter();
        function();
        ALLOCATION_COUNT.with(Cell::get)
    }

    pub fn assert_no_allocations(label: &str, function: impl FnOnce()) {
        let allocations = allocation_count_during(function);
        assert_eq!(allocations, 0, "{label} allocated {allocations} time(s)");
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn concurrent_scope_cannot_reset_another_thread() {
            let allocated = std::sync::Arc::new(Barrier::new(2));
            let release = std::sync::Arc::new(Barrier::new(2));
            let thread_allocated = allocated.clone();
            let thread_release = release.clone();
            let handle = thread::spawn(move || {
                let _scope = AllocationScope::enter();
                std::hint::black_box(vec![0_u8; 17]);
                thread_allocated.wait();
                thread_release.wait();
                ALLOCATION_COUNT.with(Cell::get)
            });

            allocated.wait();
            let _other_scope = AllocationScope::enter();
            release.wait();
            assert!(handle.join().unwrap() > 0);
        }

        #[test]
        fn nonallocating_scope_ignores_concurrent_thread_allocations() {
            let start = std::sync::Arc::new(Barrier::new(2));
            let finished = std::sync::Arc::new(Barrier::new(2));
            let thread_start = start.clone();
            let thread_finished = finished.clone();
            let handle = thread::spawn(move || {
                thread_start.wait();
                let count = allocation_count_during(|| {
                    std::hint::black_box(vec![0_u8; 23]);
                });
                thread_finished.wait();
                count
            });

            let main_count = allocation_count_during(|| {
                start.wait();
                finished.wait();
            });

            assert_eq!(main_count, 0);
            assert!(handle.join().unwrap() > 0);
        }

        #[test]
        fn nested_scope_count_isolated_and_included_by_outer_scope() {
            let outer = allocation_count_during(|| {
                let inner = allocation_count_during(|| {
                    std::hint::black_box(vec![0_u8; 11]);
                });
                assert!(inner > 0);
                std::hint::black_box(vec![0_u8; 13]);
            });

            assert!(outer >= 2);
        }
    }
}

// Re-export main types
pub use audio::{AudioProcessor, PyAudioProcessor};
pub use dsp::{Biquad, Compressor, DeEsser, Limiter, NoiseGate, ParametricEQ, RNNoiseProcessor};

#[pyfunction]
fn eq_magnitude_response(
    frequencies_hz: Vec<f64>,
    bands: Vec<(f64, f64, f64)>,
    sample_rate: f64,
) -> PyResult<Vec<f64>> {
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "sample_rate must be finite and positive",
        ));
    }
    if bands.len() != dsp::eq::NUM_BANDS {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "expected {} EQ bands, got {}",
            dsp::eq::NUM_BANDS,
            bands.len()
        )));
    }
    let nyquist = sample_rate / 2.0;
    for (index, (frequency_hz, gain_db, q)) in bands.iter().copied().enumerate() {
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 || frequency_hz >= nyquist {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "band {index} frequency must be between 0 Hz and Nyquist"
            )));
        }
        if !gain_db.is_finite() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "band {index} gain must be finite"
            )));
        }
        if !q.is_finite() || q <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "band {index} Q must be finite and positive"
            )));
        }
    }
    if frequencies_hz.iter().any(|frequency_hz| {
        !frequency_hz.is_finite() || *frequency_hz < 0.0 || *frequency_hz > nyquist
    }) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "response frequencies must be finite and between 0 Hz and Nyquist",
        ));
    }

    let mut eq = ParametricEQ::new(sample_rate);
    for (index, (frequency_hz, gain_db, q)) in bands.into_iter().enumerate() {
        eq.set_band_frequency(index, frequency_hz);
        eq.set_band_gain(index, gain_db);
        eq.set_band_q(index, q);
    }
    Ok(eq.magnitude_response_db(&frequencies_hz))
}

type PyEqBandV2 = (String, f64, f64, f64, u8, bool);

fn parse_eq_v2_bands(bands: &[PyEqBandV2], sample_rate: f64) -> PyResult<Vec<dsp::EqBandConfig>> {
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "sample_rate must be finite and positive",
        ));
    }
    if bands.len() != dsp::eq::NUM_BANDS {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "expected {} EQ bands, got {}",
            dsp::eq::NUM_BANDS,
            bands.len()
        )));
    }
    let mut configs = Vec::with_capacity(dsp::eq::NUM_BANDS);
    for (index, (filter_type, frequency_hz, gain_db, q, slope, enabled)) in bands.iter().enumerate()
    {
        let filter_type = dsp::EqFilterType::from_name(filter_type).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "band {index} has unsupported EQ filter type: {filter_type}"
            ))
        })?;
        let config = dsp::EqBandConfig {
            filter_type,
            frequency_hz: *frequency_hz,
            gain_db: *gain_db,
            q: *q,
            slope_db_per_octave: *slope,
            enabled: *enabled,
        };
        config
            .validate(index, sample_rate)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        configs.push(config);
    }
    Ok(configs)
}

#[pyfunction]
fn eq_magnitude_response_v2(
    frequencies_hz: Vec<f64>,
    bands: Vec<PyEqBandV2>,
    sample_rate: f64,
) -> PyResult<Vec<f64>> {
    let configs = parse_eq_v2_bands(&bands, sample_rate)?;
    let nyquist = sample_rate / 2.0;
    if frequencies_hz.iter().any(|frequency_hz| {
        !frequency_hz.is_finite() || *frequency_hz < 0.0 || *frequency_hz > nyquist
    }) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "response frequencies must be finite and between 0 Hz and Nyquist",
        ));
    }

    let mut eq = ParametricEQ::new(sample_rate);
    for (index, config) in configs.into_iter().enumerate() {
        eq.set_band_config(index, config);
    }
    Ok(eq.magnitude_response_db(&frequencies_hz))
}

#[pyfunction]
#[pyo3(signature = (audio, sample_rate, bands, return_output_audio=false))]
fn simulate_eq_v2(
    py: Python<'_>,
    audio: numpy::PyReadonlyArray1<'_, f32>,
    sample_rate: f64,
    bands: Vec<PyEqBandV2>,
    return_output_audio: bool,
) -> PyResult<Py<PyAny>> {
    if !sample_rate.is_finite() || !(8_000.0..=768_000.0).contains(&sample_rate) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "sample_rate must be finite and between 8000 and 768000 Hz",
        ));
    }
    let configs = parse_eq_v2_bands(&bands, sample_rate)?;
    let input = audio.as_slice()?.to_vec();
    if input.iter().any(|sample| !sample.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "audio must contain only finite samples",
        ));
    }
    let sample_count = input.len();

    let (
        output,
        input_peak,
        output_peak,
        input_true_peak,
        output_true_peak,
        input_rms,
        output_rms,
        max_response_db,
        runtime_ms,
        non_finite_output,
    ) = py.detach(move || {
        let mut eq = ParametricEQ::new(sample_rate);
        for (index, config) in configs.into_iter().enumerate() {
            eq.set_band_config(index, config);
        }
        eq.reset();

        let mut output = input.clone();
        let started = Instant::now();
        eq.process_block_inplace(&mut output);
        let runtime_ms = started.elapsed().as_secs_f64() * 1000.0;

        let input_square_sum = input
            .iter()
            .map(|sample| f64::from(*sample) * f64::from(*sample))
            .sum::<f64>();
        let output_square_sum = output
            .iter()
            .map(|sample| f64::from(*sample) * f64::from(*sample))
            .sum::<f64>();
        let divisor = input.len().max(1) as f64;
        let input_peak = input
            .iter()
            .fold(0.0_f32, |peak, sample| peak.max(sample.abs()));
        let output_peak = output
            .iter()
            .fold(0.0_f32, |peak, sample| peak.max(sample.abs()));
        let mut input_true_peak_detector = dsp::TruePeakDetector::new();
        let mut output_true_peak_detector = dsp::TruePeakDetector::new();
        let input_true_peak = input_true_peak_detector.process_block(&input);
        let output_true_peak = output_true_peak_detector.process_block(&output);
        let response_frequencies = (0..512)
            .map(|index| 20.0 * (20_000.0_f64 / 20.0).powf(index as f64 / 511.0))
            .collect::<Vec<_>>();
        let max_response_db = eq
            .magnitude_response_db(&response_frequencies)
            .into_iter()
            .fold(f64::NEG_INFINITY, f64::max);
        let input_rms = (input_square_sum / divisor).sqrt();
        let output_rms = (output_square_sum / divisor).sqrt();
        let non_finite_output = output.iter().any(|sample| !sample.is_finite());

        (
            output,
            input_peak,
            output_peak,
            input_true_peak,
            output_true_peak,
            input_rms,
            output_rms,
            max_response_db,
            runtime_ms,
            non_finite_output,
        )
    });

    let diagnostics = pyo3::types::PyDict::new(py);
    diagnostics.set_item("input_sample_peak", input_peak)?;
    diagnostics.set_item("output_sample_peak", output_peak)?;
    diagnostics.set_item("input_true_peak", input_true_peak)?;
    diagnostics.set_item("output_true_peak", output_true_peak)?;
    diagnostics.set_item("input_rms", input_rms)?;
    diagnostics.set_item("output_rms", output_rms)?;
    diagnostics.set_item("max_response_db", max_response_db)?;
    diagnostics.set_item("runtime_ms", runtime_ms)?;
    diagnostics.set_item("sample_count", sample_count)?;
    diagnostics.set_item("algorithmic_latency_samples", 0)?;
    diagnostics.set_item("non_finite_output", non_finite_output)?;
    if return_output_audio {
        diagnostics.set_item("output_audio", output)?;
    }
    Ok(diagnostics.into_any().unbind())
}

#[pyfunction]
fn measure_integrated_loudness(
    py: Python<'_>,
    audio: numpy::PyReadonlyArray1<'_, f32>,
    sample_rate: u32,
) -> PyResult<f64> {
    let samples = audio.as_slice()?.to_vec();
    py.detach(move || dsp::loudness::integrated_loudness_lufs(&samples, sample_rate))
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
}

/// Python module initialization
#[pymodule]
fn mic_eq_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Main audio processor
    m.add_class::<audio::PyAudioProcessor>()?;

    // VAD Gate Mode enum (VAD feature only)
    #[cfg(feature = "vad")]
    m.add_class::<audio::PyGateMode>()?;

    // Device enumeration
    m.add_class::<audio::DeviceInfo>()?;
    m.add_function(wrap_pyfunction!(audio::list_input_devices, m)?)?;
    m.add_function(wrap_pyfunction!(audio::list_output_devices, m)?)?;
    m.add_function(wrap_pyfunction!(
        audio::processor::simulate_auto_eq_chain,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        audio::processor::simulate_auto_makeup_control,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        audio::processor::simulate_product_resampler_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        audio::processor::product_resampler_configuration,
        m
    )?)?;
    #[cfg(feature = "vad")]
    m.add_function(wrap_pyfunction!(
        audio::processor::simulate_gate_suppressor_order,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(eq_magnitude_response, m)?)?;
    m.add_function(wrap_pyfunction!(eq_magnitude_response_v2, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_eq_v2, m)?)?;
    m.add_function(wrap_pyfunction!(measure_integrated_loudness, m)?)?;

    #[cfg(feature = "vad")]
    m.add_function(wrap_pyfunction!(
        audio::processor::analyze_vad_probabilities,
        m
    )?)?;

    #[cfg(feature = "deepfilter")]
    m.add_function(wrap_pyfunction!(configure_deepfilter_runtime_paths, m)?)?;

    Ok(())
}

#[cfg(feature = "deepfilter")]
#[pyfunction]
#[pyo3(signature = (library_path=None, model_path=None))]
fn configure_deepfilter_runtime_paths(
    library_path: Option<&str>,
    model_path: Option<&str>,
) -> PyResult<()> {
    dsp::deepfilter_ffi::configure_app_owned_paths(library_path, model_path)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}
