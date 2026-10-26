use super::{
    PROCESS_IDLE_MAX_SLEEP_US, PROCESS_IDLE_RECENT_INPUT_WINDOW_US, PROCESS_IDLE_SLEEP_US,
    PRODUCT_RESAMPLER_SINC_LEN, PRODUCT_RESAMPLER_WINDOW_NAME, RESAMPLER_CHUNK_SIZE,
};
use crate::audio::rt::{store_rt_error, FixedAudioBuffer, RtErrorCode};
use pyo3::prelude::*;
use rubato::{
    calculate_cutoff, Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::Instant;
#[cfg(not(windows))]
use thread_priority::{set_current_thread_priority, ThreadPriority};

pub(super) fn duration_samples(sample_rate: u32, duration_ms: u32) -> usize {
    (((sample_rate as u64) * duration_ms as u64 + 500) / 1000).max(1) as usize
}

pub(super) fn samples_to_micros(samples: u64, sample_rate: u32) -> u64 {
    if sample_rate == 0 {
        0
    } else {
        samples.saturating_mul(1_000_000) / sample_rate as u64
    }
}

pub(super) fn smoothing_coeff_for_time_constant(sample_rate_hz: f32, time_constant_ms: f32) -> f32 {
    if !sample_rate_hz.is_finite()
        || sample_rate_hz <= 0.0
        || !time_constant_ms.is_finite()
        || time_constant_ms <= 0.0
    {
        0.0
    } else {
        (-1.0 / (sample_rate_hz * (time_constant_ms / 1000.0))).exp()
    }
}

pub(super) fn smoothing_coeff_for_block(
    sample_rate_hz: f32,
    time_constant_ms: f32,
    block_samples: usize,
) -> f32 {
    if block_samples == 0 {
        return 1.0;
    }
    if !sample_rate_hz.is_finite()
        || sample_rate_hz <= 0.0
        || !time_constant_ms.is_finite()
        || time_constant_ms <= 0.0
    {
        return 0.0;
    }

    (-(block_samples as f32) / (sample_rate_hz * (time_constant_ms / 1000.0))).exp()
}

pub(super) fn clear_resampler_output_samples(output: &mut [Vec<f64>]) {
    for channel in output {
        channel.fill(0.0);
    }
}

pub(super) fn signal_input_resampler_failure(
    rt_error_code: &AtomicU32,
    restart_requested: &AtomicBool,
) {
    store_rt_error(rt_error_code, RtErrorCode::InputResamplerFailed);
    restart_requested.store(true, Ordering::Release);
}

pub(super) fn next_process_idle_sleep_us(
    consecutive_idle_wakeups: u32,
    input_callback_age_us: u64,
) -> u64 {
    if input_callback_age_us <= PROCESS_IDLE_RECENT_INPUT_WINDOW_US {
        return PROCESS_IDLE_SLEEP_US;
    }

    let backoff_shift = consecutive_idle_wakeups.min(4);
    PROCESS_IDLE_SLEEP_US
        .saturating_mul(1_u64 << backoff_shift)
        .min(PROCESS_IDLE_MAX_SLEEP_US)
}

#[derive(Clone, Copy)]
pub(super) struct LatencyComponents {
    pub(super) input_resampler_delay_samples: u64,
    pub(super) output_buffer_samples: u64,
    pub(super) output_sample_rate: u32,
    pub(super) output_resampler_delay_samples: u64,
    pub(super) suppressor_latency_samples: u64,
    pub(super) limiter_lookahead_samples: u64,
    pub(super) true_peak_lookahead_samples: u64,
    pub(super) processing_sample_rate: u32,
}

pub(super) fn total_reported_latency_us(
    components: LatencyComponents,
    compensation_us: u64,
) -> u64 {
    let input_resampler_latency_us = samples_to_micros(
        components.input_resampler_delay_samples,
        components.processing_sample_rate,
    );
    let output_latency_us = samples_to_micros(
        components.output_buffer_samples,
        components.output_sample_rate,
    );
    let output_resampler_latency_us = samples_to_micros(
        components.output_resampler_delay_samples,
        components.output_sample_rate,
    );
    let suppressor_latency_us = samples_to_micros(
        components.suppressor_latency_samples,
        components.processing_sample_rate,
    );
    // Both limiter delays stay in the path while limiting is disabled.
    let limiter_latency_us = samples_to_micros(
        components.limiter_lookahead_samples,
        components.processing_sample_rate,
    )
    .saturating_add(samples_to_micros(
        components.true_peak_lookahead_samples,
        components.output_sample_rate,
    ));

    output_latency_us
        .saturating_add(input_resampler_latency_us)
        .saturating_add(output_resampler_latency_us)
        .saturating_add(suppressor_latency_us)
        .saturating_add(limiter_latency_us)
        .saturating_add(compensation_us)
}

#[derive(Default)]
pub(super) struct DriftRetimer {
    next_source_position: f64,
    previous_sample: Option<f32>,
}

impl DriftRetimer {
    pub(super) fn reset(&mut self) {
        *self = Self::default();
    }
}

pub(super) fn retime_audio_block<'a, const N: usize>(
    input: &[f32],
    speed_ratio: f32,
    max_output_len: usize,
    output: &'a mut FixedAudioBuffer<f32, N>,
    state: &mut DriftRetimer,
) -> &'a [f32] {
    if input.is_empty() || max_output_len == 0 {
        output.clear();
        return output.as_slice();
    }

    let speed_ratio = speed_ratio.max(0.5) as f64;
    let output_limit = max_output_len.min(output.capacity());
    output.clear();
    loop {
        let source_position = state.next_source_position;
        let sample = if source_position < 0.0 {
            let previous = state.previous_sample.unwrap_or(input[0]);
            let fraction = (source_position + 1.0).clamp(0.0, 1.0) as f32;
            previous + (input[0] - previous) * fraction
        } else {
            let index = source_position.floor() as usize;
            if index >= input.len() {
                break;
            }
            let fraction = (source_position - index as f64) as f32;
            if fraction <= f32::EPSILON {
                input[index]
            } else if index + 1 < input.len() {
                input[index] + (input[index + 1] - input[index]) * fraction
            } else {
                break;
            }
        };

        if output.len() < output_limit {
            let _ = output.push(sample);
        }
        state.next_source_position += speed_ratio;
    }

    state.next_source_position -= input.len() as f64;
    state.previous_sample = input.last().copied();

    output.as_slice()
}

pub(super) fn build_sinc_resampler(
    input_rate: u32,
    output_rate: u32,
    chunk_size: usize,
) -> Result<SincFixedIn<f64>, String> {
    build_sinc_resampler_with_quality(
        input_rate,
        output_rate,
        chunk_size,
        PRODUCT_RESAMPLER_SINC_LEN,
        product_resampler_window(),
    )
}

fn product_resampler_window() -> WindowFunction {
    WindowFunction::Blackman
}

fn build_sinc_resampler_with_quality(
    input_rate: u32,
    output_rate: u32,
    chunk_size: usize,
    sinc_len: usize,
    window: WindowFunction,
) -> Result<SincFixedIn<f64>, String> {
    let ratio = output_rate as f64 / input_rate as f64;
    let params = SincInterpolationParameters {
        sinc_len,
        f_cutoff: calculate_cutoff(sinc_len, window),
        interpolation: SincInterpolationType::Cubic,
        oversampling_factor: 256,
        window,
    };
    SincFixedIn::<f64>::new(ratio, 1.2, params, chunk_size, 1).map_err(|e| e.to_string())
}

fn resampler_window_from_name(name: &str) -> Option<WindowFunction> {
    match name {
        "blackman_harris" => Some(WindowFunction::BlackmanHarris),
        "blackman_harris_squared" => Some(WindowFunction::BlackmanHarris2),
        "blackman" => Some(WindowFunction::Blackman),
        "blackman_squared" => Some(WindowFunction::Blackman2),
        "hann" => Some(WindowFunction::Hann),
        "hann_squared" => Some(WindowFunction::Hann2),
        _ => None,
    }
}

pub fn simulate_product_resampler(
    samples: Vec<f64>,
    input_rate: u32,
    output_rate: u32,
    chunk_size: usize,
    sinc_len: Option<usize>,
    window: Option<&str>,
) -> PyResult<(Vec<f64>, usize, usize, Vec<u64>)> {
    // Bounds the rate ratio (at most 96x), which sizes the output buffers.
    crate::validate_sample_rate(f64::from(input_rate))?;
    crate::validate_sample_rate(f64::from(output_rate))?;
    if !(1..=RESAMPLER_CHUNK_SIZE).contains(&chunk_size) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "chunk_size must be between 1 and {RESAMPLER_CHUNK_SIZE}"
        )));
    }
    let sinc_len = sinc_len.unwrap_or(PRODUCT_RESAMPLER_SINC_LEN);
    if !(32..=2048).contains(&sinc_len) || !sinc_len.is_power_of_two() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "sinc_len must be a power of two between 32 and 2048",
        ));
    }
    let window = match window {
        Some(name) => resampler_window_from_name(name).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "unsupported resampler window {name:?}"
            ))
        })?,
        None => product_resampler_window(),
    };
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "samples must be finite",
        ));
    }

    let mut resampler =
        build_sinc_resampler_with_quality(input_rate, output_rate, chunk_size, sinc_len, window)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    let delay = resampler.output_delay();
    let expected_frames =
        ((samples.len() as f64 * output_rate as f64) / input_rate as f64).round() as usize;
    let mut output_buffer = resampler.output_buffer_allocate(true);
    let mut output = Vec::with_capacity(expected_frames.saturating_add(delay));
    let mut block_times_ns = Vec::new();
    let mut remaining = samples.as_slice();

    while remaining.len() >= resampler.input_frames_next() {
        let started = Instant::now();
        let (consumed, produced) = resampler
            .process_into_buffer(&[remaining], &mut output_buffer, None)
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
        block_times_ns.push(started.elapsed().as_nanos().min(u64::MAX as u128) as u64);
        output.extend_from_slice(&output_buffer[0][..produced]);
        remaining = &remaining[consumed..];
    }
    if !remaining.is_empty() {
        let started = Instant::now();
        let (_consumed, produced) = resampler
            .process_partial_into_buffer(Some(&[remaining]), &mut output_buffer, None)
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
        block_times_ns.push(started.elapsed().as_nanos().min(u64::MAX as u128) as u64);
        output.extend_from_slice(&output_buffer[0][..produced]);
    }
    let flush_target = expected_frames.saturating_add(delay);
    while output.len() < flush_target {
        let started = Instant::now();
        let (_consumed, produced) = resampler
            .process_partial_into_buffer::<&[f64], _>(None, &mut output_buffer, None)
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
        block_times_ns.push(started.elapsed().as_nanos().min(u64::MAX as u128) as u64);
        if produced == 0 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "resampler flush produced no frames before reaching the expected length",
            ));
        }
        output.extend_from_slice(&output_buffer[0][..produced]);
    }

    Ok((output, delay, expected_frames, block_times_ns))
}

#[pyfunction(name = "simulate_product_resampler")]
#[pyo3(signature = (
    samples,
    input_rate,
    output_rate,
    chunk_size=1024,
    sinc_len=None,
    window=None
))]
pub fn simulate_product_resampler_py(
    py: Python<'_>,
    samples: Vec<f64>,
    input_rate: u32,
    output_rate: u32,
    chunk_size: usize,
    sinc_len: Option<usize>,
    window: Option<&str>,
) -> PyResult<(Vec<f64>, usize, usize, Vec<u64>)> {
    let window = window.map(str::to_owned);
    crate::catch_panic(|| {
        py.detach(move || {
            simulate_product_resampler(
                samples,
                input_rate,
                output_rate,
                chunk_size,
                sinc_len,
                window.as_deref(),
            )
        })
    })
}

#[pyfunction]
pub fn product_resampler_configuration() -> PyResult<(usize, String, String, usize, usize)> {
    crate::catch_panic(|| {
        Ok((
            PRODUCT_RESAMPLER_SINC_LEN,
            PRODUCT_RESAMPLER_WINDOW_NAME.to_string(),
            "cubic".to_string(),
            256,
            RESAMPLER_CHUNK_SIZE,
        ))
    })
}

#[inline]
pub(super) fn has_resampler_output_capacity<const N: usize>(
    scratch: &FixedAudioBuffer<f32, N>,
    outbuf: &[Vec<f64>],
) -> bool {
    outbuf
        .first()
        .map(|channel| scratch.remaining() >= channel.len())
        .unwrap_or(false)
}

#[cfg(test)]
mod product_window_tests {
    use super::*;

    /// The rejected `blackman_harris_squared` window was 10.3 dB off at 20 kHz
    /// when resampling 44.1 to 48 kHz (`evaluation/resampler-quality-report.json`);
    /// the product window holds the report's 0.25 dB passband gate.
    #[test]
    fn product_resampler_passband_is_flat_at_20_khz() {
        let (input_rate, output_rate, tone_hz) = (44_100_u32, 48_000_u32, 20_000.0_f64);
        let input: Vec<f64> = (0..input_rate as usize)
            .map(|n| {
                0.5 * (2.0 * std::f64::consts::PI * tone_hz * n as f64 / input_rate as f64).sin()
            })
            .collect();
        let (output, _delay, expected, _timings) = simulate_product_resampler(
            input,
            input_rate,
            output_rate,
            RESAMPLER_CHUNK_SIZE,
            None,
            None,
        )
        .unwrap();

        let margin = output_rate as usize / 4;
        let steady = &output[margin..expected - margin];
        let rms = (steady.iter().map(|s| s * s).sum::<f64>() / steady.len() as f64).sqrt();
        let gain_db = 20.0 * (rms / (0.5 / 2.0_f64.sqrt())).log10();

        assert!(gain_db.abs() < 0.25, "20 kHz gain {gain_db:.3} dB");
    }
}
