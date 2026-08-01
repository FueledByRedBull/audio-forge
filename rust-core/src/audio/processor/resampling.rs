fn duration_samples(sample_rate: u32, duration_ms: u32) -> usize {
    (((sample_rate as u64) * duration_ms as u64 + 500) / 1000).max(1) as usize
}

fn samples_to_micros(samples: u64, sample_rate: u32) -> u64 {
    if sample_rate == 0 {
        0
    } else {
        samples.saturating_mul(1_000_000) / sample_rate as u64
    }
}

fn smoothing_coeff_for_time_constant(sample_rate_hz: f32, time_constant_ms: f32) -> f32 {
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

fn next_process_idle_sleep_us(consecutive_idle_wakeups: u32, input_callback_age_us: u64) -> u64 {
    if input_callback_age_us <= PROCESS_IDLE_RECENT_INPUT_WINDOW_US {
        return PROCESS_IDLE_SLEEP_US;
    }

    let backoff_shift = consecutive_idle_wakeups.min(4);
    PROCESS_IDLE_SLEEP_US
        .saturating_mul(1_u64 << backoff_shift)
        .min(PROCESS_IDLE_MAX_SLEEP_US)
}

#[derive(Clone, Copy)]
struct LatencyComponents {
    output_buffer_samples: u64,
    output_sample_rate: u32,
    output_resampler_delay_samples: u64,
    suppressor_latency_samples: u64,
    limiter_lookahead_samples: u64,
    true_peak_lookahead_samples: u64,
    limiter_enabled: bool,
    processing_sample_rate: u32,
}

fn total_reported_latency_us(components: LatencyComponents, compensation_us: u64) -> u64 {
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
    let limiter_latency_us = if components.limiter_enabled {
        samples_to_micros(
            components.limiter_lookahead_samples,
            components.processing_sample_rate,
        )
        .saturating_add(samples_to_micros(
            components.true_peak_lookahead_samples,
            components.output_sample_rate,
        ))
    } else {
        0
    };

    output_latency_us
        .saturating_add(output_resampler_latency_us)
        .saturating_add(suppressor_latency_us)
        .saturating_add(limiter_latency_us)
        .saturating_add(compensation_us)
}

fn retime_audio_block<'a, const N: usize>(
    input: &'a [f32],
    speed_ratio: f32,
    max_output_len: usize,
    output: &'a mut FixedAudioBuffer<f32, N>,
) -> &'a [f32] {
    if input.is_empty() || max_output_len == 0 {
        output.clear();
        return output.as_slice();
    }

    let clamped_ratio = speed_ratio.max(0.5);
    let desired_len = ((input.len() as f32) / clamped_ratio).round().max(1.0) as usize;
    let out_len = desired_len.min(max_output_len).min(output.capacity());
    if out_len == input.len() {
        return input;
    }

    output.clear();
    if !output.set_len_zeroed(out_len) {
        return output.as_slice();
    }

    let max_src = (input.len() - 1) as f32;
    for (i, out_sample) in output.as_mut_slice().iter_mut().enumerate() {
        let src_pos = if out_len == 1 {
            0.0
        } else {
            (i as f32 * clamped_ratio).min(max_src)
        };
        let idx0 = src_pos.floor() as usize;
        let idx1 = (idx0 + 1).min(input.len() - 1);
        let frac = src_pos - idx0 as f32;
        let y0 = input[idx0];
        let y1 = input[idx1];
        *out_sample = y0 + (y1 - y0) * frac;
    }

    output.as_slice()
}

fn build_sinc_resampler(
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

#[pyfunction]
#[pyo3(signature = (
    samples,
    input_rate,
    output_rate,
    chunk_size=1024,
    sinc_len=None,
    window=None
))]
pub fn simulate_product_resampler(
    samples: Vec<f64>,
    input_rate: u32,
    output_rate: u32,
    chunk_size: usize,
    sinc_len: Option<usize>,
    window: Option<&str>,
) -> PyResult<(Vec<f64>, usize, usize, Vec<u64>)> {
    if input_rate == 0 || output_rate == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "sample rates must be positive",
        ));
    }
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

#[pyfunction]
pub fn product_resampler_configuration() -> (usize, String, String, usize, usize) {
    (
        PRODUCT_RESAMPLER_SINC_LEN,
        PRODUCT_RESAMPLER_WINDOW_NAME.to_string(),
        "cubic".to_string(),
        256,
        RESAMPLER_CHUNK_SIZE,
    )
}

#[inline]
fn has_resampler_output_capacity<const N: usize>(
    scratch: &FixedAudioBuffer<f32, N>,
    outbuf: &[Vec<f64>],
) -> bool {
    outbuf
        .first()
        .map(|channel| scratch.remaining() >= channel.len())
        .unwrap_or(false)
}
