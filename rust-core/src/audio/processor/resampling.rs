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

fn total_reported_latency_us(
    output_buffer_samples: u64,
    output_sample_rate: u32,
    suppressor_latency_samples: u64,
    limiter_lookahead_samples: u64,
    limiter_enabled: bool,
    processing_sample_rate: u32,
    compensation_us: u64,
) -> u64 {
    let output_latency_us = samples_to_micros(output_buffer_samples, output_sample_rate);
    let suppressor_latency_us =
        samples_to_micros(suppressor_latency_samples, processing_sample_rate);
    let limiter_latency_us = if limiter_enabled {
        samples_to_micros(limiter_lookahead_samples, processing_sample_rate)
    } else {
        0
    };

    output_latency_us
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
    let ratio = output_rate as f64 / input_rate as f64;
    let sinc_len = 128;
    let window = WindowFunction::BlackmanHarris2;
    let params = SincInterpolationParameters {
        sinc_len,
        f_cutoff: calculate_cutoff(sinc_len, window),
        interpolation: SincInterpolationType::Cubic,
        oversampling_factor: 256,
        window,
    };
    SincFixedIn::<f64>::new(ratio, 1.2, params, chunk_size, 1).map_err(|e| e.to_string())
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
