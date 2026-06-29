#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProcessingPath {
    RawMonitor,
    Bypass,
    Full,
}

#[derive(Default)]
struct InputPreFilterState {
    dc_x1: f32,
    dc_y1: f32,
}

#[inline]
fn select_processing_path(raw_monitor_enabled: bool, bypass_enabled: bool) -> ProcessingPath {
    if raw_monitor_enabled {
        ProcessingPath::RawMonitor
    } else if bypass_enabled {
        ProcessingPath::Bypass
    } else {
        ProcessingPath::Full
    }
}

#[inline]
fn uses_clean_write_path(path: ProcessingPath) -> bool {
    matches!(path, ProcessingPath::RawMonitor)
}

#[inline]
fn sanitize_non_finite_inplace(buffer: &mut [f32]) {
    for sample in buffer.iter_mut() {
        if !sample.is_finite() {
            *sample = 0.0;
        }
    }
}

#[inline]
fn sanitize_and_clamp_output_inplace(buffer: &mut [f32], ceiling_linear: f32) {
    let ceiling = ceiling_linear.clamp(0.0, 1.0);
    for sample in buffer.iter_mut() {
        if !sample.is_finite() {
            *sample = 0.0;
        } else {
            *sample = sample.clamp(-ceiling, ceiling);
        }
    }
}

#[inline]
fn sanitize_and_clamp_input_inplace(
    buffer: &mut [f32],
    clip_event_count: &AtomicU64,
    clip_peak_db: &AtomicU32,
) {
    for sample in buffer.iter_mut() {
        if !sample.is_finite() {
            *sample = 0.0;
            continue;
        }
        let amplitude = sample.abs();
        if amplitude > 1.0 {
            clip_event_count.fetch_add(1, Ordering::Relaxed);
            let peak_db = 20.0 * amplitude.log10();
            let current_peak = f32::from_bits(clip_peak_db.load(Ordering::Relaxed));
            if peak_db > current_peak {
                clip_peak_db.store(peak_db.to_bits(), Ordering::Relaxed);
            }
        }
        *sample = sample.clamp(-1.0, 1.0);
    }
}

#[inline]
fn apply_input_pre_filter(
    buffer: &mut [f32],
    dc_state: &mut InputPreFilterState,
    pre_filter: &mut Biquad,
) {
    for sample in buffer.iter_mut() {
        let input = *sample;
        let output = input - dc_state.dc_x1 + INPUT_DC_BLOCK_COEFF * dc_state.dc_y1;
        dc_state.dc_x1 = input;
        dc_state.dc_y1 = output;
        *sample = pre_filter.process_sample(output);
    }
}
