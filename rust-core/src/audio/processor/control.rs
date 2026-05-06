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

fn retime_audio_block<'a>(
    input: &'a [f32],
    speed_ratio: f32,
    max_output_len: usize,
    output: &'a mut Vec<f32>,
) -> &'a [f32] {
    if input.is_empty() || max_output_len == 0 {
        output.clear();
        return output.as_slice();
    }

    let clamped_ratio = speed_ratio.max(0.5);
    let desired_len = ((input.len() as f32) / clamped_ratio).round().max(1.0) as usize;
    let out_len = desired_len.min(max_output_len);
    if out_len == input.len() {
        return input;
    }

    output.clear();
    output.resize(out_len, 0.0);

    let max_src = (input.len() - 1) as f32;
    for (i, out_sample) in output.iter_mut().enumerate() {
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

#[inline]
fn release_ms_to_tenth_ms(release_ms: f64) -> u64 {
    if !release_ms.is_finite() {
        return 0;
    }
    (release_ms.max(0.0) * 10.0).round() as u64
}

#[inline]
fn clamp_control_value(value: f64, min_value: f64, max_value: f64) -> Option<f64> {
    value.is_finite().then(|| value.clamp(min_value, max_value))
}

#[inline]
fn clamp_control_value_f32(value: f32, min_value: f32, max_value: f32) -> Option<f32> {
    value.is_finite().then(|| value.clamp(min_value, max_value))
}

fn lock_rt<'a, T>(
    mutex: &'a Mutex<T>,
    lock_contention_count: &AtomicU64,
) -> Option<std::sync::MutexGuard<'a, T>> {
    match mutex.try_lock() {
        Ok(guard) => Some(guard),
        Err(std::sync::TryLockError::WouldBlock) => {
            lock_contention_count.fetch_add(1, Ordering::Relaxed);
            None
        }
        Err(std::sync::TryLockError::Poisoned(_)) => None,
    }
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

fn update_backend_diagnostics(
    available: &AtomicBool,
    failed: &AtomicBool,
    error: &Mutex<Option<String>>,
    suppressor: &NoiseSuppressionEngine,
) {
    available.store(suppressor.backend_available(), Ordering::Relaxed);
    failed.store(suppressor.backend_failed(), Ordering::Relaxed);
    if let Ok(mut guard) = error.lock() {
        *guard = suppressor.backend_error().map(str::to_string);
    }
}

#[derive(Default)]
struct StreamRecoveryState {
    last_error: Option<String>,
    last_reason: Option<String>,
    restart_count: u64,
}

#[derive(Clone)]
struct GateControlState {
    enabled: bool,
    threshold_db: f64,
    attack_ms: f64,
    release_ms: f64,
    #[cfg(feature = "vad")]
    gate_mode: GateMode,
    #[cfg(feature = "vad")]
    vad_threshold: f32,
    #[cfg(feature = "vad")]
    hold_ms: f32,
    #[cfg(feature = "vad")]
    pre_gain: f32,
    #[cfg(feature = "vad")]
    auto_threshold: bool,
    #[cfg(feature = "vad")]
    margin_db: f32,
}

impl GateControlState {
    fn new() -> Self {
        Self {
            enabled: true,
            threshold_db: -40.0,
            attack_ms: 10.0,
            release_ms: 100.0,
            #[cfg(feature = "vad")]
            gate_mode: GateMode::ThresholdOnly,
            #[cfg(feature = "vad")]
            vad_threshold: 0.4,
            #[cfg(feature = "vad")]
            hold_ms: 200.0,
            #[cfg(feature = "vad")]
            pre_gain: 1.0,
            #[cfg(feature = "vad")]
            auto_threshold: true,
            #[cfg(feature = "vad")]
            margin_db: 10.0,
        }
    }
}

#[derive(Clone)]
struct SuppressorControlState {
    enabled: bool,
    model: NoiseModel,
}

impl SuppressorControlState {
    fn new() -> Self {
        Self {
            enabled: true,
            model: NoiseModel::RNNoise,
        }
    }
}

#[derive(Clone, Copy)]
struct EqControlSnapshot {
    enabled: bool,
    bands: [(f64, f64, f64); NUM_BANDS],
}

impl EqControlSnapshot {
    fn new() -> Self {
        Self {
            enabled: true,
            bands: std::array::from_fn(|index| (DEFAULT_FREQUENCIES[index], 0.0, DEFAULT_Q)),
        }
    }
}

struct EqControlState {
    seq: AtomicU64,
    enabled: AtomicBool,
    frequency_bits: [AtomicU64; NUM_BANDS],
    gain_bits: [AtomicU64; NUM_BANDS],
    q_bits: [AtomicU64; NUM_BANDS],
}

impl EqControlState {
    fn new() -> Self {
        let snapshot = EqControlSnapshot::new();
        Self {
            seq: AtomicU64::new(0),
            enabled: AtomicBool::new(snapshot.enabled),
            frequency_bits: std::array::from_fn(|index| {
                AtomicU64::new(snapshot.bands[index].0.to_bits())
            }),
            gain_bits: std::array::from_fn(|index| {
                AtomicU64::new(snapshot.bands[index].1.to_bits())
            }),
            q_bits: std::array::from_fn(|index| AtomicU64::new(snapshot.bands[index].2.to_bits())),
        }
    }

    fn update<F>(&self, apply: F)
    where
        F: FnOnce(&Self),
    {
        self.seq.fetch_add(1, Ordering::AcqRel);
        apply(self);
        self.seq.fetch_add(1, Ordering::Release);
    }

    fn snapshot(&self) -> EqControlSnapshot {
        loop {
            let seq_before = self.seq.load(Ordering::Acquire);
            if (seq_before & 1) != 0 {
                std::hint::spin_loop();
                continue;
            }

            let enabled = self.enabled.load(Ordering::Relaxed);
            let bands = std::array::from_fn(|index| {
                let frequency = f64::from_bits(self.frequency_bits[index].load(Ordering::Relaxed));
                let gain = f64::from_bits(self.gain_bits[index].load(Ordering::Relaxed));
                let q = f64::from_bits(self.q_bits[index].load(Ordering::Relaxed));
                (frequency, gain, q)
            });

            let seq_after = self.seq.load(Ordering::Acquire);
            if seq_before == seq_after {
                return EqControlSnapshot { enabled, bands };
            }
        }
    }

    fn set_enabled(&self, enabled: bool) {
        self.update(|state| {
            state.enabled.store(enabled, Ordering::Relaxed);
        });
    }

    fn set_band_frequency(&self, band: usize, frequency: f64) {
        self.update(|state| {
            state.frequency_bits[band].store(frequency.to_bits(), Ordering::Relaxed);
        });
    }

    fn set_band_gain(&self, band: usize, gain_db: f64) {
        self.update(|state| {
            state.gain_bits[band].store(gain_db.to_bits(), Ordering::Relaxed);
        });
    }

    fn set_band_q(&self, band: usize, q: f64) {
        self.update(|state| {
            state.q_bits[band].store(q.to_bits(), Ordering::Relaxed);
        });
    }

    fn set_bands(&self, bands: &[(f64, f64, f64); NUM_BANDS]) {
        self.update(|state| {
            for (index, (frequency, gain, q)) in bands.iter().copied().enumerate() {
                state.frequency_bits[index].store(frequency.to_bits(), Ordering::Relaxed);
                state.gain_bits[index].store(gain.to_bits(), Ordering::Relaxed);
                state.q_bits[index].store(q.to_bits(), Ordering::Relaxed);
            }
        });
    }
}

#[derive(Clone)]
struct DeesserControlState {
    enabled: bool,
    auto_enabled: bool,
    auto_amount: f64,
    low_cut_hz: f64,
    high_cut_hz: f64,
    threshold_db: f64,
    ratio: f64,
    attack_ms: f64,
    release_ms: f64,
    max_reduction_db: f64,
}

impl DeesserControlState {
    fn new() -> Self {
        Self {
            enabled: false,
            auto_enabled: true,
            auto_amount: 0.5,
            low_cut_hz: 4000.0,
            high_cut_hz: 9000.0,
            threshold_db: -28.0,
            ratio: 4.0,
            attack_ms: 2.0,
            release_ms: 80.0,
            max_reduction_db: 6.0,
        }
    }
}

#[derive(Clone)]
struct CompressorControlState {
    enabled: bool,
    threshold_db: f64,
    ratio: f64,
    attack_ms: f64,
    base_release_ms: f64,
    makeup_gain_db: f64,
    adaptive_release: bool,
    auto_makeup_enabled: bool,
    target_lufs: f64,
}

impl CompressorControlState {
    fn new() -> Self {
        Self {
            enabled: true,
            threshold_db: -20.0,
            ratio: 4.0,
            attack_ms: 10.0,
            base_release_ms: 200.0,
            makeup_gain_db: 0.0,
            adaptive_release: false,
            auto_makeup_enabled: false,
            target_lufs: -18.0,
        }
    }
}

#[derive(Clone)]
struct LimiterControlState {
    enabled: bool,
    ceiling_db: f64,
    release_ms: f64,
}

impl LimiterControlState {
    fn new() -> Self {
        Self {
            enabled: true,
            ceiling_db: -0.5,
            release_ms: 50.0,
        }
    }
}

fn apply_eq_control(eq: &mut ParametricEQ, control: &EqControlSnapshot) {
    eq.set_enabled(control.enabled);
    for (index, (frequency, gain_db, q)) in control.bands.iter().copied().enumerate() {
        eq.set_band_frequency(index, frequency);
        eq.set_band_gain(index, gain_db);
        eq.set_band_q(index, q);
    }
}

fn apply_gate_control(gate: &mut NoiseGate, control: &GateControlState) {
    gate.set_enabled(control.enabled);
    gate.set_threshold(control.threshold_db);
    gate.set_attack_time(control.attack_ms);
    gate.set_release_time(control.release_ms);
    #[cfg(feature = "vad")]
    {
        gate.set_gate_mode(control.gate_mode);
        gate.set_vad_threshold(control.vad_threshold);
        gate.set_hold_time(control.hold_ms);
        gate.set_vad_pre_gain(control.pre_gain);
        gate.set_auto_threshold(control.auto_threshold);
        gate.set_margin(control.margin_db);
    }
}

fn apply_suppressor_control(
    suppressor: &mut NoiseSuppressionEngine,
    control: &SuppressorControlState,
) {
    suppressor.set_enabled(control.enabled);
}

fn swap_pending_suppressor_if_ready(
    suppressor: &mut NoiseSuppressionEngine,
    control: &SuppressorControlState,
    pending: &mut Option<NoiseSuppressionEngine>,
) -> bool {
    if suppressor.model_type() == control.model {
        return true;
    }

    let Some(candidate) = pending.as_ref() else {
        return false;
    };
    if candidate.model_type() != control.model {
        return false;
    }

    *suppressor = pending
        .take()
        .expect("pending suppressor was checked above");
    true
}

fn apply_deesser_control(deesser: &mut DeEsser, control: &DeesserControlState) {
    deesser.set_enabled(control.enabled);
    deesser.set_auto_enabled(control.auto_enabled);
    deesser.set_auto_amount(control.auto_amount);
    deesser.set_low_cut_hz(control.low_cut_hz);
    deesser.set_high_cut_hz(control.high_cut_hz);
    deesser.set_threshold_db(control.threshold_db);
    deesser.set_ratio(control.ratio);
    deesser.set_attack_ms(control.attack_ms);
    deesser.set_release_ms(control.release_ms);
    deesser.set_max_reduction_db(control.max_reduction_db);
}

fn apply_compressor_control(compressor: &mut Compressor, control: &CompressorControlState) {
    compressor.set_enabled(control.enabled);
    compressor.set_threshold(control.threshold_db);
    compressor.set_ratio(control.ratio);
    compressor.set_attack_time(control.attack_ms);
    compressor.set_base_release_time(control.base_release_ms);
    if !control.adaptive_release {
        compressor.set_release_time(control.base_release_ms);
    }
    compressor.set_makeup_gain(control.makeup_gain_db);
    compressor.set_adaptive_release(control.adaptive_release);
    compressor.set_auto_makeup_enabled(control.auto_makeup_enabled);
    compressor.set_target_lufs(control.target_lufs);
}

fn apply_limiter_control(limiter: &mut Limiter, control: &LimiterControlState) {
    limiter.set_ceiling(control.ceiling_db);
    limiter.set_release_time(control.release_ms);
    limiter.set_enabled(control.enabled);
}
