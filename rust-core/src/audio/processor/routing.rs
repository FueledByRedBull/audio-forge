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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum InputCleanupMode {
    #[default]
    Off = 0,
    Gentle = 1,
    Strong = 2,
}

impl InputCleanupMode {
    fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Off),
            1 => Some(Self::Gentle),
            2 => Some(Self::Strong),
            _ => None,
        }
    }

    fn from_id(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" => Some(Self::Off),
            "gentle" => Some(Self::Gentle),
            "strong" => Some(Self::Strong),
            _ => None,
        }
    }

    fn id(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Gentle => "gentle",
            Self::Strong => "strong",
        }
    }

    fn is_enabled(self) -> bool {
        !matches!(self, Self::Off)
    }
}

#[derive(Clone, Copy)]
struct HumBin {
    cos_phase: f32,
    sin_phase: f32,
    cos_step: f32,
    sin_step: f32,
    i_acc: f32,
    q_acc: f32,
}

impl HumBin {
    fn new(frequency_hz: f32, sample_rate: f32) -> Self {
        let omega = 2.0 * std::f32::consts::PI * frequency_hz / sample_rate.max(1.0);
        Self {
            cos_phase: 1.0,
            sin_phase: 0.0,
            cos_step: omega.cos(),
            sin_step: omega.sin(),
            i_acc: 0.0,
            q_acc: 0.0,
        }
    }

    #[inline]
    fn analyze_sample(&mut self, sample: f32) {
        self.i_acc += sample * self.cos_phase;
        self.q_acc += sample * self.sin_phase;

        let next_cos = self.cos_phase * self.cos_step - self.sin_phase * self.sin_step;
        let next_sin = self.sin_phase * self.cos_step + self.cos_phase * self.sin_step;
        self.cos_phase = next_cos;
        self.sin_phase = next_sin;
    }

    fn power_and_reset(&mut self, window_samples: usize) -> f32 {
        let n = window_samples.max(1) as f32;
        let power = (self.i_acc * self.i_acc + self.q_acc * self.q_acc) * (2.0 / (n * n));
        self.i_acc = 0.0;
        self.q_acc = 0.0;
        let phase_norm = (self.cos_phase * self.cos_phase + self.sin_phase * self.sin_phase).sqrt();
        if phase_norm > 1.0e-6 {
            self.cos_phase /= phase_norm;
            self.sin_phase /= phase_norm;
        }
        power
    }
}

const HUM_MIN_HZ: f32 = 49.0;
const HUM_MAX_HZ: f32 = 61.0;
const HUM_TRACK_STEP_HZ: f32 = 1.0;
const HUM_TRACK_BINS: usize = 13;

#[derive(Clone, Copy)]
struct NotchFilter {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl NotchFilter {
    fn new(frequency_hz: f32, q: f32, sample_rate: f32) -> Self {
        let omega = 2.0 * std::f32::consts::PI * frequency_hz / sample_rate.max(1.0);
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * q.max(1.0));
        let a0 = 1.0 + alpha;

        Self {
            b0: 1.0 / a0,
            b1: -2.0 * cos_omega / a0,
            b2: 1.0 / a0,
            a1: -2.0 * cos_omega / a0,
            a2: (1.0 - alpha) / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    #[inline]
    fn process_sample(&mut self, input: f32) -> f32 {
        let output = self.b0 * input + self.z1;
        self.z1 = self.b1 * input - self.a1 * output + self.z2;
        self.z2 = self.b2 * input - self.a2 * output;
        output
    }

    fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
}

/// A notch that retunes by crossfading parallel old/new filter states for 20 ms.
struct SmoothNotch {
    active: NotchFilter,
    pending: NotchFilter,
    frequency_hz: f32,
    pending_frequency_hz: f32,
    fade_total: usize,
    fade_remaining: usize,
    sample_rate: f32,
    q: f32,
}

impl SmoothNotch {
    fn new(frequency_hz: f32, q: f32, sample_rate: f32) -> Self {
        let filter = NotchFilter::new(frequency_hz, q, sample_rate);
        Self {
            active: filter,
            pending: filter,
            frequency_hz,
            pending_frequency_hz: frequency_hz,
            fade_total: (sample_rate * 0.020).round().max(1.0) as usize,
            fade_remaining: 0,
            sample_rate,
            q,
        }
    }

    fn retune(&mut self, frequency_hz: f32) {
        let frequency_hz = frequency_hz.clamp(20.0, self.sample_rate * 0.45);
        if (frequency_hz - self.pending_frequency_hz).abs() < 0.15 {
            return;
        }
        self.pending = NotchFilter::new(frequency_hz, self.q, self.sample_rate);
        self.pending_frequency_hz = frequency_hz;
        self.fade_remaining = self.fade_total;
    }

    #[inline]
    fn process_sample(&mut self, input: f32) -> f32 {
        let active = self.active.process_sample(input);
        if self.fade_remaining == 0 {
            return active;
        }
        let pending = self.pending.process_sample(input);
        let fade = (self.fade_total - self.fade_remaining + 1) as f32 / self.fade_total as f32;
        let output = active + (pending - active) * fade;
        self.fade_remaining -= 1;
        if self.fade_remaining == 0 {
            self.active = self.pending;
            self.frequency_hz = self.pending_frequency_hz;
        }
        output
    }

    fn reset(&mut self) {
        self.active.reset();
        self.pending.reset();
        self.fade_remaining = 0;
    }
}

/// Tracks off-nominal 49-61 Hz hum plus its harmonic and owns the selected HP.
///
/// When enabled, this state replaces the fixed 80 Hz filter so cleanup never
/// cascades two independent high-pass responses.
struct AdaptiveInputCleanupState {
    sample_rate: f32,
    mode: InputCleanupMode,
    lowpass_state: f32,
    low_env: f32,
    slow_low_env: f32,
    broadband_env: f32,
    rumble_hold_samples: u32,
    hum_bins: [HumBin; HUM_TRACK_BINS],
    hum_harmonic_bins: [HumBin; HUM_TRACK_BINS],
    hum_window_samples: usize,
    hum_window_pos: usize,
    hum_windows_observed: u32,
    hum_candidate_windows: u8,
    hum_total_energy: f32,
    hum_hold_samples: u32,
    hum_line_hz: f32,
    hum_strength: f32,
    harmonic_strength: f32,
    adaptive_highpass: Biquad,
    adaptive_highpass_hz: f32,
    hum_notch: SmoothNotch,
    harmonic_notch: SmoothNotch,
    hum_detected: bool,
    rumble_detected: bool,
    selected_high_pass_hz: f32,
}

impl AdaptiveInputCleanupState {
    fn new(sample_rate: f32) -> Self {
        let notch_q = 36.0;
        Self {
            sample_rate,
            mode: InputCleanupMode::Off,
            lowpass_state: 0.0,
            low_env: 0.0,
            slow_low_env: 0.0,
            broadband_env: 0.0,
            rumble_hold_samples: 0,
            hum_bins: std::array::from_fn(|index| {
                HumBin::new(HUM_MIN_HZ + index as f32 * HUM_TRACK_STEP_HZ, sample_rate)
            }),
            hum_harmonic_bins: std::array::from_fn(|index| {
                HumBin::new(
                    2.0 * (HUM_MIN_HZ + index as f32 * HUM_TRACK_STEP_HZ),
                    sample_rate,
                )
            }),
            hum_window_samples: (sample_rate * 0.25).round().max(1.0) as usize,
            hum_window_pos: 0,
            hum_windows_observed: 0,
            hum_candidate_windows: 0,
            hum_total_energy: 0.0,
            hum_hold_samples: 0,
            hum_line_hz: 0.0,
            hum_strength: 0.0,
            harmonic_strength: 0.0,
            adaptive_highpass: Biquad::new(
                BiquadType::HighPass,
                INPUT_PREFILTER_HZ,
                0.0,
                INPUT_PREFILTER_Q,
                sample_rate as f64,
            ),
            adaptive_highpass_hz: INPUT_PREFILTER_HZ as f32,
            hum_notch: SmoothNotch::new(55.0, notch_q, sample_rate),
            harmonic_notch: SmoothNotch::new(110.0, notch_q, sample_rate),
            hum_detected: false,
            rumble_detected: false,
            selected_high_pass_hz: INPUT_PREFILTER_HZ as f32,
        }
    }

    fn set_mode(&mut self, mode: InputCleanupMode) {
        if self.mode == mode {
            return;
        }
        self.mode = mode;
        if !mode.is_enabled() {
            self.reset_dynamic_state();
        }
    }

    fn reset_dynamic_state(&mut self) {
        self.lowpass_state = 0.0;
        self.low_env = 0.0;
        self.slow_low_env = 0.0;
        self.broadband_env = 0.0;
        self.rumble_hold_samples = 0;
        self.hum_window_pos = 0;
        self.hum_windows_observed = 0;
        self.hum_candidate_windows = 0;
        self.hum_total_energy = 0.0;
        self.hum_hold_samples = 0;
        self.hum_line_hz = 0.0;
        self.hum_strength = 0.0;
        self.harmonic_strength = 0.0;
        self.adaptive_highpass.reset();
        self.adaptive_highpass_hz = f32::NAN;
        self.hum_notch.reset();
        self.harmonic_notch.reset();
        self.hum_detected = false;
        self.rumble_detected = false;
        self.selected_high_pass_hz = INPUT_PREFILTER_HZ as f32;
    }

    fn analyze_input(&mut self, buffer: &[f32]) {
        if !self.mode.is_enabled() {
            return;
        }

        let lowpass_coeff = (2.0 * std::f32::consts::PI * 150.0 / self.sample_rate)
            .clamp(0.0, 1.0);
        let fast_attack = 0.08;
        let fast_release = 0.006;
        let slow_coeff = 0.0012;
        let broadband_coeff = 0.02;

        for &sample in buffer {
            self.hum_total_energy += sample * sample;
            for bin in self.hum_bins.iter_mut() {
                bin.analyze_sample(sample);
            }
            for bin in self.hum_harmonic_bins.iter_mut() {
                bin.analyze_sample(sample);
            }
            self.hum_window_pos += 1;
            if self.hum_window_pos >= self.hum_window_samples {
                self.finish_hum_window();
            }

            self.lowpass_state += lowpass_coeff * (sample - self.lowpass_state);
            let low_abs = self.lowpass_state.abs();
            let low_coeff = if low_abs > self.low_env {
                fast_attack
            } else {
                fast_release
            };
            self.low_env += low_coeff * (low_abs - self.low_env);
            self.slow_low_env += slow_coeff * (low_abs - self.slow_low_env);
            self.broadband_env += broadband_coeff * (sample.abs() - self.broadband_env);

            let burst_ratio = self.low_env / self.slow_low_env.max(0.006);
            let low_dominance = self.low_env / self.broadband_env.max(0.01);
            let threshold = match self.mode {
                InputCleanupMode::Off => f32::INFINITY,
                InputCleanupMode::Gentle => 0.055,
                InputCleanupMode::Strong => 0.035,
            };
            let ratio_threshold = match self.mode {
                InputCleanupMode::Off => f32::INFINITY,
                InputCleanupMode::Gentle => 2.8,
                InputCleanupMode::Strong => 2.1,
            };
            let startup_burst = self.hum_windows_observed == 0 && self.low_env > 0.45;
            let established_floor = self.hum_windows_observed > 0 && self.slow_low_env > 0.012;
            if (startup_burst || established_floor)
                && self.hum_hold_samples == 0
                && self.hum_candidate_windows == 0
                && self.low_env > threshold
                && burst_ratio > ratio_threshold
                && low_dominance > 0.62
            {
                self.rumble_hold_samples = match self.mode {
                    InputCleanupMode::Off => 0,
                    InputCleanupMode::Gentle => (self.sample_rate * 0.18).round() as u32,
                    InputCleanupMode::Strong => (self.sample_rate * 0.30).round() as u32,
                };
            } else {
                self.rumble_hold_samples = self.rumble_hold_samples.saturating_sub(1);
            }
            self.hum_hold_samples = self.hum_hold_samples.saturating_sub(1);
        }
    }

    fn finish_hum_window(&mut self) {
        let mut best_frequency_hz = 0.0_f32;
        let mut best_primary_power = 0.0_f32;
        let mut best_harmonic_power = 0.0_f32;
        let mut best_score = 0.0_f32;
        for index in 0..HUM_TRACK_BINS {
            let primary_power = self.hum_bins[index].power_and_reset(self.hum_window_samples);
            let harmonic_power =
                self.hum_harmonic_bins[index].power_and_reset(self.hum_window_samples);
            let score = primary_power + harmonic_power * 0.65;
            if score > best_score {
                best_score = score;
                best_primary_power = primary_power;
                best_harmonic_power = harmonic_power;
                best_frequency_hz = HUM_MIN_HZ + index as f32 * HUM_TRACK_STEP_HZ;
            }
        }
        let total_power = self.hum_total_energy / self.hum_window_samples.max(1) as f32 + 1.0e-9;
        self.hum_window_pos = 0;
        self.hum_windows_observed = self.hum_windows_observed.saturating_add(1);
        self.hum_total_energy = 0.0;

        let primary_ratio = best_primary_power / total_power;
        let harmonic_ratio = best_harmonic_power / total_power;
        let (ratio_threshold, power_threshold) = match self.mode {
            InputCleanupMode::Off => (f32::INFINITY, f32::INFINITY),
            InputCleanupMode::Gentle => (0.075, 1.8e-5),
            InputCleanupMode::Strong => (0.040, 8.0e-6),
        };

        let candidate = (best_primary_power > power_threshold
            || best_harmonic_power > power_threshold * 0.70)
            && (primary_ratio > ratio_threshold || harmonic_ratio > ratio_threshold * 0.85)
            && best_frequency_hz > 0.0;
        if candidate {
            self.hum_candidate_windows = self.hum_candidate_windows.saturating_add(1).min(3);
        } else {
            self.hum_candidate_windows = 0;
        }
        if self.hum_candidate_windows >= 2 {
            self.hum_hold_samples = (self.sample_rate * 0.75).round() as u32;
            self.hum_line_hz = if self.hum_line_hz <= 0.0 {
                best_frequency_hz
            } else {
                self.hum_line_hz + 0.35 * (best_frequency_hz - self.hum_line_hz)
            }
            .clamp(HUM_MIN_HZ, HUM_MAX_HZ);
        }
    }

    fn process_block(&mut self, buffer: &mut [f32]) {
        if !self.mode.is_enabled() {
            return;
        }

        self.hum_detected = self.hum_hold_samples > 0;
        self.rumble_detected = self.rumble_hold_samples > 0;
        self.selected_high_pass_hz = match (self.mode, self.rumble_detected) {
            (InputCleanupMode::Off, _) => INPUT_PREFILTER_HZ as f32,
            (InputCleanupMode::Gentle, true) => 100.0,
            (InputCleanupMode::Strong, true) => 120.0,
            (InputCleanupMode::Gentle, false) | (InputCleanupMode::Strong, false) => {
                INPUT_PREFILTER_HZ as f32
            }
        };

        if (self.selected_high_pass_hz - self.adaptive_highpass_hz).abs() > 0.5 {
            self.adaptive_highpass
                .set_frequency(self.selected_high_pass_hz as f64);
            self.adaptive_highpass_hz = self.selected_high_pass_hz;
        }

        let hum_attack = match self.mode {
            InputCleanupMode::Off => 0.0,
            InputCleanupMode::Gentle => 0.22,
            InputCleanupMode::Strong => 0.34,
        };
        let hum_release = 0.035;
        let target_hum = if self.hum_detected {
            match self.mode {
                InputCleanupMode::Off => 0.0,
                InputCleanupMode::Gentle => 0.55,
                InputCleanupMode::Strong => 0.85,
            }
        } else {
            0.0
        };
        let target_harmonic = if self.hum_detected {
            match self.mode {
                InputCleanupMode::Off | InputCleanupMode::Gentle => 0.0,
                InputCleanupMode::Strong => 0.60,
            }
        } else {
            0.0
        };
        self.hum_strength = smooth_toward(self.hum_strength, target_hum, hum_attack, hum_release);
        self.harmonic_strength =
            smooth_toward(self.harmonic_strength, target_harmonic, hum_attack, hum_release);
        if self.hum_line_hz > 0.0 {
            self.hum_notch.retune(self.hum_line_hz);
            self.harmonic_notch.retune(self.hum_line_hz * 2.0);
        }

        for sample in buffer.iter_mut() {
            let mut y = *sample;
            let primary_notched = self.hum_notch.process_sample(y);
            y += (primary_notched - y) * self.hum_strength.clamp(0.0, 1.0);
            let harmonic_notched = self.harmonic_notch.process_sample(y);
            y += (harmonic_notched - y) * self.harmonic_strength.clamp(0.0, 1.0);
            y = self.adaptive_highpass.process_sample(y);
            *sample = y;
        }
    }
}

#[inline]
fn smooth_toward(current: f32, target: f32, attack: f32, release: f32) -> f32 {
    let coeff = if target > current { attack } else { release };
    current + coeff * (target - current)
}

#[inline]
fn update_decaying_peak_db(value_db: f32, history: &AtomicU32, decay_db_per_update: f32) {
    let previous = f32::from_bits(history.load(Ordering::Relaxed)).max(0.0);
    let decayed = (previous - decay_db_per_update.max(0.0)).max(0.0);
    history.store(value_db.max(decayed).to_bits(), Ordering::Relaxed);
}

#[inline]
fn publish_input_cleanup_diagnostics(
    state: &AdaptiveInputCleanupState,
    hum_detected: &AtomicBool,
    rumble_detected: &AtomicBool,
    selected_high_pass_hz: &AtomicU32,
) {
    hum_detected.store(state.hum_detected, Ordering::Relaxed);
    rumble_detected.store(state.rumble_detected, Ordering::Relaxed);
    selected_high_pass_hz.store(state.selected_high_pass_hz.to_bits(), Ordering::Relaxed);
}

#[inline]
fn publish_input_cleanup_bypassed(
    hum_detected: &AtomicBool,
    rumble_detected: &AtomicBool,
    selected_high_pass_hz: &AtomicU32,
) {
    hum_detected.store(false, Ordering::Relaxed);
    rumble_detected.store(false, Ordering::Relaxed);
    selected_high_pass_hz.store((INPUT_PREFILTER_HZ as f32).to_bits(), Ordering::Relaxed);
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

#[derive(Debug, Clone, Copy)]
struct MeterBlockStats {
    peak_db: f32,
    rms_db: f32,
    crest_factor_db: f32,
    mean_power: f32,
}

#[inline]
fn update_meter_block_stats(
    buffer: &[f32],
    rms_acc: &mut f32,
    meter_coeff: f32,
) -> MeterBlockStats {
    let mut peak: f32 = 0.0;
    let mut block_power: f32 = 0.0;
    for &sample in buffer.iter() {
        let abs = sample.abs();
        if abs > peak {
            peak = abs;
        }
        block_power += sample * sample;
        *rms_acc = meter_coeff * *rms_acc + (1.0 - meter_coeff) * (sample * sample);
    }

    let peak_db = if peak > 0.0 {
        20.0 * peak.log10()
    } else {
        -120.0
    };
    let rms_db = if *rms_acc > 0.0 {
        10.0 * rms_acc.log10()
    } else {
        -120.0
    };
    let crest_factor_db = (peak_db - rms_db).clamp(0.0, 80.0);
    let mean_power = if buffer.is_empty() {
        0.0
    } else {
        block_power / buffer.len() as f32
    };
    MeterBlockStats {
        peak_db,
        rms_db,
        crest_factor_db,
        mean_power,
    }
}

#[cfg(test)]
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
fn sanitize_and_clamp_output_inplace_with_metrics(
    buffer: &mut [f32],
    ceiling_linear: f32,
    output_clip_event_count: &AtomicU64,
    output_clip_peak_db: &AtomicU32,
) {
    let ceiling = ceiling_linear.clamp(0.0, 1.0);
    let mut clipped_samples = 0_u64;
    let mut max_clipped_amplitude = 0.0_f32;

    for sample in buffer.iter_mut() {
        if !sample.is_finite() {
            *sample = 0.0;
            continue;
        }
        let amplitude = sample.abs();
        if amplitude > ceiling {
            clipped_samples = clipped_samples.saturating_add(1);
            max_clipped_amplitude = max_clipped_amplitude.max(amplitude);
        }
        *sample = sample.clamp(-ceiling, ceiling);
    }

    if clipped_samples > 0 {
        output_clip_event_count.fetch_add(clipped_samples, Ordering::Relaxed);
        let peak_db = 20.0 * max_clipped_amplitude.log10();
        let current_peak = f32::from_bits(output_clip_peak_db.load(Ordering::Relaxed));
        if peak_db > current_peak {
            output_clip_peak_db.store(peak_db.to_bits(), Ordering::Relaxed);
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
    apply_fixed_highpass: bool,
) {
    for sample in buffer.iter_mut() {
        let input = *sample;
        let output = input - dc_state.dc_x1 + INPUT_DC_BLOCK_COEFF * dc_state.dc_y1;
        dc_state.dc_x1 = input;
        dc_state.dc_y1 = output;
        *sample = if apply_fixed_highpass {
            pre_filter.process_sample(output)
        } else {
            output
        };
    }
}
