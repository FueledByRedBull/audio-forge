include!("routing.rs");
include!("resampling.rs");
include!("diagnostics.rs");

const CONTROL_SNAPSHOT_MAX_RETRIES: usize = 32;

fn locked_control_update<F>(writer: &Mutex<()>, seq: &AtomicU64, apply: F)
where
    F: FnOnce(),
{
    let _writer = writer
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    seq.fetch_add(1, Ordering::AcqRel);
    apply();
    seq.fetch_add(1, Ordering::Release);
}

fn stable_control_snapshot<T, F>(seq: &AtomicU64, mut read: F) -> Option<T>
where
    F: FnMut() -> T,
{
    for _ in 0..CONTROL_SNAPSHOT_MAX_RETRIES {
        let seq_before = seq.load(Ordering::Acquire);
        if (seq_before & 1) != 0 {
            std::hint::spin_loop();
            continue;
        }

        let state = read();
        let seq_after = seq.load(Ordering::Acquire);
        if seq_before == seq_after {
            return Some(state);
        }
        std::hint::spin_loop();
    }

    None
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

struct AtomicGateControlState {
    writer: Mutex<()>,
    seq: AtomicU64,
    enabled: AtomicBool,
    threshold_db_bits: AtomicU64,
    attack_ms_bits: AtomicU64,
    release_ms_bits: AtomicU64,
    #[cfg(feature = "vad")]
    gate_mode: AtomicU8,
    #[cfg(feature = "vad")]
    vad_threshold_bits: AtomicU32,
    #[cfg(feature = "vad")]
    hold_ms_bits: AtomicU32,
    #[cfg(feature = "vad")]
    pre_gain_bits: AtomicU32,
    #[cfg(feature = "vad")]
    auto_threshold: AtomicBool,
    #[cfg(feature = "vad")]
    margin_db_bits: AtomicU32,
}

impl AtomicGateControlState {
    fn new() -> Self {
        let state = GateControlState::new();
        Self {
            writer: Mutex::new(()),
            seq: AtomicU64::new(0),
            enabled: AtomicBool::new(state.enabled),
            threshold_db_bits: AtomicU64::new(state.threshold_db.to_bits()),
            attack_ms_bits: AtomicU64::new(state.attack_ms.to_bits()),
            release_ms_bits: AtomicU64::new(state.release_ms.to_bits()),
            #[cfg(feature = "vad")]
            gate_mode: AtomicU8::new(state.gate_mode as u8),
            #[cfg(feature = "vad")]
            vad_threshold_bits: AtomicU32::new(state.vad_threshold.to_bits()),
            #[cfg(feature = "vad")]
            hold_ms_bits: AtomicU32::new(state.hold_ms.to_bits()),
            #[cfg(feature = "vad")]
            pre_gain_bits: AtomicU32::new(state.pre_gain.to_bits()),
            #[cfg(feature = "vad")]
            auto_threshold: AtomicBool::new(state.auto_threshold),
            #[cfg(feature = "vad")]
            margin_db_bits: AtomicU32::new(state.margin_db.to_bits()),
        }
    }

    fn update<F>(&self, apply: F)
    where
        F: FnOnce(&Self),
    {
        locked_control_update(&self.writer, &self.seq, || apply(self));
    }

    fn snapshot(&self) -> Option<GateControlState> {
        stable_control_snapshot(&self.seq, || GateControlState {
                enabled: self.enabled.load(Ordering::Relaxed),
                threshold_db: f64::from_bits(self.threshold_db_bits.load(Ordering::Relaxed)),
                attack_ms: f64::from_bits(self.attack_ms_bits.load(Ordering::Relaxed)),
                release_ms: f64::from_bits(self.release_ms_bits.load(Ordering::Relaxed)),
                #[cfg(feature = "vad")]
                gate_mode: match self.gate_mode.load(Ordering::Relaxed) {
                    1 => GateMode::VadAssisted,
                    2 => GateMode::VadOnly,
                    _ => GateMode::ThresholdOnly,
                },
                #[cfg(feature = "vad")]
                vad_threshold: f32::from_bits(self.vad_threshold_bits.load(Ordering::Relaxed)),
                #[cfg(feature = "vad")]
                hold_ms: f32::from_bits(self.hold_ms_bits.load(Ordering::Relaxed)),
                #[cfg(feature = "vad")]
                pre_gain: f32::from_bits(self.pre_gain_bits.load(Ordering::Relaxed)),
                #[cfg(feature = "vad")]
                auto_threshold: self.auto_threshold.load(Ordering::Relaxed),
                #[cfg(feature = "vad")]
                margin_db: f32::from_bits(self.margin_db_bits.load(Ordering::Relaxed)),
            })
    }

    fn set_enabled(&self, enabled: bool) {
        self.update(|state| state.enabled.store(enabled, Ordering::Relaxed));
    }

    fn set_threshold_db(&self, threshold_db: f64) {
        self.update(|state| {
            state
                .threshold_db_bits
                .store(threshold_db.to_bits(), Ordering::Relaxed);
        });
    }

    fn set_attack_ms(&self, attack_ms: f64) {
        self.update(|state| {
            state
                .attack_ms_bits
                .store(attack_ms.to_bits(), Ordering::Relaxed);
        });
    }

    fn set_release_ms(&self, release_ms: f64) {
        self.update(|state| {
            state
                .release_ms_bits
                .store(release_ms.to_bits(), Ordering::Relaxed);
        });
    }

    #[cfg(feature = "vad")]
    fn set_gate_mode(&self, gate_mode: GateMode) {
        self.update(|state| state.gate_mode.store(gate_mode as u8, Ordering::Relaxed));
    }

    #[cfg(feature = "vad")]
    fn set_vad_threshold(&self, threshold: f32) {
        self.update(|state| {
            state
                .vad_threshold_bits
                .store(threshold.to_bits(), Ordering::Relaxed);
        });
    }

    #[cfg(feature = "vad")]
    fn set_hold_ms(&self, hold_ms: f32) {
        self.update(|state| {
            state
                .hold_ms_bits
                .store(hold_ms.to_bits(), Ordering::Relaxed);
        });
    }

    #[cfg(feature = "vad")]
    fn set_pre_gain(&self, pre_gain: f32) {
        self.update(|state| {
            state
                .pre_gain_bits
                .store(pre_gain.to_bits(), Ordering::Relaxed);
        });
    }

    #[cfg(feature = "vad")]
    fn set_auto_threshold(&self, enabled: bool) {
        self.update(|state| state.auto_threshold.store(enabled, Ordering::Relaxed));
    }

    #[cfg(feature = "vad")]
    fn set_margin_db(&self, margin_db: f32) {
        self.update(|state| {
            state
                .margin_db_bits
                .store(margin_db.to_bits(), Ordering::Relaxed);
        });
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

fn noise_model_from_u8(value: u8) -> NoiseModel {
    match value {
        #[cfg(feature = "deepfilter")]
        1 => NoiseModel::DeepFilterNetLL,
        #[cfg(feature = "deepfilter")]
        2 => NoiseModel::DeepFilterNet,
        _ => NoiseModel::RNNoise,
    }
}

struct AtomicSuppressorControlState {
    writer: Mutex<()>,
    seq: AtomicU64,
    enabled: AtomicBool,
    model: AtomicU8,
}

impl AtomicSuppressorControlState {
    fn new() -> Self {
        let state = SuppressorControlState::new();
        Self {
            writer: Mutex::new(()),
            seq: AtomicU64::new(0),
            enabled: AtomicBool::new(state.enabled),
            model: AtomicU8::new(state.model as u8),
        }
    }

    fn update<F>(&self, apply: F)
    where
        F: FnOnce(&Self),
    {
        locked_control_update(&self.writer, &self.seq, || apply(self));
    }

    fn snapshot(&self) -> Option<SuppressorControlState> {
        stable_control_snapshot(&self.seq, || SuppressorControlState {
                enabled: self.enabled.load(Ordering::Relaxed),
                model: noise_model_from_u8(self.model.load(Ordering::Relaxed)),
            })
    }

    fn set_enabled(&self, enabled: bool) {
        self.update(|state| state.enabled.store(enabled, Ordering::Relaxed));
    }

    fn set_model(&self, model: NoiseModel) {
        self.update(|state| state.model.store(model as u8, Ordering::Relaxed));
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
    writer: Mutex<()>,
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
            writer: Mutex::new(()),
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
        locked_control_update(&self.writer, &self.seq, || apply(self));
    }

    fn snapshot(&self) -> Option<EqControlSnapshot> {
        stable_control_snapshot(&self.seq, || {
            let enabled = self.enabled.load(Ordering::Relaxed);
            let bands = std::array::from_fn(|index| {
                let frequency = f64::from_bits(self.frequency_bits[index].load(Ordering::Relaxed));
                let gain = f64::from_bits(self.gain_bits[index].load(Ordering::Relaxed));
                let q = f64::from_bits(self.q_bits[index].load(Ordering::Relaxed));
                (frequency, gain, q)
            });

            EqControlSnapshot { enabled, bands }
        })
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

struct AtomicDeesserControlState {
    writer: Mutex<()>,
    seq: AtomicU64,
    enabled: AtomicBool,
    auto_enabled: AtomicBool,
    auto_amount_bits: AtomicU64,
    low_cut_hz_bits: AtomicU64,
    high_cut_hz_bits: AtomicU64,
    threshold_db_bits: AtomicU64,
    ratio_bits: AtomicU64,
    attack_ms_bits: AtomicU64,
    release_ms_bits: AtomicU64,
    max_reduction_db_bits: AtomicU64,
}

impl AtomicDeesserControlState {
    fn new() -> Self {
        let state = DeesserControlState::new();
        Self {
            writer: Mutex::new(()),
            seq: AtomicU64::new(0),
            enabled: AtomicBool::new(state.enabled),
            auto_enabled: AtomicBool::new(state.auto_enabled),
            auto_amount_bits: AtomicU64::new(state.auto_amount.to_bits()),
            low_cut_hz_bits: AtomicU64::new(state.low_cut_hz.to_bits()),
            high_cut_hz_bits: AtomicU64::new(state.high_cut_hz.to_bits()),
            threshold_db_bits: AtomicU64::new(state.threshold_db.to_bits()),
            ratio_bits: AtomicU64::new(state.ratio.to_bits()),
            attack_ms_bits: AtomicU64::new(state.attack_ms.to_bits()),
            release_ms_bits: AtomicU64::new(state.release_ms.to_bits()),
            max_reduction_db_bits: AtomicU64::new(state.max_reduction_db.to_bits()),
        }
    }

    fn update<F>(&self, apply: F)
    where
        F: FnOnce(&Self),
    {
        locked_control_update(&self.writer, &self.seq, || apply(self));
    }

    fn snapshot(&self) -> Option<DeesserControlState> {
        stable_control_snapshot(&self.seq, || DeesserControlState {
                enabled: self.enabled.load(Ordering::Relaxed),
                auto_enabled: self.auto_enabled.load(Ordering::Relaxed),
                auto_amount: f64::from_bits(self.auto_amount_bits.load(Ordering::Relaxed)),
                low_cut_hz: f64::from_bits(self.low_cut_hz_bits.load(Ordering::Relaxed)),
                high_cut_hz: f64::from_bits(self.high_cut_hz_bits.load(Ordering::Relaxed)),
                threshold_db: f64::from_bits(self.threshold_db_bits.load(Ordering::Relaxed)),
                ratio: f64::from_bits(self.ratio_bits.load(Ordering::Relaxed)),
                attack_ms: f64::from_bits(self.attack_ms_bits.load(Ordering::Relaxed)),
                release_ms: f64::from_bits(self.release_ms_bits.load(Ordering::Relaxed)),
                max_reduction_db: f64::from_bits(
                    self.max_reduction_db_bits.load(Ordering::Relaxed),
                ),
            })
    }

    fn set_enabled(&self, enabled: bool) {
        self.update(|state| state.enabled.store(enabled, Ordering::Relaxed));
    }

    fn set_auto_enabled(&self, enabled: bool) {
        self.update(|state| state.auto_enabled.store(enabled, Ordering::Relaxed));
    }

    fn set_auto_amount(&self, value: f64) {
        self.update(|state| state.auto_amount_bits.store(value.to_bits(), Ordering::Relaxed));
    }

    fn set_low_cut_hz(&self, value: f64) {
        self.update(|state| state.low_cut_hz_bits.store(value.to_bits(), Ordering::Relaxed));
    }

    fn set_high_cut_hz(&self, value: f64) {
        self.update(|state| state.high_cut_hz_bits.store(value.to_bits(), Ordering::Relaxed));
    }

    fn set_threshold_db(&self, value: f64) {
        self.update(|state| state.threshold_db_bits.store(value.to_bits(), Ordering::Relaxed));
    }

    fn set_ratio(&self, value: f64) {
        self.update(|state| state.ratio_bits.store(value.to_bits(), Ordering::Relaxed));
    }

    fn set_attack_ms(&self, value: f64) {
        self.update(|state| state.attack_ms_bits.store(value.to_bits(), Ordering::Relaxed));
    }

    fn set_release_ms(&self, value: f64) {
        self.update(|state| state.release_ms_bits.store(value.to_bits(), Ordering::Relaxed));
    }

    fn set_max_reduction_db(&self, value: f64) {
        self.update(|state| {
            state
                .max_reduction_db_bits
                .store(value.to_bits(), Ordering::Relaxed);
        });
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
    sidechain_highpass_enabled: bool,
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
            sidechain_highpass_enabled: true,
        }
    }
}

struct AtomicCompressorControlState {
    writer: Mutex<()>,
    seq: AtomicU64,
    enabled: AtomicBool,
    threshold_db_bits: AtomicU64,
    ratio_bits: AtomicU64,
    attack_ms_bits: AtomicU64,
    base_release_ms_bits: AtomicU64,
    makeup_gain_db_bits: AtomicU64,
    adaptive_release: AtomicBool,
    auto_makeup_enabled: AtomicBool,
    target_lufs_bits: AtomicU64,
    sidechain_highpass_enabled: AtomicBool,
}

impl AtomicCompressorControlState {
    fn new() -> Self {
        let state = CompressorControlState::new();
        Self {
            writer: Mutex::new(()),
            seq: AtomicU64::new(0),
            enabled: AtomicBool::new(state.enabled),
            threshold_db_bits: AtomicU64::new(state.threshold_db.to_bits()),
            ratio_bits: AtomicU64::new(state.ratio.to_bits()),
            attack_ms_bits: AtomicU64::new(state.attack_ms.to_bits()),
            base_release_ms_bits: AtomicU64::new(state.base_release_ms.to_bits()),
            makeup_gain_db_bits: AtomicU64::new(state.makeup_gain_db.to_bits()),
            adaptive_release: AtomicBool::new(state.adaptive_release),
            auto_makeup_enabled: AtomicBool::new(state.auto_makeup_enabled),
            target_lufs_bits: AtomicU64::new(state.target_lufs.to_bits()),
            sidechain_highpass_enabled: AtomicBool::new(state.sidechain_highpass_enabled),
        }
    }

    fn update<F>(&self, apply: F)
    where
        F: FnOnce(&Self),
    {
        locked_control_update(&self.writer, &self.seq, || apply(self));
    }

    fn snapshot(&self) -> Option<CompressorControlState> {
        stable_control_snapshot(&self.seq, || CompressorControlState {
                enabled: self.enabled.load(Ordering::Relaxed),
                threshold_db: f64::from_bits(self.threshold_db_bits.load(Ordering::Relaxed)),
                ratio: f64::from_bits(self.ratio_bits.load(Ordering::Relaxed)),
                attack_ms: f64::from_bits(self.attack_ms_bits.load(Ordering::Relaxed)),
                base_release_ms: f64::from_bits(
                    self.base_release_ms_bits.load(Ordering::Relaxed),
                ),
                makeup_gain_db: f64::from_bits(
                    self.makeup_gain_db_bits.load(Ordering::Relaxed),
                ),
                adaptive_release: self.adaptive_release.load(Ordering::Relaxed),
                auto_makeup_enabled: self.auto_makeup_enabled.load(Ordering::Relaxed),
                target_lufs: f64::from_bits(self.target_lufs_bits.load(Ordering::Relaxed)),
                sidechain_highpass_enabled: self
                    .sidechain_highpass_enabled
                    .load(Ordering::Relaxed),
            })
    }

    fn set_enabled(&self, enabled: bool) {
        self.update(|state| state.enabled.store(enabled, Ordering::Relaxed));
    }

    fn set_threshold_db(&self, value: f64) {
        self.update(|state| state.threshold_db_bits.store(value.to_bits(), Ordering::Relaxed));
    }

    fn set_ratio(&self, value: f64) {
        self.update(|state| state.ratio_bits.store(value.to_bits(), Ordering::Relaxed));
    }

    fn set_attack_ms(&self, value: f64) {
        self.update(|state| state.attack_ms_bits.store(value.to_bits(), Ordering::Relaxed));
    }

    fn set_base_release_ms(&self, value: f64) {
        self.update(|state| {
            state
                .base_release_ms_bits
                .store(value.to_bits(), Ordering::Relaxed);
        });
    }

    fn set_makeup_gain_db(&self, value: f64) {
        self.update(|state| {
            state
                .makeup_gain_db_bits
                .store(value.to_bits(), Ordering::Relaxed);
        });
    }

    fn set_adaptive_release(&self, enabled: bool) {
        self.update(|state| state.adaptive_release.store(enabled, Ordering::Relaxed));
    }

    fn set_auto_makeup_enabled(&self, enabled: bool) {
        self.update(|state| state.auto_makeup_enabled.store(enabled, Ordering::Relaxed));
    }

    fn set_target_lufs(&self, value: f64) {
        self.update(|state| state.target_lufs_bits.store(value.to_bits(), Ordering::Relaxed));
    }

    fn set_sidechain_highpass_enabled(&self, enabled: bool) {
        self.update(|state| {
            state
                .sidechain_highpass_enabled
                .store(enabled, Ordering::Relaxed);
        });
    }
}

#[derive(Clone)]
struct LimiterControlState {
    enabled: bool,
    ceiling_db: f64,
    release_ms: f64,
    careful_output_enabled: bool,
}

const CAREFUL_OUTPUT_CEILING_DB: f64 = -1.5;

impl LimiterControlState {
    fn new() -> Self {
        Self {
            enabled: true,
            ceiling_db: -0.5,
            release_ms: 50.0,
            careful_output_enabled: true,
        }
    }
}

struct AtomicLimiterControlState {
    writer: Mutex<()>,
    seq: AtomicU64,
    enabled: AtomicBool,
    ceiling_db_bits: AtomicU64,
    release_ms_bits: AtomicU64,
    careful_output_enabled: AtomicBool,
}

impl AtomicLimiterControlState {
    fn new() -> Self {
        let state = LimiterControlState::new();
        Self {
            writer: Mutex::new(()),
            seq: AtomicU64::new(0),
            enabled: AtomicBool::new(state.enabled),
            ceiling_db_bits: AtomicU64::new(state.ceiling_db.to_bits()),
            release_ms_bits: AtomicU64::new(state.release_ms.to_bits()),
            careful_output_enabled: AtomicBool::new(state.careful_output_enabled),
        }
    }

    fn update<F>(&self, apply: F)
    where
        F: FnOnce(&Self),
    {
        locked_control_update(&self.writer, &self.seq, || apply(self));
    }

    fn snapshot(&self) -> Option<LimiterControlState> {
        stable_control_snapshot(&self.seq, || LimiterControlState {
            enabled: self.enabled.load(Ordering::Relaxed),
            ceiling_db: f64::from_bits(self.ceiling_db_bits.load(Ordering::Relaxed)),
            release_ms: f64::from_bits(self.release_ms_bits.load(Ordering::Relaxed)),
            careful_output_enabled: self.careful_output_enabled.load(Ordering::Relaxed),
        })
    }

    fn set_enabled(&self, enabled: bool) {
        self.update(|state| state.enabled.store(enabled, Ordering::Relaxed));
    }

    fn set_ceiling_db(&self, value: f64) {
        self.update(|state| state.ceiling_db_bits.store(value.to_bits(), Ordering::Relaxed));
    }

    fn set_release_ms(&self, value: f64) {
        self.update(|state| state.release_ms_bits.store(value.to_bits(), Ordering::Relaxed));
    }

    fn set_careful_output_enabled(&self, enabled: bool) {
        self.update(|state| {
            state
                .careful_output_enabled
                .store(enabled, Ordering::Relaxed);
        });
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
    compressor.set_sidechain_highpass_enabled(control.sidechain_highpass_enabled);
}

fn effective_limiter_ceiling_db(ceiling_db: f64, careful_output_enabled: bool) -> f64 {
    if careful_output_enabled {
        ceiling_db.min(CAREFUL_OUTPUT_CEILING_DB)
    } else {
        ceiling_db
    }
}

fn apply_limiter_control(limiter: &mut Limiter, control: &LimiterControlState) {
    limiter.set_ceiling(effective_limiter_ceiling_db(
        control.ceiling_db,
        control.careful_output_enabled,
    ));
    limiter.set_release_time(control.release_ms);
    limiter.set_enabled(control.enabled);
}
