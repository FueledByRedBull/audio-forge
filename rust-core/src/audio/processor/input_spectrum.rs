//! Input analyser: the spectrum of the microphone input, for display behind the
//! EQ curve. The DSP loop copies each input block into a bounded queue
//! (`enqueue_input_spectrum_block`); a worker thread turns the queue into
//! log-spaced bands that readers take from atomics. Nothing here feeds back
//! into the audio path.

use super::lifecycle::Streams;
use super::shared::InputSpectrumShared;
use super::{AudioProcessor, RT_PROCESS_BUFFER_CAPACITY};
use crate::audio::buffer::AudioConsumer;
use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

/// Bands of the published spectrum.
pub const INPUT_SPECTRUM_BIN_COUNT: usize = 96;
/// Lowest and highest band edges, in Hz.
const LOWEST_EDGE_HZ: f64 = 20.0;
const HIGHEST_EDGE_HZ: f64 = 20_000.0;
/// Every band reads within this range, in dBFS.
pub(super) const INPUT_SPECTRUM_FLOOR_DB: f32 = -120.0;
const CEILING_DB: f32 = 0.0;
const FFT_SIZE: usize = 2048;
/// A new frame every half window (50% overlap).
const HOP: usize = FFT_SIZE / 2;
/// A band falls at most this fast: 20 dB per 300 ms. It rises at once.
const RELEASE_DB_PER_SECOND: f32 = 20.0 / 0.3;
/// Equivalent noise bandwidth of the Hann window, in FFT bins.
const HANN_NOISE_BANDWIDTH_BINS: f32 = 1.5;
/// Queue between the DSP loop and the worker: two of the largest blocks the
/// loop hands on (about 340 ms at 48 kHz).
pub(super) const INPUT_SPECTRUM_QUEUE_SAMPLES: usize = 2 * RT_PROCESS_BUFFER_CAPACITY;
/// Input samples are clamped to this magnitude, well above full scale.
const MAX_INPUT_MAGNITUDE: f32 = 16.0;
/// How long the worker sleeps when the queue is empty.
const WORKER_IDLE_SLEEP: Duration = Duration::from_millis(5);

/// Centre frequency of every band, in Hz (geometric centre of its edges).
pub fn input_spectrum_bin_frequencies() -> [f32; INPUT_SPECTRUM_BIN_COUNT] {
    std::array::from_fn(|band| band_edge_hz(band as f64 + 0.5) as f32)
}

/// Edge `position` of the log-spaced bands; 0 is the lowest edge and
/// `INPUT_SPECTRUM_BIN_COUNT` the highest.
fn band_edge_hz(position: f64) -> f64 {
    LOWEST_EDGE_HZ
        * (HIGHEST_EDGE_HZ / LOWEST_EDGE_HZ).powf(position / INPUT_SPECTRUM_BIN_COUNT as f64)
}

/// Where a band takes its power from.
#[derive(Clone, Copy, Debug, PartialEq)]
enum BandSource {
    /// The FFT bins `start..end` have their centres inside the band.
    Bins { start: usize, end: usize },
    /// No bin centre lies inside the band: interpolate between bin `lower`
    /// and the next one at the band's centre.
    Interpolated { lower: usize, fraction: f32 },
}

fn band_sources(sample_rate: u32) -> [BandSource; INPUT_SPECTRUM_BIN_COUNT] {
    let bin_hz = f64::from(sample_rate) / FFT_SIZE as f64;
    let nyquist_bin = FFT_SIZE / 2;
    std::array::from_fn(|band| {
        let bin_at = |hz: f64| ((hz / bin_hz).ceil() as usize).min(nyquist_bin + 1);
        let start = bin_at(band_edge_hz(band as f64));
        let end = bin_at(band_edge_hz(band as f64 + 1.0));
        if start < end {
            return BandSource::Bins { start, end };
        }
        let position = (band_edge_hz(band as f64 + 0.5) / bin_hz).min(nyquist_bin as f64);
        let lower = (position.floor() as usize).min(nyquist_bin - 1);
        BandSource::Interpolated {
            lower,
            fraction: (position - lower as f64) as f32,
        }
    })
}

/// The worker's analysis state. `new` allocates everything; `push` does not.
pub(super) struct SpectrumAnalyser {
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    window_sum: f32,
    /// The latest `FFT_SIZE` input samples; new ones fill the upper `HOP`.
    history: Vec<f32>,
    /// New samples in the upper `HOP` of `history`.
    filled: usize,
    /// The lower half of `history` holds input, not start-up zeros.
    primed: bool,
    fft_input: Vec<f32>,
    fft_output: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
    /// Per-bin power, relative to a full-scale sine.
    power: Vec<f32>,
    bands: [BandSource; INPUT_SPECTRUM_BIN_COUNT],
    /// Turns `|X|^2` into power relative to a full-scale sine centred on a bin.
    power_scale: f32,
    release_db_per_frame: f32,
    display_db: [f32; INPUT_SPECTRUM_BIN_COUNT],
}

impl SpectrumAnalyser {
    pub(super) fn new(sample_rate: u32) -> Self {
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(FFT_SIZE);
        let window: Vec<f32> = (0..FFT_SIZE)
            .map(|n| {
                let phase = std::f64::consts::TAU * n as f64 / FFT_SIZE as f64;
                (0.5 - 0.5 * phase.cos()) as f32
            })
            .collect();
        let window_sum: f32 = window.iter().sum();
        let amplitude_scale = 2.0 / window_sum;
        Self {
            fft_input: fft.make_input_vec(),
            fft_output: fft.make_output_vec(),
            scratch: fft.make_scratch_vec(),
            fft,
            window,
            window_sum,
            history: vec![0.0; FFT_SIZE],
            filled: 0,
            primed: false,
            power: vec![0.0; FFT_SIZE / 2 + 1],
            bands: band_sources(sample_rate),
            power_scale: amplitude_scale * amplitude_scale,
            release_db_per_frame: RELEASE_DB_PER_SECOND * HOP as f32 / sample_rate as f32,
            display_db: [INPUT_SPECTRUM_FLOOR_DB; INPUT_SPECTRUM_BIN_COUNT],
        }
    }

    /// Take input samples; `publish` gets the bands of every frame they
    /// complete. The first frame comes once a whole window has arrived.
    pub(super) fn push(
        &mut self,
        mut samples: &[f32],
        mut publish: impl FnMut(&[f32; INPUT_SPECTRUM_BIN_COUNT]),
    ) {
        while !samples.is_empty() {
            let take = samples.len().min(HOP - self.filled);
            let at = FFT_SIZE - HOP + self.filled;
            for (slot, &sample) in self.history[at..at + take].iter_mut().zip(samples) {
                // The tap sits before the loop's sanitizer. A finite but huge
                // sample would overflow the mean and the FFT and read as
                // silence; clamped, a clipping input reads as loud.
                *slot = if sample.is_finite() {
                    sample.clamp(-MAX_INPUT_MAGNITUDE, MAX_INPUT_MAGNITUDE)
                } else {
                    0.0
                };
            }
            self.filled += take;
            samples = &samples[take..];
            if self.filled < HOP {
                break;
            }
            if self.primed {
                self.analyse();
                publish(&self.display_db);
            }
            self.history.copy_within(HOP.., 0);
            self.filled = 0;
            self.primed = true;
        }
    }

    fn analyse(&mut self) {
        // The chain's pre-filter removes DC; do not show an offset as 20 Hz
        // content. The window-weighted mean is the DC the FFT sees, so removing
        // it leaves a tone that ends mid-cycle in the frame untouched.
        let mean = self
            .history
            .iter()
            .zip(&self.window)
            .map(|(sample, weight)| sample * weight)
            .sum::<f32>()
            / self.window_sum;
        for ((input, &sample), &weight) in self
            .fft_input
            .iter_mut()
            .zip(&self.history)
            .zip(&self.window)
        {
            *input = (sample - mean) * weight;
        }
        self.fft
            .process_with_scratch(&mut self.fft_input, &mut self.fft_output, &mut self.scratch)
            .expect("only a length mismatch fails, and `new` sized every buffer from the plan");
        for (power, bin) in self.power.iter_mut().zip(&self.fft_output) {
            *power = bin.norm_sqr() * self.power_scale;
        }
        for (display, source) in self.display_db.iter_mut().zip(&self.bands) {
            let power = match *source {
                // Band power: a sine reads its level whatever the band's width.
                // A one-bin band reads that bin, so a sine on it reads 0 dB too.
                BandSource::Bins { start, end } => {
                    let sum: f32 = self.power[start..end].iter().sum();
                    sum / ((end - start) as f32).min(HANN_NOISE_BANDWIDTH_BINS)
                }
                BandSource::Interpolated { lower, fraction } => {
                    self.power[lower] * (1.0 - fraction) + self.power[lower + 1] * fraction
                }
            };
            let level =
                (10.0 * power.max(1.0e-20).log10()).clamp(INPUT_SPECTRUM_FLOOR_DB, CEILING_DB);
            *display = if level >= *display {
                level
            } else {
                (*display - self.release_db_per_frame).max(level)
            };
        }
    }
}

impl InputSpectrumShared {
    fn publish(&self, bands: &[f32; INPUT_SPECTRUM_BIN_COUNT]) {
        for (slot, value) in self.bands_db.iter().zip(bands) {
            slot.store(value.to_bits(), Ordering::Relaxed);
        }
        self.sequence.fetch_add(1, Ordering::Release);
        self.has_frame.store(true, Ordering::Release);
    }

    /// No spectrum: every band at the floor, no frame to report.
    pub(super) fn clear(&self) {
        self.has_frame.store(false, Ordering::Release);
        for slot in &self.bands_db {
            slot.store(INPUT_SPECTRUM_FLOOR_DB.to_bits(), Ordering::Relaxed);
        }
    }
}

/// Run the worker until it is stopped. A panic ends it, clears the display
/// and is reported through `worker_failed`; the audio goes on.
fn run_input_spectrum_worker(
    spectrum: &InputSpectrumShared,
    consumer: AudioConsumer,
    sample_rate: u32,
) {
    let analysed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        analyse_input_spectrum(spectrum, consumer, sample_rate)
    }));
    if analysed.is_err() {
        spectrum.worker_failed.store(true, Ordering::Release);
        spectrum.clear();
    }
}

fn analyse_input_spectrum(
    spectrum: &InputSpectrumShared,
    mut consumer: AudioConsumer,
    sample_rate: u32,
) {
    let mut analyser = SpectrumAnalyser::new(sample_rate);
    let mut chunk = vec![0.0_f32; HOP];
    while spectrum.worker_running.load(Ordering::Acquire) {
        #[cfg(test)]
        if spectrum.inject_worker_panic.load(Ordering::Acquire) {
            panic!("injected input analyser failure");
        }
        let read = consumer.read(&mut chunk);
        if read == 0 {
            std::thread::sleep(WORKER_IDLE_SLEEP);
            continue;
        }
        analyser.push(&chunk[..read], |bands| spectrum.publish(bands));
    }
}

impl AudioProcessor {
    /// Start the analyser worker on the queue the DSP loop feeds. It only
    /// feeds the display, so a failed spawn does not stop the audio: it is
    /// reported through `worker_failed`, and the tap counts every block it
    /// then drops.
    pub(super) fn start_input_spectrum_worker(
        &self,
        streams: &mut Streams,
        consumer: AudioConsumer,
    ) {
        let spectrum = &self.shared.input_spectrum;
        spectrum.worker_running.store(true, Ordering::Release);
        match self.spawn_input_spectrum_worker(consumer) {
            Ok(handle) => streams.input_spectrum_thread = Some(handle),
            Err(_) => {
                spectrum.worker_running.store(false, Ordering::Release);
                spectrum.worker_failed.store(true, Ordering::Release);
            }
        }
    }

    fn spawn_input_spectrum_worker(
        &self,
        consumer: AudioConsumer,
    ) -> std::io::Result<std::thread::JoinHandle<()>> {
        #[cfg(test)]
        if self
            .shared
            .input_spectrum
            .inject_spawn_failure
            .load(Ordering::Acquire)
        {
            return Err(std::io::Error::other("injected spawn failure"));
        }
        let shared = Arc::clone(&self.shared);
        let sample_rate = self.sample_rate;
        std::thread::Builder::new()
            .spawn(move || run_input_spectrum_worker(&shared.input_spectrum, consumer, sample_rate))
    }

    /// Stop the analyser worker and wait for it to exit.
    pub(super) fn stop_input_spectrum_worker(&self, streams: &mut Streams) {
        self.shared
            .input_spectrum
            .worker_running
            .store(false, Ordering::Release);
        if let Some(handle) = streams.input_spectrum_thread.take() {
            // The worker catches its own panics; one that still escapes only
            // loses the display, so it is reported and the stop goes on.
            if handle.join().is_err() {
                self.shared
                    .input_spectrum
                    .worker_failed
                    .store(true, Ordering::Release);
            }
        }
    }

    /// The analyser worker could not start, or panicked, this run.
    pub fn get_input_spectrum_worker_failed(&self) -> bool {
        self.shared
            .input_spectrum
            .worker_failed
            .load(Ordering::Acquire)
    }

    /// Copy the latest input spectrum (dBFS per band, see
    /// `input_spectrum_bin_frequencies`) into `out`. Returns the frame's
    /// sequence number, which rises with every frame and across restarts, or
    /// 0 (with every band at the floor) before the first frame of a run and
    /// while stopped. Bands of one call may come from two adjacent frames.
    pub fn read_input_spectrum(&self, out: &mut [f32; INPUT_SPECTRUM_BIN_COUNT]) -> u64 {
        let spectrum = &self.shared.input_spectrum;
        if !spectrum.has_frame.load(Ordering::Acquire) {
            out.fill(INPUT_SPECTRUM_FLOOR_DB);
            return 0;
        }
        let sequence = spectrum.sequence.load(Ordering::Acquire);
        for (value, slot) in out.iter_mut().zip(&spectrum.bands_db) {
            *value = f32::from_bits(slot.load(Ordering::Relaxed));
        }
        sequence
    }

    /// Input blocks the analyser queue dropped because its worker fell behind.
    pub fn get_input_spectrum_dropped_blocks(&self) -> u64 {
        self.shared
            .input_spectrum
            .dropped_blocks
            .load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::buffer::AudioRingBuffer;
    use crate::audio::processor::dsp_loop::enqueue_input_spectrum_block;
    use std::sync::atomic::AtomicU64;

    fn sine(sample_rate: u32, frequency: f32, amplitude: f32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|n| {
                // f64 phase: an f32 one this large would add its own noise floor.
                let phase = std::f64::consts::TAU * f64::from(frequency) * n as f64
                    / f64::from(sample_rate);
                amplitude * phase.sin() as f32
            })
            .collect()
    }

    /// The bands of the last frame `samples` complete.
    fn last_frame(sample_rate: u32, samples: &[f32]) -> [f32; INPUT_SPECTRUM_BIN_COUNT] {
        let mut analyser = SpectrumAnalyser::new(sample_rate);
        let mut last = None;
        // Odd chunk sizes, like the loop's blocks.
        for chunk in samples.chunks(480) {
            analyser.push(chunk, |bands| last = Some(*bands));
        }
        last.expect("at least one frame")
    }

    fn band_of(frequency: f32) -> usize {
        let edges = (0..=INPUT_SPECTRUM_BIN_COUNT).map(|edge| band_edge_hz(edge as f64) as f32);
        edges
            .collect::<Vec<_>>()
            .windows(2)
            .position(|pair| pair[0] <= frequency && frequency < pair[1])
            .unwrap()
    }

    #[test]
    fn bin_frequencies_rise_from_20_hz_to_20_khz() {
        let frequencies = input_spectrum_bin_frequencies();
        assert!(frequencies.windows(2).all(|pair| pair[0] < pair[1]));
        assert!((20.0..22.0).contains(&frequencies[0]), "{}", frequencies[0]);
        let last = frequencies[INPUT_SPECTRUM_BIN_COUNT - 1];
        assert!((18_000.0..20_000.0).contains(&last), "{last}");
    }

    #[test]
    fn every_band_maps_inside_the_spectrum_at_both_rates() {
        for rate in [44_100, 48_000] {
            for source in band_sources(rate) {
                match source {
                    BandSource::Bins { start, end } => {
                        assert!(start < end && end <= FFT_SIZE / 2 + 1)
                    }
                    BandSource::Interpolated { lower, fraction } => {
                        assert!(lower < FFT_SIZE / 2 && (0.0..=1.0).contains(&fraction))
                    }
                }
            }
            // Low bands are narrower than a bin; high ones span many.
            assert!(matches!(
                band_sources(rate)[0],
                BandSource::Interpolated { .. }
            ));
            assert!(matches!(
                band_sources(rate)[INPUT_SPECTRUM_BIN_COUNT - 1],
                BandSource::Bins { start, end } if end - start > 10
            ));
        }
    }

    #[test]
    fn a_sine_reads_its_level_in_its_band_at_48_and_44_1_khz() {
        for rate in [48_000, 44_100] {
            for (frequency, amplitude, expected_db) in [
                (100.0, 1.0, 0.0),
                (1_000.0, 1.0, 0.0),
                (1_000.0, 0.1, -20.0),
                (8_000.0, 0.5, -6.02),
            ] {
                let bands = last_frame(rate, &sine(rate, frequency, amplitude, rate as usize));
                let band = band_of(frequency);
                let peak = (0..INPUT_SPECTRUM_BIN_COUNT)
                    .max_by(|&a, &b| bands[a].total_cmp(&bands[b]))
                    .unwrap();
                assert!(
                    peak.abs_diff(band) <= 1,
                    "{rate} Hz rate, {frequency} Hz: peak in band {peak}, expected {band}"
                );
                assert!(
                    (bands[peak] - expected_db).abs() < 2.0,
                    "{rate} Hz rate, {frequency} Hz: {} dB, expected {expected_db}",
                    bands[peak]
                );
                // Far from the tone the bands sit well below it.
                let far = if band > 48 { 10 } else { 85 };
                assert!(
                    bands[far] < expected_db - 60.0,
                    "band {far}: {}",
                    bands[far]
                );
            }
        }
    }

    #[test]
    fn silence_reads_at_the_floor() {
        let bands = last_frame(48_000, &vec![0.0; 48_000]);
        assert!(bands.iter().all(|&band| band == INPUT_SPECTRUM_FLOOR_DB));
        // A DC offset is not shown as low-frequency content.
        let bands = last_frame(48_000, &vec![0.25; 48_000]);
        assert!(bands.iter().all(|&band| band == INPUT_SPECTRUM_FLOOR_DB));
    }

    #[test]
    fn non_finite_input_never_reaches_the_bands() {
        let mut samples = sine(48_000, 1_000.0, 0.5, 48_000);
        samples[30_000] = f32::NAN;
        samples[30_001] = f32::INFINITY;
        let bands = last_frame(48_000, &samples);
        assert!(bands
            .iter()
            .all(|band| (INPUT_SPECTRUM_FLOOR_DB..=CEILING_DB).contains(band)));
    }

    #[test]
    fn a_band_rises_at_once_and_falls_20_db_in_about_300_ms() {
        let rate = 48_000;
        let mut analyser = SpectrumAnalyser::new(rate);
        let mut frames = Vec::new();
        let band = band_of(1_000.0);
        analyser.push(&sine(rate, 1_000.0, 1.0, FFT_SIZE * 2), |bands| {
            frames.push(bands[band])
        });
        assert!(frames.last().unwrap() > &-2.0, "{frames:?}");
        frames.clear();
        // 300 ms of silence after the tone.
        analyser.push(&vec![0.0; rate as usize * 3 / 10 + FFT_SIZE], |bands| {
            frames.push(bands[band])
        });
        let per_frame = RELEASE_DB_PER_SECOND * HOP as f32 / rate as f32;
        assert!(frames
            .windows(2)
            .all(|pair| (pair[0] - pair[1] - per_frame).abs() < 1.0e-3
                || pair[1] == INPUT_SPECTRUM_FLOOR_DB));
        let frames_in_300_ms = (0.3 * rate as f32 / HOP as f32).round() as usize;
        let fallen = frames[0] - frames[frames_in_300_ms];
        assert!((fallen - 20.0).abs() < 1.5, "fell {fallen} dB in 300 ms");
    }

    #[test]
    fn the_first_frame_waits_for_a_whole_window() {
        let mut analyser = SpectrumAnalyser::new(48_000);
        let mut frames = 0;
        analyser.push(&vec![0.1; FFT_SIZE - 1], |_| frames += 1);
        assert_eq!(frames, 0);
        analyser.push(&[0.1], |_| frames += 1);
        assert_eq!(frames, 1);
        analyser.push(&vec![0.1; HOP], |_| frames += 1);
        assert_eq!(frames, 2);
    }

    #[test]
    fn a_full_queue_drops_whole_blocks_and_counts_them() {
        let (mut producer, mut consumer) = AudioRingBuffer::new(1_000).split();
        let dropped = AtomicU64::new(0);
        let block = [0.5_f32; 480];
        enqueue_input_spectrum_block(&mut producer, &block, &dropped);
        enqueue_input_spectrum_block(&mut producer, &block, &dropped);
        assert_eq!(dropped.load(Ordering::Relaxed), 0);
        // 960 of 1000 used: the next block does not fit and is dropped whole.
        enqueue_input_spectrum_block(&mut producer, &block, &dropped);
        enqueue_input_spectrum_block(&mut producer, &block, &dropped);
        assert_eq!(dropped.load(Ordering::Relaxed), 2);
        assert_eq!(consumer.len(), 960);
        let mut drained = [0.0_f32; 1_000];
        assert_eq!(consumer.read(&mut drained), 960);
        enqueue_input_spectrum_block(&mut producer, &block, &dropped);
        assert_eq!(consumer.len(), 480);
        assert_eq!(dropped.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn the_loop_taps_once_before_the_paths_split_and_before_any_processing() {
        let region =
            crate::rt_scan::marked_region(include_str!("dsp_loop.rs"), "dsp_processing_loop");
        assert_eq!(region.matches("enqueue_input_spectrum_block(").count(), 1);
        let block =
            crate::rt_scan::source_between(region, "fn process_block(", "fn record_raw_tap(");
        let tap = block.find("enqueue_input_spectrum_block(").unwrap();
        for later in [
            "select_processing_path(",
            "sanitize_and_clamp_input_inplace(",
            "apply_input_pre_filter(",
        ] {
            assert!(
                tap < block.find(later).unwrap(),
                "the tap must precede {later}"
            );
        }
    }

    #[test]
    fn the_tap_does_not_allocate() {
        let (mut producer, _consumer) = AudioRingBuffer::new(1_000).split();
        let dropped = AtomicU64::new(0);
        let block = [0.5_f32; 480];
        crate::test_alloc::assert_no_allocations("input spectrum tap", || {
            for _ in 0..4 {
                enqueue_input_spectrum_block(&mut producer, &block, &dropped);
            }
        });
        assert_eq!(dropped.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn the_worker_publishes_frames_then_stops_joins_and_clears_on_stop() {
        let processor = AudioProcessor::new();
        let (mut producer, consumer) = AudioRingBuffer::new(INPUT_SPECTRUM_QUEUE_SAMPLES).split();
        let mut bands = [0.0_f32; INPUT_SPECTRUM_BIN_COUNT];
        assert_eq!(processor.read_input_spectrum(&mut bands), 0);
        let mut streams = Streams::default();
        processor.start_input_spectrum_worker(&mut streams, consumer);
        assert!(!processor.get_input_spectrum_worker_failed());
        let tone = sine(48_000, 1_000.0, 0.5, 4 * FFT_SIZE);
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        let mut fed = 0;
        let sequence = loop {
            if fed < tone.len() {
                fed += producer.write(&tone[fed..]);
            }
            let sequence = processor.read_input_spectrum(&mut bands);
            if sequence >= 3 {
                break sequence;
            }
            assert!(std::time::Instant::now() < deadline, "no frames published");
            std::thread::sleep(Duration::from_millis(2));
        };
        assert!(sequence >= 3);
        assert!(bands[band_of(1_000.0)] > -10.0);

        processor.stop_input_spectrum_worker(&mut streams);
        assert!(streams.input_spectrum_thread.is_none());
        // `stop` clears the published values once the worker is gone.
        processor.shared.clear_run_values();
        assert_eq!(processor.read_input_spectrum(&mut bands), 0);
        assert!(bands.iter().all(|&band| band == INPUT_SPECTRUM_FLOOR_DB));
        // The worker is gone: nothing publishes any more.
        producer.write(&tone);
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(processor.read_input_spectrum(&mut bands), 0);
    }

    #[test]
    fn a_huge_finite_input_reads_as_loud_not_as_silence() {
        // A broken driver's samples near f32::MAX would overflow the mean
        // and the FFT unclamped.
        let samples = sine(48_000, 1_000.0, f32::MAX * 0.9, 48_000);
        let bands = last_frame(48_000, &samples);
        assert!(bands.iter().all(|band| band.is_finite()));
        assert_eq!(bands[band_of(1_000.0)], CEILING_DB);
    }

    #[test]
    fn a_panicking_worker_is_reported_and_clears_the_display() {
        let processor = AudioProcessor::new();
        let (mut producer, consumer) = AudioRingBuffer::new(INPUT_SPECTRUM_QUEUE_SAMPLES).split();
        let mut streams = Streams::default();
        processor.start_input_spectrum_worker(&mut streams, consumer);
        let mut bands = [0.0_f32; INPUT_SPECTRUM_BIN_COUNT];
        let tone = sine(48_000, 1_000.0, 0.5, 4 * FFT_SIZE);
        producer.write(&tone);
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while processor.read_input_spectrum(&mut bands) == 0 {
            assert!(std::time::Instant::now() < deadline, "no frames published");
            std::thread::sleep(Duration::from_millis(2));
        }
        assert!(!processor.get_input_spectrum_worker_failed());

        processor
            .shared
            .input_spectrum
            .inject_worker_panic
            .store(true, Ordering::Release);
        while !processor.get_input_spectrum_worker_failed() {
            assert!(std::time::Instant::now() < deadline, "panic not reported");
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(processor.read_input_spectrum(&mut bands), 0);
        assert!(bands.iter().all(|&band| band == INPUT_SPECTRUM_FLOOR_DB));
        // The stop still joins the worker, and the next run starts clean.
        processor.stop_input_spectrum_worker(&mut streams);
        assert!(streams.input_spectrum_thread.is_none());
        assert!(processor.get_input_spectrum_worker_failed());
        processor.shared.reset_for_start();
        assert!(!processor.get_input_spectrum_worker_failed());
    }

    #[test]
    fn a_failed_spawn_is_reported_and_the_tap_counts_its_drops() {
        let processor = AudioProcessor::new();
        processor
            .shared
            .input_spectrum
            .inject_spawn_failure
            .store(true, Ordering::Release);
        let (mut producer, consumer) = AudioRingBuffer::new(INPUT_SPECTRUM_QUEUE_SAMPLES).split();
        let mut streams = Streams::default();
        processor.start_input_spectrum_worker(&mut streams, consumer);
        assert!(streams.input_spectrum_thread.is_none());
        assert!(processor.get_input_spectrum_worker_failed());
        assert!(!processor
            .shared
            .input_spectrum
            .worker_running
            .load(Ordering::Acquire));

        let dropped = &processor.shared.input_spectrum.dropped_blocks;
        let block = [0.5_f32; 480];
        for _ in 0..INPUT_SPECTRUM_QUEUE_SAMPLES / block.len() + 4 {
            enqueue_input_spectrum_block(&mut producer, &block, dropped);
        }
        assert!(processor.get_input_spectrum_dropped_blocks() >= 4);
        processor.stop_input_spectrum_worker(&mut streams);
    }
}
