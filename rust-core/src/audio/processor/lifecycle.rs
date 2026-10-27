//! Starting and stopping the audio streams and the DSP thread.

use super::control::SuppressorControlState;
#[cfg(windows)]
use super::dsp_loop::tolerate_scheduling_failure;
use super::dsp_loop::{
    ensure_resampler_scratch_capacity, record_dsp_thread_panic, startup_control, DspLoop,
    DspLoopSetup,
};
use super::resampling::{build_sinc_resampler, duration_samples};
#[cfg(feature = "vad")]
use super::VAD_WORKER_MAX_BUFFER_SAMPLES;
use super::{
    AudioProcessor, DSP_THREAD_READY_TIMEOUT_MS, MAX_OUTPUT_PROBE_SECONDS, MAX_RECORDING_SECONDS,
    OUTPUT_PRIME_MS, RESAMPLER_CHUNK_SIZE, RT_OUTPUT_SCRATCH_CAPACITY, RT_PROCESS_BUFFER_CAPACITY,
};
use crate::audio::buffer::{AudioConsumer, AudioProducer, AudioRingBuffer};
use crate::audio::clock::now_micros;
use crate::audio::input::{AudioInput, InputStreamOptions};
use crate::audio::output::{AudioOutput, OutputProbeControl, OutputUnderruns};
#[cfg(windows)]
use crate::audio::scheduling::ProAudioThread;
use crate::dsp::noise_suppressor::{
    new_realtime_noise_suppression_engine as new_noise_suppression_engine,
    ControlledNoiseSuppressionEngine as NoiseSuppressionEngine,
};
use ringbuf::traits::Consumer;
use rubato::SincFixedIn;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::MutexGuard;
use std::sync::{mpsc, Arc};
use std::time::Duration;

/// Stream and worker-thread handles. `AudioProcessor::streams` is the lock that
/// serializes start, stop and recovery.
#[derive(Default)]
pub(super) struct Streams {
    input: Option<AudioInput>,
    output: Option<AudioOutput>,
    /// The DSP thread.
    pub(super) process_thread: Option<std::thread::JoinHandle<()>>,
    /// Watchdog for callback-based recovery detection.
    pub(super) supervisor_thread: Option<std::thread::JoinHandle<()>>,
    /// Non-realtime VAD worker.
    #[cfg(feature = "vad")]
    pub(super) vad_worker_thread: Option<std::thread::JoinHandle<()>>,
}

/// What an open stream reports about its endpoint.
pub(super) struct OpenStream {
    pub(super) endpoint_id: Option<String>,
    pub(super) sample_rate: u32,
}

/// Resampler between a device rate and the engine rate, if they differ.
type DeviceResampler = Option<SincFixedIn<f64>>;

/// The DSP thread and the channel on which it reports readiness.
type SpawnedDsp = (
    std::thread::JoinHandle<()>,
    mpsc::Receiver<Result<(), String>>,
);

/// The capture side of a start: the stream and the ring it feeds.
struct OpenedInput {
    stream: AudioInput,
    ring: AudioConsumer,
    /// Dropped-sample counter of the capture ring.
    dropped: Arc<AtomicU64>,
    sample_rate: u32,
}

/// The playback side of a start: the stream and its primed ring.
struct OpenedOutput {
    stream: AudioOutput,
    producer: AudioProducer,
    sample_rate: u32,
}

/// The devices the streams are bound to. It is kept after `stop` so recovery
/// restarts on the same devices; the `*_stream` entries exist only while the
/// stream is open.
#[derive(Default)]
pub(super) struct ActiveDevices {
    pub(super) input_name: Option<String>,
    /// Zero-based occurrence of the selected input friendly name.
    pub(super) input_name_ordinal: u32,
    /// Stable platform endpoint ID of the selected input, when available.
    pub(super) input_endpoint_id: Option<String>,
    pub(super) input_stream: Option<OpenStream>,
    pub(super) output_name: Option<String>,
    pub(super) output_name_ordinal: u32,
    pub(super) output_endpoint_id: Option<String>,
    pub(super) output_stream: Option<OpenStream>,
}

impl AudioProcessor {
    /// Take the lifecycle lock. A panic in an earlier start must not brick later
    /// calls, so a poisoned lock is used as is.
    pub(super) fn lock_streams(&self) -> MutexGuard<'_, Streams> {
        self.streams
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub(super) fn lock_devices(&self) -> MutexGuard<'_, ActiveDevices> {
        self.devices
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub(super) fn drain_retired_suppressors(&self) {
        if let Ok(mut rx_guard) = self.shared.suppressor.retired_rx.lock() {
            if let Some(rx) = rx_guard.as_mut() {
                while let Some(_engine) = rx.try_pop() {}
            }
        }
    }

    fn check_dsp_startup(
        &self,
        streams: &mut Streams,
        readiness: Result<Result<(), String>, mpsc::RecvTimeoutError>,
    ) -> Result<(), String> {
        let result = readiness
            .unwrap_or_else(|error| Err(format!("DSP thread failed to become ready: {error}")));
        if result.is_err() {
            self.stop_locked(streams);
        }
        result
    }

    /// Start audio processing
    ///
    /// # Arguments
    /// * `input_device` - Input device name (None for default)
    /// * `output_device` - Output device name (None for default)
    pub fn start(
        &self,
        input_device: Option<&str>,
        output_device: Option<&str>,
    ) -> Result<String, String> {
        self.start_with_device_ordinals_and_ids(input_device, 0, None, output_device, 0, None)
    }

    /// Start audio processing using stable platform endpoint IDs when present.
    ///
    /// The friendly name and ordinal remain a compatibility fallback for
    /// platforms without stable endpoint IDs.
    #[allow(clippy::too_many_arguments)]
    pub fn start_with_device_ordinals_and_ids(
        &self,
        input_device: Option<&str>,
        input_device_name_ordinal: u32,
        input_device_endpoint_id: Option<&str>,
        output_device: Option<&str>,
        output_device_name_ordinal: u32,
        output_device_endpoint_id: Option<&str>,
    ) -> Result<String, String> {
        let mut streams = self.lock_streams();
        self.start_locked(
            &mut streams,
            input_device,
            input_device_name_ordinal,
            input_device_endpoint_id,
            output_device,
            output_device_name_ordinal,
            output_device_endpoint_id,
        )
    }

    /// `start_with_device_ordinals_and_ids` with the lifecycle lock already held.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn start_locked(
        &self,
        streams: &mut Streams,
        input_device: Option<&str>,
        input_device_name_ordinal: u32,
        input_device_endpoint_id: Option<&str>,
        output_device: Option<&str>,
        output_device_name_ordinal: u32,
        output_device_endpoint_id: Option<&str>,
    ) -> Result<String, String> {
        // A redundant start must not discard a pending restart or the back-off.
        if self.shared.stream.running.load(Ordering::SeqCst) {
            return Err("Already running".to_string());
        }
        self.ensure_supervisor(streams);
        self.shared
            .stream
            .restart_requested
            .store(false, Ordering::Release);
        if !self.shared.stream.recovering.load(Ordering::Acquire) {
            self.shared
                .stream
                .restart_backoff_index
                .store(0, Ordering::Release);
        }

        self.start_with_cleanup(streams, |processor, streams| {
            processor.start_streams_unguarded(
                streams,
                input_device,
                input_device_name_ordinal,
                input_device_endpoint_id,
                output_device,
                output_device_name_ordinal,
                output_device_endpoint_id,
            )
        })
    }

    /// Run a start attempt and, if it fails, leave the same clean stopped state
    /// whatever step failed, instead of the half-built attempt's published
    /// values (buffer sizes, recording ring, probe producer, streams).
    pub(super) fn start_with_cleanup(
        &self,
        streams: &mut Streams,
        attempt: impl FnOnce(&Self, &mut Streams) -> Result<String, String>,
    ) -> Result<String, String> {
        let previous_output_rate = self.shared.output.sample_rate.load(Ordering::Relaxed);
        let started = attempt(self, streams);
        if started.is_err() {
            self.stop_locked(streams);
            self.shared
                .output
                .sample_rate
                .store(previous_output_rate, Ordering::Relaxed);
        }
        started
    }

    #[allow(clippy::too_many_arguments)]
    fn start_streams_unguarded(
        &self,
        streams: &mut Streams,
        input_device: Option<&str>,
        input_device_name_ordinal: u32,
        input_device_endpoint_id: Option<&str>,
        output_device: Option<&str>,
        output_device_name_ordinal: u32,
        output_device_endpoint_id: Option<&str>,
    ) -> Result<String, String> {
        let recording_producer = self.reset_recording_tap();
        self.shared.reset_for_start();

        let input = self.open_input(
            input_device,
            input_device_name_ordinal,
            input_device_endpoint_id,
        )?;
        let output = self.open_output(
            output_device,
            output_device_name_ordinal,
            output_device_endpoint_id,
        )?;

        let input_device_name = input.stream.device_info().name.clone();
        let output_device_name = output.stream.device_info().name.clone();
        {
            let mut devices = self.lock_devices();
            devices.input_name = Some(input_device_name.clone());
            devices.input_name_ordinal = input_device_name_ordinal;
            // Bind recovery to the endpoints actually opened, including a user's
            // initial default-device choice, even if another same-name device appears.
            devices.input_endpoint_id = input
                .stream
                .device_info()
                .endpoint_id
                .clone()
                .or_else(|| input_device_endpoint_id.map(ToOwned::to_owned));
            devices.input_stream = Some(OpenStream {
                endpoint_id: input.stream.device_info().endpoint_id.clone(),
                sample_rate: input.stream.device_info().sample_rate,
            });
            devices.output_name = Some(output_device_name.clone());
            devices.output_name_ordinal = output_device_name_ordinal;
            devices.output_endpoint_id = output
                .stream
                .device_info()
                .endpoint_id
                .clone()
                .or_else(|| output_device_endpoint_id.map(ToOwned::to_owned));
            devices.output_stream = Some(OpenStream {
                endpoint_id: output.stream.device_info().endpoint_id.clone(),
                sample_rate: output.stream.device_info().sample_rate,
            });
        }
        let input_sample_rate_for_thread = input.sample_rate;
        let output_sample_rate_for_thread = output.sample_rate;
        streams.input = Some(input.stream);
        streams.output = Some(output.stream);

        let (input_resampler, output_resampler) =
            self.build_resamplers(input_sample_rate_for_thread, output_sample_rate_for_thread)?;

        #[cfg(feature = "vad")]
        let (vad_queue, vad_worker_consumer) =
            AudioRingBuffer::new(VAD_WORKER_MAX_BUFFER_SAMPLES).split();
        #[cfg(feature = "vad")]
        self.ensure_vad_worker(streams, vad_worker_consumer);

        let (suppressor_rx, retired_suppressor_tx) = self.shared.suppressor.open_queues();
        let (suppressor_initial_control, suppressor_initial_engine) = self.initial_suppressor();

        // Start processing thread
        self.mark_running();

        let setup = DspLoopSetup {
            shared: Arc::clone(&self.shared),
            sample_rate: self.sample_rate,
            input_sample_rate: input_sample_rate_for_thread,
            output_sample_rate: output_sample_rate_for_thread,
            input: input.ring,
            input_dropped: input.dropped,
            output: output.producer,
            recording: recording_producer,
            #[cfg(feature = "vad")]
            vad_queue,
            input_resampler,
            output_resampler,
            suppressor: suppressor_initial_engine,
            suppressor_control: suppressor_initial_control,
            suppressor_commands: suppressor_rx,
            suppressor_retire: retired_suppressor_tx,
        };

        let (handle, dsp_ready_rx) = Self::spawn_dsp_thread(setup)?;
        streams.process_thread = Some(handle);
        self.check_dsp_startup(
            streams,
            dsp_ready_rx.recv_timeout(Duration::from_millis(DSP_THREAD_READY_TIMEOUT_MS)),
        )?;

        // A model selected after the first engine was built found the stream
        // stopped and queued nothing; the stream is marked running now, so
        // settle it here. A failure undoes the selection and is counted in the
        // diagnostics; it does not fail the start. If the selection also saw the
        // running stream, both queue an engine and the loop swaps both in.
        self.settle_selection_after_start(self.engine_builder());

        if let Some(output) = streams.output.as_ref() {
            if let Err(e) = output.start() {
                self.stop_locked(streams);
                return Err(format!("Failed to start output stream: {}", e));
            }
        }

        if let Some(input) = streams.input.as_ref() {
            if let Err(e) = input.start() {
                self.stop_locked(streams);
                return Err(format!("Failed to start input stream: {}", e));
            }
        }

        Ok(format!(
            "Started: {} -> {}",
            input_device_name, output_device_name
        ))
    }

    /// Clear stale capture state (an explicit output mute survives restarts) and
    /// give the raw recording tap a fresh ring. Its capacity is fixed so the DSP
    /// thread can write without locking during steady-state capture.
    pub(super) fn reset_recording_tap(&self) -> AudioProducer {
        self.shared.recording.clear_capture();
        if let Ok(mut consumer_guard) = self.shared.recording.consumer.lock() {
            *consumer_guard = None;
        }

        let recording_rb = AudioRingBuffer::new(self.sample_rate as usize * MAX_RECORDING_SECONDS);
        let (recording_producer, recording_consumer) = recording_rb.split();
        recording_producer.reset_dropped_count();
        if let Ok(mut consumer_guard) = self.shared.recording.consumer.lock() {
            *consumer_guard = Some(recording_consumer);
        }
        recording_producer
    }

    /// Open the capture stream (not yet started) and the ring it feeds.
    fn open_input(
        &self,
        device: Option<&str>,
        name_ordinal: u32,
        endpoint_id: Option<&str>,
    ) -> Result<OpenedInput, String> {
        let last_callback_time_us = Arc::clone(&self.shared.input.callback_time_us);
        let callback_error_count = Arc::clone(&self.shared.input.callback_error_count);
        let rt_error_code = Arc::clone(&self.shared.dsp.rt_error_code);
        let options = InputStreamOptions::new(
            Arc::clone(&self.shared.input.channel_mode),
            Arc::clone(&self.shared.input.stereo_correlation),
            Arc::clone(&self.shared.input.phase_warning_count),
            Arc::clone(&self.shared.input.phase_rescue_strategy),
            Arc::clone(&self.shared.input.phase_estimated_delay_samples),
            Arc::clone(&self.shared.input.phase_polarity_flipped),
        );
        let setup = match device {
            Some(name) => {
                AudioInput::from_named_device_identity_setup(name, name_ordinal, endpoint_id)
            }
            None => AudioInput::from_default_device_setup(),
        }
        .map_err(|e| format!("Failed to start audio input: {}", e))?;
        let sample_rate = setup.device_info.sample_rate;

        let (producer, ring, dropped) = self.input_ring(sample_rate);
        let stream = AudioInput::from_setup(
            setup,
            producer,
            last_callback_time_us,
            callback_error_count,
            rt_error_code,
            Arc::clone(&self.shared.stream.restart_requested),
            options,
        )
        .map_err(|e| format!("Failed to start audio input: {}", e))?;

        self.shared
            .input
            .fixed_buffer_frames
            .store(stream.fixed_buffer_frames().unwrap_or(0), Ordering::Relaxed);
        Ok(OpenedInput {
            stream,
            ring,
            dropped,
            sample_rate,
        })
    }

    /// Open the playback stream (not yet started) with its probe tap, and prime
    /// its ring with silence.
    fn open_output(
        &self,
        device: Option<&str>,
        name_ordinal: u32,
        endpoint_id: Option<&str>,
    ) -> Result<OpenedOutput, String> {
        let setup = match device {
            Some(name) => {
                AudioOutput::from_named_device_identity_setup(name, name_ordinal, endpoint_id)
            }
            None => AudioOutput::from_default_device_setup(),
        }
        .map_err(|e| format!("Failed to resolve audio output: {}", e))?;

        let sample_rate = setup.device_info.sample_rate;
        let (output_producer, output_consumer) = self.output_ring(sample_rate);
        let output_probe_rb = AudioRingBuffer::new(sample_rate as usize * MAX_OUTPUT_PROBE_SECONDS);
        let (output_probe_producer, output_probe_consumer) = output_probe_rb.split();
        {
            let mut producer = self
                .shared
                .output
                .probe_producer
                .lock()
                .map_err(|_| "Failed to initialize output probe buffer".to_string())?;
            *producer = Some(output_probe_producer);
        }
        self.shared.output.reset_probe();
        let output_probe = OutputProbeControl {
            consumer: output_probe_consumer,
            active: Arc::clone(&self.shared.output.probe_active),
            complete: Arc::clone(&self.shared.output.probe_complete),
            cancel_requested: Arc::clone(&self.shared.output.probe_cancel_requested),
        };

        let recording_active = Arc::clone(&self.shared.recording.active);
        let output_muted = Arc::clone(&self.shared.output.muted);
        let last_output_callback_time_us = Arc::clone(&self.shared.output.callback_time_us);
        let output_underruns = OutputUnderruns {
            streak: Arc::clone(&self.shared.output.underrun_streak),
            total: Arc::clone(&self.shared.output.underrun_total),
            priming_total: Arc::clone(&self.shared.output.priming_underrun_total),
            audio_queued_at_us: Arc::clone(&self.shared.output.last_write_time_us),
            priming_started_at_us: Arc::clone(&self.shared.output.priming_started_at_us),
            primed: Arc::clone(&self.shared.output.primed),
        };
        let output_callback_error_count = Arc::clone(&self.shared.output.callback_error_count);
        let rt_error_code_for_output = Arc::clone(&self.shared.dsp.rt_error_code);
        let stream = AudioOutput::from_setup(
            setup,
            output_consumer,
            recording_active,
            output_muted,
            last_output_callback_time_us,
            output_underruns,
            output_callback_error_count,
            rt_error_code_for_output,
            Arc::clone(&self.shared.stream.restart_requested),
            Some(output_probe),
        )
        .map_err(|e| format!("Failed to start audio output: {}", e))?;
        self.shared
            .output
            .fixed_buffer_frames
            .store(stream.fixed_buffer_frames().unwrap_or(0), Ordering::Relaxed);
        Ok(OpenedOutput {
            stream,
            producer: output_producer,
            sample_rate,
        })
    }

    /// The capture ring and its dropped-sample counter. The ring carries
    /// device-rate samples, so its 2 second headroom and the backlog thresholds
    /// scale with the device rate, not the engine rate.
    pub(super) fn input_ring(
        &self,
        sample_rate: u32,
    ) -> (AudioProducer, AudioConsumer, Arc<AtomicU64>) {
        let input_rb = AudioRingBuffer::new(sample_rate as usize * 2);
        let (producer, ring) = input_rb.split();
        let dropped = producer.dropped_counter();
        self.shared.input.set_dropped_counter(Arc::clone(&dropped));
        (producer, ring, dropped)
    }

    /// The playback ring, sized for the playback device rate and primed with
    /// silence. The prime goes straight to the producer: it neither marks real
    /// audio as queued nor primes the stream.
    pub(super) fn output_ring(&self, sample_rate: u32) -> (AudioProducer, AudioConsumer) {
        self.shared
            .output
            .sample_rate
            .store(sample_rate, Ordering::Relaxed);
        let output_rb = AudioRingBuffer::new(sample_rate as usize * 2);
        let (mut output_producer, output_consumer) = output_rb.split();
        let output_prime_samples = duration_samples(sample_rate, OUTPUT_PRIME_MS);
        let silence = vec![0.0f32; output_prime_samples];
        output_producer.write(&silence);
        self.shared
            .output
            .buffer_len
            .store(output_prime_samples as u32, Ordering::Relaxed);
        (output_producer, output_consumer)
    }

    /// The first suppressor engine and its control state. Built on the calling
    /// thread: a DLL or model load must not run on the DSP thread, where a slow
    /// one would outlast the readiness timeout and stop() would then block
    /// joining it.
    pub(super) fn initial_suppressor(&self) -> (SuppressorControlState, NoiseSuppressionEngine) {
        let control = startup_control(
            self.shared.suppressor.rt_control.snapshot(),
            SuppressorControlState::new,
            &self.shared.suppressor.dirty,
            &self.shared.dsp.lock_contention_count,
        );
        let engine = new_noise_suppression_engine(
            control.model,
            Arc::clone(&self.shared.suppressor.strength),
        );
        // Before the stream is marked running: a model selected from now on has
        // to reach the loop as a queued engine, and one selected earlier has.
        self.shared
            .suppressor
            .engine_model
            .store(control.model as u8, Ordering::Release);
        (control, engine)
    }

    /// Publish a run as started, right before its DSP loop exists.
    pub(super) fn mark_running(&self) {
        self.shared.stream.running.store(true, Ordering::SeqCst);
        self.shared
            .stream
            .last_start_time_us
            .store(now_micros(), Ordering::Release);
    }

    /// Sinc resamplers between each device rate and the engine rate; `None` for
    /// a side that already runs at the engine rate.
    pub(super) fn build_resamplers(
        &self,
        input_sample_rate: u32,
        output_sample_rate: u32,
    ) -> Result<(DeviceResampler, DeviceResampler), String> {
        let build = |side: &str, from: u32, to: u32| -> Result<DeviceResampler, String> {
            if from == to {
                return Ok(None);
            }
            build_sinc_resampler(from, to, RESAMPLER_CHUNK_SIZE)
                .map(Some)
                .map_err(|e| {
                    format!("Failed to initialize {side} resampler ({from} Hz -> {to} Hz): {e}")
                })
        };
        let input_resampler = build("input", input_sample_rate, self.sample_rate)?;
        let output_resampler = build("output", self.sample_rate, output_sample_rate)?;
        if let Some(resampler) = input_resampler.as_ref() {
            ensure_resampler_scratch_capacity(resampler, RT_PROCESS_BUFFER_CAPACITY, "input")?;
        }
        if let Some(resampler) = output_resampler.as_ref() {
            ensure_resampler_scratch_capacity(resampler, RT_OUTPUT_SCRATCH_CAPACITY, "output")?;
        }
        self.shared
            .input
            .resampler_active
            .store(input_resampler.is_some(), Ordering::Relaxed);
        self.shared
            .output
            .resampler_active
            .store(output_resampler.is_some(), Ordering::Relaxed);
        Ok((input_resampler, output_resampler))
    }

    /// Spawn the DSP thread. The returned channel reports readiness once every
    /// worker buffer and DSP state is initialized.
    fn spawn_dsp_thread(setup: DspLoopSetup) -> Result<SpawnedDsp, String> {
        let (dsp_ready_tx, dsp_ready_rx) = mpsc::channel();
        let spawned = std::thread::Builder::new().spawn(move || {
            #[cfg(windows)]
            let shared = Arc::clone(&setup.shared);
            let dsp_loop = DspLoop::new(setup);
            // Register only after every worker buffer and DSP state is initialized.
            // A missing or disabled MMCSS "Pro Audio" task only costs scheduling
            // priority, so it degrades to normal priority and is reported.
            #[cfg(windows)]
            let scheduling = tolerate_scheduling_failure(
                ProAudioThread::enter(),
                &shared.dsp.scheduling_degraded,
            );
            let _ = dsp_ready_tx.send(Ok(()));
            dsp_loop.run();
            #[cfg(windows)]
            if let Some(Err(error)) = scheduling.map(ProAudioThread::finish) {
                eprintln!("Failed to revert DSP thread Pro Audio scheduling: {error}");
            }
        });
        let handle = spawned.map_err(|e| format!("Failed to spawn DSP thread: {e}"))?;
        Ok((handle, dsp_ready_rx))
    }

    /// Stop audio processing
    pub fn stop(&self) {
        let mut streams = self.lock_streams();
        self.stop_locked(&mut streams);
    }

    /// `stop` with the lifecycle lock already held.
    pub(super) fn stop_locked(&self, streams: &mut Streams) {
        self.shared.stream.stop_epoch.fetch_add(1, Ordering::AcqRel);
        self.shared.stream.running.store(false, Ordering::SeqCst);
        self.shared.output.reset_probe();
        self.shared
            .input
            .fixed_buffer_frames
            .store(0, Ordering::Relaxed);
        self.shared
            .output
            .fixed_buffer_frames
            .store(0, Ordering::Relaxed);
        if let Ok(mut producer) = self.shared.output.probe_producer.lock() {
            *producer = None;
        }
        self.shared
            .stream
            .last_start_time_us
            .store(0, Ordering::Release);

        if let Some(handle) = streams.process_thread.take() {
            if let Err(payload) = handle.join() {
                record_dsp_thread_panic(
                    payload.as_ref(),
                    &self.shared.stream.recovery_state,
                    false,
                );
            }
        }
        #[cfg(feature = "vad")]
        self.stop_vad_worker(streams);

        if let Some(input) = &streams.input {
            let _ = input.pause();
        }

        if let Some(output) = &streams.output {
            let _ = output.pause();
        }

        streams.input = None;
        streams.output = None;
        {
            let mut devices = self.lock_devices();
            devices.input_stream = None;
            devices.output_stream = None;
        }

        // Clear the restart request only after the DSP thread has been joined and
        // the streams dropped: either can still raise one while stopping, and a
        // stale request would restart a processor the user just stopped.
        self.shared
            .stream
            .restart_requested
            .store(false, Ordering::Release);
        if let Ok(mut state) = self.shared.stream.recovery_state.lock() {
            state.reason_pending = false;
        }
        self.shared
            .stream
            .recovering
            .store(false, Ordering::Release);

        // Clear capture state without changing the caller's output mute.
        self.shared.recording.clear_capture();
        self.shared
            .suppressor
            .latency_samples
            .store(0, Ordering::Relaxed);
        self.shared
            .input
            .resampler_active
            .store(false, Ordering::Relaxed);
        self.shared
            .output
            .resampler_active
            .store(false, Ordering::Relaxed);
        if let Ok(mut consumer_guard) = self.shared.recording.consumer.lock() {
            *consumer_guard = None;
        }
        if let Ok(mut tx) = self.shared.suppressor.pending_tx.lock() {
            *tx = None;
        }
        if let Ok(mut rx) = self.shared.suppressor.retired_rx.lock() {
            *rx = None;
        }

        // Reset DSP state so stop/start can recover from stuck envelopes.
        if let Ok(mut e) = self.shared.eq.correction_shadow.lock() {
            e.reset();
        }
        if let Ok(mut e) = self.shared.eq.tone_shadow.lock() {
            e.reset();
        }
        if let Ok(mut c) = self.shared.compressor.shadow.lock() {
            c.reset();
        }
        if let Ok(mut d) = self.shared.deesser.shadow.lock() {
            d.reset();
        }
        if let Ok(mut l) = self.shared.limiter.shadow.lock() {
            l.reset();
        }
    }
}

#[cfg(test)]
mod startup_tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    #[test]
    fn failed_dsp_readiness_stops_joins_and_clears_capture_state() {
        for failure in ["registration", "disconnected", "timeout"] {
            let processor = AudioProcessor::new();
            processor
                .shared
                .stream
                .running
                .store(true, Ordering::SeqCst);
            processor
                .shared
                .recording
                .active
                .store(true, Ordering::Release);
            processor
                .shared
                .output
                .probe_active
                .store(true, Ordering::Release);
            processor
                .shared
                .input
                .fixed_buffer_frames
                .store(480, Ordering::Relaxed);
            processor
                .shared
                .stream
                .restart_requested
                .store(true, Ordering::Release);
            let shared = Arc::clone(&processor.shared);
            let exited = Arc::new(AtomicBool::new(false));
            let worker_exited = Arc::clone(&exited);
            processor.lock_streams().process_thread = Some(std::thread::spawn(move || {
                while shared.stream.running.load(Ordering::SeqCst) {
                    std::thread::yield_now();
                }
                worker_exited.store(true, Ordering::Release);
            }));
            let (ready_tx, ready_rx) = mpsc::channel();
            match failure {
                "registration" => ready_tx
                    .send(Err("MMCSS test startup failure".into()))
                    .unwrap(),
                "disconnected" => drop(ready_tx),
                _ => {}
            }
            let error = processor
                .check_dsp_startup(
                    &mut processor.lock_streams(),
                    ready_rx.recv_timeout(Duration::ZERO),
                )
                .unwrap_err();
            if failure == "registration" {
                assert_eq!(error, "MMCSS test startup failure");
            } else {
                assert!(error.starts_with("DSP thread failed to become ready:"));
            }
            assert!(!processor.shared.stream.running.load(Ordering::SeqCst));
            assert!(exited.load(Ordering::Acquire));
            assert!(processor.lock_streams().process_thread.is_none());
            assert!(!processor.shared.recording.active.load(Ordering::Acquire));
            assert!(!processor.shared.output.probe_active.load(Ordering::Acquire));
            assert_eq!(
                processor
                    .shared
                    .input
                    .fixed_buffer_frames
                    .load(Ordering::Relaxed),
                0
            );
            assert!(!processor
                .shared
                .stream
                .restart_requested
                .load(Ordering::Acquire));
            assert!(processor.lock_streams().input.is_none());
            assert!(processor.lock_streams().output.is_none());
        }
    }

    #[test]
    fn every_start_re_enters_output_priming_before_the_stream_exists() {
        // Restart and device-change recovery both go through this start function.
        let source = include_str!("lifecycle.rs");
        let source = &source[..source
            .find(
                "#[cfg(test)]
mod startup_tests",
            )
            .unwrap()];
        let start = source.find("fn start_streams_unguarded(").unwrap();
        let start = &source[start..];
        let counters = include_str!("shared.rs");
        let counters = &counters[counters.find("fn reset_for_start(").unwrap()..];
        let counters: String = counters.split_whitespace().collect();
        for reset in [
            "self.output.last_write_time_us.store(0,Ordering::Relaxed);",
            "self.output.primed.store(false,Ordering::Relaxed);",
            "self.output.priming_started_at_us",
        ] {
            assert!(counters.contains(reset), "{reset}");
        }
        let reset = start.find("self.shared.reset_for_start();").unwrap();
        let build = start.find("AudioOutput::from_setup(").unwrap();
        assert!(reset < build);
        // The silent prime goes straight to the producer: it must neither mark
        // real audio as queued nor prime the stream.
        let prime = &start[start.find("let output_prime_samples").unwrap()..];
        let prime = &prime[..prime.find("\n    }\n").unwrap()];
        assert!(prime.contains("output_producer.write(&silence)"));
        assert!(!prime.contains("last_write_time_us") && !prime.contains(".primed"));
        // Recovery and a normal start run the same function.
        let recovery = include_str!("recovery.rs");
        assert!(recovery.contains("processor.start_locked("));
        assert!(source.contains(
            "self.start_locked(
            &mut streams,"
        ));
    }

    #[test]
    fn startup_failure_check_precedes_both_stream_starts() {
        let source = include_str!("lifecycle.rs");
        let source = &source[..source
            .find(
                "#[cfg(test)]
mod startup_tests",
            )
            .unwrap()];
        let worker_published = source
            .rfind("streams.process_thread = Some(handle);")
            .unwrap();
        let startup = &source[worker_published..];
        assert!(
            startup.find("self.check_dsp_startup(").unwrap()
                < startup.find("output.start()").unwrap()
        );
        assert!(startup.find("output.start()").unwrap() < startup.find("input.start()").unwrap());
    }
}
