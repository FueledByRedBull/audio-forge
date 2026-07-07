impl AudioProcessor {
    fn drain_retired_suppressors(&self) {
        if let Ok(mut rx_guard) = self.retired_suppressor_rx.lock() {
            if let Some(rx) = rx_guard.as_mut() {
                while let Some(_engine) = rx.pop() {}
            }
        }
    }

    /// Start audio processing
    ///
    /// # Arguments
    /// * `input_device` - Input device name (None for default)
    /// * `output_device` - Output device name (None for default)
    pub fn start(
        &mut self,
        input_device: Option<&str>,
        output_device: Option<&str>,
    ) -> Result<String, String> {
        self.ensure_supervisor();
        self.restart_requested.store(false, Ordering::Release);
        if !self.recovering.load(Ordering::Acquire) {
            self.restart_backoff_index.store(0, Ordering::Release);
        }

        if self.running.load(Ordering::SeqCst) {
            return Err("Already running".to_string());
        }

        // Ensure any stale recording/mute state is cleared before starting.
        self.recording_active.store(false, Ordering::Release);
        self.output_muted.store(false, Ordering::Release);
        self.raw_recording_pos.store(0, Ordering::Release);
        self.raw_recording_target.store(0, Ordering::Release);
        self.recording_level_db
            .store((-120.0_f32).to_bits(), Ordering::Relaxed);
        if let Ok(mut consumer_guard) = self.raw_recording_consumer.lock() {
            *consumer_guard = None;
        }

        // Create input ring buffer (2 seconds capacity)
        let input_rb = AudioRingBuffer::new(self.sample_rate as usize * 2);
        let (input_producer, input_consumer) = input_rb.split();

        // Create a dedicated raw recording tap buffer. Capacity is fixed so the DSP thread
        // can write without locking during steady-state capture.
        let recording_rb = AudioRingBuffer::new(self.sample_rate as usize * MAX_RECORDING_SECONDS);
        let (mut recording_producer, recording_consumer) = recording_rb.split();
        recording_producer.reset_dropped_count();
        if let Ok(mut consumer_guard) = self.raw_recording_consumer.lock() {
            *consumer_guard = Some(recording_consumer);
        }

        // Capture the dropped counter before producer is moved
        let input_dropped_counter = input_producer.dropped_counter();
        self.input_dropped = input_dropped_counter;

        // Reset the dropped counter at start
        self.input_dropped.store(0, Ordering::Relaxed);
        self.output_underrun_streak.store(0, Ordering::Relaxed);
        self.output_underrun_total.store(0, Ordering::Relaxed);
        self.jitter_dropped_samples.store(0, Ordering::Relaxed);
        self.output_retime_adjustment_count
            .store(0, Ordering::Relaxed);
        self.output_recovery_event_count.store(0, Ordering::Relaxed);
        self.output_short_write_dropped_samples
            .store(0, Ordering::Relaxed);
        self.dsp_idle_wakeup_count.store(0, Ordering::Relaxed);
        self.dsp_idle_sleep_us
            .store(PROCESS_IDLE_SLEEP_US, Ordering::Relaxed);
        self.lock_contention_count.store(0, Ordering::Relaxed);
        self.suppressor_non_finite_count.store(0, Ordering::Relaxed);
        self.input_backlog_recovery_count
            .store(0, Ordering::Relaxed);
        self.input_backlog_dropped_samples
            .store(0, Ordering::Relaxed);
        self.clip_event_count.store(0, Ordering::Relaxed);
        self.clip_peak_db
            .store((-120.0_f32).to_bits(), Ordering::Relaxed);
        self.rt_error_code
            .store(RtErrorCode::None as u32, Ordering::Relaxed);
        self.input_callback_error_count.store(0, Ordering::Relaxed);
        self.output_callback_error_count.store(0, Ordering::Relaxed);
        self.rt_buffer_overflow_count.store(0, Ordering::Relaxed);
        self.last_input_callback_time_us.store(0, Ordering::Relaxed);
        self.last_output_callback_time_us
            .store(0, Ordering::Relaxed);
        self.last_output_write_time.store(0, Ordering::Relaxed);

        // Start audio input
        let last_input_callback_time_us = Arc::clone(&self.last_input_callback_time_us);
        let input_callback_error_count = Arc::clone(&self.input_callback_error_count);
        let rt_error_code_for_input = Arc::clone(&self.rt_error_code);
        let input = match input_device {
            Some(name) => AudioInput::from_device_name(
                name,
                input_producer,
                last_input_callback_time_us,
                input_callback_error_count,
                rt_error_code_for_input,
            ),
            None => AudioInput::from_default_device(
                input_producer,
                last_input_callback_time_us,
                input_callback_error_count,
                rt_error_code_for_input,
            ),
        }
        .map_err(|e| format!("Failed to start audio input: {}", e))?;

        let input_device_name = input.device_info().name.clone();
        let input_sample_rate_for_thread = input.device_info().sample_rate;

        let output_setup = match output_device {
            Some(name) => AudioOutput::from_named_device_setup(name),
            None => AudioOutput::from_default_device_setup(),
        };
        let output_setup = match output_setup {
            Ok(setup) => setup,
            Err(e) => {
                return Err(format!("Failed to resolve audio output: {}", e));
            }
        };

        let output_device_name = output_setup.device_info.name.clone();
        let output_sample_rate_for_thread = output_setup.device_info.sample_rate;
        self.output_sample_rate
            .store(output_sample_rate_for_thread, Ordering::Relaxed);

        // Create output ring buffer sized for the actual playback device rate.
        let output_rb = AudioRingBuffer::new(output_sample_rate_for_thread as usize * 2);
        let (output_producer, output_consumer) = output_rb.split();

        // Start audio output
        let recording_active = Arc::clone(&self.recording_active);
        let output_muted = Arc::clone(&self.output_muted);
        let last_output_callback_time_us = Arc::clone(&self.last_output_callback_time_us);
        let output_underrun_streak = Arc::clone(&self.output_underrun_streak);
        let output_underrun_total = Arc::clone(&self.output_underrun_total);
        let output_callback_error_count = Arc::clone(&self.output_callback_error_count);
        let rt_error_code_for_output = Arc::clone(&self.rt_error_code);
        let output_result = AudioOutput::from_setup(
            output_setup,
            output_consumer,
            recording_active.clone(),
            output_muted.clone(),
            last_output_callback_time_us,
            output_underrun_streak,
            output_underrun_total,
            output_callback_error_count,
            rt_error_code_for_output,
        );
        let output = match output_result {
            Ok(output) => output,
            Err(e) => {
                return Err(format!("Failed to start audio output: {}", e));
            }
        };

        let output_prime_samples = duration_samples(output_sample_rate_for_thread, OUTPUT_PRIME_MS);
        let mut prod = output_producer;
        let silence = vec![0.0f32; output_prime_samples];
        prod.write(&silence);
        self.output_buffer_len
            .store(output_prime_samples as u32, Ordering::Relaxed);
        let output_producer = prod;

        self.input_device_name = Some(input_device_name.clone());
        self.output_device_name = Some(output_device_name.clone());
        self.audio_input = Some(input);
        self.audio_output = Some(output);

        const RESAMPLER_CHUNK_SIZE: usize = 1024;
        let input_resampler = if input_sample_rate_for_thread != self.sample_rate {
            Some(
                build_sinc_resampler(
                    input_sample_rate_for_thread,
                    self.sample_rate,
                    RESAMPLER_CHUNK_SIZE,
                )
                .map_err(|e| {
                    self.stop();
                    format!(
                        "Failed to initialize input resampler ({} Hz -> {} Hz): {}",
                        input_sample_rate_for_thread, self.sample_rate, e
                    )
                })?,
            )
        } else {
            None
        };
        let output_resampler = if output_sample_rate_for_thread != self.sample_rate {
            Some(
                build_sinc_resampler(
                    self.sample_rate,
                    output_sample_rate_for_thread,
                    RESAMPLER_CHUNK_SIZE,
                )
                .map_err(|e| {
                    self.stop();
                    format!(
                        "Failed to initialize output resampler ({} Hz -> {} Hz): {}",
                        self.sample_rate, output_sample_rate_for_thread, e
                    )
                })?,
            )
        } else {
            None
        };
        self.input_resampler_active
            .store(input_resampler.is_some(), Ordering::Relaxed);
        self.output_resampler_active
            .store(output_resampler.is_some(), Ordering::Relaxed);

        #[cfg(feature = "vad")]
        let (mut vad_worker_producer, vad_worker_consumer) =
            AudioRingBuffer::new(VAD_WORKER_MAX_BUFFER_SAMPLES).split();
        #[cfg(feature = "vad")]
        self.ensure_vad_worker(vad_worker_consumer);

        let suppressor_queue =
            RtCommandQueue::<NoiseSuppressionEngine, RT_SUPPRESSOR_COMMAND_CAPACITY>::new();
        let (suppressor_tx, mut suppressor_rx) = suppressor_queue.split();
        if let Ok(mut tx) = self.pending_suppressor_tx.lock() {
            *tx = Some(suppressor_tx);
        }
        let retired_suppressor_queue =
            RtCommandQueue::<NoiseSuppressionEngine, RT_SUPPRESSOR_RETIRE_CAPACITY>::new();
        let (mut retired_suppressor_tx, retired_suppressor_rx) = retired_suppressor_queue.split();
        if let Ok(mut rx) = self.retired_suppressor_rx.lock() {
            *rx = Some(retired_suppressor_rx);
        }

        // Start processing thread
        self.running.store(true, Ordering::SeqCst);
        self.last_start_time_us
            .store(now_micros(), Ordering::Release);

        let gate_enabled = Arc::clone(&self.gate_enabled);
        let gate_rt_control = Arc::clone(&self.gate_rt_control);
        let gate_dirty = Arc::clone(&self.gate_dirty);
        let suppressor_enabled = Arc::clone(&self.suppressor_enabled);
        let suppressor_rt_control = Arc::clone(&self.suppressor_rt_control);
        let suppressor_dirty = Arc::clone(&self.suppressor_dirty);
        let suppressor_reset_requested = Arc::clone(&self.suppressor_reset_requested);
        let eq_enabled = Arc::clone(&self.eq_enabled);
        let eq_control = Arc::clone(&self.eq_control);
        let eq_dirty = Arc::clone(&self.eq_dirty);
        let compressor_enabled = Arc::clone(&self.compressor_enabled);
        let compressor_rt_control = Arc::clone(&self.compressor_rt_control);
        let compressor_dirty = Arc::clone(&self.compressor_dirty);
        let deesser_enabled = Arc::clone(&self.deesser_enabled);
        let deesser_rt_control = Arc::clone(&self.deesser_rt_control);
        let deesser_dirty = Arc::clone(&self.deesser_dirty);
        let limiter_enabled = Arc::clone(&self.limiter_enabled);
        let limiter_rt_control = Arc::clone(&self.limiter_rt_control);
        let limiter_dirty = Arc::clone(&self.limiter_dirty);
        let mut output_producer = output_producer;
        let running = Arc::clone(&self.running);
        let bypass = Arc::clone(&self.bypass);
        let raw_monitor_enabled = Arc::clone(&self.raw_monitor_enabled);

        // Clone metering atomics
        let input_peak = Arc::clone(&self.input_peak);
        let input_rms = Arc::clone(&self.input_rms);
        let output_peak = Arc::clone(&self.output_peak);
        let output_rms = Arc::clone(&self.output_rms);
        let compressor_gain_reduction = Arc::clone(&self.compressor_gain_reduction);
        let deesser_gain_reduction = Arc::clone(&self.deesser_gain_reduction);
        let gate_gain_meter = Arc::clone(&self.gate_gain_meter);
        #[cfg(feature = "vad")]
        let vad_probability = Arc::clone(&self.vad_probability);
        #[cfg(feature = "vad")]
        let gate_noise_floor_db = Arc::clone(&self.gate_noise_floor_db);
        #[cfg(feature = "vad")]
        let gate_fused_score = Arc::clone(&self.gate_fused_score);
        #[cfg(feature = "vad")]
        let vad_available = Arc::clone(&self.vad_available);
        #[cfg(feature = "vad")]
        let vad_last_update_us = Arc::clone(&self.vad_last_update_us);
        let compressor_current_release_ms = Arc::clone(&self.compressor_current_release_ms);
        let compressor_current_lufs = Arc::clone(&self.compressor_current_lufs);
        let compressor_current_makeup_gain = Arc::clone(&self.compressor_current_makeup_gain);
        let latency_us = Arc::clone(&self.latency_us);
        let latency_compensation_us = Arc::clone(&self.latency_compensation_us);
        let sample_rate_for_latency = self.sample_rate;
        let output_sample_rate_for_latency = output_sample_rate_for_thread;

        // Clone DSP performance metric atomics
        let dsp_time_us = Arc::clone(&self.dsp_time_us);
        let input_buffer_len = Arc::clone(&self.input_buffer_len);
        let smoothed_input_buffer_len = Arc::clone(&self.smoothed_input_buffer_len);
        let output_buffer_len = Arc::clone(&self.output_buffer_len);
        let suppressor_buffer_len = Arc::clone(&self.suppressor_buffer_len);
        let suppressor_latency_samples = Arc::clone(&self.suppressor_latency_samples);
        let last_output_write_time = Arc::clone(&self.last_output_write_time);
        let last_input_callback_time_us_for_dsp = Arc::clone(&self.last_input_callback_time_us);
        let dsp_time_smoothed_us = Arc::clone(&self.dsp_time_smoothed_us);
        let smoothed_buffer_len = Arc::clone(&self.smoothed_buffer_len);
        let jitter_dropped_samples = Arc::clone(&self.jitter_dropped_samples);
        let output_retime_adjustment_count = Arc::clone(&self.output_retime_adjustment_count);
        let output_recovery_event_count = Arc::clone(&self.output_recovery_event_count);
        let output_short_write_dropped_samples =
            Arc::clone(&self.output_short_write_dropped_samples);
        let dsp_idle_wakeup_count = Arc::clone(&self.dsp_idle_wakeup_count);
        let dsp_idle_sleep_us = Arc::clone(&self.dsp_idle_sleep_us);
        let suppressor_non_finite_count = Arc::clone(&self.suppressor_non_finite_count);
        let input_backlog_recovery_count = Arc::clone(&self.input_backlog_recovery_count);
        let input_backlog_dropped_samples = Arc::clone(&self.input_backlog_dropped_samples);
        let clip_event_count = Arc::clone(&self.clip_event_count);
        let clip_peak_db = Arc::clone(&self.clip_peak_db);
        let noise_backend_available = Arc::clone(&self.noise_backend_available);
        let noise_backend_failed = Arc::clone(&self.noise_backend_failed);
        let rt_error_code = Arc::clone(&self.rt_error_code);
        let rt_buffer_overflow_count = Arc::clone(&self.rt_buffer_overflow_count);
        let suppressor_strength = Arc::clone(&self.suppressor_strength);
        let recording_active_thread = Arc::clone(&recording_active);

        // Clone raw recording buffer atomics
        let raw_recording_pos = Arc::clone(&self.raw_recording_pos);
        let raw_recording_target = Arc::clone(&self.raw_recording_target);
        let recording_level_db = Arc::clone(&self.recording_level_db);

        let (dsp_ready_tx, dsp_ready_rx) = mpsc::channel();
        let handle = std::thread::spawn(move || {
            let mut consumer = input_consumer;
            let mut input_buffer = FixedAudioBuffer::<f32, RT_INPUT_CHUNK_CAPACITY>::new();
            let mut temp_buffer = FixedAudioBuffer::<f32, RT_PROCESS_BUFFER_CAPACITY>::new();
            let mut rnnoise_output = FixedAudioBuffer::<f32, RT_SUPPRESSOR_OUTPUT_CAPACITY>::new();
            let mut resample_input =
                crate::audio::rt::FixedAudioRing::<f64, RT_RESAMPLE_QUEUE_CAPACITY>::new();
            let mut resampler_input_frame = [0.0f64; RESAMPLER_CHUNK_SIZE];
            let mut resampler = input_resampler;
            let mut resampler_out = resampler.as_ref().map(|r| r.output_buffer_allocate(true));
            let mut output_resample_input =
                crate::audio::rt::FixedAudioRing::<f64, RT_RESAMPLE_QUEUE_CAPACITY>::new();
            let mut output_resampler_input_frame = [0.0f64; RESAMPLER_CHUNK_SIZE];
            let mut output_resampler = output_resampler;
            let mut output_resampler_out = output_resampler
                .as_ref()
                .map(|r| r.output_buffer_allocate(true));
            let mut output_resampled_scratch =
                FixedAudioBuffer::<f32, RT_OUTPUT_SCRATCH_CAPACITY>::new();
            let mut output_queue_control_scratch =
                FixedAudioBuffer::<f32, RT_OUTPUT_SCRATCH_CAPACITY>::new();
            let mut discontinuity_fade_scratch =
                FixedAudioBuffer::<f32, RT_OUTPUT_SCRATCH_CAPACITY>::new();
            let mut output_safety_scratch =
                FixedAudioBuffer::<f32, RT_OUTPUT_SCRATCH_CAPACITY>::new();
            let mut gate_rt = NoiseGate::new(-40.0, 10.0, 100.0, sample_rate_for_latency as f64);
            #[cfg(feature = "vad")]
            {
                let control = gate_rt_control
                    .snapshot()
                    .unwrap_or_else(GateControlState::new);
                let vad_auto_gate =
                    VadAutoGate::without_backend(sample_rate_for_latency, control.vad_threshold);
                gate_rt.set_vad_auto_gate(Some(vad_auto_gate));
            }
            let mut eq_rt = ParametricEQ::new(sample_rate_for_latency as f64);
            let mut compressor_rt = Compressor::default_voice(sample_rate_for_latency as f64);
            let mut deesser_rt = DeEsser::new(sample_rate_for_latency as f64);
            let mut limiter_rt = Limiter::default_settings(sample_rate_for_latency as f64);
            let suppressor_initial_control = suppressor_rt_control
                .snapshot()
                .unwrap_or_else(SuppressorControlState::new);
            let mut suppressor_rt = NoiseSuppressionEngine::new(
                suppressor_initial_control.model,
                Arc::clone(&suppressor_strength),
            );
            let mut deferred_suppressor_retire: Option<NoiseSuppressionEngine> = None;
            let output_ceiling_linear =
                Cell::new(10.0_f32.powf(limiter_rt.ceiling_db() as f32 / 20.0));
            let limiter_lookahead_samples =
                Arc::new(AtomicU64::new(limiter_rt.lookahead_samples() as u64));
            let limiter_lookahead_samples_for_chain = Arc::clone(&limiter_lookahead_samples);
            let gate_snapshot = gate_rt_control
                .snapshot()
                .unwrap_or_else(GateControlState::new);
            apply_gate_control(&mut gate_rt, &gate_snapshot);
            #[cfg(feature = "vad")]
            {
                gate_noise_floor_db.store(gate_rt.noise_floor().to_bits(), Ordering::Relaxed);
                vad_available.store(gate_rt.is_vad_available(), Ordering::Relaxed);
            }
            apply_suppressor_control(&mut suppressor_rt, &suppressor_initial_control);
            update_backend_status_rt(
                &noise_backend_available,
                &noise_backend_failed,
                rt_error_code.as_ref(),
                &suppressor_rt,
            );
            let eq_snapshot = eq_control.snapshot().unwrap_or_else(EqControlSnapshot::new);
            apply_eq_control(&mut eq_rt, &eq_snapshot);
            let compressor_snapshot = compressor_rt_control
                .snapshot()
                .unwrap_or_else(CompressorControlState::new);
            apply_compressor_control(&mut compressor_rt, &compressor_snapshot);
            let deesser_snapshot = deesser_rt_control
                .snapshot()
                .unwrap_or_else(DeesserControlState::new);
            apply_deesser_control(&mut deesser_rt, &deesser_snapshot);
            let limiter_snapshot = limiter_rt_control
                .snapshot()
                .unwrap_or_else(LimiterControlState::new);
            apply_limiter_control(&mut limiter_rt, &limiter_snapshot);
            output_ceiling_linear.set(10.0_f32.powf(limiter_rt.ceiling_db() as f32 / 20.0));

            // Set high thread priority only after all AudioForge-owned buffers and
            // DSP state have been allocated and initialized.
            if let Err(_e) = set_current_thread_priority(ThreadPriority::Max) {
                processor_debug_log!("Warning: Could not set audio thread priority: {:?}", _e);
            }

            let mut pre_filter_state = InputPreFilterState::default();

            // Pre-filter at 80Hz to remove rumble before gate/suppressor stages.
            let mut pre_filter = Biquad::new(
                BiquadType::HighPass,
                INPUT_PREFILTER_HZ,
                0.0,
                INPUT_PREFILTER_Q,
                sample_rate_for_latency as f64,
            );

            // Metering state (IIR smoothing for RMS)
            let mut input_rms_acc: f32 = 0.0;
            let mut output_rms_acc: f32 = 0.0;
            let meter_coeff =
                smoothing_coeff_for_time_constant(sample_rate_for_latency as f32, 100.0);

            // Latency tracking
            let mut last_latency_update = Instant::now();
            let mut last_heartbeat = Instant::now();
            let latency_update_interval = std::time::Duration::from_millis(100); // Update every 100ms
            const HEARTBEAT_INTERVAL: std::time::Duration = std::time::Duration::from_secs(1);
            const SUPPRESSOR_STARVATION_MS: u64 = 400; // Reset suppressor if no output this long
            const SUPPRESSOR_RECOVERY_COOLDOWN_MS: u64 = 2000;
            const NON_FINITE_REBUILD_THRESHOLD: u32 = 3;
            const NON_FINITE_REBUILD_WINDOW_MS: u64 = 2000;
            let mut last_suppressor_recovery =
                Instant::now() - std::time::Duration::from_millis(SUPPRESSOR_RECOVERY_COOLDOWN_MS);
            let mut suppressor_soft_reset_pending = false;
            let mut non_finite_window_started_at: Option<Instant> = None;
            let mut non_finite_window_count: u32 = 0;

            // Helper: compute peak and RMS from buffer, update atomics
            let measure_levels = |buffer: &[f32],
                                  rms_acc: &mut f32,
                                  peak_atomic: &AtomicU32,
                                  rms_atomic: &AtomicU32| {
                let mut peak: f32 = 0.0;
                for &sample in buffer.iter() {
                    let abs = sample.abs();
                    if abs > peak {
                        peak = abs;
                    }
                    // IIR RMS accumulator
                    *rms_acc = meter_coeff * *rms_acc + (1.0 - meter_coeff) * (sample * sample);
                }
                // Convert to dB
                let peak_db = if peak > 0.0 {
                    20.0 * peak.log10()
                } else {
                    -120.0
                };
                let rms_db = if *rms_acc > 0.0 {
                    10.0 * rms_acc.log10() // RMS is sqrt of mean squared, so 10*log10 not 20
                } else {
                    -120.0
                };
                peak_atomic.store(peak_db.to_bits(), Ordering::Relaxed);
                rms_atomic.store(rms_db.to_bits(), Ordering::Relaxed);
            };

            macro_rules! apply_downstream_chain_rt {
                ($buffer:expr) => {{
                    if deesser_dirty.swap(false, Ordering::AcqRel) {
                        if let Some(control) = deesser_rt_control.snapshot() {
                            apply_deesser_control(&mut deesser_rt, &control);
                        } else {
                            deesser_dirty.store(true, Ordering::Release);
                        }
                    }
                    if eq_dirty.swap(false, Ordering::AcqRel) {
                        if let Some(eq_snapshot) = eq_control.snapshot() {
                            apply_eq_control(&mut eq_rt, &eq_snapshot);
                        } else {
                            eq_dirty.store(true, Ordering::Release);
                        }
                    }
                    if compressor_dirty.swap(false, Ordering::AcqRel) {
                        if let Some(control) = compressor_rt_control.snapshot() {
                            apply_compressor_control(&mut compressor_rt, &control);
                        } else {
                            compressor_dirty.store(true, Ordering::Release);
                        }
                    }
                    if limiter_dirty.swap(false, Ordering::AcqRel) {
                        if let Some(control) = limiter_rt_control.snapshot() {
                            apply_limiter_control(&mut limiter_rt, &control);
                            output_ceiling_linear
                                .set(10.0_f32.powf(limiter_rt.ceiling_db() as f32 / 20.0));
                            limiter_lookahead_samples_for_chain
                                .store(limiter_rt.lookahead_samples() as u64, Ordering::Relaxed);
                        } else {
                            limiter_dirty.store(true, Ordering::Release);
                        }
                    }

                    if deesser_enabled.load(Ordering::Acquire) {
                        deesser_rt.process_block_inplace($buffer);
                        deesser_gain_reduction.store(
                            deesser_rt.current_gain_reduction_db().to_bits(),
                            Ordering::Relaxed,
                        );
                    } else {
                        deesser_gain_reduction.store(0.0_f32.to_bits(), Ordering::Relaxed);
                    }

                    if eq_enabled.load(Ordering::Acquire) {
                        eq_rt.process_block_inplace($buffer);
                    }

                    if compressor_enabled.load(Ordering::Acquire) {
                        compressor_rt.process_block_inplace($buffer);
                        compressor_gain_reduction.store(
                            (compressor_rt.current_gain_reduction() as f32).to_bits(),
                            Ordering::Relaxed,
                        );
                        let current_release = compressor_rt.current_release_time();
                        compressor_current_release_ms
                            .store(release_ms_to_tenth_ms(current_release), Ordering::Relaxed);
                        compressor_current_lufs
                            .store(compressor_rt.current_lufs().to_bits(), Ordering::Relaxed);
                        compressor_current_makeup_gain.store(
                            compressor_rt.current_makeup_gain().to_bits(),
                            Ordering::Relaxed,
                        );
                    } else {
                        compressor_gain_reduction.store(0.0_f32.to_bits(), Ordering::Relaxed);
                        compressor_current_release_ms
                            .store(COMPRESSOR_DEFAULT_RELEASE_TENTH_MS, Ordering::Relaxed);
                        compressor_current_lufs.store((-100.0_f64).to_bits(), Ordering::Relaxed);
                        compressor_current_makeup_gain.store(0.0_f64.to_bits(), Ordering::Relaxed);
                    }

                    if limiter_enabled.load(Ordering::Acquire) {
                        limiter_rt.process_block_inplace($buffer);
                    }
                }};
            }

            // Time-based EMA smoothing for metrics (200ms time constant)
            const TAU_MS: f32 = 200.0; // Time constant in milliseconds
            let dt_ms = 10.0; // Processing interval (480 samples @ 48kHz)
            let alpha = 1.0 - (-dt_ms / TAU_MS).exp(); // Smoothing factor

            let smooth_dsp_time = |raw_us: u64, prev_smoothed: u64| -> u64 {
                let raw_f = raw_us as f32;
                let prev_f = prev_smoothed as f32;
                (alpha * raw_f + (1.0 - alpha) * prev_f) as u64
            };

            let smooth_buffer = |raw: u32, prev_smoothed: u32| -> u32 {
                let raw_f = raw as f32;
                let prev_f = prev_smoothed as f32;
                (alpha * raw_f + (1.0 - alpha) * prev_f) as u32
            };

            // Jitter-buffer write control: keep output queue in a healthy range.
            let output_target_low_samples =
                duration_samples(output_sample_rate_for_latency, OUTPUT_PRIME_MS);
            let output_target_high_samples =
                duration_samples(output_sample_rate_for_latency, OUTPUT_TARGET_HIGH_MS);
            let output_target_center_samples =
                (output_target_low_samples + output_target_high_samples) / 2;
            let output_hard_backlog_samples =
                duration_samples(output_sample_rate_for_latency, OUTPUT_HARD_BACKLOG_MS);
            const OUTPUT_MAX_CATCHUP_RATIO: f32 = 1.03;
            const OUTPUT_MAX_EMERGENCY_CATCHUP_RATIO: f32 = 1.06;
            let input_backlog_high_samples = duration_samples(input_sample_rate_for_thread, 250);
            let input_backlog_low_samples = duration_samples(input_sample_rate_for_thread, 100);
            let discontinuity_fade_samples =
                duration_samples(output_sample_rate_for_latency, 6).max(1);
            let discontinuity_fade_remaining = Cell::new(0usize);
            let mut output_drift_error_ema = 0.0_f32;
            let mut output_writer = OutputWriteContext {
                output_producer: &mut output_producer,
                output_queue_control_scratch: &mut output_queue_control_scratch,
                discontinuity_fade_scratch: &mut discontinuity_fade_scratch,
                output_safety_scratch: &mut output_safety_scratch,
                drift_error_ema: &mut output_drift_error_ema,
                discontinuity_fade_remaining: &discontinuity_fade_remaining,
                limiter_enabled: limiter_enabled.as_ref(),
                output_ceiling_linear: &output_ceiling_linear,
                counters: OutputWriteCounters {
                    jitter_dropped_samples: jitter_dropped_samples.as_ref(),
                    output_retime_adjustment_count: output_retime_adjustment_count.as_ref(),
                    output_recovery_event_count: output_recovery_event_count.as_ref(),
                    output_short_write_dropped_samples: output_short_write_dropped_samples.as_ref(),
                    rt_buffer_overflow_count: rt_buffer_overflow_count.as_ref(),
                    rt_error_code: rt_error_code.as_ref(),
                    output_buffer_len: output_buffer_len.as_ref(),
                    last_output_write_time: last_output_write_time.as_ref(),
                },
                limits: OutputWriteLimits {
                    output_target_center_samples,
                    output_hard_backlog_samples,
                    discontinuity_fade_samples,
                    max_catchup_ratio: OUTPUT_MAX_CATCHUP_RATIO,
                    max_emergency_catchup_ratio: OUTPUT_MAX_EMERGENCY_CATCHUP_RATIO,
                },
            };
            let mut write_output = |samples: &[f32], clean_path: bool| {
                if let Some(output_resampler) = output_resampler.as_mut() {
                    for &sample in samples {
                        if !output_resample_input.push(sample as f64) {
                            rt_buffer_overflow_count.fetch_add(1, Ordering::Relaxed);
                            store_rt_error(
                                rt_error_code.as_ref(),
                                RtErrorCode::FixedBufferOverflow,
                            );
                            break;
                        }
                    }

                    loop {
                        output_resampled_scratch.clear();
                        let input_frames_needed = output_resampler.input_frames_next();
                        while output_resample_input.len() >= input_frames_needed
                            && input_frames_needed <= output_resampler_input_frame.len()
                        {
                            let Some(outbuf) = output_resampler_out.as_mut() else {
                                break;
                            };
                            if !has_resampler_output_capacity(&output_resampled_scratch, outbuf) {
                                break;
                            }
                            let read = output_resample_input
                                .pop_into(&mut output_resampler_input_frame[..input_frames_needed]);
                            if read != input_frames_needed {
                                break;
                            }
                            let in_slices = [&output_resampler_input_frame[..input_frames_needed]];
                            if let Ok((_nbr_in, nbr_out)) =
                                output_resampler.process_into_buffer(&in_slices, outbuf, None)
                            {
                                for &sample in outbuf[0].iter().take(nbr_out) {
                                    if !output_resampled_scratch.push(sample as f32) {
                                        rt_buffer_overflow_count.fetch_add(1, Ordering::Relaxed);
                                        store_rt_error(
                                            rt_error_code.as_ref(),
                                            RtErrorCode::FixedBufferOverflow,
                                        );
                                        break;
                                    }
                                }
                            }
                        }

                        if !output_writer
                            .write_chunk(output_resampled_scratch.as_slice(), clean_path)
                        {
                            break;
                        }
                        if output_resample_input.len() < input_frames_needed {
                            break;
                        }
                    }
                } else {
                    let _ = output_writer.write_chunk(samples, clean_path);
                }
            };
            let mut previous_processing_path = select_processing_path(
                raw_monitor_enabled.load(Ordering::Acquire),
                bypass.load(Ordering::SeqCst),
            );
            let mut consecutive_idle_wakeups = 0u32;
            let _ = dsp_ready_tx.send(());

            // Run entire processing loop with denormals flushed to zero
            // This prevents tiny floating point values from causing CPU stalls and audio artifacts
            // SAFETY: This only modifies floating point control flags for this thread
            unsafe {
                no_denormals::no_denormals(|| {
                    // RT_REGION_START: dsp_processing_loop
                    while running.load(Ordering::SeqCst) {
                        // Record input buffer fill level (samples waiting to be processed)
                        let mut raw_input_len = consumer.len();
                        if raw_input_len > input_backlog_high_samples {
                            let mut to_drop =
                                raw_input_len.saturating_sub(input_backlog_low_samples);
                            let mut dropped_total = 0usize;
                            while to_drop > 0 {
                                let batch = to_drop.min(input_buffer.capacity());
                                let dropped = consumer
                                    .read(&mut input_buffer.as_mut_capacity_slice()[..batch]);
                                if dropped == 0 {
                                    break;
                                }
                                to_drop = to_drop.saturating_sub(dropped);
                                dropped_total += dropped;
                            }
                            if dropped_total > 0 {
                                input_backlog_recovery_count.fetch_add(1, Ordering::Relaxed);
                                input_backlog_dropped_samples
                                    .fetch_add(dropped_total as u64, Ordering::Relaxed);
                                store_rt_error(
                                    rt_error_code.as_ref(),
                                    RtErrorCode::InputBacklogDropped,
                                );
                                resample_input.clear();
                                if let Some(outbuf) = resampler_out.as_mut() {
                                    outbuf[0].clear();
                                }
                                if !raw_monitor_enabled.load(Ordering::Acquire) {
                                    discontinuity_fade_remaining.set(discontinuity_fade_samples);
                                }
                                raw_input_len = consumer.len();
                            }
                        }
                        let raw_input_len = raw_input_len as u32;
                        input_buffer_len.store(raw_input_len, Ordering::Relaxed);
                        let smoothed_input = smooth_buffer(
                            raw_input_len,
                            smoothed_input_buffer_len.load(Ordering::Relaxed),
                        );
                        smoothed_input_buffer_len.store(smoothed_input, Ordering::Relaxed);

                        // Read audio samples
                        let n_raw = consumer.read(input_buffer.as_mut_capacity_slice());

                        if n_raw > 0 {
                            let n = if let Some(resampler) = resampler.as_mut() {
                                temp_buffer.clear();
                                for &sample in input_buffer.as_mut_capacity_slice()[..n_raw].iter()
                                {
                                    if !resample_input.push(sample as f64) {
                                        rt_buffer_overflow_count.fetch_add(1, Ordering::Relaxed);
                                        store_rt_error(
                                            rt_error_code.as_ref(),
                                            RtErrorCode::FixedBufferOverflow,
                                        );
                                        break;
                                    }
                                }

                                let input_frames_needed = resampler.input_frames_next();
                                while resample_input.len() >= input_frames_needed
                                    && input_frames_needed <= resampler_input_frame.len()
                                {
                                    let Some(outbuf) = resampler_out.as_mut() else {
                                        break;
                                    };
                                    if !has_resampler_output_capacity(&temp_buffer, outbuf) {
                                        break;
                                    }
                                    let read = resample_input.pop_into(
                                        &mut resampler_input_frame[..input_frames_needed],
                                    );
                                    if read != input_frames_needed {
                                        break;
                                    }
                                    let in_slices = [&resampler_input_frame[..input_frames_needed]];
                                    if let Ok((_nbr_in, nbr_out)) =
                                        resampler.process_into_buffer(&in_slices, outbuf, None)
                                    {
                                        let channel_out = &outbuf[0];
                                        for &sample in channel_out.iter().take(nbr_out) {
                                            if !temp_buffer.push(sample as f32) {
                                                rt_buffer_overflow_count
                                                    .fetch_add(1, Ordering::Relaxed);
                                                store_rt_error(
                                                    rt_error_code.as_ref(),
                                                    RtErrorCode::FixedBufferOverflow,
                                                );
                                                break;
                                            }
                                        }
                                    }
                                }
                                temp_buffer.len()
                            } else {
                                temp_buffer.clear();
                                let written = temp_buffer.extend_from_slice(
                                    &input_buffer.as_mut_capacity_slice()[..n_raw],
                                );
                                if written < n_raw {
                                    rt_buffer_overflow_count.fetch_add(1, Ordering::Relaxed);
                                    store_rt_error(
                                        rt_error_code.as_ref(),
                                        RtErrorCode::FixedBufferOverflow,
                                    );
                                }
                                written
                            };

                            if n == 0 {
                                consecutive_idle_wakeups =
                                    consecutive_idle_wakeups.saturating_add(1);
                                let last_input_callback =
                                    last_input_callback_time_us_for_dsp.load(Ordering::Relaxed);
                                let input_callback_age_us = if last_input_callback == 0 {
                                    u64::MAX
                                } else {
                                    now_micros().saturating_sub(last_input_callback)
                                };
                                let idle_sleep_us = next_process_idle_sleep_us(
                                    consecutive_idle_wakeups,
                                    input_callback_age_us,
                                );
                                dsp_idle_wakeup_count.fetch_add(1, Ordering::Relaxed);
                                dsp_idle_sleep_us.store(idle_sleep_us, Ordering::Relaxed);
                                std::thread::sleep(std::time::Duration::from_micros(idle_sleep_us));
                                continue;
                            }
                            consecutive_idle_wakeups = 0;

                            let buffer = temp_buffer.as_mut_slice();

                            // Start DSP timing
                            let dsp_start = Instant::now();
                            let processing_path = select_processing_path(
                                raw_monitor_enabled.load(Ordering::Acquire),
                                bypass.load(Ordering::SeqCst),
                            );
                            if processing_path != previous_processing_path {
                                pre_filter_state = InputPreFilterState::default();
                                input_rms_acc = 0.0;
                                output_rms_acc = 0.0;
                                gate_gain_meter.store(1.0_f32.to_bits(), Ordering::Relaxed);
                                compressor_gain_reduction
                                    .store(0.0_f32.to_bits(), Ordering::Relaxed);
                                deesser_gain_reduction.store(0.0_f32.to_bits(), Ordering::Relaxed);
                                compressor_current_release_ms
                                    .store(COMPRESSOR_DEFAULT_RELEASE_TENTH_MS, Ordering::Relaxed);
                                compressor_current_lufs
                                    .store((-100.0_f64).to_bits(), Ordering::Relaxed);
                                compressor_current_makeup_gain
                                    .store(0.0_f64.to_bits(), Ordering::Relaxed);
                                suppressor_buffer_len.store(0, Ordering::Relaxed);
                                suppressor_latency_samples.store(0, Ordering::Relaxed);
                                smoothed_buffer_len.store(0, Ordering::Relaxed);
                                discontinuity_fade_remaining.set(discontinuity_fade_samples);

                                gate_rt.reset();
                                if let Some(control) = gate_rt_control.snapshot() {
                                    apply_gate_control(&mut gate_rt, &control);
                                } else {
                                    gate_dirty.store(true, Ordering::Release);
                                }
                                eq_rt.reset();
                                compressor_rt.reset();
                                deesser_rt.reset();
                                limiter_rt.reset();
                                suppressor_rt.soft_reset();
                                update_backend_status_rt(
                                    &noise_backend_available,
                                    &noise_backend_failed,
                                    rt_error_code.as_ref(),
                                    &suppressor_rt,
                                );
                                previous_processing_path = processing_path;
                            }

                            if processing_path == ProcessingPath::RawMonitor {
                                sanitize_non_finite_inplace(buffer);

                                measure_levels(buffer, &mut input_rms_acc, &input_peak, &input_rms);

                                if recording_active.load(Ordering::Relaxed) {
                                    let target = raw_recording_target.load(Ordering::Acquire);
                                    let pos = raw_recording_pos.load(Ordering::Acquire);
                                    if pos < target {
                                        let remaining = target - pos;
                                        let to_copy = n.min(remaining);
                                        let written = recording_producer.write(&buffer[..to_copy]);
                                        let new_pos = pos.saturating_add(written);
                                        raw_recording_pos.store(new_pos, Ordering::Release);

                                        let window_len =
                                            (sample_rate_for_latency as usize / 10).max(1);
                                        let level_start = to_copy.saturating_sub(window_len);
                                        let level_slice = &buffer[level_start..to_copy];
                                        let level_rms = if level_slice.is_empty() {
                                            -120.0
                                        } else {
                                            let sum_sq: f32 = level_slice
                                                .iter()
                                                .map(|sample| sample * sample)
                                                .sum();
                                            let rms = (sum_sq / level_slice.len() as f32).sqrt();
                                            if rms > 1e-6 {
                                                20.0 * rms.log10()
                                            } else {
                                                -120.0
                                            }
                                        };
                                        recording_level_db
                                            .store(level_rms.to_bits(), Ordering::Relaxed);
                                    }
                                }

                                measure_levels(
                                    buffer,
                                    &mut output_rms_acc,
                                    &output_peak,
                                    &output_rms,
                                );
                                compressor_gain_reduction
                                    .store(0.0_f32.to_bits(), Ordering::Relaxed);
                                deesser_gain_reduction.store(0.0_f32.to_bits(), Ordering::Relaxed);
                                gate_gain_meter.store(1.0_f32.to_bits(), Ordering::Relaxed);
                                compressor_current_release_ms
                                    .store(COMPRESSOR_DEFAULT_RELEASE_TENTH_MS, Ordering::Relaxed);
                                compressor_current_lufs
                                    .store((-100.0_f64).to_bits(), Ordering::Relaxed);
                                compressor_current_makeup_gain
                                    .store(0.0_f64.to_bits(), Ordering::Relaxed);
                                suppressor_buffer_len.store(0, Ordering::Relaxed);
                                suppressor_latency_samples.store(0, Ordering::Relaxed);
                                smoothed_buffer_len.store(0, Ordering::Relaxed);

                                write_output(buffer, uses_clean_write_path(processing_path));
                                let raw_dsp_us = dsp_start.elapsed().as_micros() as u64;
                                let prev_smoothed = dsp_time_smoothed_us.load(Ordering::Relaxed);
                                let smoothed = smooth_dsp_time(raw_dsp_us, prev_smoothed);
                                dsp_time_us.store(raw_dsp_us, Ordering::Relaxed);
                                dsp_time_smoothed_us.store(smoothed, Ordering::Relaxed);
                                continue;
                            }

                            sanitize_and_clamp_input_inplace(
                                buffer,
                                &clip_event_count,
                                &clip_peak_db,
                            );
                            apply_input_pre_filter(buffer, &mut pre_filter_state, &mut pre_filter);

                            // Measure INPUT levels (after pre-filter, before main processing)
                            measure_levels(buffer, &mut input_rms_acc, &input_peak, &input_rms);

                            // === RAW AUDIO RECORDING TAP (for calibration) ===
                            // Capture audio AFTER pre-filter, BEFORE noise gate
                            // This is the raw microphone response needed for EQ analysis
                            if recording_active.load(Ordering::Relaxed) {
                                let target = raw_recording_target.load(Ordering::Acquire);
                                let pos = raw_recording_pos.load(Ordering::Acquire);
                                if pos < target {
                                    let remaining = target - pos;
                                    let to_copy = n.min(remaining);
                                    let written = recording_producer.write(&buffer[..to_copy]);
                                    let new_pos = pos.saturating_add(written);
                                    raw_recording_pos.store(new_pos, Ordering::Release);

                                    let window_len = (sample_rate_for_latency as usize / 10).max(1);
                                    let level_start = to_copy.saturating_sub(window_len);
                                    let level_slice = &buffer[level_start..to_copy];
                                    let level_rms = if level_slice.is_empty() {
                                        -120.0
                                    } else {
                                        let sum_sq: f32 =
                                            level_slice.iter().map(|sample| sample * sample).sum();
                                        let rms = (sum_sq / level_slice.len() as f32).sqrt();
                                        if rms > 1e-6 {
                                            20.0 * rms.log10()
                                        } else {
                                            -120.0
                                        }
                                    };
                                    recording_level_db
                                        .store(level_rms.to_bits(), Ordering::Relaxed);
                                }
                            }
                            // === END RECORDING TAP ===

                            if processing_path == ProcessingPath::Bypass {
                                // Bypass mode: measure output = input, send directly
                                measure_levels(
                                    buffer,
                                    &mut output_rms_acc,
                                    &output_peak,
                                    &output_rms,
                                );
                                compressor_gain_reduction
                                    .store(0.0_f32.to_bits(), Ordering::Relaxed);
                                deesser_gain_reduction.store(0.0_f32.to_bits(), Ordering::Relaxed);
                                compressor_current_release_ms
                                    .store(COMPRESSOR_DEFAULT_RELEASE_TENTH_MS, Ordering::Relaxed); // Default 200ms
                                compressor_current_lufs
                                    .store((-100.0_f64).to_bits(), Ordering::Relaxed);
                                compressor_current_makeup_gain
                                    .store(0.0_f64.to_bits(), Ordering::Relaxed);
                                suppressor_buffer_len.store(0, Ordering::Relaxed);

                                write_output(buffer, uses_clean_write_path(processing_path));
                                // Record DSP processing time
                                let raw_dsp_us = dsp_start.elapsed().as_micros() as u64;
                                let prev_smoothed = dsp_time_smoothed_us.load(Ordering::Relaxed);
                                let smoothed = smooth_dsp_time(raw_dsp_us, prev_smoothed);
                                dsp_time_us.store(raw_dsp_us, Ordering::Relaxed);
                                dsp_time_smoothed_us.store(smoothed, Ordering::Relaxed);
                            } else {
                                // Stage 1: Noise Gate
                                if gate_enabled.load(Ordering::Acquire) {
                                    if gate_dirty.swap(false, Ordering::AcqRel) {
                                        if let Some(control) = gate_rt_control.snapshot() {
                                            apply_gate_control(&mut gate_rt, &control);
                                        } else {
                                            gate_dirty.store(true, Ordering::Release);
                                        }
                                    }

                                    #[cfg(feature = "vad")]
                                    {
                                        let written = vad_worker_producer.write(buffer);
                                        if written < buffer.len() {
                                            rt_buffer_overflow_count
                                                .fetch_add(1, Ordering::Relaxed);
                                            store_rt_error(
                                                rt_error_code.as_ref(),
                                                RtErrorCode::FixedBufferOverflow,
                                            );
                                        }
                                        let latest_prob =
                                            f32::from_bits(vad_probability.load(Ordering::Acquire));
                                        let last_update =
                                            vad_last_update_us.load(Ordering::Acquire);
                                        let fresh = last_update > 0
                                            && now_micros().saturating_sub(last_update)
                                                <= VAD_PROBABILITY_STALE_US;
                                        let worker_available =
                                            vad_available.load(Ordering::Acquire) && fresh;
                                        gate_rt.set_external_vad_probability(
                                            latest_prob,
                                            worker_available,
                                        );
                                    }
                                    gate_rt.process_block_inplace(buffer);
                                    gate_gain_meter.store(
                                        gate_rt.current_gain().clamp(0.0, 1.0).to_bits(),
                                        Ordering::Relaxed,
                                    );
                                    #[cfg(feature = "vad")]
                                    {
                                        let prob = gate_rt.get_vad_probability();
                                        vad_probability.store(prob.to_bits(), Ordering::Relaxed);
                                        gate_noise_floor_db.store(
                                            gate_rt.noise_floor().to_bits(),
                                            Ordering::Relaxed,
                                        );
                                        gate_fused_score.store(
                                            gate_rt.fused_gate_score().to_bits(),
                                            Ordering::Relaxed,
                                        );
                                        let last_update =
                                            vad_last_update_us.load(Ordering::Acquire);
                                        let fresh = last_update > 0
                                            && now_micros().saturating_sub(last_update)
                                                <= VAD_PROBABILITY_STALE_US;
                                        vad_available.store(
                                            gate_rt.is_vad_available() && fresh,
                                            Ordering::Relaxed,
                                        );
                                    }
                                }

                                // Stage 2: Noise Suppression (RNNoise or DeepFilterNet)
                                let use_suppressor = suppressor_enabled.load(Ordering::Acquire);
                                if use_suppressor {
                                    if let Some(retired) = deferred_suppressor_retire.take() {
                                        if let Err(retired) = retired_suppressor_tx.push(retired) {
                                            deferred_suppressor_retire = Some(retired);
                                        }
                                    }

                                    if suppressor_dirty.swap(false, Ordering::AcqRel) {
                                        if let Some(control) = suppressor_rt_control.snapshot() {
                                            while deferred_suppressor_retire.is_none()
                                                && !retired_suppressor_tx.is_full()
                                            {
                                                let candidate_matches =
                                                    match suppressor_rx.iter().next() {
                                                        Some(candidate) => {
                                                            candidate.model_type() == control.model
                                                        }
                                                        None => break,
                                                    };
                                                let Some(candidate) = suppressor_rx.pop() else {
                                                    break;
                                                };

                                                if candidate_matches
                                                    && suppressor_rt.model_type() != control.model
                                                {
                                                    let retired = std::mem::replace(
                                                        &mut suppressor_rt,
                                                        candidate,
                                                    );
                                                    if let Err(retired) =
                                                        retired_suppressor_tx.push(retired)
                                                    {
                                                        deferred_suppressor_retire = Some(retired);
                                                        rt_buffer_overflow_count
                                                            .fetch_add(1, Ordering::Relaxed);
                                                        store_rt_error(
                                                            rt_error_code.as_ref(),
                                                            RtErrorCode::FixedBufferOverflow,
                                                        );
                                                        break;
                                                    }
                                                } else if let Err(candidate) =
                                                    retired_suppressor_tx.push(candidate)
                                                {
                                                    deferred_suppressor_retire = Some(candidate);
                                                    rt_buffer_overflow_count
                                                        .fetch_add(1, Ordering::Relaxed);
                                                    store_rt_error(
                                                        rt_error_code.as_ref(),
                                                        RtErrorCode::FixedBufferOverflow,
                                                    );
                                                    break;
                                                }
                                            }
                                            if suppressor_rt.model_type() == control.model {
                                                apply_suppressor_control(
                                                    &mut suppressor_rt,
                                                    &control,
                                                );
                                                update_backend_status_rt(
                                                    &noise_backend_available,
                                                    &noise_backend_failed,
                                                    rt_error_code.as_ref(),
                                                    &suppressor_rt,
                                                );
                                            } else {
                                                suppressor_dirty.store(true, Ordering::Release);
                                            }
                                        } else {
                                            suppressor_dirty.store(true, Ordering::Release);
                                        }
                                    }
                                    if suppressor_reset_requested.swap(false, Ordering::AcqRel) {
                                        suppressor_rt.soft_reset();
                                        update_backend_status_rt(
                                            &noise_backend_available,
                                            &noise_backend_failed,
                                            rt_error_code.as_ref(),
                                            &suppressor_rt,
                                        );
                                    }
                                    {
                                        // Always feed suppressor first so frame accumulation is correct.
                                        let accepted = suppressor_rt.push_samples(buffer);
                                        if accepted < buffer.len() {
                                            rt_buffer_overflow_count
                                                .fetch_add(1, Ordering::Relaxed);
                                            store_rt_error(
                                                rt_error_code.as_ref(),
                                                RtErrorCode::FixedBufferOverflow,
                                            );
                                        }
                                        suppressor_rt.process_frames();
                                        update_backend_status_rt(
                                            &noise_backend_available,
                                            &noise_backend_failed,
                                            rt_error_code.as_ref(),
                                            &suppressor_rt,
                                        );

                                        // Track suppressor internal buffer fill level after processing.
                                        let suppressor_buffered = suppressor_rt.pending_input()
                                            + suppressor_rt.available_samples();
                                        let suppressor_latency = suppressor_rt.latency_samples()
                                            + suppressor_rt.pending_input();
                                        let raw_buffer = suppressor_buffered as u32;
                                        let prev_smoothed =
                                            smoothed_buffer_len.load(Ordering::Relaxed);
                                        let smoothed = smooth_buffer(raw_buffer, prev_smoothed);
                                        suppressor_buffer_len.store(raw_buffer, Ordering::Relaxed);
                                        suppressor_latency_samples
                                            .store(suppressor_latency as u32, Ordering::Relaxed);
                                        smoothed_buffer_len.store(smoothed, Ordering::Relaxed);

                                        let available = suppressor_rt.available_samples();
                                        if available == 0 {
                                            let pending = suppressor_rt.pending_input();
                                            if pending >= RNNOISE_FRAME_SIZE
                                                && !recording_active_thread.load(Ordering::Relaxed)
                                            {
                                                let last_write =
                                                    last_output_write_time.load(Ordering::Relaxed);
                                                if last_write > 0 {
                                                    let now = now_micros();
                                                    let since_write_ms =
                                                        now.saturating_sub(last_write) / 1000;
                                                    if since_write_ms > SUPPRESSOR_STARVATION_MS
                                                        && last_suppressor_recovery
                                                            .elapsed()
                                                            .as_millis()
                                                            as u64
                                                            > SUPPRESSOR_RECOVERY_COOLDOWN_MS
                                                    {
                                                        if suppressor_soft_reset_pending {
                                                            suppressor_rt.soft_reset();
                                                            suppressor_soft_reset_pending = false;
                                                        } else {
                                                            suppressor_rt.soft_reset();
                                                            suppressor_soft_reset_pending = true;
                                                        }
                                                        update_backend_status_rt(
                                                            &noise_backend_available,
                                                            &noise_backend_failed,
                                                            rt_error_code.as_ref(),
                                                            &suppressor_rt,
                                                        );
                                                        last_suppressor_recovery = Instant::now();
                                                    }
                                                }
                                            }
                                        }
                                        if available > 0 {
                                            suppressor_soft_reset_pending = false;
                                            rnnoise_output.clear();
                                            let count = available.min(rnnoise_output.capacity());
                                            let _ = rnnoise_output.set_len_zeroed(count);
                                            let processed = suppressor_rt
                                                .pop_samples_into(rnnoise_output.as_mut_slice());
                                            let _ = rnnoise_output.set_len_zeroed(processed);
                                            let output_slice = rnnoise_output.as_mut_slice();

                                            let mut detected_non_finite = false;
                                            for sample in output_slice.iter_mut() {
                                                if !sample.is_finite() {
                                                    *sample = 0.0;
                                                    detected_non_finite = true;
                                                }
                                            }
                                            if detected_non_finite {
                                                suppressor_non_finite_count
                                                    .fetch_add(1, Ordering::Relaxed);
                                                store_rt_error(
                                                    rt_error_code.as_ref(),
                                                    RtErrorCode::SuppressorNonFinite,
                                                );
                                                let now = Instant::now();
                                                match non_finite_window_started_at {
                                                    Some(start)
                                                        if now.duration_since(start).as_millis()
                                                            as u64
                                                            <= NON_FINITE_REBUILD_WINDOW_MS => {}
                                                    _ => {
                                                        non_finite_window_started_at = Some(now);
                                                        non_finite_window_count = 0;
                                                    }
                                                }
                                                non_finite_window_count =
                                                    non_finite_window_count.saturating_add(1);
                                                if non_finite_window_count
                                                    >= NON_FINITE_REBUILD_THRESHOLD
                                                {
                                                    suppressor_rt.soft_reset();
                                                    non_finite_window_started_at = None;
                                                    non_finite_window_count = 0;
                                                }
                                                update_backend_status_rt(
                                                    &noise_backend_available,
                                                    &noise_backend_failed,
                                                    rt_error_code.as_ref(),
                                                    &suppressor_rt,
                                                );
                                            }

                                            apply_downstream_chain_rt!(output_slice);

                                            // Measure OUTPUT levels
                                            measure_levels(
                                                output_slice,
                                                &mut output_rms_acc,
                                                &output_peak,
                                                &output_rms,
                                            );

                                            // Send processed samples to output
                                            write_output(output_slice, false);
                                        }

                                        // Record DSP processing time
                                        let raw_dsp_us = dsp_start.elapsed().as_micros() as u64;
                                        let prev_smoothed =
                                            dsp_time_smoothed_us.load(Ordering::Relaxed);
                                        let smoothed = smooth_dsp_time(raw_dsp_us, prev_smoothed);
                                        dsp_time_us.store(raw_dsp_us, Ordering::Relaxed);
                                        dsp_time_smoothed_us.store(smoothed, Ordering::Relaxed);
                                    }
                                } else {
                                    // Suppressor disabled: clear buffer counter
                                    suppressor_buffer_len.store(0, Ordering::Relaxed);
                                    suppressor_latency_samples.store(0, Ordering::Relaxed);
                                    smoothed_buffer_len.store(0, Ordering::Relaxed);
                                    if suppressor_reset_requested.swap(false, Ordering::AcqRel) {
                                        suppressor_rt.soft_reset();
                                        update_backend_status_rt(
                                            &noise_backend_available,
                                            &noise_backend_failed,
                                            rt_error_code.as_ref(),
                                            &suppressor_rt,
                                        );
                                    }
                                    // Suppressor disabled: apply remaining stages directly

                                    apply_downstream_chain_rt!(buffer);

                                    // Measure OUTPUT levels
                                    measure_levels(
                                        buffer,
                                        &mut output_rms_acc,
                                        &output_peak,
                                        &output_rms,
                                    );

                                    // Send to output
                                    write_output(buffer, false);
                                    // Record DSP processing time
                                    let raw_dsp_us = dsp_start.elapsed().as_micros() as u64;
                                    let prev_smoothed =
                                        dsp_time_smoothed_us.load(Ordering::Relaxed);
                                    let smoothed = smooth_dsp_time(raw_dsp_us, prev_smoothed);
                                    dsp_time_us.store(raw_dsp_us, Ordering::Relaxed);
                                    dsp_time_smoothed_us.store(smoothed, Ordering::Relaxed);
                                }
                            }
                            // Update latency periodically
                            if last_latency_update.elapsed() >= latency_update_interval {
                                last_latency_update = Instant::now();

                                let output_buffer_samples =
                                    output_buffer_len.load(Ordering::Relaxed) as u64;

                                let suppressor_latency_samples =
                                    if suppressor_enabled.load(Ordering::Acquire) {
                                        suppressor_latency_samples.load(Ordering::Relaxed) as u64
                                    } else {
                                        0
                                    };
                                let total_latency = total_reported_latency_us(
                                    output_buffer_samples,
                                    output_sample_rate_for_latency,
                                    suppressor_latency_samples,
                                    limiter_lookahead_samples.load(Ordering::Relaxed),
                                    limiter_enabled.load(Ordering::Acquire),
                                    sample_rate_for_latency,
                                    latency_compensation_us.load(Ordering::Relaxed),
                                );
                                latency_us.store(total_latency, Ordering::Relaxed);
                            }
                        } else {
                            // No data available, sleep briefly to avoid busy-wait
                            if last_heartbeat.elapsed() >= HEARTBEAT_INTERVAL {
                                last_heartbeat = Instant::now();
                            }
                            consecutive_idle_wakeups = consecutive_idle_wakeups.saturating_add(1);
                            let last_input_callback =
                                last_input_callback_time_us_for_dsp.load(Ordering::Relaxed);
                            let input_callback_age_us = if last_input_callback == 0 {
                                u64::MAX
                            } else {
                                now_micros().saturating_sub(last_input_callback)
                            };
                            let idle_sleep_us = next_process_idle_sleep_us(
                                consecutive_idle_wakeups,
                                input_callback_age_us,
                            );
                            dsp_idle_wakeup_count.fetch_add(1, Ordering::Relaxed);
                            dsp_idle_sleep_us.store(idle_sleep_us, Ordering::Relaxed);
                            std::thread::sleep(std::time::Duration::from_micros(idle_sleep_us));
                        }
                    }
                    // RT_REGION_END: dsp_processing_loop
                }); // End no_denormals block
            } // End unsafe block
        });

        self.process_thread = Some(handle);
        if let Err(e) =
            dsp_ready_rx.recv_timeout(Duration::from_millis(DSP_THREAD_READY_TIMEOUT_MS))
        {
            self.stop();
            return Err(format!("DSP thread failed to become ready: {}", e));
        }

        if let Some(output) = self.audio_output.as_ref() {
            if let Err(e) = output.start() {
                self.stop();
                return Err(format!("Failed to start output stream: {}", e));
            }
        }

        if let Some(input) = self.audio_input.as_ref() {
            if let Err(e) = input.start() {
                self.stop();
                return Err(format!("Failed to start input stream: {}", e));
            }
        }

        Ok(format!(
            "Started: {} -> {}",
            input_device_name, output_device_name
        ))
    }

    /// Stop audio processing
    pub fn stop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        self.restart_requested.store(false, Ordering::Release);
        self.recovering.store(false, Ordering::Release);
        self.last_start_time_us.store(0, Ordering::Release);

        if let Some(handle) = self.process_thread.take() {
            let _ = handle.join();
        }
        #[cfg(feature = "vad")]
        self.stop_vad_worker();

        if let Some(input) = &self.audio_input {
            let _ = input.pause();
        }

        if let Some(output) = &self.audio_output {
            let _ = output.pause();
        }

        self.audio_input = None;
        self.audio_output = None;

        // Ensure output is unmuted and recording state is cleared.
        self.recording_active.store(false, Ordering::Release);
        self.output_muted.store(false, Ordering::Release);
        self.raw_recording_pos.store(0, Ordering::Release);
        self.raw_recording_target.store(0, Ordering::Release);
        self.recording_level_db
            .store((-120.0_f32).to_bits(), Ordering::Relaxed);
        self.suppressor_latency_samples.store(0, Ordering::Relaxed);
        self.input_resampler_active.store(false, Ordering::Relaxed);
        self.output_resampler_active.store(false, Ordering::Relaxed);
        if let Ok(mut consumer_guard) = self.raw_recording_consumer.lock() {
            *consumer_guard = None;
        }
        if let Ok(mut tx) = self.pending_suppressor_tx.lock() {
            *tx = None;
        }
        if let Ok(mut rx) = self.retired_suppressor_rx.lock() {
            *rx = None;
        }

        // Reinitialize suppressor state so stop/start can recover from poisoned model state.
        if let Ok(mut s) = self.suppressor.lock() {
            let was_enabled = s.is_enabled();
            let model = s.model_type();
            *s = NoiseSuppressionEngine::new(model, Arc::clone(&self.suppressor_strength));
            s.set_enabled(was_enabled);
            update_backend_diagnostics(
                &self.noise_backend_available,
                &self.noise_backend_failed,
                self.noise_backend_error.as_ref(),
                &s,
            );
        }

        // Reset DSP state so stop/start can recover from stuck envelopes.
        if let Ok(mut g) = self.gate.lock() {
            g.reset();
            if let Ok(control) = self.gate_control.lock() {
                apply_gate_control(&mut g, &control);
            }
        }
        if let Ok(mut e) = self.eq.lock() {
            e.reset();
        }
        if let Ok(mut c) = self.compressor.lock() {
            c.reset();
        }
        if let Ok(mut d) = self.deesser.lock() {
            d.reset();
        }
        if let Ok(mut l) = self.limiter.lock() {
            l.reset();
        }
    }
}
