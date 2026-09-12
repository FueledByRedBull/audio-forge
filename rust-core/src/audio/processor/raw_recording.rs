impl AudioProcessor {
    /// Start recording raw audio for calibration
    /// Taps audio after the fixed pre-filter while preserving the historical
    /// calibration behavior (adaptive cleanup is paused during capture).
    pub fn start_raw_recording(&mut self, duration_secs: f64) -> Result<(), String> {
        self.start_raw_recording_with_tap(duration_secs, false)
    }

    /// Start a raw capture with an explicit tap position relative to adaptive
    /// input cleanup. The default calibration path remains after the fixed
    /// pre-filter and pauses adaptive cleanup; setting `before_cleanup` keeps
    /// cleanup running while capturing the original input for an audition.
    pub fn start_raw_recording_with_tap(
        &mut self,
        duration_secs: f64,
        before_cleanup: bool,
    ) -> Result<(), String> {
        if !duration_secs.is_finite() || duration_secs <= 0.0 {
            return Err("Recording duration must be a finite positive value".to_string());
        }

        let num_samples_f = duration_secs * self.sample_rate as f64;
        if !num_samples_f.is_finite() || num_samples_f < 1.0 {
            return Err("Recording duration is too short".to_string());
        }

        let num_samples = num_samples_f.round() as usize;
        let max_samples = self.sample_rate as usize * MAX_RECORDING_SECONDS;
        if num_samples > max_samples {
            return Err(format!(
                "Requested recording length exceeds max supported capture window ({}s)",
                MAX_RECORDING_SECONDS
            ));
        }

        let mut drained = vec![0.0f32; 4096];
        if let Ok(mut consumer_guard) = self.raw_recording_consumer.lock() {
            let Some(ref mut consumer) = *consumer_guard else {
                return Err("Recording buffer unavailable. Start the processor first.".to_string());
            };
            while !consumer.is_empty() {
                let read = consumer.read(&mut drained);
                if read == 0 {
                    break;
                }
            }
            self.raw_recording_pos.store(0, Ordering::Release);
            self.raw_recording_target
                .store(num_samples, Ordering::Release);
            self.recording_level_db
                .store((-120.0_f32).to_bits(), Ordering::Relaxed);
            self.raw_recording_before_cleanup
                .store(before_cleanup, Ordering::Release);
            self.recording_active.store(true, Ordering::Release);
            Ok(())
        } else {
            Err("Failed to access recording buffer".to_string())
        }
    }

    /// Stop recording and return captured audio (truncated to actual length)
    pub fn stop_raw_recording(&mut self) -> Option<Vec<f32>> {
        self.recording_active.store(false, Ordering::Release);
        self.raw_recording_before_cleanup
            .store(false, Ordering::Release);
        self.recording_level_db
            .store((-120.0_f32).to_bits(), Ordering::Relaxed);
        if let Ok(mut consumer_guard) = self.raw_recording_consumer.lock() {
            if let Some(ref mut consumer) = *consumer_guard {
                let pos = self.raw_recording_pos.load(Ordering::Acquire);
                let mut buffer = vec![0.0f32; pos];
                let mut read_total = 0usize;
                while read_total < pos {
                    let read = consumer.read(&mut buffer[read_total..]);
                    if read == 0 {
                        break;
                    }
                    read_total += read;
                }
                buffer.truncate(read_total);
                self.raw_recording_target.store(0, Ordering::Release);
                return Some(buffer);
            }
        }
        None
    }

    /// Check if recording is complete
    pub fn is_recording_complete(&self) -> bool {
        let target = self.raw_recording_target.load(Ordering::Acquire);
        let pos = self.raw_recording_pos.load(Ordering::Acquire);
        target > 0 && pos >= target
    }

    /// Get recording progress (0.0 to 1.0)
    pub fn recording_progress(&self) -> f32 {
        let target = self.raw_recording_target.load(Ordering::Acquire);
        if target == 0 {
            return 0.0;
        }
        let pos = self.raw_recording_pos.load(Ordering::Acquire);
        (pos as f32 / target as f32).min(1.0)
    }

    /// Get current recording level as RMS in dB (for level meter visualization)
    /// Returns -inf dB if no recording is active
    pub fn recording_level_db(&self) -> f32 {
        f32::from_bits(self.recording_level_db.load(Ordering::Relaxed))
    }

    /// Manually set output mute state (useful for calibration workflow)
    pub fn set_output_mute(&self, muted: bool) {
        self.output_muted.store(muted, Ordering::Release);
    }

    /// Queue a mono calibration probe for the selected output stream.
    ///
    /// The output callback duplicates the mono samples across the active
    /// endpoint's channels and temporarily drains the normal monitor path.
    pub fn queue_output_probe(&self, samples: &[f32]) -> Result<(), String> {
        if !self.is_running() {
            return Err("Audio processor must be running before queuing a probe".to_string());
        }
        if self.output_muted.load(Ordering::Acquire) {
            return Err("Unmute output before playing a calibration probe".to_string());
        }
        if samples.is_empty() {
            return Err("Output probe must contain at least one sample".to_string());
        }
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err("Output probe contains non-finite samples".to_string());
        }
        if self.output_probe_active.load(Ordering::Acquire)
            || self
                .output_probe_cancel_requested
                .load(Ordering::Acquire)
        {
            return Err("An output probe is already active".to_string());
        }

        let max_samples =
            self.output_sample_rate() as usize * MAX_OUTPUT_PROBE_SECONDS;
        if samples.len() > max_samples {
            return Err(format!(
                "Output probe exceeds the {} second limit",
                MAX_OUTPUT_PROBE_SECONDS
            ));
        }

        let mut producer_guard = self
            .output_probe_producer
            .lock()
            .map_err(|_| "Failed to access output probe buffer".to_string())?;
        let producer = producer_guard
            .as_mut()
            .ok_or_else(|| "Output probe buffer is unavailable".to_string())?;
        if producer.free_len() != producer.capacity() {
            return Err("Output probe buffer still contains stale samples".to_string());
        }
        let written = producer.write(samples);
        if written != samples.len() {
            return Err(format!(
                "Output probe write was incomplete ({written}/{})",
                samples.len()
            ));
        }

        self.output_probe_complete.store(false, Ordering::Release);
        self.output_probe_cancel_requested
            .store(false, Ordering::Release);
        self.output_probe_active.store(true, Ordering::Release);
        Ok(())
    }

    /// Whether the last queued selected-route probe has finished rendering.
    pub fn is_output_probe_complete(&self) -> bool {
        self.output_probe_complete.load(Ordering::Acquire)
    }

    /// Cancel the selected-route probe and ask the output callback to drain it.
    pub fn cancel_output_probe(&self) {
        if self.output_probe_active.load(Ordering::Acquire) {
            self.output_probe_cancel_requested
                .store(true, Ordering::Release);
        }
    }
}
