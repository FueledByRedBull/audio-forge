use super::AudioProcessor;
use crate::audio::clock::now_micros;
use crate::audio::rt::RtErrorCode;
use crate::dsp::Compressor;
use std::sync::atomic::Ordering;
#[cfg(not(windows))]
use thread_priority::{set_current_thread_priority, ThreadPriority};

impl AudioProcessor {
    // === Metering ===

    /// Get input peak level in dB
    pub fn get_input_peak_db(&self) -> f32 {
        f32::from_bits(self.shared.input.peak.load(Ordering::Relaxed))
    }

    /// Get input RMS level in dB
    pub fn get_input_rms_db(&self) -> f32 {
        f32::from_bits(self.shared.input.rms.load(Ordering::Relaxed))
    }

    pub fn get_input_crest_factor_db(&self) -> f32 {
        f32::from_bits(self.shared.input.crest_factor_db.load(Ordering::Relaxed))
    }

    /// Get output peak level in dB
    pub fn get_output_peak_db(&self) -> f32 {
        f32::from_bits(self.shared.output.peak.load(Ordering::Relaxed))
    }

    /// Get output RMS level in dB
    pub fn get_output_rms_db(&self) -> f32 {
        f32::from_bits(self.shared.output.rms.load(Ordering::Relaxed))
    }

    pub fn get_output_crest_factor_db(&self) -> f32 {
        f32::from_bits(self.shared.output.crest_factor_db.load(Ordering::Relaxed))
    }

    // Peak "since last read" values are linear amplitudes. Each take swaps the
    // atomic back to 0, so a peak the DSP thread publishes between the read
    // and the reset lands in the next take. Only one caller may take them.

    /// Largest absolute input sample since the previous take (0.0 = silence).
    pub fn take_input_peak_since_read(&self) -> f32 {
        f32::from_bits(self.shared.input.peak_hold.swap(0, Ordering::Relaxed))
    }

    /// Largest absolute sample of the OUT meter tap (before the final
    /// true-peak limiter and clamp) since the previous take.
    pub fn take_output_pre_limiter_peak_since_read(&self) -> f32 {
        f32::from_bits(
            self.shared
                .output
                .pre_limiter_peak_hold
                .swap(0, Ordering::Relaxed),
        )
    }

    /// Largest absolute sample queued for the output device (after the final
    /// limiter, clamp and fade) since the previous take.
    pub fn take_output_peak_since_read(&self) -> f32 {
        f32::from_bits(
            self.shared
                .output
                .final_peak_hold
                .swap(0, Ordering::Relaxed),
        )
    }

    /// Input samples at or above full scale, cumulative since start.
    pub fn get_input_full_scale_count(&self) -> u64 {
        self.shared.input.full_scale_count.load(Ordering::Relaxed)
    }

    /// Queued output samples at or above full scale, cumulative since start.
    pub fn get_output_full_scale_count(&self) -> u64 {
        self.shared.output.full_scale_count.load(Ordering::Relaxed)
    }

    pub fn get_output_short_term_lufs(&self) -> f32 {
        f32::from_bits(self.shared.output.short_term_lufs.load(Ordering::Relaxed))
    }

    /// Latest stereo input correlation (-1.0 to +1.0).
    pub fn get_input_stereo_correlation(&self) -> f32 {
        f32::from_bits(self.shared.input.stereo_correlation.load(Ordering::Relaxed))
    }

    /// Number of stereo input blocks with strong negative correlation.
    pub fn get_input_phase_warning_count(&self) -> u64 {
        self.shared
            .input
            .phase_warning_count
            .load(Ordering::Relaxed)
    }

    /// Get compressor gain reduction in dB
    pub fn get_compressor_gain_reduction_db(&self) -> f32 {
        f32::from_bits(
            self.shared
                .compressor
                .gain_reduction
                .load(Ordering::Relaxed),
        )
    }

    /// Get current processing latency in milliseconds
    pub fn get_latency_ms(&self) -> f32 {
        let latency_us = self.shared.dsp.latency_us.load(Ordering::Relaxed);
        latency_us as f32 / 1000.0
    }

    /// Get engine latency excluding measured external route compensation.
    pub fn get_engine_latency_ms(&self) -> f32 {
        let latency_us = self.shared.dsp.latency_us.load(Ordering::Relaxed);
        let compensation_us = self
            .shared
            .dsp
            .latency_compensation_us
            .load(Ordering::Relaxed);
        latency_us.saturating_sub(compensation_us) as f32 / 1000.0
    }

    /// Set user latency compensation in milliseconds (added to reported latency).
    pub fn set_latency_compensation_ms(&self, compensation_ms: f32) {
        let Some(clamped_ms) = self.clamp_live_control_f32(compensation_ms, 0.0, 5000.0) else {
            return;
        };
        let compensation_us = (clamped_ms * 1000.0) as u64;
        self.shared
            .dsp
            .latency_compensation_us
            .store(compensation_us, Ordering::Relaxed);
    }

    // === DSP Performance Metrics ===

    /// Get smoothed input buffer fill level in samples
    pub fn get_input_buffer_smoothed_samples(&self) -> u32 {
        self.shared
            .input
            .smoothed_buffer_len
            .load(Ordering::Relaxed)
    }

    /// Get output buffer fill level in samples
    pub fn get_output_buffer_samples(&self) -> u32 {
        self.shared.output.buffer_len.load(Ordering::Relaxed)
    }

    /// Get active output device sample rate in Hz.
    pub fn output_sample_rate(&self) -> u32 {
        self.shared.output.sample_rate.load(Ordering::Relaxed)
    }

    /// Negotiated fixed CPAL output buffer size, or zero for driver default.
    pub fn output_fixed_buffer_frames(&self) -> u32 {
        self.shared
            .output
            .fixed_buffer_frames
            .load(Ordering::Relaxed)
    }

    /// Negotiated fixed CPAL input buffer size, or zero for driver default.
    pub fn input_fixed_buffer_frames(&self) -> u32 {
        self.shared
            .input
            .fixed_buffer_frames
            .load(Ordering::Relaxed)
    }

    // === Dropped Sample Tracking ===

    /// Get dropped sample count (samples lost due to buffer overflow)
    pub fn get_dropped_samples(&self) -> u64 {
        self.shared
            .input
            .dropped_counter()
            .load(Ordering::Relaxed)
            .saturating_sub(self.shared.input.dropped_baseline.load(Ordering::Relaxed))
    }

    /// Reset dropped sample counter
    pub fn reset_dropped_samples(&self) {
        self.shared.input.dropped_baseline.store(
            self.shared.input.dropped_counter().load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
    }

    /// Real-time control-snapshot reads that stayed unstable (a writer was
    /// mid-update for the whole retry budget) and were re-armed for the next pass.
    pub fn get_lock_contention_count(&self) -> u64 {
        self.shared
            .dsp
            .lock_contention_count
            .load(Ordering::Relaxed)
    }

    /// Blocks where the downstream chain produced non-finite samples.
    pub fn get_downstream_non_finite_count(&self) -> u64 {
        self.shared
            .dsp
            .downstream_non_finite_count
            .load(Ordering::Relaxed)
    }

    /// Output resampler failures since start.
    pub fn get_output_resampler_error_count(&self) -> u64 {
        self.shared
            .output
            .resampler_error_count
            .load(Ordering::Relaxed)
    }

    /// Whether the DSP thread runs without MMCSS Pro Audio scheduling.
    pub fn is_dsp_scheduling_degraded(&self) -> bool {
        self.shared.dsp.scheduling_degraded.load(Ordering::Relaxed)
    }

    /// Why the live compressor's auto makeup is off or not tracking, if it is.
    pub fn get_auto_makeup_error(&self) -> Option<&'static str> {
        Compressor::auto_makeup_error_text(
            self.shared
                .compressor
                .auto_makeup_error
                .load(Ordering::Relaxed),
        )
    }

    pub fn get_suppressor_non_finite_count(&self) -> u64 {
        self.shared
            .suppressor
            .non_finite_count
            .load(Ordering::Relaxed)
    }

    pub fn get_rt_error_code(&self) -> u32 {
        self.shared.dsp.rt_error_code.load(Ordering::Relaxed)
    }

    pub fn get_rt_error_name(&self) -> &'static str {
        RtErrorCode::from_u32(self.get_rt_error_code()).as_str()
    }

    pub fn get_input_callback_error_count(&self) -> u64 {
        self.shared
            .input
            .callback_error_count
            .load(Ordering::Relaxed)
    }

    pub fn get_output_callback_error_count(&self) -> u64 {
        self.shared
            .output
            .callback_error_count
            .load(Ordering::Relaxed)
    }

    pub fn get_rt_buffer_overflow_count(&self) -> u64 {
        self.shared
            .dsp
            .rt_buffer_overflow_count
            .load(Ordering::Relaxed)
    }

    /// Age of last input callback heartbeat in milliseconds.
    pub fn get_input_callback_age_ms(&self) -> u64 {
        let last = self.shared.input.callback_time_us.load(Ordering::Relaxed);
        if last == 0 {
            return u64::MAX;
        }
        let now = now_micros();
        now.saturating_sub(last) / 1000
    }

    /// Age of last output callback heartbeat in milliseconds.
    pub fn get_output_callback_age_ms(&self) -> u64 {
        let last = self.shared.output.callback_time_us.load(Ordering::Relaxed);
        if last == 0 {
            return u64::MAX;
        }
        let now = now_micros();
        now.saturating_sub(last) / 1000
    }

    /// Current consecutive output underrun streak.
    pub fn get_output_underrun_streak(&self) -> u32 {
        self.shared.output.underrun_streak.load(Ordering::Relaxed)
    }

    /// Total output underrun callbacks since start.
    pub fn get_output_underrun_total(&self) -> u64 {
        self.shared.output.underrun_total.load(Ordering::Relaxed)
    }

    /// Output callbacks that ran dry before the stream was primed (before a
    /// callback was fully served after real audio was queued). Start-up
    /// priming, not counted as dropouts.
    pub fn get_output_priming_underrun_total(&self) -> u64 {
        self.shared
            .output
            .priming_underrun_total
            .load(Ordering::Relaxed)
    }

    /// Samples dropped by DSP-side jitter control.
    pub fn get_jitter_dropped_samples(&self) -> u64 {
        self.shared
            .output
            .jitter_dropped_samples
            .load(Ordering::Relaxed)
    }

    /// Number of normal drift-retime adjustments.
    pub fn get_output_retime_adjustment_count(&self) -> u64 {
        self.shared
            .output
            .retime_adjustment_count
            .load(Ordering::Relaxed)
    }

    /// Number of true output recovery events.
    pub fn get_output_recovery_event_count(&self) -> u64 {
        self.shared
            .output
            .recovery_event_count
            .load(Ordering::Relaxed)
    }

    pub fn get_dsp_idle_wakeup_count(&self) -> u64 {
        self.shared.dsp.idle_wakeup_count.load(Ordering::Relaxed)
    }

    pub fn get_dsp_idle_sleep_us(&self) -> u64 {
        self.shared.dsp.idle_sleep_us.load(Ordering::Relaxed)
    }

    /// Whether the currently selected suppressor backend is operational.
    pub fn is_noise_backend_available(&self) -> bool {
        self.shared
            .suppressor
            .backend_available
            .load(Ordering::Relaxed)
    }

    /// Whether the suppressor backend has failed after startup.
    pub fn noise_backend_failed(&self) -> bool {
        self.shared
            .suppressor
            .backend_failed
            .load(Ordering::Relaxed)
    }

    /// Last suppressor backend error, if any.
    pub fn noise_backend_error(&self) -> Option<String> {
        let stored = self
            .shared
            .suppressor
            .backend_error
            .lock()
            .ok()
            .and_then(|error| error.clone());
        stored.or_else(|| {
            let code = RtErrorCode::from_u32(self.shared.dsp.rt_error_code.load(Ordering::Relaxed));
            (code == RtErrorCode::SuppressorBackendFailed).then(|| code.as_str().to_string())
        })
    }

    /// Suppress watchdog-driven stream recovery during intrusive UI workflows.
    pub fn set_recovery_suppressed(&self, suppressed: bool) {
        self.shared
            .stream
            .recovery_suppressed
            .store(suppressed, Ordering::Release);
    }

    /// Whether watchdog-driven stream recovery is currently suppressed.
    pub fn is_recovery_suppressed(&self) -> bool {
        self.shared
            .stream
            .recovery_suppressed
            .load(Ordering::Acquire)
    }
}
