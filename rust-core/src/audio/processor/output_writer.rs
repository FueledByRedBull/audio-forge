struct OutputWriteCounters<'a> {
    jitter_dropped_samples: &'a AtomicU64,
    output_retime_adjustment_count: &'a AtomicU64,
    output_recovery_event_count: &'a AtomicU64,
    output_short_write_dropped_samples: &'a AtomicU64,
    rt_buffer_overflow_count: &'a AtomicU64,
    rt_error_code: &'a AtomicU32,
    output_buffer_len: &'a AtomicU32,
    last_output_write_time: &'a AtomicU64,
    output_clip_event_count: &'a AtomicU64,
    output_clip_peak_db: &'a AtomicU32,
    output_true_peak_event_count: &'a AtomicU64,
    output_true_peak_db: &'a AtomicU32,
    output_true_peak_input_db: &'a AtomicU32,
    output_true_peak_gain_reduction_db: &'a AtomicU32,
    output_true_peak_gain_reduction_history_db: &'a AtomicU32,
    output_true_peak_headroom_db: &'a AtomicU32,
}

#[derive(Clone, Copy)]
struct OutputWriteLimits {
    output_target_center_samples: usize,
    output_hard_backlog_samples: usize,
    discontinuity_fade_samples: usize,
    max_catchup_ratio: f32,
    max_emergency_catchup_ratio: f32,
}

struct OutputWriteContext<
    'a,
    const OUTPUT_QUEUE_CONTROL_CAPACITY: usize,
    const DISCONTINUITY_FADE_CAPACITY: usize,
    const OUTPUT_SAFETY_CAPACITY: usize,
> {
    output_producer: &'a mut AudioProducer,
    output_queue_control_scratch: &'a mut FixedAudioBuffer<f32, OUTPUT_QUEUE_CONTROL_CAPACITY>,
    discontinuity_fade_scratch: &'a mut FixedAudioBuffer<f32, DISCONTINUITY_FADE_CAPACITY>,
    output_safety_scratch: &'a mut FixedAudioBuffer<f32, OUTPUT_SAFETY_CAPACITY>,
    true_peak_detector: &'a mut TruePeakDetector,
    true_peak_limiter: &'a mut TruePeakLimiter,
    drift_error_ema: &'a mut f32,
    drift_retimer: &'a mut DriftRetimer,
    discontinuity_fade_remaining: &'a Cell<usize>,
    /// Fade-in owed to the output stream after a short write dropped samples.
    /// It applies after the true-peak delay, where the dropout happened.
    output_fade_remaining: usize,
    limiter_enabled: &'a AtomicBool,
    output_ceiling_linear: &'a Cell<f32>,
    counters: OutputWriteCounters<'a>,
    limits: OutputWriteLimits,
}

impl<
        'a,
        const OUTPUT_QUEUE_CONTROL_CAPACITY: usize,
        const DISCONTINUITY_FADE_CAPACITY: usize,
        const OUTPUT_SAFETY_CAPACITY: usize,
    >
    OutputWriteContext<
        'a,
        OUTPUT_QUEUE_CONTROL_CAPACITY,
        DISCONTINUITY_FADE_CAPACITY,
        OUTPUT_SAFETY_CAPACITY,
    >
{
    fn write_chunk(&mut self, write_source: &[f32], clean_path: bool) -> bool {
        if write_source.is_empty() {
            return false;
        }

        let capacity = self.output_producer.capacity();
        let free = self.output_producer.free_len();
        let fill = capacity.saturating_sub(free);

        let mut write_slice = write_source;
        if !clean_path {
            write_slice = Self::apply_drift_retime(
                write_source,
                (fill, capacity),
                self.drift_error_ema,
                self.output_queue_control_scratch,
                self.drift_retimer,
                &self.counters,
                self.limits,
            );
            write_slice = Self::apply_discontinuity_fade(
                write_slice,
                self.discontinuity_fade_scratch,
                self.discontinuity_fade_remaining,
                self.limits.discontinuity_fade_samples,
                &self.counters,
            );
        } else {
            self.drift_retimer.reset();
        }

        Self::sanitize_and_limit(
            write_slice,
            self.output_safety_scratch,
            self.true_peak_detector,
            self.true_peak_limiter,
            self.limiter_enabled,
            self.output_ceiling_linear,
            &self.counters,
        );
        fade_in(
            self.output_safety_scratch.as_mut_slice(),
            &mut self.output_fade_remaining,
            self.limits.discontinuity_fade_samples,
        );
        Self::write_to_output_queue(
            self.output_safety_scratch.as_slice(),
            free,
            self.output_producer,
            &mut self.output_fade_remaining,
            self.limits.discontinuity_fade_samples,
            &self.counters,
        );
        Self::update_output_fill(self.output_producer, &self.counters);
        true
    }

    fn apply_drift_retime<'b>(
        write_source: &'b [f32],
        queue: (usize, usize),
        drift_error_ema: &mut f32,
        output_queue_control_scratch: &'b mut FixedAudioBuffer<f32, OUTPUT_QUEUE_CONTROL_CAPACITY>,
        drift_retimer: &mut DriftRetimer,
        counters: &OutputWriteCounters<'_>,
        limits: OutputWriteLimits,
    ) -> &'b [f32] {
        let (fill, producer_capacity) = queue;
        let error = fill as f32 - limits.output_target_center_samples as f32;
        *drift_error_ema = *drift_error_ema * 0.85 + error * 0.15;
        let positive_zone = limits
            .output_hard_backlog_samples
            .saturating_sub(limits.output_target_center_samples)
            .max(1) as f32;
        let negative_zone = limits.output_target_center_samples.max(1) as f32;
        let normalized_error = if *drift_error_ema >= 0.0 {
            (*drift_error_ema / positive_zone).clamp(0.0, 1.0)
        } else {
            (*drift_error_ema / negative_zone).clamp(-1.0, 0.0)
        };
        let mut queue_speed_ratio = (1.0 + normalized_error * OUTPUT_DRIFT_MAX_RATIO_ADJUST)
            .clamp(OUTPUT_DRIFT_MAX_EXPANSION_RATIO, limits.max_catchup_ratio);
        if fill >= limits.output_hard_backlog_samples {
            queue_speed_ratio = limits.max_emergency_catchup_ratio;
        }

        let adjusted_slice = retime_audio_block(
            write_source,
            queue_speed_ratio,
            producer_capacity
                .max(1)
                .min(output_queue_control_scratch.capacity()),
            output_queue_control_scratch,
            drift_retimer,
        );
        if adjusted_slice.len() != write_source.len() {
            let delta = write_source.len().abs_diff(adjusted_slice.len());
            if adjusted_slice.len() < write_source.len() {
                counters
                    .jitter_dropped_samples
                    .fetch_add(delta as u64, Ordering::Relaxed);
            }
            counters
                .output_retime_adjustment_count
                .fetch_add(1, Ordering::Relaxed);
        }
        adjusted_slice
    }

    fn apply_discontinuity_fade<'b>(
        write_slice: &'b [f32],
        discontinuity_fade_scratch: &'b mut FixedAudioBuffer<f32, DISCONTINUITY_FADE_CAPACITY>,
        discontinuity_fade_remaining: &Cell<usize>,
        discontinuity_fade_samples: usize,
        counters: &OutputWriteCounters<'_>,
    ) -> &'b [f32] {
        let fade_remaining = discontinuity_fade_remaining.get();
        if fade_remaining == 0 || write_slice.is_empty() {
            return write_slice;
        }

        discontinuity_fade_scratch.clear();
        let written = discontinuity_fade_scratch.extend_from_slice(write_slice);
        if written < write_slice.len() {
            Self::record_fixed_buffer_overflow(counters);
        }
        let mut remaining = fade_remaining;
        fade_in(
            discontinuity_fade_scratch.as_mut_slice(),
            &mut remaining,
            discontinuity_fade_samples,
        );
        discontinuity_fade_remaining.set(remaining);
        discontinuity_fade_scratch.as_slice()
    }

    fn sanitize_and_limit<'b>(
        write_slice: &[f32],
        output_safety_scratch: &'b mut FixedAudioBuffer<f32, OUTPUT_SAFETY_CAPACITY>,
        true_peak_detector: &mut TruePeakDetector,
        true_peak_limiter: &mut TruePeakLimiter,
        limiter_enabled: &AtomicBool,
        output_ceiling_linear: &Cell<f32>,
        counters: &OutputWriteCounters<'_>,
    ) -> &'b [f32] {
        output_safety_scratch.clear();
        let safety_written = output_safety_scratch.extend_from_slice(write_slice);
        if safety_written < write_slice.len() {
            Self::record_fixed_buffer_overflow(counters);
        }
        let limiting = limiter_enabled.load(Ordering::Acquire);
        let output_ceiling = if limiting {
            output_ceiling_linear.get()
        } else {
            1.0
        };
        sanitize_non_finite_inplace(output_safety_scratch.as_mut_slice());
        // The stage always runs so its delay never leaves or re-enters the
        // output timeline when the limiter is toggled.
        if limiting {
            true_peak_limiter.set_ceiling_linear(output_ceiling);
        }
        true_peak_limiter.set_enabled(limiting);
        let stats = true_peak_limiter.process_block_inplace(output_safety_scratch.as_mut_slice());
        if limiting {
            Self::record_true_peak_limiter_stats(stats, output_ceiling, counters);
        } else {
            counters
                .output_true_peak_gain_reduction_db
                .store(0.0_f32.to_bits(), Ordering::Relaxed);
            update_decaying_peak_db(
                0.0,
                counters.output_true_peak_gain_reduction_history_db,
                0.15,
            );
        }
        sanitize_and_clamp_output_inplace_with_metrics(
            output_safety_scratch.as_mut_slice(),
            output_ceiling,
            counters.output_clip_event_count,
            counters.output_clip_peak_db,
        );
        Self::record_true_peak(
            true_peak_detector,
            output_safety_scratch.as_slice(),
            output_ceiling,
            counters,
        );
        output_safety_scratch.as_slice()
    }

    fn record_true_peak(
        true_peak_detector: &mut TruePeakDetector,
        samples: &[f32],
        output_ceiling: f32,
        counters: &OutputWriteCounters<'_>,
    ) {
        let true_peak = true_peak_detector.process_block(samples);
        let peak_db = 20.0 * true_peak.max(1e-10).log10();
        counters
            .output_true_peak_db
            .store(peak_db.to_bits(), Ordering::Relaxed);
        let headroom_db = 20.0 * (output_ceiling.max(1e-10) / true_peak.max(1e-10)).log10();
        counters
            .output_true_peak_headroom_db
            .store(headroom_db.to_bits(), Ordering::Relaxed);
    }

    fn record_true_peak_limiter_stats(
        stats: TruePeakLimiterBlockStats,
        output_ceiling: f32,
        counters: &OutputWriteCounters<'_>,
    ) {
        if stats.limited_events > 0 {
            counters
                .output_true_peak_event_count
                .fetch_add(stats.limited_events, Ordering::Relaxed);
        }
        let input_peak_db = 20.0 * stats.input_true_peak.max(1e-10).log10();
        counters
            .output_true_peak_input_db
            .store(input_peak_db.to_bits(), Ordering::Relaxed);
        counters
            .output_true_peak_gain_reduction_db
            .store(stats.max_gain_reduction_db.to_bits(), Ordering::Relaxed);
        update_decaying_peak_db(
            stats.max_gain_reduction_db,
            counters.output_true_peak_gain_reduction_history_db,
            0.15,
        );
        let headroom_db =
            20.0 * (output_ceiling.max(1e-10) / stats.output_true_peak.max(1e-10)).log10();
        counters
            .output_true_peak_headroom_db
            .store(headroom_db.to_bits(), Ordering::Relaxed);
    }

    fn write_to_output_queue(
        pending_slice: &[f32],
        free: usize,
        output_producer: &mut AudioProducer,
        output_fade_remaining: &mut usize,
        discontinuity_fade_samples: usize,
        counters: &OutputWriteCounters<'_>,
    ) {
        let mut pending_slice = pending_slice;
        if pending_slice.len() > free {
            let dropped = pending_slice.len() - free;
            counters
                .output_short_write_dropped_samples
                .fetch_add(dropped as u64, Ordering::Relaxed);
            counters
                .output_recovery_event_count
                .fetch_add(1, Ordering::Relaxed);
            *output_fade_remaining = discontinuity_fade_samples;
            pending_slice = &pending_slice[..free];
        }

        if pending_slice.is_empty() {
            return;
        }

        let written = output_producer.write(pending_slice);
        if written > 0 {
            counters
                .last_output_write_time
                .store(now_micros(), Ordering::Relaxed);
        }
        if written < pending_slice.len() {
            let dropped = pending_slice.len() - written;
            counters
                .output_short_write_dropped_samples
                .fetch_add(dropped as u64, Ordering::Relaxed);
            counters
                .output_recovery_event_count
                .fetch_add(1, Ordering::Relaxed);
            *output_fade_remaining = discontinuity_fade_samples;
        }
    }

    fn update_output_fill(
        output_producer: &AudioProducer,
        counters: &OutputWriteCounters<'_>,
    ) {
        let new_fill = output_producer
            .capacity()
            .saturating_sub(output_producer.free_len());
        counters
            .output_buffer_len
            .store(new_fill as u32, Ordering::Relaxed);
    }

    fn record_fixed_buffer_overflow(counters: &OutputWriteCounters<'_>) {
        counters
            .rt_buffer_overflow_count
            .fetch_add(1, Ordering::Relaxed);
        store_rt_error(counters.rt_error_code, RtErrorCode::FixedBufferOverflow);
    }
}

/// Multiply the start of `samples` by the remaining part of a linear fade-in.
fn fade_in(samples: &mut [f32], remaining: &mut usize, total: usize) {
    let count = (*remaining).min(samples.len());
    let elapsed = total.saturating_sub(*remaining);
    for (i, sample) in samples.iter_mut().take(count).enumerate() {
        *sample *= ((elapsed + i + 1) as f32 / total as f32).clamp(0.0, 1.0);
    }
    *remaining -= count;
}
