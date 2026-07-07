//! Lightweight true-peak style detector.
//!
//! This is a measurement path, not a limiter. It uses 4x Catmull-Rom
//! interpolation over adjacent samples to catch likely inter-sample overs.

#[derive(Debug, Clone)]
pub struct TruePeakDetector {
    history: [f32; 3],
    filled: usize,
    last_peak: f32,
}

impl TruePeakDetector {
    pub fn new() -> Self {
        Self {
            history: [0.0; 3],
            filled: 0,
            last_peak: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.history = [0.0; 3];
        self.filled = 0;
        self.last_peak = 0.0;
    }

    pub fn process_block(&mut self, samples: &[f32]) -> f32 {
        let mut peak = 0.0_f32;

        for sample in samples.iter().copied() {
            let sample = if sample.is_finite() { sample } else { 0.0 };
            peak = peak.max(sample.abs());

            if self.filled < self.history.len() {
                self.history[self.filled] = sample;
                self.filled += 1;
                continue;
            }

            let x0 = self.history[0];
            let x1 = self.history[1];
            let x2 = self.history[2];
            let x3 = sample;

            peak = peak.max(Self::segment_true_peak(x0, x1, x2, x3));
            self.history = [x1, x2, x3];
        }

        self.last_peak = peak;
        peak
    }

    pub fn last_peak(&self) -> f32 {
        self.last_peak
    }

    #[inline]
    fn segment_true_peak(x0: f32, x1: f32, x2: f32, x3: f32) -> f32 {
        const SUBSAMPLE_POINTS: [f32; 3] = [0.25, 0.5, 0.75];
        let mut peak = x1.abs().max(x2.abs());
        for t in SUBSAMPLE_POINTS {
            peak = peak.max(Self::catmull_rom(x0, x1, x2, x3, t).abs());
        }
        peak
    }

    #[inline]
    fn catmull_rom(x0: f32, x1: f32, x2: f32, x3: f32, t: f32) -> f32 {
        let t2 = t * t;
        let t3 = t2 * t;
        0.5 * ((2.0 * x1)
            + (-x0 + x2) * t
            + (2.0 * x0 - 5.0 * x1 + 4.0 * x2 - x3) * t2
            + (-x0 + 3.0 * x1 - 3.0 * x2 + x3) * t3)
    }
}

impl Default for TruePeakDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_signal_matches_sample_peak() {
        let mut detector = TruePeakDetector::new();
        let peak = detector.process_block(&[0.5; 16]);

        assert!((peak - 0.5).abs() < 1e-6);
        assert!((detector.last_peak() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn cubic_oversampling_detects_intersample_overshoot() {
        let mut detector = TruePeakDetector::new();
        let peak = detector.process_block(&[0.0, 1.0, 1.0, 0.0, 0.0]);

        assert!(peak > 1.0);
    }

    #[test]
    fn reset_clears_history_and_peak() {
        let mut detector = TruePeakDetector::new();
        assert!(detector.process_block(&[0.0, 1.0, 1.0, 0.0, 0.0]) > 1.0);

        detector.reset();

        assert_eq!(detector.last_peak(), 0.0);
        assert_eq!(detector.process_block(&[0.25]), 0.25);
    }
}
