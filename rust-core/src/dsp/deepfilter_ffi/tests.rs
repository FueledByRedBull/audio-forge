#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::{Mutex, OnceLock};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn temp_test_dir(name: &str) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("audioforge-{}-{}", name, suffix));
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn test_ffi_availability() {
        let processor = DeepFilterProcessor::default();
        // This test passes regardless of whether FFI is available
        // If not available, the processor will use passthrough
        if processor.is_ffi_available() {
            println!("DeepFilterNet C FFI is available");
        } else {
            println!("DeepFilterNet C FFI is not available (using passthrough)");
        }
    }

    #[test]
    fn test_frame_size() {
        assert_eq!(DEEPFILTER_FRAME_SIZE, 480);
    }

    #[test]
    fn test_strength_operations() {
        let processor = DeepFilterProcessor::default();
        processor.set_strength(0.5);
        assert_eq!(processor.get_strength(), 0.5);
    }

    #[test]
    fn test_buffering_logic() {
        let mut processor = DeepFilterProcessor::default();
        assert_eq!(processor.pending_input(), 0);

        processor.push_samples(&[0.1; 100]);
        assert_eq!(processor.pending_input(), 100);

        processor.push_samples(&[0.2; 400]);
        assert_eq!(processor.pending_input(), 500); // Accumulated

        processor.process_frames();
        assert_eq!(processor.pending_input(), 20); // 500 - 480 = 20 remaining
    }

    #[test]
    fn test_latency_samples() {
        let processor = DeepFilterProcessor::default();
        assert_eq!(processor.latency_samples(), 480);
    }

    #[test]
    fn test_enable_disable() {
        let mut processor = DeepFilterProcessor::default();
        assert!(processor.is_enabled());

        processor.set_enabled(false);
        assert!(!processor.is_enabled());

        processor.set_enabled(true);
        assert!(processor.is_enabled());
    }

    #[test]
    fn test_soft_reset() {
        let mut processor = DeepFilterProcessor::default();
        processor.push_samples(&[0.1; 500]);
        processor.process_frames();

        processor.soft_reset();
        assert_eq!(processor.pending_input(), 0);
        assert_eq!(processor.available_samples(), 0);
    }

    #[test]
    fn test_passthrough_processing() {
        let mut processor = DeepFilterProcessor::default();
        let input = vec![0.5; 1000];
        processor.push_samples(&input);
        processor.process_frames();

        let output = processor.pop_all_samples();
        assert!(!output.is_empty(), "Should produce output");

        // Output should contain complete frames (1000 samples = 2 frames of 480 + 40 remaining)
        assert_eq!(output.len(), 960); // Two complete frames
    }

    #[test]
    fn test_find_model_path_accepts_explicit_file() {
        let _guard = env_lock().lock().unwrap();
        let temp_dir = temp_test_dir("deepfilter-file");
        let model_path = temp_dir.join("custom-model.tar.gz");
        fs::write(&model_path, []).unwrap();

        std::env::set_var("DEEPFILTER_MODEL_PATH", &model_path);
        let resolved = find_model_path(DeepFilterModel::LowLatency);
        std::env::remove_var("DEEPFILTER_MODEL_PATH");

        assert_eq!(resolved, Some(model_path));
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_find_model_path_accepts_directory() {
        let _guard = env_lock().lock().unwrap();
        let temp_dir = temp_test_dir("deepfilter-dir");
        let model_path = temp_dir.join(DeepFilterModel::LowLatency.filename());
        fs::write(&model_path, []).unwrap();

        std::env::set_var("DEEPFILTER_MODEL_PATH", &temp_dir);
        let resolved = find_model_path(DeepFilterModel::LowLatency);
        std::env::remove_var("DEEPFILTER_MODEL_PATH");

        assert_eq!(resolved, Some(model_path));
        let _ = fs::remove_dir_all(temp_dir);
    }
}
