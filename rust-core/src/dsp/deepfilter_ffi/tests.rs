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
        // Test fixtures contain no privileged or secret data; uniqueness only
        // prevents parallel test collisions and is not a security boundary.
        // nosemgrep: rust.lang.security.temp-dir.temp-dir
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
    fn test_external_model_path_requires_explicit_opt_in() {
        let _guard = env_lock().lock().unwrap();
        let temp_dir = temp_test_dir("deepfilter-file");
        let model_path = temp_dir.join("custom-model.tar.gz");
        fs::write(&model_path, []).unwrap();

        std::env::set_var("DEEPFILTER_MODEL_PATH", &model_path);
        std::env::remove_var("AUDIOFORGE_ALLOW_EXTERNAL_DF");
        let blocked = find_model_path(
            DeepFilterModel::LowLatency,
            &AppOwnedDeepFilterPaths::default(),
            external_deepfilter_paths_allowed(),
        );
        std::env::set_var("AUDIOFORGE_ALLOW_EXTERNAL_DF", "1");
        let allowed = find_model_path(
            DeepFilterModel::LowLatency,
            &AppOwnedDeepFilterPaths::default(),
            external_deepfilter_paths_allowed(),
        );
        std::env::remove_var("DEEPFILTER_MODEL_PATH");
        std::env::remove_var("AUDIOFORGE_ALLOW_EXTERNAL_DF");

        assert_ne!(blocked, Some(model_path.clone()));
        assert_eq!(allowed, Some(model_path));
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_app_owned_model_path_does_not_require_external_opt_in() {
        let _guard = env_lock().lock().unwrap();
        let temp_dir = temp_test_dir("deepfilter-dir");
        let model_path = temp_dir.join(DeepFilterModel::LowLatency.filename());
        fs::write(&model_path, []).unwrap();

        let app_paths = AppOwnedDeepFilterPaths {
            library: None,
            model: Some(temp_dir.clone()),
        };
        let resolved = find_model_path(DeepFilterModel::LowLatency, &app_paths, false);

        assert_eq!(resolved, Some(model_path));
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_opted_in_external_model_overrides_app_owned_model() {
        let _guard = env_lock().lock().unwrap();
        let temp_dir = temp_test_dir("deepfilter-model-precedence");
        let bundled_dir = temp_dir.join("bundled");
        fs::create_dir_all(&bundled_dir).unwrap();
        let bundled_model = bundled_dir.join(DeepFilterModel::LowLatency.filename());
        let external_model = temp_dir.join("external-model.tar.gz");
        fs::write(&bundled_model, []).unwrap();
        fs::write(&external_model, []).unwrap();
        let app_paths = AppOwnedDeepFilterPaths {
            library: None,
            model: Some(bundled_dir),
        };

        std::env::set_var("DEEPFILTER_MODEL_PATH", &external_model);
        let default_path = find_model_path(DeepFilterModel::LowLatency, &app_paths, false);
        let overridden_path = find_model_path(DeepFilterModel::LowLatency, &app_paths, true);
        std::env::remove_var("DEEPFILTER_MODEL_PATH");

        assert_eq!(default_path, Some(bundled_model));
        assert_eq!(overridden_path, Some(external_model));
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_external_library_path_requires_explicit_opt_in() {
        let _guard = env_lock().lock().unwrap();
        let temp_dir = temp_test_dir("deepfilter-library");
        let library_path = temp_dir.join(if cfg!(target_os = "windows") {
            "df.dll"
        } else if cfg!(target_os = "macos") {
            "libdf.dylib"
        } else {
            "libdf.so"
        });
        fs::write(&library_path, []).unwrap();
        let library_name = library_path.file_name().unwrap().to_str().unwrap();

        std::env::set_var("DEEPFILTER_LIB_PATH", &library_path);
        let blocked = DeepFilterLib::trusted_library_candidates(
            library_name,
            &AppOwnedDeepFilterPaths::default(),
            false,
        );
        let allowed = DeepFilterLib::trusted_library_candidates(
            library_name,
            &AppOwnedDeepFilterPaths::default(),
            true,
        );
        std::env::remove_var("DEEPFILTER_LIB_PATH");

        let canonical = library_path.canonicalize().unwrap();
        assert!(!blocked.contains(&canonical));
        assert!(allowed.contains(&canonical));
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_opted_in_external_library_precedes_app_owned_library() {
        let _guard = env_lock().lock().unwrap();
        let temp_dir = temp_test_dir("deepfilter-library-precedence");
        let library_name = if cfg!(target_os = "windows") {
            "df.dll"
        } else if cfg!(target_os = "macos") {
            "libdf.dylib"
        } else {
            "libdf.so"
        };
        let bundled_dir = temp_dir.join("bundled");
        let external_dir = temp_dir.join("external");
        fs::create_dir_all(&bundled_dir).unwrap();
        fs::create_dir_all(&external_dir).unwrap();
        let bundled_library = bundled_dir.join(library_name);
        let external_library = external_dir.join(library_name);
        fs::write(&bundled_library, []).unwrap();
        fs::write(&external_library, []).unwrap();
        let app_paths = AppOwnedDeepFilterPaths {
            library: Some(bundled_library.clone()),
            model: None,
        };

        std::env::set_var("DEEPFILTER_LIB_PATH", &external_library);
        let default_candidates =
            DeepFilterLib::trusted_library_candidates(library_name, &app_paths, false);
        let override_candidates =
            DeepFilterLib::trusted_library_candidates(library_name, &app_paths, true);
        std::env::remove_var("DEEPFILTER_LIB_PATH");

        assert_eq!(default_candidates.first(), bundled_library.canonicalize().ok().as_ref());
        assert_eq!(override_candidates.first(), external_library.canonicalize().ok().as_ref());
        let _ = fs::remove_dir_all(temp_dir);
    }
}
