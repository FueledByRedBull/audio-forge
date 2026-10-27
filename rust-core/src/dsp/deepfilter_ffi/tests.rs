#[cfg(test)]
mod tests {
    use super::fake::{processor_on, Fake};
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    // The path-resolution tests below pass external overrides as arguments
    // instead of setting DEEPFILTER_* variables: mutating the process
    // environment would race with every other test that builds a processor.

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
        // Passes with or without the library, but an unavailable backend
        // (passthrough) must always say why.
        assert_eq!(
            processor.is_ffi_available(),
            processor.backend_error().is_none()
        );
    }

    /// The passthrough tests above pass without the library; this one needs the
    /// real load wherever the development df.dll sits at the repository root.
    #[test]
    fn repository_df_dll_loads_when_present() {
        let library = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("rust-core sits under the repository root")
            .join("df.dll");
        if !cfg!(all(windows, debug_assertions)) || !library.is_file() {
            eprintln!(
                "SKIPPED repository_df_dll_loads_when_present: {} absent or not a debug Windows build",
                library.display()
            );
            return;
        }
        let loaded = DeepFilterLib::try_load(&AppOwnedDeepFilterPaths::default(), false);
        assert!(
            loaded.is_ok(),
            "{} must load: {:?}",
            library.display(),
            loaded.err()
        );
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
    fn successful_inference_count_excludes_passthrough_frames() {
        let mut processor = DeepFilterProcessor::default();
        processor.push_samples(&[0.1; DEEPFILTER_FRAME_SIZE]);
        processor.process_frames();

        assert_eq!(
            processor.successful_inference_frames(),
            u64::from(processor.is_ffi_available())
        );
    }

    #[test]
    fn test_latency_samples() {
        assert_eq!(
            DeepFilterModel::LowLatency.latency_samples(),
            DEEPFILTER_FRAME_SIZE
        );
        assert_eq!(
            DeepFilterModel::Standard.latency_samples(),
            DEEPFILTER_FRAME_SIZE * 3
        );

        let ll = DeepFilterProcessor::new(
            Arc::new(AtomicU32::new(1.0_f32.to_bits())),
            DeepFilterModel::LowLatency,
        );
        let standard = DeepFilterProcessor::new(
            Arc::new(AtomicU32::new(1.0_f32.to_bits())),
            DeepFilterModel::Standard,
        );
        assert_eq!(ll.latency_samples(), 480);
        assert_eq!(
            standard.latency_samples(),
            if standard.df.is_some() { 1440 } else { 480 }
        );
    }

    #[test]
    fn disabled_deepfilter_reports_only_passthrough_frame_latency() {
        let mut processor = DeepFilterProcessor::new(
            Arc::new(AtomicU32::new(1.0_f32.to_bits())),
            DeepFilterModel::Standard,
        );

        processor.set_enabled(false);

        assert_eq!(processor.latency_samples(), DEEPFILTER_FRAME_SIZE);
    }

    #[test]
    fn runtime_config_rejects_invalid_values() {
        assert!(DeepFilterRuntimeConfig::try_new(12.0, 0.0).is_ok());
        assert!(DeepFilterRuntimeConfig::try_new(0.0, 0.0).is_err());
        assert!(DeepFilterRuntimeConfig::try_new(101.0, 0.0).is_err());
        assert!(DeepFilterRuntimeConfig::try_new(20.0, -0.01).is_err());
        assert!(DeepFilterRuntimeConfig::try_new(20.0, 0.051).is_err());
    }

    #[test]
    fn runtime_config_defaults_match_retained_evaluation() {
        let config = DeepFilterRuntimeConfig::default();
        assert_eq!(config.attenuation_limit_db(), 30.0);
        assert_eq!(config.post_filter_beta(), 0.0);
    }

    #[test]
    fn dry_delay_matches_each_model_latency_and_resets() {
        for model in [DeepFilterModel::LowLatency, DeepFilterModel::Standard] {
            let mut processor =
                DeepFilterProcessor::new(Arc::new(AtomicU32::new(0.5_f32.to_bits())), model);
            processor.dry_frame.fill(0.0);
            processor.dry_frame[0] = 1.0;
            processor.delay_dry_frame();
            assert!(processor
                .aligned_dry_frame
                .iter()
                .all(|sample| *sample == 0.0));

            let frame_count = model.latency_samples() / DEEPFILTER_FRAME_SIZE;
            for frame in 1..=frame_count {
                processor.dry_frame.fill(0.0);
                processor.delay_dry_frame();
                if frame < frame_count {
                    assert!(processor
                        .aligned_dry_frame
                        .iter()
                        .all(|sample| *sample == 0.0));
                } else {
                    assert_eq!(processor.aligned_dry_frame[0], 1.0);
                }
            }

            processor.soft_reset();
            processor.dry_frame.fill(0.0);
            processor.delay_dry_frame();
            assert!(processor
                .aligned_dry_frame
                .iter()
                .all(|sample| *sample == 0.0));
        }
    }

    #[test]
    fn dry_delay_has_no_realtime_allocations() {
        let mut processor = DeepFilterProcessor::new(
            Arc::new(AtomicU32::new(0.5_f32.to_bits())),
            DeepFilterModel::Standard,
        );
        processor.dry_frame.fill(0.25);

        crate::test_alloc::assert_no_allocations("DeepFilter dry delay", || {
            processor.delay_dry_frame();
        });
    }

    #[test]
    fn wet_dry_mix_contract_covers_zero_half_and_full_strength() {
        let dry = [0.25; DEEPFILTER_FRAME_SIZE];
        for (strength, expected) in [(0.0, 0.25), (0.5, 0.625), (1.0, 1.0)] {
            let mut wet = [1.0; DEEPFILTER_FRAME_SIZE];
            mix_wet_with_aligned_dry(&mut wet, &dry, strength);
            assert!(wet.iter().all(|sample| *sample == expected));
        }
    }

    #[test]
    fn soft_reset_and_failure_preserve_dry_alignment_without_allocating() {
        for model in [DeepFilterModel::LowLatency, DeepFilterModel::Standard] {
            let mut processor =
                DeepFilterProcessor::new(Arc::new(AtomicU32::new(0.37_f32.to_bits())), model);
            let loaded = processor.df.is_some();
            processor.push_samples(&[0.1; 600]);
            processor.process_frames();
            assert_eq!(processor.pending_input(), 120);
            assert_eq!(processor.available_samples(), 480);
            let strength = processor.smoothed_strength;
            let successes = processor.successful_inference_frames();
            if loaded {
                processor.mark_backend_failed_rt(DeepFilterProcessError::NonFiniteOutput);
            }
            let latency = processor.latency_samples();
            crate::test_alloc::assert_no_allocations("DeepFilter recovery", || {
                processor.soft_reset();
                assert_eq!(processor.pending_input(), 0);
                assert_eq!(processor.available_samples(), 0);
                assert_eq!(processor.smoothed_strength, strength);
                for frame in 0..5 {
                    let gated = [if frame == 0 { 0.25 } else { 0.0 }; 480];
                    processor.push_samples(&gated);
                    processor.process_frames();
                    let mut output = [0.0; 480];
                    assert_eq!(processor.pop_samples_into(&mut output), 480);
                    let dry_frame = if loaded {
                        model.latency_samples() / 480
                    } else {
                        0
                    };
                    assert_eq!(output, [if frame == dry_frame { 0.25 } else { 0.0 }; 480]);
                    assert_eq!(processor.latency_samples(), latency);
                    assert_eq!(processor.successful_inference_frames(), successes);
                    assert_eq!(processor.backend_failed(), loaded);
                }
            });
            processor.soft_reset();
            processor.set_enabled(false);
            processor.push_samples(&[0.2; 480]);
            processor.process_frames();
            let mut output = [0.0; 480];
            assert_eq!(processor.pop_samples_into(&mut output), 480);
            assert_eq!(output, [0.2; 480]);
        }
    }

    #[test]
    fn controlled_unavailable_and_failed_backends_never_compensate_dry_output() {
        for model in [DeepFilterModel::LowLatency, DeepFilterModel::Standard] {
            for unavailable in [true, false] {
                let mut ordinary =
                    DeepFilterProcessor::new(Arc::new(AtomicU32::new(1.0_f32.to_bits())), model);
                let mut controlled =
                    DeepFilterProcessor::new(Arc::new(AtomicU32::new(1.0_f32.to_bits())), model);
                if !unavailable && !controlled.is_ffi_available() {
                    assert!(
                        !crate::dsp::noise_suppressor::deepfilter_experimental_enabled(),
                        "enabled DeepFilter recovery test needs its runtime assets"
                    );
                    continue;
                }
                if unavailable {
                    ordinary.df = None;
                    controlled.df = None;
                }
                for frame in 0..12 {
                    if frame == 4 && !unavailable {
                        ordinary.mark_backend_failed_rt(DeepFilterProcessError::NonFiniteOutput);
                        controlled.mark_backend_failed_rt(DeepFilterProcessError::NonFiniteOutput);
                    }
                    if frame == 6 {
                        ordinary.soft_reset();
                        controlled.soft_reset();
                    }
                    ordinary.set_enabled(frame != 9);
                    controlled.set_enabled(frame != 9);
                    let raw = std::array::from_fn::<_, 480, _>(|index| {
                        ((index + frame * 480) as f32 * 0.05).sin() * 0.2
                    });
                    let controls = [GateControl {
                        gain: if frame % 2 == 0 { 0.02 } else { 1.0 },
                        open: true,
                    }; 480];
                    ordinary.push_samples(&raw);
                    controlled.push_controlled_samples(&raw, &controls);
                    ordinary.process_frames();
                    controlled.process_frames();
                    let mut expected = [0.0; 480];
                    let mut actual = [0.0; 480];
                    assert_eq!(ordinary.pop_samples_into(&mut expected), 480);
                    assert_eq!(controlled.pop_samples_into(&mut actual), 480);
                    if unavailable || frame >= 4 {
                        assert_eq!(actual, expected);
                    }
                    assert_eq!(ordinary.backend_failed(), controlled.backend_failed());
                    assert_eq!(ordinary.latency_samples(), controlled.latency_samples());
                }
            }
        }
    }

    #[test]
    fn native_soft_reset_mutes_only_the_reported_model_overlap() {
        let before = std::array::from_fn::<_, 480, _>(|index| (index as f32 * 0.047).sin() * 0.25);
        let after = std::array::from_fn::<_, 480, _>(|index| (index as f32 * 0.091).sin() * -0.17);
        for model in [DeepFilterModel::LowLatency, DeepFilterModel::Standard] {
            for gain in [0.3, 1.0] {
                for strength in [0.0_f32, 0.37, 1.0] {
                    let mut processor = DeepFilterProcessor::new(
                        Arc::new(AtomicU32::new(strength.to_bits())),
                        model,
                    );
                    if !processor.is_ffi_available() {
                        eprintln!(
                            "Skipping native reset regression: {:?}",
                            processor.backend_error()
                        );
                        return;
                    }
                    let mut output = [0.0; 480];
                    for _ in 0..20 {
                        processor.push_samples(&before);
                        processor.process_frames();
                        processor.pop_samples_into(&mut output);
                    }
                    processor.push_samples(&before[..123]);
                    processor.soft_reset();
                    assert_eq!(processor.pending_input(), 0);
                    let dry = after.map(|sample| sample * gain);
                    let delay_frames = model.latency_samples() / 480;
                    for frame in 0..=delay_frames {
                        processor.push_samples(&dry);
                        processor.process_frames();
                        assert_eq!(processor.pop_samples_into(&mut output), 480);
                        assert!(!processor.backend_failed());
                        if frame < delay_frames {
                            assert!(output.iter().all(|sample| *sample == 0.0));
                        } else if strength == 0.0 {
                            assert_eq!(output, dry);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn long_digital_silence_keeps_ffi_finite_under_rt_fp_mode() {
        const SILENCE_FRAMES: usize = 7_800;
        const TONE_FRAMES: usize = 20;
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let app_paths = AppOwnedDeepFilterPaths {
            library: Some(repo_root.join(if cfg!(target_os = "windows") {
                "df.dll"
            } else if cfg!(target_os = "macos") {
                "libdf.dylib"
            } else {
                "libdf.so"
            })),
            model: Some(repo_root.join("models")),
        };
        let lib = match DeepFilterLib::try_load(&app_paths, false) {
            Ok(lib) => Arc::new(lib),
            Err(error) => {
                eprintln!("Skipping native DeepFilter silence regression: {error}");
                return;
            }
        };

        for model in [DeepFilterModel::LowLatency, DeepFilterModel::Standard] {
            let mut ffi = match DeepFilterFFI::new(
                lib.clone(),
                model,
                DeepFilterRuntimeConfig::default(),
                &app_paths,
                false,
                None,
            ) {
                Ok(ffi) => ffi,
                Err(error) => {
                    eprintln!("Skipping native DeepFilter silence regression: {error}");
                    return;
                }
            };
            let mut silence = [0.0_f32; DEEPFILTER_FRAME_SIZE];
            let mut output = [0.0_f32; DEEPFILTER_FRAME_SIZE];
            // SAFETY: This test intentionally exercises the same scoped floating-point
            // mode as the realtime DSP loop, using only stack-owned test buffers.
            // nosemgrep: rust.lang.security.unsafe-usage.unsafe-usage
            let failed_silence_frame = unsafe {
                no_denormals::no_denormals(|| {
                    (0..SILENCE_FRAMES).find(|_| {
                        ffi.process_into(&mut silence, &mut output).is_err()
                            || output.iter().any(|sample| !sample.is_finite())
                    })
                })
            };
            assert_eq!(
                failed_silence_frame, None,
                "{model:?} became non-finite during sustained digital silence"
            );

            for (index, sample) in silence.iter_mut().enumerate() {
                *sample = (index as f32 * 0.071).sin() * 0.05;
            }
            let mut tone_nonzero = false;
            // SAFETY: This test intentionally exercises the same scoped floating-point
            // mode as the realtime DSP loop, using only stack-owned test buffers.
            // nosemgrep: rust.lang.security.unsafe-usage.unsafe-usage
            let failed_tone_frame = unsafe {
                no_denormals::no_denormals(|| {
                    (0..TONE_FRAMES).find(|_| match ffi.process_into(&mut silence, &mut output) {
                        Ok(_) => {
                            assert!(
                                output.iter().all(|sample| sample.is_finite()),
                                "{model:?} produced non-finite output after silence"
                            );
                            tone_nonzero |= output.iter().any(|sample| *sample != 0.0);
                            false
                        }
                        Err(_) => true,
                    })
                })
            };
            assert!(
                failed_tone_frame.is_none(),
                "{model:?} did not recover for a finite tone after silence at frame {failed_tone_frame:?}"
            );
            assert!(
                tone_nonzero,
                "{model:?} produced only zero output for {TONE_FRAMES} finite tone frames after silence"
            );
        }
    }

    #[test]
    fn strength_smoothing_is_independent_of_backlog_chunking() {
        let frame = [0.0; DEEPFILTER_FRAME_SIZE];
        let frames = 4;

        let batched_strength = Arc::new(AtomicU32::new(0.0_f32.to_bits()));
        let mut batched =
            DeepFilterProcessor::new(batched_strength.clone(), DeepFilterModel::LowLatency);
        batched_strength.store(1.0_f32.to_bits(), Ordering::Relaxed);
        for _ in 0..frames {
            assert_eq!(batched.push_samples(&frame), DEEPFILTER_FRAME_SIZE);
        }
        batched.process_frames();

        let streamed_strength = Arc::new(AtomicU32::new(0.0_f32.to_bits()));
        let mut streamed =
            DeepFilterProcessor::new(streamed_strength.clone(), DeepFilterModel::LowLatency);
        streamed_strength.store(1.0_f32.to_bits(), Ordering::Relaxed);
        for _ in 0..frames {
            assert_eq!(streamed.push_samples(&frame), DEEPFILTER_FRAME_SIZE);
            streamed.process_frames();
        }

        assert!((batched.smoothed_strength - streamed.smoothed_strength).abs() < 1.0e-6);
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

        let mut output = vec![0.0_f32; processor.available_samples()];
        let output_len = processor.pop_samples_into(&mut output);
        assert!(output_len > 0, "Should produce output");

        // Output should contain complete frames (1000 samples = 2 frames of 480 + 40 remaining)
        assert_eq!(output_len, 960); // Two complete frames
    }

    #[test]
    fn test_external_model_path_requires_explicit_opt_in() {
        let temp_dir = temp_test_dir("deepfilter-file");
        let model_path = temp_dir.join("custom-model.tar.gz");
        fs::write(&model_path, []).unwrap();

        let blocked = find_model_path(
            DeepFilterModel::LowLatency,
            &AppOwnedDeepFilterPaths::default(),
            parse_external_opt_in(""),
            Some(&model_path),
        );
        let allowed = find_model_path(
            DeepFilterModel::LowLatency,
            &AppOwnedDeepFilterPaths::default(),
            parse_external_opt_in(" 1 "),
            Some(&model_path),
        );

        assert_ne!(blocked, Some(model_path.clone()));
        assert_eq!(allowed, Some(model_path));
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_external_opt_in_accepts_only_explicit_truthy_values() {
        for value in ["1", "true", "YES", " on "] {
            assert!(parse_external_opt_in(value), "{value:?}");
        }
        for value in ["", "0", "false", "enable", "2"] {
            assert!(!parse_external_opt_in(value), "{value:?}");
        }
    }

    #[test]
    fn test_app_owned_model_path_does_not_require_external_opt_in() {
        let temp_dir = temp_test_dir("deepfilter-dir");
        let model_path = temp_dir.join(DeepFilterModel::LowLatency.filename());
        fs::write(&model_path, []).unwrap();

        let app_paths = AppOwnedDeepFilterPaths {
            library: None,
            model: Some(temp_dir.clone()),
        };
        let resolved = find_model_path(DeepFilterModel::LowLatency, &app_paths, false, None);

        assert_eq!(resolved, Some(model_path));
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_opted_in_external_model_overrides_app_owned_model() {
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

        let default_path = find_model_path(
            DeepFilterModel::LowLatency,
            &app_paths,
            false,
            Some(&external_model),
        );
        let overridden_path = find_model_path(
            DeepFilterModel::LowLatency,
            &app_paths,
            true,
            Some(&external_model),
        );

        assert_eq!(default_path, Some(bundled_model));
        assert_eq!(overridden_path, Some(external_model));
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_external_library_path_requires_explicit_opt_in() {
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

        let blocked = DeepFilterLib::trusted_library_candidates(
            library_name,
            &AppOwnedDeepFilterPaths::default(),
            false,
            Some(&library_path),
        );
        let allowed = DeepFilterLib::trusted_library_candidates(
            library_name,
            &AppOwnedDeepFilterPaths::default(),
            true,
            Some(&library_path),
        );

        let canonical = library_path.canonicalize().unwrap();
        assert!(!blocked.contains(&canonical));
        assert!(allowed.contains(&canonical));
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_opted_in_external_library_precedes_app_owned_library() {
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

        let default_candidates = DeepFilterLib::trusted_library_candidates(
            library_name,
            &app_paths,
            false,
            Some(&external_library),
        );
        let override_candidates = DeepFilterLib::trusted_library_candidates(
            library_name,
            &app_paths,
            true,
            Some(&external_library),
        );

        assert_eq!(
            default_candidates.first(),
            bundled_library.canonicalize().ok().as_ref()
        );
        assert_eq!(
            override_candidates.first(),
            external_library.canonicalize().ok().as_ref()
        );
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_non_finite_strength_never_reaches_the_wet_dry_mix() {
        let bits = Arc::new(AtomicU32::new(f32::NAN.to_bits()));
        let mut processor = DeepFilterProcessor::new(Arc::clone(&bits), DeepFilterModel::Standard);
        assert!(processor.smoothed_strength.is_finite());

        processor.set_strength(0.5);
        NoiseSuppressor::set_strength(&processor, f32::NAN);
        assert_eq!(processor.get_strength(), 0.5);

        bits.store(f32::NAN.to_bits(), Ordering::Relaxed);
        processor.push_samples(&[0.1; DEEPFILTER_FRAME_SIZE]);
        processor.process_frames();
        assert!(processor.smoothed_strength.is_finite());
        let mut output = [0.0_f32; DEEPFILTER_FRAME_SIZE];
        assert_eq!(
            processor.pop_samples_into(&mut output),
            DEEPFILTER_FRAME_SIZE
        );
        assert!(output.iter().all(|sample| sample.is_finite()));
    }

    // The pinned library aborts the whole process when it cannot load a model
    // or process a frame, and nothing on this side can catch that. The failure
    // paths are therefore driven through the stand-in library of `fake`, which
    // has the library's ABI: a failing load, a failing frame and a model that
    // is not the pinned one must each leave the process alive, the audio
    // passing through unchanged and the reason in `backend_error`.

    /// Push `frames` constant frames (0.1, 0.2, ...) through the processor and
    /// return everything that came out.
    fn run_frames(processor: &mut DeepFilterProcessor, frames: usize) -> Vec<f32> {
        let mut output = Vec::new();
        for frame in 0..frames {
            let input = [0.1 * (frame + 1) as f32; DEEPFILTER_FRAME_SIZE];
            processor.push_samples(&input);
            processor.process_frames();
            let mut block = [0.0_f32; DEEPFILTER_FRAME_SIZE];
            assert_eq!(
                processor.pop_samples_into(&mut block),
                DEEPFILTER_FRAME_SIZE
            );
            output.extend_from_slice(&block);
        }
        assert_eq!(processor.pending_input(), 0);
        assert_eq!(processor.available_samples(), 0);
        output
    }

    fn assert_dry_passthrough(output: &[f32]) {
        for (frame, block) in output.chunks(DEEPFILTER_FRAME_SIZE).enumerate() {
            let expected = 0.1 * (frame + 1) as f32;
            assert!(block.iter().all(|sample| *sample == expected), "{frame}");
        }
    }

    #[test]
    fn a_model_that_is_not_the_pinned_one_never_reaches_the_library() {
        let fake = Fake::new();
        let mut processor = processor_on(&fake, DeepFilterModel::LowLatency, 1.0, true);
        assert_eq!(fake.created.load(Ordering::Acquire), 0);
        assert!(processor
            .backend_error()
            .is_some_and(|error| error.contains("is not the pinned model")));
        assert!(!processor.is_ffi_available());
        assert_dry_passthrough(&run_frames(&mut processor, 3));
    }

    #[test]
    fn a_failing_model_load_leaves_dry_passthrough_and_says_why() {
        let fake = Fake::new();
        fake.fail_create.store(true, Ordering::Release);
        let mut processor = processor_on(&fake, DeepFilterModel::LowLatency, 1.0, false);
        assert_eq!(fake.created.load(Ordering::Acquire), 1);
        assert!(!fake.freed.load(Ordering::Acquire));
        assert!(processor.backend_error().is_some_and(|error| error
            .starts_with("DeepFilterNet init failed: Failed to create DeepFilterNet instance")));
        assert!(!processor.is_ffi_available());
        assert_dry_passthrough(&run_frames(&mut processor, 3));
        assert_eq!(fake.started.load(Ordering::Acquire), 0);
    }

    #[test]
    fn a_frame_size_mismatch_frees_the_state_and_falls_back() {
        let fake = Fake::new();
        fake.frame_length.store(0, Ordering::Release);
        let mut processor = processor_on(&fake, DeepFilterModel::LowLatency, 1.0, false);
        assert_eq!(fake.created.load(Ordering::Acquire), 1);
        assert!(fake.freed.load(Ordering::Acquire));
        assert!(processor
            .backend_error()
            .is_some_and(|error| error.contains("frame size mismatch: expected 480, got 0")));
        assert_dry_passthrough(&run_frames(&mut processor, 2));
    }

    #[test]
    fn a_failing_inference_falls_back_without_losing_or_corrupting_samples() {
        let fake = Fake::new();
        fake.fail_from.store(2, Ordering::Release);
        let mut processor = processor_on(&fake, DeepFilterModel::LowLatency, 1.0, false);
        assert!(processor.is_ffi_available());
        let output = run_frames(&mut processor, 6);

        assert_eq!(output.len(), 6 * DEEPFILTER_FRAME_SIZE);
        assert!(output.iter().all(|sample| sample.is_finite()));
        assert!(processor.backend_failed());
        assert!(!processor.is_ffi_available());
        assert_eq!(
            processor.backend_error(),
            Some("DeepFilterNet processing failed with non-finite SNR")
        );
        assert_eq!(processor.successful_inference_frames(), 2);
        let frames: Vec<&[f32]> = output.chunks(DEEPFILTER_FRAME_SIZE).collect();
        // Frames the library handled are its (here, halved) output; once it
        // failed, the dry input follows with the model's one-frame delay kept.
        assert!(frames[0].iter().all(|sample| *sample == 0.1 * 0.5));
        assert!(frames[1].iter().all(|sample| *sample == 0.2 * 0.5));
        for (frame, block) in frames.iter().enumerate().skip(2) {
            let delayed = 0.1 * frame as f32;
            assert!(block.iter().all(|sample| *sample == delayed));
        }
        drop(processor);
        assert!(fake.freed.load(Ordering::Acquire));
    }

    #[test]
    fn non_finite_input_is_zeroed_before_it_reaches_the_library() {
        let fake = Fake::new();
        let mut processor = processor_on(&fake, DeepFilterModel::LowLatency, 1.0, false);
        let mut input = [0.2_f32; DEEPFILTER_FRAME_SIZE];
        input[3] = f32::NAN;
        input[7] = f32::INFINITY;
        processor.push_samples(&input);
        processor.process_frames();
        let mut output = [0.0_f32; DEEPFILTER_FRAME_SIZE];
        assert_eq!(
            processor.pop_samples_into(&mut output),
            DEEPFILTER_FRAME_SIZE
        );
        assert_eq!(fake.started.load(Ordering::Acquire), 1);
        assert!(!fake.non_finite_input_seen.load(Ordering::Acquire));
        assert!(output.iter().all(|sample| sample.is_finite()));
        assert!(processor.is_ffi_available());
    }

    /// The real library ends the process on a model file it cannot parse, so
    /// a corrupt bundled model must be refused before the library is called.
    #[test]
    fn a_corrupt_bundled_model_is_refused_before_the_real_library_sees_it() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let library = repo_root.join("df.dll");
        if !cfg!(all(windows, debug_assertions)) || !library.is_file() {
            eprintln!(
                "SKIPPED: {} absent or not a debug Windows build",
                library.display()
            );
            return;
        }
        let model = temp_test_dir("corrupt-bundled-model").join("model.tar.gz");
        fs::write(&model, b"truncated download").unwrap();
        let app_paths = AppOwnedDeepFilterPaths {
            library: Some(library),
            model: Some(model),
        };
        for model_kind in [DeepFilterModel::LowLatency, DeepFilterModel::Standard] {
            let mut processor = DeepFilterProcessor::from_library(
                Arc::new(AtomicU32::new(1.0_f32.to_bits())),
                model_kind,
                DeepFilterRuntimeConfig::default(),
                DeepFilterLib::try_load(&app_paths, false),
                &app_paths,
                false,
                None,
            );
            assert!(processor
                .backend_error()
                .is_some_and(|error| error.contains("is not the pinned model")));
            assert!(!processor.is_ffi_available());
            assert_dry_passthrough(&run_frames(&mut processor, 2));
        }
    }

    /// The shipped library must answer a model it cannot load with a null
    /// state. A library without the abort-free C API ends the process
    /// instead, so the check runs in a child process: such a library fails
    /// this test and leaves the rest of the run alone.
    #[test]
    fn the_real_library_reports_a_corrupt_model_instead_of_aborting() {
        const CHILD: &str = "AUDIOFORGE_TEST_CORRUPT_MODEL_CHILD";
        let library = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join(if cfg!(windows) { "df.dll" } else { "libdf.so" });
        if std::env::var_os(CHILD).is_none() {
            if !library.is_file() {
                eprintln!("SKIPPED: {} is absent", library.display());
                return;
            }
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "--nocapture"])
                .arg("dsp::deepfilter_ffi::tests::the_real_library_reports_a_corrupt_model_instead_of_aborting")
                .env(CHILD, "1")
                .status()
                .unwrap();
            assert!(
                status.success(),
                "{} ended the process on a corrupt model ({status}); rebuild it with build_deepfilter.ps1",
                library.display()
            );
            return;
        }

        let app_paths = AppOwnedDeepFilterPaths {
            library: Some(library),
            model: None,
        };
        let lib = Arc::new(DeepFilterLib::try_load(&app_paths, false).expect("the library loads"));
        let corrupt = temp_test_dir("corrupt-model").join("DeepFilterNet3_ll_onnx.tar.gz");
        fs::write(&corrupt, b"truncated download").unwrap();
        let error = DeepFilterFFI::open(lib, DeepFilterRuntimeConfig::default(), &corrupt, None)
            .err()
            .expect("a corrupt model is an error");
        assert!(
            error.contains("Failed to create DeepFilterNet instance"),
            "{error}"
        );
    }
}
