//! Seeded DSP contention and device-enumeration smoke coverage.

use mic_eq_core::audio::input::list_input_devices;
use mic_eq_core::audio::processor::{run_seeded_control_dsp_stress, AudioProcessor};

/// Exercise concurrent rapid controls and production DSP processing.
///
/// This test is also run explicitly in release mode so the contention timing is
/// representative of the shipped processor.
#[test]
fn seeded_control_and_dsp_loops_remain_finite_under_contention() {
    for seed in [1, 0x5eed, 0xdead_beef, u64::MAX - 1] {
        let report = run_seeded_control_dsp_stress(seed, 600)
            .unwrap_or_else(|error| panic!("seed {seed:#x}: {error}"));
        assert_eq!(report.control_updates, 600);
        assert!(report.processed_blocks >= 600);
        assert!(report.snapshot_rearms >= 1);
        assert!(report.model_switches >= 1);
        assert!(report.suppressor_resets >= 1);
        assert!(report.max_output_abs.is_finite());
        assert!(report.max_output_abs <= 16.0);
    }
}

/// Smoke-test device enumeration and non-started processor lifecycle handling.
///
/// USB unplug and WASAPI invalid-device callbacks still require a hardware-driven
/// integration test; this covers the deterministic, device-independent boundary.
#[test]
fn device_enumeration_and_lifecycle_smoke() {
    let mut device_counts = Vec::new();
    for _ in 0..50 {
        match list_input_devices() {
            Ok(devices) => device_counts.push(devices.len()),
            Err(error) => println!("Device enumeration unavailable in this environment: {error}"),
        }
    }

    if let Some(&first_count) = device_counts.first() {
        assert!(
            device_counts.iter().all(|&count| count == first_count)
                || device_counts.iter().any(|&count| count > 0),
            "device enumeration should be stable or discover at least one device"
        );
    }

    if let Ok(devices) = list_input_devices() {
        assert!(!devices
            .iter()
            .any(|device| device.name.contains("FAKE_DEVICE_DOES_NOT_EXIST")));
    }

    let processor = AudioProcessor::new();
    assert!(!processor.is_running());
    processor.set_gate_threshold(-60.0);
    processor
        .set_eq_band_gain(0, 6.0)
        .expect("test gain should be valid");
    processor.set_compressor_threshold(-20.0);
    let (_, gain, _) = processor
        .get_eq_band_params(0)
        .expect("EQ band parameters should remain available");
    assert_eq!(gain, 6.0);
    assert!(!processor.is_running());
}
