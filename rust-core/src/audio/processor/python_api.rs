/// Gate operating modes
#[cfg(feature = "vad")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[pyclass(eq, eq_int, skip_from_py_object)]
pub enum PyGateMode {
    /// Traditional gate using only level threshold
    ThresholdOnly = 0,
    /// Hybrid: gate opens when level exceeded OR speech detected
    VadAssisted = 1,
    /// VAD-only: gate opens solely based on speech probability
    VadOnly = 2,
}

fn py_dict_bool(
    settings: Option<&Bound<'_, pyo3::types::PyDict>>,
    key: &str,
    default: bool,
) -> PyResult<bool> {
    if let Some(settings) = settings {
        if let Some(value) = settings.get_item(key)? {
            return value.extract::<bool>();
        }
    }
    Ok(default)
}

fn py_dict_f64(
    settings: Option<&Bound<'_, pyo3::types::PyDict>>,
    key: &str,
    default: f64,
) -> PyResult<f64> {
    if let Some(settings) = settings {
        if let Some(value) = settings.get_item(key)? {
            return value.extract::<f64>();
        }
    }
    Ok(default)
}

fn linear_to_db(value: f32) -> f32 {
    20.0 * value.max(1.0e-12).log10()
}

fn percentile_f32(values: &mut [f32], percentile: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f32::total_cmp);
    let position = (values.len().saturating_sub(1) as f32) * percentile.clamp(0.0, 1.0);
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        values[lower]
    } else {
        let fraction = position - lower as f32;
        values[lower] + fraction * (values[upper] - values[lower])
    }
}

fn compressor_pumping_score(gr_trace_db: &[f32], cadence_hz: f32) -> f32 {
    if gr_trace_db.len() < 3 || !cadence_hz.is_finite() || cadence_hz <= 0.0 {
        return 0.0;
    }
    let dt = 1.0 / cadence_hz;
    let highpass_rc = 1.0 / (2.0 * std::f32::consts::PI * 2.0);
    let lowpass_rc = 1.0 / (2.0 * std::f32::consts::PI * 8.0);
    let highpass_alpha = highpass_rc / (highpass_rc + dt);
    let lowpass_alpha = dt / (lowpass_rc + dt);
    let mut previous_input = gr_trace_db[0];
    let mut highpass = 0.0_f32;
    let mut bandpass = 0.0_f32;
    let mut bandpass_abs = Vec::with_capacity(gr_trace_db.len());
    let mut deltas = Vec::with_capacity(gr_trace_db.len().saturating_sub(1));

    for &value in gr_trace_db.iter().skip(1) {
        if !value.is_finite() {
            return f32::INFINITY;
        }
        highpass = highpass_alpha * (highpass + value - previous_input);
        bandpass += lowpass_alpha * (highpass - bandpass);
        bandpass_abs.push(bandpass.abs());
        deltas.push((value - previous_input).abs());
        previous_input = value;
    }
    let robust_limit = percentile_f32(&mut bandpass_abs.clone(), 0.95);
    let robust_rms = if bandpass_abs.is_empty() {
        0.0
    } else {
        (bandpass_abs
            .iter()
            .map(|value| value.min(robust_limit).powi(2))
            .sum::<f32>()
            / bandpass_abs.len() as f32)
            .sqrt()
    };
    robust_rms + percentile_f32(&mut deltas, 0.95)
}

/// Stream a capture through the production compressor auto-makeup controller.
///
/// VAD probabilities are supplied at the fixed 10 ms control cadence so the
/// evaluation harness can use the exact shared Silero posterior while keeping
/// model inference outside the compressor implementation.
#[pyfunction]
#[pyo3(signature = (
    audio,
    sample_rate,
    vad_probabilities,
    noise_floor_db,
    noise_reliability,
    settings=None
))]
pub fn simulate_auto_makeup_control(
    py: Python<'_>,
    audio: numpy::PyReadonlyArray1<'_, f32>,
    sample_rate: f64,
    vad_probabilities: Vec<f64>,
    noise_floor_db: f64,
    noise_reliability: f64,
    settings: Option<&Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<Py<PyAny>> {
    const CONTROL_BLOCK_SIZE: usize = 480;
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "sample_rate must be positive and finite",
        ));
    }
    if !noise_floor_db.is_finite()
        || !noise_reliability.is_finite()
        || !(0.0..=1.0).contains(&noise_reliability)
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "noise evidence must be finite and reliability must be between 0 and 1",
        ));
    }
    if vad_probabilities
        .iter()
        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "VAD probabilities must be finite and between 0 and 1",
        ));
    }

    let audio = audio.as_slice()?;
    let block_count = audio.len().div_ceil(CONTROL_BLOCK_SIZE);
    if !vad_probabilities.is_empty() && vad_probabilities.len() != block_count {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "expected {block_count} VAD probabilities at the 10 ms control cadence, got {}",
            vad_probabilities.len()
        )));
    }

    let mut compressor = Compressor::new(
        py_dict_f64(settings, "threshold_db", -24.0)?,
        py_dict_f64(settings, "ratio", 3.0)?,
        py_dict_f64(settings, "attack_ms", 10.0)?,
        py_dict_f64(settings, "release_ms", 180.0)?,
        py_dict_f64(settings, "makeup_gain_db", 0.0)?,
        6.0,
        sample_rate,
    );
    compressor.set_auto_makeup_enabled(true);
    compressor.set_target_lufs(py_dict_f64(settings, "target_lufs", -18.0)?);
    compressor.set_noise_reference_reliability(noise_reliability);
    compressor.set_adaptive_release(py_dict_bool(settings, "adaptive_release", true)?);
    compressor.set_sidechain_highpass_enabled(py_dict_bool(
        settings,
        "sidechain_highpass_enabled",
        true,
    )?);
    let vad_reliability = py_dict_f64(settings, "vad_reliability", 1.0)?;
    if !vad_reliability.is_finite() || !(0.0..=1.0).contains(&vad_reliability) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "vad_reliability must be finite and between 0 and 1",
        ));
    }
    let return_output_audio = py_dict_bool(settings, "return_output_audio", false)?;

    let mut output_audio = if return_output_audio {
        Vec::with_capacity(audio.len())
    } else {
        Vec::new()
    };
    let mut makeup_gain_db = Vec::with_capacity(block_count);
    let mut activity = Vec::with_capacity(block_count);
    let mut reliability = Vec::with_capacity(block_count);
    let mut gain_reduction_db = Vec::with_capacity(block_count);
    let mut input_rms_db = Vec::with_capacity(block_count);
    let mut output_rms_db = Vec::with_capacity(block_count);
    let mut block_runtime_ms = Vec::with_capacity(block_count);

    for (block_index, chunk) in audio.chunks(CONTROL_BLOCK_SIZE).enumerate() {
        let mut block = chunk.to_vec();
        let input_rms = (block
            .iter()
            .map(|sample| (*sample as f64) * (*sample as f64))
            .sum::<f64>()
            / block.len().max(1) as f64)
            .sqrt() as f32;
        let evidence = vad_probabilities
            .get(block_index)
            .copied()
            .map(|vad_probability| AutoMakeupActivityInput {
                vad_probability,
                vad_reliability,
                noise_floor_db,
                live_noise_reliability: noise_reliability,
            });
        let started = Instant::now();
        compressor.process_block_inplace_with_activity_control(&mut block, evidence);
        block_runtime_ms.push(started.elapsed().as_secs_f32() * 1000.0);
        let output_rms = (block
            .iter()
            .map(|sample| (*sample as f64) * (*sample as f64))
            .sum::<f64>()
            / block.len().max(1) as f64)
            .sqrt() as f32;
        input_rms_db.push(linear_to_db(input_rms));
        output_rms_db.push(linear_to_db(output_rms));
        makeup_gain_db.push(compressor.current_makeup_gain() as f32);
        activity.push(compressor.auto_makeup_activity() as f32);
        reliability.push(compressor.auto_makeup_activity_reliability() as f32);
        gain_reduction_db.push(compressor.current_gain_reduction() as f32);
        if return_output_audio {
            output_audio.extend_from_slice(&block);
        }
    }

    let diagnostics = pyo3::types::PyDict::new(py);
    diagnostics.set_item("control_block_size", CONTROL_BLOCK_SIZE)?;
    diagnostics.set_item(
        "control_cadence_hz",
        sample_rate / CONTROL_BLOCK_SIZE as f64,
    )?;
    diagnostics.set_item("processed_samples", audio.len())?;
    diagnostics.set_item("makeup_gain_db", makeup_gain_db)?;
    diagnostics.set_item("activity", activity)?;
    diagnostics.set_item("reliability", reliability)?;
    diagnostics.set_item("gain_reduction_db", gain_reduction_db)?;
    diagnostics.set_item("input_rms_db", input_rms_db)?;
    diagnostics.set_item("output_rms_db", output_rms_db)?;
    diagnostics.set_item(
        "p95_block_runtime_ms",
        percentile_f32(&mut block_runtime_ms.clone(), 0.95),
    )?;
    diagnostics.set_item(
        "p99_block_runtime_ms",
        percentile_f32(&mut block_runtime_ms.clone(), 0.99),
    )?;
    diagnostics.set_item(
        "max_block_runtime_ms",
        block_runtime_ms
            .into_iter()
            .max_by(f32::total_cmp)
            .unwrap_or(0.0),
    )?;
    if return_output_audio {
        diagnostics.set_item("output_audio", output_audio)?;
    }
    Ok(diagnostics.into_any().unbind())
}

/// Evaluate the existing gate/RNNoise stages in either application order.
#[cfg(feature = "vad")]
#[pyfunction]
#[pyo3(signature = (
    audio,
    vad_probabilities,
    suppressor_before_gate,
    suppressor_strength=1.0,
    settings=None
))]
pub fn simulate_gate_suppressor_order(
    py: Python<'_>,
    audio: numpy::PyReadonlyArray1<'_, f32>,
    vad_probabilities: Vec<f32>,
    suppressor_before_gate: bool,
    suppressor_strength: f32,
    settings: Option<&Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<Py<PyAny>> {
    if !suppressor_strength.is_finite() || !(0.0..=1.0).contains(&suppressor_strength) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "suppressor_strength must be finite and between 0 and 1",
        ));
    }
    let audio = audio.as_slice()?;
    let block_count = audio.len().div_ceil(RNNOISE_FRAME_SIZE);
    if vad_probabilities.len() != block_count
        || vad_probabilities
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "expected {block_count} finite VAD probabilities at the 10 ms RNNoise cadence"
        )));
    }

    let mut gate = NoiseGate::new(
        py_dict_f64(settings, "gate_threshold_db", -40.0)?,
        py_dict_f64(settings, "gate_attack_ms", 10.0)?,
        py_dict_f64(settings, "gate_release_ms", 100.0)?,
        48_000.0,
    );
    gate.set_gate_mode(GateMode::VadAssisted);
    let strength = Arc::new(AtomicU32::new(suppressor_strength.to_bits()));
    let mut suppressor = NoiseSuppressionEngine::new(NoiseModel::RNNoise, strength);
    let mut output = Vec::with_capacity(audio.len());
    let mut gate_gain = Vec::with_capacity(block_count);
    let started = Instant::now();

    for (block_index, chunk) in audio.chunks(RNNOISE_FRAME_SIZE).enumerate() {
        let mut frame = [0.0_f32; RNNOISE_FRAME_SIZE];
        frame[..chunk.len()].copy_from_slice(chunk);
        gate.set_external_vad_probability(vad_probabilities[block_index], true);
        if suppressor_before_gate {
            if suppressor.push_samples(&frame) != RNNOISE_FRAME_SIZE {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "RNNoise rejected a complete input frame",
                ));
            }
            suppressor.process_frames();
            if suppressor.pop_samples_into(&mut frame) != RNNOISE_FRAME_SIZE {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "RNNoise did not produce a complete output frame",
                ));
            }
            gate.process_block_inplace(&mut frame);
        } else {
            gate.process_block_inplace(&mut frame);
            if suppressor.push_samples(&frame) != RNNOISE_FRAME_SIZE {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "RNNoise rejected a complete input frame",
                ));
            }
            suppressor.process_frames();
            if suppressor.pop_samples_into(&mut frame) != RNNOISE_FRAME_SIZE {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "RNNoise did not produce a complete output frame",
                ));
            }
        }
        output.extend_from_slice(&frame[..chunk.len()]);
        gate_gain.push(gate.current_gain());
    }

    let diagnostics = pyo3::types::PyDict::new(py);
    diagnostics.set_item("output_audio", output)?;
    diagnostics.set_item("gate_gain", gate_gain)?;
    diagnostics.set_item("gate_chatter_event_count", gate.chatter_event_count())?;
    diagnostics.set_item("gate_noise_floor_db", gate.noise_floor())?;
    diagnostics.set_item(
        "gate_noise_floor_reliability",
        gate.noise_floor_reliability(),
    )?;
    diagnostics.set_item("suppressor_latency_samples", suppressor.latency_samples())?;
    diagnostics.set_item(
        "runtime_ms",
        started.elapsed().as_secs_f64() * 1000.0,
    )?;
    Ok(diagnostics.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (audio, sample_rate, bands, settings=None))]
pub fn simulate_auto_eq_chain(
    py: Python<'_>,
    audio: numpy::PyReadonlyArray1<'_, f32>,
    sample_rate: f64,
    bands: Vec<(f64, f64, f64)>,
    settings: Option<&Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<Py<PyAny>> {
    let simulation_started = Instant::now();
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "sample_rate must be positive and finite",
        ));
    }
    if bands.len() != NUM_BANDS {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "expected {NUM_BANDS} EQ bands, got {}",
            bands.len()
        )));
    }

    let mut processor = OfflineDspBlockProcessor::new(sample_rate);
    processor.set_eq_enabled(true);
    for (index, (frequency, gain_db, q)) in bands.iter().copied().enumerate() {
        processor.eq_mut().set_band_frequency(index, frequency);
        processor.eq_mut().set_band_gain(index, gain_db);
        processor.eq_mut().set_band_q(index, q);
    }

    let deesser_enabled = py_dict_bool(settings, "deesser_enabled", false)?;
    let return_output_audio = py_dict_bool(settings, "return_output_audio", false)?;
    processor.set_eq_before_deesser(py_dict_bool(
        settings,
        "eq_before_deesser",
        false,
    )?);
    processor.set_deesser_enabled(deesser_enabled);
    if deesser_enabled {
        let deesser = processor.deesser_mut();
        deesser.set_auto_enabled(py_dict_bool(settings, "deesser_auto_enabled", true)?);
        deesser.set_auto_amount(py_dict_f64(settings, "deesser_auto_amount", 0.5)?);
        deesser.set_low_cut_hz(py_dict_f64(settings, "deesser_low_cut_hz", 4000.0)?);
        deesser.set_high_cut_hz(py_dict_f64(settings, "deesser_high_cut_hz", 11_000.0)?);
        deesser.set_threshold_db(py_dict_f64(settings, "deesser_threshold_db", -28.0)?);
        deesser.set_ratio(py_dict_f64(settings, "deesser_ratio", 4.0)?);
        deesser.set_attack_ms(py_dict_f64(settings, "deesser_attack_ms", 2.0)?);
        deesser.set_release_ms(py_dict_f64(settings, "deesser_release_ms", 80.0)?);
        deesser.set_max_reduction_db(py_dict_f64(settings, "deesser_max_reduction_db", 6.0)?);
    }

    let compressor_enabled = py_dict_bool(settings, "compressor_enabled", true)?;
    processor.set_compressor_enabled(compressor_enabled);
    if compressor_enabled {
        let compressor = processor.compressor_mut();
        compressor.set_threshold(py_dict_f64(settings, "compressor_threshold_db", -20.0)?);
        compressor.set_ratio(py_dict_f64(settings, "compressor_ratio", 4.0)?);
        compressor.set_attack_time(py_dict_f64(settings, "compressor_attack_ms", 10.0)?);
        compressor.set_release_time(py_dict_f64(settings, "compressor_release_ms", 200.0)?);
        compressor.set_makeup_gain(py_dict_f64(settings, "compressor_makeup_gain_db", 0.0)?);
        compressor.set_adaptive_release(py_dict_bool(
            settings,
            "compressor_adaptive_release",
            false,
        )?);
        compressor.set_base_release_time(py_dict_f64(
            settings,
            "compressor_base_release_ms",
            50.0,
        )?);
        compressor.set_auto_makeup_enabled(py_dict_bool(
            settings,
            "compressor_auto_makeup_enabled",
            false,
        )?);
        compressor.set_target_lufs(py_dict_f64(settings, "compressor_target_lufs", -18.0)?);
        compressor.set_sidechain_highpass_enabled(py_dict_bool(
            settings,
            "compressor_sidechain_highpass_enabled",
            true,
        )?);
    }

    let limiter_enabled = py_dict_bool(settings, "limiter_enabled", true)?;
    processor.set_limiter_enabled(limiter_enabled);
    let limiter_ceiling_db = py_dict_f64(settings, "limiter_ceiling_db", -0.5)?;
    let careful_output_enabled = py_dict_bool(settings, "limiter_careful_output_enabled", true)?;
    let effective_ceiling_db =
        effective_limiter_ceiling_db(limiter_ceiling_db, careful_output_enabled) as f32;
    if limiter_enabled {
        processor
            .limiter_mut()
            .set_lookahead_ms(py_dict_f64(settings, "limiter_lookahead_ms", 2.0)?);
        processor
            .limiter_mut()
            .set_ceiling(effective_ceiling_db as f64);
        processor
            .limiter_mut()
            .set_release_time(py_dict_f64(settings, "limiter_release_ms", 50.0)?);
        processor
            .true_peak_limiter_mut()
            .set_release_ms(py_dict_f64(settings, "limiter_release_ms", 50.0)? as f32);
    }

    let mut output = FixedAudioBuffer::<f32, RT_PROCESS_BUFFER_CAPACITY>::new();
    let mut input_square_sum = 0.0_f64;
    let mut output_square_sum = 0.0_f64;
    let mut input_samples = 0_usize;
    let mut output_samples = 0_usize;
    let mut input_sample_peak = 0.0_f32;
    let mut output_sample_peak = 0.0_f32;
    let mut pre_limiter_true_peak = 0.0_f32;
    let mut output_true_peak = 0.0_f32;
    let mut limiter_peak_gain_reduction_db = 0.0_f32;
    let mut true_peak_limiter_gain_reduction_db = 0.0_f32;
    let mut compressor_gain_reduction_db = 0.0_f32;
    let mut deesser_gain_reduction_db = 0.0_f32;
    let mut true_peak_limited_events = 0_u64;
    let mut analysis_rows: Vec<(f32, f32, f32, f32)> = Vec::new();
    let mut non_finite_output = false;
    let audio = audio.as_slice()?;
    let mut rendered_audio = if return_output_audio {
        Vec::with_capacity(audio.len())
    } else {
        Vec::new()
    };

    let analysis_block_samples = ((sample_rate * 0.020).round() as usize)
        .clamp(1, RT_PROCESS_BUFFER_CAPACITY);
    for chunk in audio.chunks(analysis_block_samples) {
        let mut block = chunk.to_vec();
        let mut block_input_square_sum = 0.0_f64;
        for sample in block.iter_mut() {
            if !sample.is_finite() {
                *sample = 0.0;
            }
            input_square_sum += (*sample as f64) * (*sample as f64);
            block_input_square_sum += (*sample as f64) * (*sample as f64);
            input_samples += 1;
        }

        let stats = processor.process_block_with_stats(&mut block, &mut output);
        let block_input_rms = if chunk.is_empty() {
            0.0
        } else {
            (block_input_square_sum / chunk.len() as f64).sqrt() as f32
        };
        let block_output_square_sum = output
            .as_slice()
            .iter()
            .map(|sample| {
                if !sample.is_finite() {
                    non_finite_output = true;
                    0.0
                } else {
                    (*sample as f64) * (*sample as f64)
                }
            })
            .sum::<f64>();
        let block_output_rms = if output.as_slice().is_empty() {
            0.0
        } else {
            (block_output_square_sum / output.as_slice().len() as f64).sqrt() as f32
        };
        analysis_rows.push((
            linear_to_db(block_input_rms),
            linear_to_db(block_output_rms),
            stats.compressor_gain_reduction_db,
            stats.deesser_gain_reduction_db,
        ));
        input_sample_peak = input_sample_peak.max(stats.input_sample_peak);
        output_sample_peak = output_sample_peak.max(stats.output_sample_peak);
        pre_limiter_true_peak = pre_limiter_true_peak.max(stats.true_peak_limiter_input_peak);
        output_true_peak = output_true_peak.max(stats.output_true_peak);
        limiter_peak_gain_reduction_db =
            limiter_peak_gain_reduction_db.max(stats.limiter_peak_gain_reduction_db);
        true_peak_limiter_gain_reduction_db = true_peak_limiter_gain_reduction_db
            .max(stats.true_peak_limiter_gain_reduction_db);
        compressor_gain_reduction_db =
            compressor_gain_reduction_db.max(stats.compressor_gain_reduction_db);
        deesser_gain_reduction_db = deesser_gain_reduction_db.max(stats.deesser_gain_reduction_db);
        true_peak_limited_events =
            true_peak_limited_events.saturating_add(stats.true_peak_limited_events);

        for &sample in output.as_slice() {
            output_square_sum += (sample as f64) * (sample as f64);
            output_samples += 1;
        }
        if return_output_audio {
            rendered_audio.extend_from_slice(output.as_slice());
        }
    }

    let input_rms = if input_samples > 0 {
        (input_square_sum / input_samples as f64).sqrt() as f32
    } else {
        0.0
    };
    let output_rms = if output_samples > 0 {
        (output_square_sum / output_samples as f64).sqrt() as f32
    } else {
        0.0
    };
    let output_sample_peak_db = linear_to_db(output_sample_peak);
    let pre_limiter_true_peak_db = linear_to_db(pre_limiter_true_peak);
    let output_true_peak_db = linear_to_db(output_true_peak);
    let mut input_rms_rows: Vec<f32> = analysis_rows.iter().map(|row| row.0).collect();
    let input_floor_db = percentile_f32(&mut input_rms_rows.clone(), 0.20);
    let input_p90_db = percentile_f32(&mut input_rms_rows, 0.90);
    let active_threshold_db = (input_floor_db + 6.0).max(input_p90_db - 24.0).max(-60.0);
    let mut active_compressor_reduction: Vec<f32> = analysis_rows
        .iter()
        .filter(|row| row.0 >= active_threshold_db)
            .map(|row| row.2.max(0.0))
        .collect();
    let mut active_deesser_reduction: Vec<f32> = analysis_rows
        .iter()
        .filter(|row| row.0 >= active_threshold_db)
            .map(|row| row.3.max(0.0))
        .collect();
    if active_compressor_reduction.len() < 3 {
        active_compressor_reduction = analysis_rows.iter().map(|row| row.2.max(0.0)).collect();
        active_deesser_reduction = analysis_rows.iter().map(|row| row.3.max(0.0)).collect();
    }
    let active_block_count = active_compressor_reduction.len();
    let compressor_active_ratio = if active_block_count > 0 {
        active_compressor_reduction
            .iter()
            .filter(|reduction| **reduction >= 0.10)
            .count() as f32
            / active_block_count as f32
    } else {
        0.0
    };
    let compressor_reduction_median_db =
        percentile_f32(&mut active_compressor_reduction.clone(), 0.50);
    let compressor_reduction_p95_db =
        percentile_f32(&mut active_compressor_reduction, 0.95);
    let deesser_reduction_median_db =
        percentile_f32(&mut active_deesser_reduction.clone(), 0.50);
    let deesser_reduction_p95_db = percentile_f32(&mut active_deesser_reduction, 0.95);
    let mut active_output_gain_db: Vec<f32> = analysis_rows
        .iter()
        .filter(|row| row.0 >= active_threshold_db && row.0 > -100.0)
        .map(|row| row.1 - row.0)
        .collect();
    let mut silence_level_delta_db: Vec<f32> = analysis_rows
        .iter()
        .filter(|row| row.0 < active_threshold_db && row.0 > -100.0)
        .map(|row| row.1 - row.0)
        .collect();
    let mut silence_output_gain_db: Vec<f32> = analysis_rows
        .iter()
        .filter(|row| row.0 < active_threshold_db)
        .map(|row| -row.2.max(0.0))
        .collect();
    let active_output_gain_db = percentile_f32(&mut active_output_gain_db, 0.50);
    let silence_output_gain_db = percentile_f32(&mut silence_output_gain_db, 0.50);
    let silence_level_delta_db = percentile_f32(&mut silence_level_delta_db, 0.50);
    let compressor_gr_trace = analysis_rows
        .iter()
        .map(|row| row.2.max(0.0))
        .collect::<Vec<_>>();
    let compressor_pumping_score_db = compressor_pumping_score(&compressor_gr_trace, 50.0);
    let diagnostics = pyo3::types::PyDict::new(py);
    diagnostics.set_item("input_sample_peak_db", linear_to_db(input_sample_peak))?;
    diagnostics.set_item("input_rms_db", linear_to_db(input_rms))?;
    diagnostics.set_item("output_sample_peak_db", output_sample_peak_db)?;
    diagnostics.set_item("pre_limiter_true_peak_db", pre_limiter_true_peak_db)?;
    diagnostics.set_item("output_true_peak_db", output_true_peak_db)?;
    diagnostics.set_item("output_rms_db", linear_to_db(output_rms))?;
    diagnostics.set_item("limiter_effective_ceiling_db", effective_ceiling_db)?;
    diagnostics.set_item(
        "sample_headroom_db",
        effective_ceiling_db - output_sample_peak_db,
    )?;
    diagnostics.set_item(
        "pre_limiter_true_peak_headroom_db",
        effective_ceiling_db - pre_limiter_true_peak_db,
    )?;
    diagnostics.set_item(
        "true_peak_headroom_db",
        effective_ceiling_db - output_true_peak_db,
    )?;
    diagnostics.set_item("limiter_gain_reduction_db", limiter_peak_gain_reduction_db)?;
    diagnostics.set_item(
        "true_peak_limiter_gain_reduction_db",
        true_peak_limiter_gain_reduction_db,
    )?;
    diagnostics.set_item("true_peak_limited_events", true_peak_limited_events)?;
    diagnostics.set_item("compressor_gain_reduction_db", compressor_gain_reduction_db)?;
    diagnostics.set_item("deesser_gain_reduction_db", deesser_gain_reduction_db)?;
    diagnostics.set_item(
        "compressor_gain_reduction_median_db",
        compressor_reduction_median_db,
    )?;
    diagnostics.set_item(
        "compressor_gain_reduction_p95_db",
        compressor_reduction_p95_db,
    )?;
    diagnostics.set_item(
        "compressor_gain_reduction_active_ratio",
        compressor_active_ratio,
    )?;
    diagnostics.set_item("active_output_gain_db", active_output_gain_db)?;
    diagnostics.set_item("silence_output_gain_db", silence_output_gain_db)?;
    diagnostics.set_item("silence_level_delta_db", silence_level_delta_db)?;
    diagnostics.set_item("compressor_pumping_score_db", compressor_pumping_score_db)?;
    diagnostics.set_item("non_finite_output", non_finite_output)?;
    diagnostics.set_item(
        "candidate_runtime_ms",
        simulation_started.elapsed().as_secs_f64() * 1000.0,
    )?;
    diagnostics.set_item(
        "deesser_gain_reduction_median_db",
        deesser_reduction_median_db,
    )?;
    diagnostics.set_item(
        "deesser_gain_reduction_p95_db",
        deesser_reduction_p95_db,
    )?;
    diagnostics.set_item("analysis_block_ms", 20.0_f32)?;
    diagnostics.set_item("active_analysis_threshold_db", active_threshold_db)?;
    diagnostics.set_item("active_analysis_block_count", active_block_count)?;
    diagnostics.set_item("processed_samples", output_samples)?;
    if return_output_audio {
        diagnostics.set_item("output_audio", rendered_audio)?;
    }
    Ok(diagnostics.into_any().unbind())
}

/// Analyze an offline capture with the same stateful Silero VAD used by the
/// realtime gate. The returned probabilities are one value per model window;
/// the final partial window is zero-padded so callers can map every sample of
/// a capture to a posterior without inventing a second VAD implementation.
#[cfg(feature = "vad")]
#[pyfunction]
#[pyo3(signature = (audio, sample_rate, threshold=0.48))]
pub fn analyze_vad_probabilities(
    audio: numpy::PyReadonlyArray1<'_, f32>,
    sample_rate: u32,
    threshold: f32,
) -> PyResult<Vec<f32>> {
    if sample_rate == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "sample_rate must be positive",
        ));
    }

    let samples = audio.as_slice().map_err(|error| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "audio must be a contiguous float32 array: {error}"
        ))
    })?;
    let mut vad = SileroVAD::new(sample_rate, threshold).map_err(|error| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Silero VAD is unavailable for offline analysis: {error}"
        ))
    })?;
    let window_size = vad.window_size().max(1);
    let frame_count = samples.len().div_ceil(window_size);
    let mut frame = vec![0.0_f32; window_size];
    let mut probabilities = Vec::with_capacity(frame_count);

    for chunk in samples.chunks(window_size) {
        frame.fill(0.0);
        frame[..chunk.len()].copy_from_slice(chunk);
        let probability = vad.process(&frame).map_err(|error| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Silero VAD offline inference failed: {error}"
            ))
        })?;
        probabilities.push(probability.clamp(0.0, 1.0));
    }

    Ok(probabilities)
}

#[cfg(test)]
mod compressor_metric_tests {
    use super::compressor_pumping_score;

    #[test]
    fn pumping_score_is_zero_for_steady_gain_reduction() {
        let trace = vec![3.0_f32; 250];
        assert_eq!(compressor_pumping_score(&trace, 50.0), 0.0);
    }

    #[test]
    fn pumping_score_focuses_on_fast_gain_modulation() {
        let fast = (0..500)
            .map(|index| {
                3.0 + (2.0 * std::f32::consts::PI * 4.0 * index as f32 / 50.0).sin()
            })
            .collect::<Vec<_>>();
        let slow = (0..500)
            .map(|index| {
                3.0 + (2.0 * std::f32::consts::PI * 0.2 * index as f32 / 50.0).sin()
            })
            .collect::<Vec<_>>();

        assert!(
            compressor_pumping_score(&fast, 50.0)
                > 2.0 * compressor_pumping_score(&slow, 50.0)
        );
    }
}

/// Python-exposed audio processor
#[pyclass(name = "AudioProcessor", unsendable)]
pub struct PyAudioProcessor {
    processor: AudioProcessor,
}

#[pymethods]
impl PyAudioProcessor {
    #[new]
    fn new() -> Self {
        Self {
            processor: AudioProcessor::new(),
        }
    }

    /// Start audio processing
    #[pyo3(signature = (input_device=None, output_device=None))]
    fn start(
        &mut self,
        input_device: Option<&str>,
        output_device: Option<&str>,
    ) -> PyResult<String> {
        self.processor
            .start(input_device, output_device)
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)
    }

    /// Stop audio processing
    fn stop(&mut self) {
        self.processor.stop();
    }

    /// Check if running
    fn is_running(&self) -> bool {
        self.processor.is_running()
    }

    /// Get active input device name for the running stream.
    fn get_active_input_device(&self) -> Option<String> {
        self.processor.active_input_device_name()
    }

    /// Get active output device name for the running stream.
    fn get_active_output_device(&self) -> Option<String> {
        self.processor.active_output_device_name()
    }

    /// Get sample rate
    fn sample_rate(&self) -> u32 {
        self.processor.sample_rate()
    }

    /// Set master bypass
    fn set_bypass(&self, bypass: bool) {
        self.processor.set_bypass(bypass);
    }

    /// Get bypass state
    fn is_bypass(&self) -> bool {
        self.processor.is_bypass()
    }

    fn set_raw_monitor_enabled(&self, enabled: bool) {
        self.processor.set_raw_monitor_enabled(enabled);
    }

    fn is_raw_monitor_enabled(&self) -> bool {
        self.processor.is_raw_monitor_enabled()
    }

    fn set_input_channel_mode(&self, mode: &str) -> PyResult<()> {
        let mode = InputChannelMode::from_id(mode).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid input channel mode: {mode}"
            ))
        })?;
        self.processor.set_input_channel_mode(mode);
        Ok(())
    }

    fn get_input_channel_mode(&self) -> String {
        self.processor.input_channel_mode().id().to_string()
    }

    fn set_input_cleanup_mode(&self, mode: &str) -> PyResult<()> {
        let mode = InputCleanupMode::from_id(mode).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid input cleanup mode: {mode}"
            ))
        })?;
        self.processor.set_input_cleanup_mode(mode);
        Ok(())
    }

    fn get_input_cleanup_mode(&self) -> String {
        self.processor.input_cleanup_mode().id().to_string()
    }

    // === Noise Gate ===

    fn set_gate_enabled(&self, enabled: bool) {
        self.processor.set_gate_enabled(enabled);
    }

    fn is_gate_enabled(&self) -> bool {
        self.processor.is_gate_enabled()
    }

    fn get_gate_chatter_event_count(&self) -> u64 {
        self.processor.get_gate_chatter_event_count()
    }

    fn set_gate_threshold(&self, threshold_db: f64) {
        self.processor.set_gate_threshold(threshold_db);
    }

    fn set_gate_attack(&self, attack_ms: f64) {
        self.processor.set_gate_attack(attack_ms);
    }

    fn set_gate_release(&self, release_ms: f64) {
        self.processor.set_gate_release(release_ms);
    }

    // === VAD Gate Controls ===

    /// Set gate mode (0 = ThresholdOnly, 1 = VadAssisted, 2 = VadOnly)
    #[cfg(feature = "vad")]
    #[pyo3(signature = (mode))]
    fn set_gate_mode(&self, mode: u8) -> PyResult<()> {
        self.processor
            .set_gate_mode(mode)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
    }

    /// Get VAD speech probability (0.0-1.0)
    #[cfg(feature = "vad")]
    fn get_vad_probability(&self) -> f32 {
        self.processor.get_vad_probability()
    }

    /// Get fused gate open score (0.0-1.0)
    #[cfg(feature = "vad")]
    fn get_gate_fused_score(&self) -> f32 {
        self.processor.get_gate_fused_score()
    }

    /// Check whether VAD backend is available (model/runtime loaded)
    #[cfg(feature = "vad")]
    fn is_vad_available(&self) -> bool {
        self.processor.is_vad_available()
    }

    /// Set VAD probability threshold (0.0-1.0)
    #[cfg(feature = "vad")]
    fn set_vad_threshold(&self, threshold: f32) {
        self.processor.set_vad_threshold(threshold);
    }

    /// Set VAD hold time in milliseconds
    #[cfg(feature = "vad")]
    fn set_vad_hold_time(&self, hold_ms: f32) {
        self.processor.set_vad_hold_time(hold_ms);
    }

    /// Set VAD pre-gain to boost weak signals for better speech detection
    /// Default is 1.0 (no gain). Values > 1.0 boost the signal.
    /// This helps with quiet microphones where VAD can't detect speech.
    #[cfg(feature = "vad")]
    fn set_vad_pre_gain(&self, gain: f32) {
        self.processor.set_vad_pre_gain(gain);
    }

    /// Enable/disable auto-threshold mode (automatically adjusts gate threshold based on noise floor)
    #[cfg(feature = "vad")]
    fn set_auto_threshold(&self, enabled: bool) {
        self.processor.set_auto_threshold(enabled);
    }

    /// Set margin above noise floor for auto-threshold (in dB)
    #[cfg(feature = "vad")]
    fn set_gate_margin(&self, margin_db: f32) {
        self.processor.set_gate_margin(margin_db);
    }

    /// Get current noise floor estimate (in dB)
    #[cfg(feature = "vad")]
    fn get_noise_floor(&self) -> f32 {
        self.processor.get_noise_floor()
    }

    /// Get current gate margin (in dB)
    #[cfg(feature = "vad")]
    fn gate_margin(&self) -> f32 {
        self.processor.gate_margin()
    }

    /// Check if auto-threshold is enabled
    #[cfg(feature = "vad")]
    fn auto_threshold_enabled(&self) -> bool {
        self.processor.auto_threshold_enabled()
    }

    /// Get current VAD pre-gain
    #[cfg(feature = "vad")]
    fn vad_pre_gain(&self) -> f32 {
        self.processor.vad_pre_gain()
    }

    // === RNNoise ===

    fn set_rnnoise_enabled(&self, enabled: bool) {
        self.processor.set_rnnoise_enabled(enabled);
    }

    fn is_rnnoise_enabled(&self) -> bool {
        self.processor.is_rnnoise_enabled()
    }

    /// Set RNNoise wet/dry mix strength (0.0 = fully dry, 1.0 = fully wet)
    fn set_rnnoise_strength(&self, strength: f64) {
        self.processor.set_rnnoise_strength(strength as f32);
    }

    /// Get current RNNoise strength
    fn get_rnnoise_strength(&self) -> f64 {
        self.processor.get_rnnoise_strength() as f64
    }

    /// Set noise suppression model by name ("rnnoise" or "deepfilter")
    fn set_noise_model(&self, model: &str) -> bool {
        match NoiseModel::from_id(model) {
            Some(m) => self.processor.set_noise_model(m),
            None => false,
        }
    }

    /// Get current noise model name
    fn get_noise_model(&self) -> String {
        self.processor.get_noise_model().id().to_string()
    }

    /// Get current noise model display name
    fn get_noise_model_display_name(&self) -> String {
        self.processor.get_noise_model().display_name().to_string()
    }

    /// List available noise models: [(id, display_name), ...]
    fn list_noise_models(&self) -> Vec<(String, String)> {
        self.processor.list_noise_models()
    }

    // === EQ ===

    fn set_eq_enabled(&self, enabled: bool) {
        self.processor.set_eq_enabled(enabled);
    }

    fn is_eq_enabled(&self) -> bool {
        self.processor.is_eq_enabled()
    }

    fn set_eq_band_gain(&self, band: usize, gain_db: f64) -> PyResult<()> {
        self.processor
            .set_eq_band_gain(band, gain_db)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
    }

    fn set_eq_band_frequency(&self, band: usize, frequency: f64) -> PyResult<()> {
        self.processor
            .set_eq_band_frequency(band, frequency)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
    }

    fn set_eq_band_q(&self, band: usize, q: f64) -> PyResult<()> {
        self.processor
            .set_eq_band_q(band, q)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
    }

    fn get_eq_band_params(&self, band: usize) -> Option<(f64, f64, f64)> {
        self.processor.get_eq_band_params(band)
    }

    /// Apply EQ settings for all 10 bands in a single atomic call
    ///
    /// Args:
    ///     bands: List of (frequency_hz, gain_db, q) tuples for each band (must be 10)
    ///
    /// Raises:
    ///     ValueError: If band count is not 10 or parameters are out of range
    fn apply_eq_settings(&self, bands: Vec<(f64, f64, f64)>) -> PyResult<()> {
        self.processor.apply_eq_settings(bands)
    }

    // === De-Esser ===

    fn set_deesser_enabled(&self, enabled: bool) {
        self.processor.set_deesser_enabled(enabled);
    }

    fn is_deesser_enabled(&self) -> bool {
        self.processor.is_deesser_enabled()
    }

    fn set_deesser_low_cut_hz(&self, hz: f64) {
        self.processor.set_deesser_low_cut_hz(hz);
    }

    fn set_deesser_high_cut_hz(&self, hz: f64) {
        self.processor.set_deesser_high_cut_hz(hz);
    }

    fn set_deesser_threshold_db(&self, threshold_db: f64) {
        self.processor.set_deesser_threshold_db(threshold_db);
    }

    fn set_deesser_ratio(&self, ratio: f64) {
        self.processor.set_deesser_ratio(ratio);
    }

    fn set_deesser_attack_ms(&self, attack_ms: f64) {
        self.processor.set_deesser_attack_ms(attack_ms);
    }

    fn set_deesser_release_ms(&self, release_ms: f64) {
        self.processor.set_deesser_release_ms(release_ms);
    }

    fn set_deesser_max_reduction_db(&self, max_reduction_db: f64) {
        self.processor
            .set_deesser_max_reduction_db(max_reduction_db);
    }

    fn set_deesser_auto_enabled(&self, auto_enabled: bool) {
        self.processor.set_deesser_auto_enabled(auto_enabled);
    }

    fn is_deesser_auto_enabled(&self) -> bool {
        self.processor.is_deesser_auto_enabled()
    }

    fn set_deesser_auto_amount(&self, amount: f64) {
        self.processor.set_deesser_auto_amount(amount);
    }

    fn get_deesser_low_cut_hz(&self) -> f64 {
        self.processor.get_deesser_low_cut_hz()
    }

    fn get_deesser_high_cut_hz(&self) -> f64 {
        self.processor.get_deesser_high_cut_hz()
    }

    fn get_deesser_threshold_db(&self) -> f64 {
        self.processor.get_deesser_threshold_db()
    }

    fn get_deesser_ratio(&self) -> f64 {
        self.processor.get_deesser_ratio()
    }

    fn get_deesser_max_reduction_db(&self) -> f64 {
        self.processor.get_deesser_max_reduction_db()
    }

    fn get_deesser_auto_amount(&self) -> f64 {
        self.processor.get_deesser_auto_amount()
    }

    fn get_deesser_gain_reduction_db(&self) -> f32 {
        self.processor.get_deesser_gain_reduction_db()
    }

    fn get_deesser_detector_confidence(&self) -> f32 {
        self.processor.get_deesser_detector_confidence()
    }

    // === Compressor ===

    fn set_compressor_enabled(&self, enabled: bool) {
        self.processor.set_compressor_enabled(enabled);
    }

    fn is_compressor_enabled(&self) -> bool {
        self.processor.is_compressor_enabled()
    }

    fn set_compressor_threshold(&self, threshold_db: f64) {
        self.processor.set_compressor_threshold(threshold_db);
    }

    fn set_compressor_ratio(&self, ratio: f64) {
        self.processor.set_compressor_ratio(ratio);
    }

    fn set_compressor_attack(&self, attack_ms: f64) {
        self.processor.set_compressor_attack(attack_ms);
    }

    fn set_compressor_release(&self, release_ms: f64) {
        self.processor.set_compressor_release(release_ms);
    }

    /// Get compressor release time.
    ///
    /// Note: When adaptive release is enabled, this returns the base release time.
    /// Use get_compressor_current_release() for the actual adaptive release time.
    fn get_compressor_release(&self) -> f64 {
        self.processor.get_compressor_release()
    }

    fn set_compressor_makeup_gain(&self, makeup_gain_db: f64) {
        self.processor.set_compressor_makeup_gain(makeup_gain_db);
    }

    /// Set compressor adaptive release mode
    fn set_compressor_adaptive_release(&self, enabled: bool) {
        self.processor.set_compressor_adaptive_release(enabled);
    }

    /// Get compressor adaptive release mode
    fn get_compressor_adaptive_release(&self) -> bool {
        self.processor.get_compressor_adaptive_release()
    }

    /// Set compressor base release time (milliseconds)
    fn set_compressor_base_release(&self, release_ms: f64) {
        self.processor.set_compressor_base_release(release_ms);
    }

    /// Get compressor base release time (milliseconds)
    fn get_compressor_base_release(&self) -> f64 {
        self.processor.get_compressor_base_release()
    }

    /// Set compressor detector sidechain high-pass mode
    fn set_compressor_sidechain_highpass_enabled(&self, enabled: bool) {
        self.processor
            .set_compressor_sidechain_highpass_enabled(enabled);
    }

    /// Get compressor detector sidechain high-pass mode
    fn get_compressor_sidechain_highpass_enabled(&self) -> bool {
        self.processor.get_compressor_sidechain_highpass_enabled()
    }

    /// Get current compressor release time (adaptive or base, in milliseconds)
    fn get_compressor_current_release(&self) -> f64 {
        let release_raw = self
            .processor
            .compressor_current_release_ms
            .load(Ordering::Relaxed);
        release_raw as f64 / 10.0 // Convert back from 0.1ms resolution
    }

    // === Auto Makeup Gain ===

    /// Set compressor auto makeup gain mode
    fn set_compressor_auto_makeup_enabled(&self, enabled: bool) {
        self.processor.set_compressor_auto_makeup_enabled(enabled);
    }

    /// Get compressor auto makeup gain mode
    fn get_compressor_auto_makeup_enabled(&self) -> bool {
        self.processor.get_compressor_auto_makeup_enabled()
    }

    /// Set compressor target LUFS
    fn set_compressor_target_lufs(&self, target_lufs: f64) {
        self.processor.set_compressor_target_lufs(target_lufs);
    }

    /// Get compressor target LUFS
    fn get_compressor_target_lufs(&self) -> f64 {
        self.processor.get_compressor_target_lufs()
    }

    /// Set confidence in Auto Voice Setup's room-noise reference (0.0-1.0).
    fn set_compressor_noise_reference_reliability(&self, reliability: f64) {
        self.processor
            .set_compressor_noise_reference_reliability(reliability);
    }

    /// Get compressor current LUFS
    fn get_compressor_current_lufs(&self) -> f64 {
        self.processor.get_compressor_current_lufs()
    }

    /// Get compressor current makeup gain
    fn get_compressor_current_makeup_gain(&self) -> f64 {
        self.processor.get_compressor_current_makeup_gain()
    }

    // === Limiter ===

    fn set_limiter_enabled(&self, enabled: bool) {
        self.processor.set_limiter_enabled(enabled);
    }

    fn is_limiter_enabled(&self) -> bool {
        self.processor.is_limiter_enabled()
    }

    fn set_limiter_ceiling(&self, ceiling_db: f64) {
        self.processor.set_limiter_ceiling(ceiling_db);
    }

    fn set_limiter_release(&self, release_ms: f64) {
        self.processor.set_limiter_release(release_ms);
    }

    fn set_limiter_careful_output_enabled(&self, enabled: bool) {
        self.processor
            .set_limiter_careful_output_enabled(enabled);
    }

    fn is_limiter_careful_output_enabled(&self) -> bool {
        self.processor.is_limiter_careful_output_enabled()
    }

    fn get_limiter_effective_ceiling_db(&self) -> f64 {
        self.processor.limiter_effective_ceiling_db()
    }

    // === Metering ===

    fn get_input_peak_db(&self) -> f32 {
        self.processor.get_input_peak_db()
    }

    fn get_input_rms_db(&self) -> f32 {
        self.processor.get_input_rms_db()
    }

    fn get_input_crest_factor_db(&self) -> f32 {
        self.processor.get_input_crest_factor_db()
    }

    fn get_output_peak_db(&self) -> f32 {
        self.processor.get_output_peak_db()
    }

    fn get_output_rms_db(&self) -> f32 {
        self.processor.get_output_rms_db()
    }

    fn get_output_crest_factor_db(&self) -> f32 {
        self.processor.get_output_crest_factor_db()
    }

    fn get_output_short_term_lufs(&self) -> f32 {
        self.processor.get_output_short_term_lufs()
    }

    fn get_input_stereo_correlation(&self) -> f32 {
        self.processor.get_input_stereo_correlation()
    }

    fn get_input_phase_warning_count(&self) -> u64 {
        self.processor.get_input_phase_warning_count()
    }

    fn get_compressor_gain_reduction_db(&self) -> f32 {
        self.processor.get_compressor_gain_reduction_db()
    }

    fn get_latency_ms(&self) -> f32 {
        self.processor.get_latency_ms()
    }

    fn get_engine_latency_ms(&self) -> f32 {
        self.processor.get_engine_latency_ms()
    }

    fn set_latency_compensation_ms(&self, compensation_ms: f32) {
        self.processor.set_latency_compensation_ms(compensation_ms);
    }

    fn get_latency_compensation_ms(&self) -> f32 {
        self.processor.get_latency_compensation_ms()
    }

    // === DSP Performance Metrics ===

    fn get_dsp_time_ms(&self) -> f32 {
        self.processor.get_dsp_time_ms()
    }

    fn get_input_buffer_samples(&self) -> u32 {
        self.processor.get_input_buffer_samples()
    }

    fn get_input_buffer_smoothed_samples(&self) -> u32 {
        self.processor.get_input_buffer_smoothed_samples()
    }

    fn get_output_buffer_samples(&self) -> u32 {
        self.processor.get_output_buffer_samples()
    }

    fn output_sample_rate(&self) -> u32 {
        self.processor.output_sample_rate()
    }

    fn output_fixed_buffer_frames(&self) -> u32 {
        self.processor.output_fixed_buffer_frames()
    }

    fn input_fixed_buffer_frames(&self) -> u32 {
        self.processor.input_fixed_buffer_frames()
    }

    fn get_rnnoise_buffer_samples(&self) -> u32 {
        self.processor.get_rnnoise_buffer_samples()
    }

    /// Get smoothed DSP processing time in milliseconds
    fn get_dsp_time_smoothed_ms(&self) -> f32 {
        let us = self.processor.dsp_time_smoothed_us.load(Ordering::Relaxed);
        us as f32 / 1000.0
    }

    /// Get smoothed suppressor buffer fill level in samples
    fn get_buffer_smoothed_samples(&self) -> u32 {
        self.processor.smoothed_buffer_len.load(Ordering::Relaxed)
    }

    // === Dropped Sample Tracking ===

    fn get_dropped_samples(&self) -> u64 {
        self.processor.get_dropped_samples()
    }

    fn reset_dropped_samples(&self) {
        self.processor.reset_dropped_samples();
    }

    fn get_lock_contention_count(&self) -> u64 {
        self.processor.get_lock_contention_count()
    }

    fn reset_lock_contention_count(&self) {
        self.processor.reset_lock_contention_count();
    }

    fn get_input_callback_age_ms(&self) -> u64 {
        self.processor.get_input_callback_age_ms()
    }

    fn get_output_callback_age_ms(&self) -> u64 {
        self.processor.get_output_callback_age_ms()
    }

    fn get_output_underrun_streak(&self) -> u32 {
        self.processor.get_output_underrun_streak()
    }

    fn get_output_underrun_total(&self) -> u64 {
        self.processor.get_output_underrun_total()
    }

    fn get_jitter_dropped_samples(&self) -> u64 {
        self.processor.get_jitter_dropped_samples()
    }

    fn get_output_retime_adjustment_count(&self) -> u64 {
        self.processor.get_output_retime_adjustment_count()
    }

    fn get_output_recovery_event_count(&self) -> u64 {
        self.processor.get_output_recovery_event_count()
    }

    fn get_output_recovery_count(&self) -> u64 {
        self.processor.get_output_recovery_count()
    }

    fn get_suppressor_non_finite_count(&self) -> u64 {
        self.processor.get_suppressor_non_finite_count()
    }

    fn get_rt_error_code(&self) -> u32 {
        self.processor.get_rt_error_code()
    }

    fn get_rt_error_name(&self) -> &'static str {
        self.processor.get_rt_error_name()
    }

    fn get_input_callback_error_count(&self) -> u64 {
        self.processor.get_input_callback_error_count()
    }

    fn get_output_callback_error_count(&self) -> u64 {
        self.processor.get_output_callback_error_count()
    }

    fn get_rt_buffer_overflow_count(&self) -> u64 {
        self.processor.get_rt_buffer_overflow_count()
    }

    fn is_noise_backend_available(&self) -> bool {
        self.processor.is_noise_backend_available()
    }

    fn noise_backend_failed(&self) -> bool {
        self.processor.noise_backend_failed()
    }

    fn noise_backend_error(&self) -> Option<String> {
        self.processor.noise_backend_error()
    }

    fn set_recovery_suppressed(&self, suppressed: bool) {
        self.processor.set_recovery_suppressed(suppressed);
    }

    fn is_recovery_suppressed(&self) -> bool {
        self.processor.is_recovery_suppressed()
    }

    fn get_runtime_diagnostics(&self, py: Python) -> PyResult<Py<PyAny>> {
        let diagnostics = pyo3::types::PyDict::new(py);
        let noise_model = self.processor.get_noise_model();
        diagnostics.set_item("noise_model", noise_model.id())?;
        #[cfg(feature = "deepfilter")]
        {
            let deepfilter_config = if matches!(
                noise_model,
                NoiseModel::DeepFilterNetLL | NoiseModel::DeepFilterNet
            ) {
                Some(crate::dsp::deepfilter_ffi::DeepFilterRuntimeConfig::default())
            } else {
                None
            };
            diagnostics.set_item(
                "noise_attenuation_limit_db",
                deepfilter_config.map(|config| config.attenuation_limit_db()),
            )?;
            diagnostics.set_item(
                "noise_post_filter_beta",
                deepfilter_config.map(|config| config.post_filter_beta()),
            )?;
        }
        diagnostics.set_item(
            "noise_backend_available",
            self.processor.is_noise_backend_available(),
        )?;
        diagnostics.set_item(
            "noise_backend_failed",
            self.processor.noise_backend_failed(),
        )?;
        diagnostics.set_item("noise_backend_error", self.processor.noise_backend_error())?;
        diagnostics.set_item(
            "input_dropped_samples",
            self.processor.get_dropped_samples(),
        )?;
        diagnostics.set_item(
            "input_backlog_recovery_count",
            self.processor
                .input_backlog_recovery_count
                .load(Ordering::Relaxed),
        )?;
        diagnostics.set_item(
            "input_backlog_dropped_samples",
            self.processor
                .input_backlog_dropped_samples
                .load(Ordering::Relaxed),
        )?;
        diagnostics.set_item(
            "lock_contention_count",
            self.processor.get_lock_contention_count(),
        )?;
        diagnostics.set_item(
            "output_underrun_total",
            self.processor.get_output_underrun_total(),
        )?;
        diagnostics.set_item(
            "output_underrun_streak",
            self.processor.get_output_underrun_streak(),
        )?;
        diagnostics.set_item(
            "jitter_dropped_samples",
            self.processor.get_jitter_dropped_samples(),
        )?;
        diagnostics.set_item(
            "output_retime_adjustment_count",
            self.processor.get_output_retime_adjustment_count(),
        )?;
        diagnostics.set_item(
            "output_recovery_event_count",
            self.processor.get_output_recovery_event_count(),
        )?;
        diagnostics.set_item(
            "output_recovery_count",
            self.processor.get_output_recovery_event_count(),
        )?;
        diagnostics.set_item(
            "dsp_idle_wakeup_count",
            self.processor.get_dsp_idle_wakeup_count(),
        )?;
        diagnostics.set_item("dsp_idle_sleep_us", self.processor.get_dsp_idle_sleep_us())?;
        diagnostics.set_item(
            "output_short_write_dropped_samples",
            self.processor
                .output_short_write_dropped_samples
                .load(Ordering::Relaxed),
        )?;
        diagnostics.set_item(
            "input_channel_mode",
            self.processor.input_channel_mode().id(),
        )?;
        diagnostics.set_item(
            "input_cleanup_mode",
            self.processor.input_cleanup_mode().id(),
        )?;
        diagnostics.set_item(
            "input_cleanup_hum_detected",
            self.processor
                .input_cleanup_hum_detected
                .load(Ordering::Relaxed),
        )?;
        diagnostics.set_item(
            "input_cleanup_rumble_detected",
            self.processor
                .input_cleanup_rumble_detected
                .load(Ordering::Relaxed),
        )?;
        diagnostics.set_item(
            "input_cleanup_high_pass_hz",
            f32::from_bits(
                self.processor
                    .input_cleanup_high_pass_hz
                    .load(Ordering::Relaxed),
            ),
        )?;
        diagnostics.set_item(
            "input_crest_factor_db",
            self.processor.get_input_crest_factor_db(),
        )?;
        diagnostics.set_item(
            "output_crest_factor_db",
            self.processor.get_output_crest_factor_db(),
        )?;
        diagnostics.set_item(
            "output_short_term_lufs",
            self.processor.get_output_short_term_lufs(),
        )?;
        diagnostics.set_item(
            "input_stereo_correlation",
            self.processor.get_input_stereo_correlation(),
        )?;
        diagnostics.set_item(
            "input_phase_warning_count",
            self.processor.get_input_phase_warning_count(),
        )?;
        let phase_strategy = crate::audio::input::PhaseRescueStrategy::from_u8(
            self.processor
                .input_phase_rescue_strategy
                .load(Ordering::Relaxed),
        );
        diagnostics.set_item("input_phase_rescue_strategy", phase_strategy.name())?;
        diagnostics.set_item(
            "input_phase_estimated_delay_samples",
            f32::from_bits(
                self.processor
                    .input_phase_estimated_delay_samples
                    .load(Ordering::Relaxed),
            ),
        )?;
        diagnostics.set_item(
            "input_phase_polarity_flipped",
            self.processor
                .input_phase_polarity_flipped
                .load(Ordering::Relaxed),
        )?;
        diagnostics.set_item(
            "stream_restart_count",
            self.processor.get_stream_restart_count(),
        )?;
        diagnostics.set_item(
            "last_restart_reason",
            self.processor.get_last_restart_reason(),
        )?;
        diagnostics.set_item("last_stream_error", self.processor.get_last_stream_error())?;
        diagnostics.set_item(
            "suppressor_non_finite_count",
            self.processor.get_suppressor_non_finite_count(),
        )?;
        diagnostics.set_item("rt_error_code", self.processor.get_rt_error_code())?;
        diagnostics.set_item("rt_error_name", self.processor.get_rt_error_name())?;
        diagnostics.set_item(
            "input_callback_error_count",
            self.processor.get_input_callback_error_count(),
        )?;
        diagnostics.set_item(
            "output_callback_error_count",
            self.processor.get_output_callback_error_count(),
        )?;
        diagnostics.set_item(
            "rt_buffer_overflow_count",
            self.processor.get_rt_buffer_overflow_count(),
        )?;
        diagnostics.set_item(
            "clip_event_count",
            self.processor.clip_event_count.load(Ordering::Relaxed),
        )?;
        diagnostics.set_item(
            "clip_peak_db",
            f32::from_bits(self.processor.clip_peak_db.load(Ordering::Relaxed)),
        )?;
        diagnostics.set_item(
            "output_clip_event_count",
            self.processor
                .output_clip_event_count
                .load(Ordering::Relaxed),
        )?;
        diagnostics.set_item(
            "output_clip_peak_db",
            f32::from_bits(
                self.processor
                    .output_clip_peak_db
                    .load(Ordering::Relaxed),
            ),
        )?;
        diagnostics.set_item(
            "output_true_peak_event_count",
            self.processor
                .output_true_peak_event_count
                .load(Ordering::Relaxed),
        )?;
        diagnostics.set_item(
            "output_true_peak_db",
            f32::from_bits(
                self.processor
                    .output_true_peak_db
                    .load(Ordering::Relaxed),
            ),
        )?;
        diagnostics.set_item(
            "output_true_peak_input_db",
            f32::from_bits(
                self.processor
                    .output_true_peak_input_db
                    .load(Ordering::Relaxed),
            ),
        )?;
        diagnostics.set_item(
            "output_true_peak_gain_reduction_db",
            f32::from_bits(
                self.processor
                    .output_true_peak_gain_reduction_db
                    .load(Ordering::Relaxed),
            ),
        )?;
        diagnostics.set_item(
            "output_true_peak_gain_reduction_history_db",
            f32::from_bits(
                self.processor
                    .output_true_peak_gain_reduction_history_db
                    .load(Ordering::Relaxed),
            ),
        )?;
        diagnostics.set_item(
            "output_true_peak_headroom_db",
            f32::from_bits(
                self.processor
                    .output_true_peak_headroom_db
                    .load(Ordering::Relaxed),
            ),
        )?;
        diagnostics.set_item(
            "limiter_gain_reduction_db",
            f32::from_bits(
                self.processor
                    .limiter_gain_reduction_db
                    .load(Ordering::Relaxed),
            ),
        )?;
        diagnostics.set_item(
            "limiter_peak_gain_reduction_db",
            f32::from_bits(
                self.processor
                    .limiter_peak_gain_reduction_db
                    .load(Ordering::Relaxed),
            ),
        )?;
        diagnostics.set_item(
            "limiter_gain_reduction_history_db",
            f32::from_bits(
                self.processor
                    .limiter_gain_reduction_history_db
                    .load(Ordering::Relaxed),
            ),
        )?;
        diagnostics.set_item(
            "limiter_careful_output_enabled",
            self.processor.is_limiter_careful_output_enabled(),
        )?;
        diagnostics.set_item(
            "limiter_effective_ceiling_db",
            self.processor.limiter_effective_ceiling_db(),
        )?;
        diagnostics.set_item(
            "gate_chatter_event_count",
            self.processor.get_gate_chatter_event_count(),
        )?;
        diagnostics.set_item(
            "gate_auto_relax_active",
            self.processor.is_gate_auto_relax_active(),
        )?;
        diagnostics.set_item(
            "deesser_detector_confidence",
            self.processor.get_deesser_detector_confidence(),
        )?;
        diagnostics.set_item(
            "input_resampler_active",
            self.processor
                .input_resampler_active
                .load(Ordering::Relaxed),
        )?;
        diagnostics.set_item(
            "output_resampler_active",
            self.processor
                .output_resampler_active
                .load(Ordering::Relaxed),
        )?;
        diagnostics.set_item("output_sample_rate", self.processor.output_sample_rate())?;
        diagnostics.set_item(
            "output_fixed_buffer_frames",
            self.processor.output_fixed_buffer_frames(),
        )?;
        diagnostics.set_item(
            "input_fixed_buffer_frames",
            self.processor.input_fixed_buffer_frames(),
        )?;
        diagnostics.set_item("engine_latency_ms", self.processor.get_engine_latency_ms())?;
        diagnostics.set_item("total_latency_ms", self.processor.get_latency_ms())?;
        diagnostics.set_item(
            "recovery_suppressed",
            self.processor.is_recovery_suppressed(),
        )?;
        diagnostics.set_item(
            "raw_monitor_enabled",
            self.processor.is_raw_monitor_enabled(),
        )?;
        #[cfg(feature = "vad")]
        diagnostics.set_item("gate_fused_score", self.processor.get_gate_fused_score())?;
        Ok(diagnostics.into_any().unbind())
    }

    // === Stream Recovery Status ===

    /// Service pending recovery requests (returns None if no attempt).
    fn service_recovery(&mut self) -> Option<bool> {
        self.processor.service_recovery()
    }

    fn is_recovery_requested(&self) -> bool {
        self.processor.is_recovery_requested()
    }

    fn is_recovering(&self) -> bool {
        self.processor.is_recovering()
    }

    fn get_stream_restart_count(&self) -> u64 {
        self.processor.get_stream_restart_count()
    }

    fn get_last_stream_error(&self) -> Option<String> {
        self.processor.get_last_stream_error()
    }

    fn get_last_restart_reason(&self) -> Option<String> {
        self.processor.get_last_restart_reason()
    }

    // === RAW AUDIO RECORDING (for calibration) ===

    /// Start recording raw audio for calibration (10 seconds @ 48kHz)
    fn start_raw_recording(&mut self, duration_secs: f64) -> PyResult<()> {
        self.processor
            .start_raw_recording(duration_secs)
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)
    }

    /// Stop recording and return audio data as NumPy array
    fn stop_raw_recording(&mut self, py: Python) -> PyResult<Py<PyAny>> {
        if let Some(audio) = self.processor.stop_raw_recording() {
            // Zero-copy transfer to NumPy
            use numpy::PyArray1;
            let array = PyArray1::from_vec(py, audio);
            Ok(array.into_any().unbind())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "No recording in progress",
            ))
        }
    }

    /// Check if recording is complete
    fn is_recording_complete(&mut self) -> bool {
        self.processor.is_recording_complete()
    }

    /// Get recording progress (0.0 to 1.0)
    fn recording_progress(&mut self) -> f32 {
        self.processor.recording_progress()
    }

    /// Get current recording level as RMS in dB (for level meter visualization)
    fn recording_level_db(&mut self) -> f32 {
        self.processor.recording_level_db()
    }

    /// Manually set output mute state (useful for calibration workflow)
    fn set_output_mute(&mut self, muted: bool) {
        self.processor.set_output_mute(muted);
    }

    /// Queue a mono calibration probe on the selected CPAL output route.
    fn queue_output_probe(
        &self,
        samples: numpy::PyReadonlyArray1<'_, f32>,
    ) -> PyResult<()> {
        let samples = samples.as_slice().map_err(|error| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "output probe must be a contiguous float32 array: {error}"
            ))
        })?;
        self.processor
            .queue_output_probe(samples)
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)
    }

    fn is_output_probe_complete(&self) -> bool {
        self.processor.is_output_probe_complete()
    }

    fn cancel_output_probe(&self) {
        self.processor.cancel_output_probe();
    }
}
