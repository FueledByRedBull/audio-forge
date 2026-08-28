use mic_eq_core::dsp::deepfilter_ffi::{
    configure_app_owned_paths, DeepFilterModel, DeepFilterProcessor, DeepFilterRuntimeConfig,
    DEEPFILTER_FRAME_SIZE,
};
use mic_eq_core::dsp::NoiseSuppressor;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::atomic::AtomicU32;
use std::sync::Arc;
use std::time::{Duration, Instant};

fn percentile_seconds(durations: &mut [Duration], percentile: f64) -> f64 {
    if durations.is_empty() {
        return 0.0;
    }
    durations.sort_unstable();
    let index = ((durations.len() - 1) as f64 * percentile)
        .round()
        .clamp(0.0, (durations.len() - 1) as f64) as usize;
    durations[index].as_secs_f64()
}

fn arguments() -> Result<HashMap<String, String>, String> {
    let mut values = HashMap::new();
    let mut args = std::env::args().skip(1);
    while let Some(name) = args.next() {
        if !name.starts_with("--") {
            return Err(format!("Unexpected positional argument: {name}"));
        }
        let value = args
            .next()
            .ok_or_else(|| format!("Missing value for {name}"))?;
        values.insert(name, value);
    }
    Ok(values)
}

fn required<'a>(args: &'a HashMap<String, String>, name: &str) -> Result<&'a str, String> {
    args.get(name)
        .map(String::as_str)
        .ok_or_else(|| format!("Missing required argument {name}"))
}

fn parse_f32(args: &HashMap<String, String>, name: &str, default: f32) -> Result<f32, String> {
    args.get(name)
        .map(|value| {
            value
                .parse::<f32>()
                .map_err(|error| format!("Invalid {name} value {value:?}: {error}"))
        })
        .unwrap_or(Ok(default))
}

fn read_f32_le(path: &Path) -> Result<Vec<f32>, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("Could not read {}: {error}", path.display()))?;
    if bytes.len() % size_of::<f32>() != 0 {
        return Err(format!(
            "Input {} has {} bytes, which is not valid raw f32 audio",
            path.display(),
            bytes.len()
        ));
    }
    Ok(bytes
        .as_chunks::<{ size_of::<f32>() }>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect())
}

fn write_f32_le(path: &Path, samples: &[f32]) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(samples));
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    fs::write(path, bytes).map_err(|error| format!("Could not write {}: {error}", path.display()))
}

fn main() -> Result<(), String> {
    let args = arguments()?;
    let model = match required(&args, "--model")? {
        "ll" => DeepFilterModel::LowLatency,
        "standard" => DeepFilterModel::Standard,
        value => return Err(format!("Unknown --model {value:?}; use ll or standard")),
    };
    let input_path = Path::new(required(&args, "--input")?);
    let output_path = Path::new(required(&args, "--output")?);
    let library_path = required(&args, "--library")?;
    let model_dir = required(&args, "--model-dir")?;
    let attenuation_limit_db = parse_f32(&args, "--attenuation-db", 80.0)?;
    let post_filter_beta = parse_f32(&args, "--post-filter-beta", 0.0)?;
    let strength = parse_f32(&args, "--strength", 1.0)?.clamp(0.0, 1.0);
    let config = DeepFilterRuntimeConfig::try_new(attenuation_limit_db, post_filter_beta)?;

    configure_app_owned_paths(Some(library_path), Some(model_dir))?;
    std::env::set_var("AUDIOFORGE_ENABLE_DEEPFILTER", "1");

    let mut input = read_f32_le(input_path)?;
    let source_samples = input.len();
    let latency_samples = model.latency_samples();
    input.resize(source_samples + latency_samples, 0.0);
    let padded_samples = input.len().div_ceil(DEEPFILTER_FRAME_SIZE) * DEEPFILTER_FRAME_SIZE;
    input.resize(padded_samples, 0.0);

    let initialization_started = Instant::now();
    let strength_bits = Arc::new(AtomicU32::new(strength.to_bits()));
    let mut processor = DeepFilterProcessor::new_with_config(strength_bits, model, config);
    let initialization_seconds = initialization_started.elapsed().as_secs_f64();
    if !processor.is_ffi_available() {
        return Err(format!(
            "DeepFilter backend unavailable: {}",
            processor.backend_error().unwrap_or("unknown error")
        ));
    }

    let processing_started = Instant::now();
    let mut output = Vec::with_capacity(input.len());
    let mut produced = vec![0.0_f32; DEEPFILTER_FRAME_SIZE];
    let mut frame_durations = Vec::with_capacity(input.len().div_ceil(DEEPFILTER_FRAME_SIZE));
    for frame in input.as_chunks::<DEEPFILTER_FRAME_SIZE>().0 {
        let frame_started = Instant::now();
        if processor.push_samples(frame) != DEEPFILTER_FRAME_SIZE {
            return Err("DeepFilter input buffer rejected a complete frame".to_string());
        }
        processor.process_frames();
        let produced_len = processor.pop_samples_into(&mut produced);
        if produced_len != DEEPFILTER_FRAME_SIZE {
            return Err(format!(
                "DeepFilter produced {} samples for a {}-sample frame",
                produced_len, DEEPFILTER_FRAME_SIZE
            ));
        }
        output.extend_from_slice(&produced[..produced_len]);
        frame_durations.push(frame_started.elapsed());
    }
    let processing_seconds = processing_started.elapsed().as_secs_f64();
    write_f32_le(output_path, &output)?;

    let audio_seconds = input.len() as f64 / 48_000.0;
    let realtime_factor = processing_seconds / audio_seconds.max(f64::EPSILON);
    let mut p95_durations = frame_durations.clone();
    let mut p99_durations = frame_durations.clone();
    let p95_frame_seconds = percentile_seconds(&mut p95_durations, 0.95);
    let p99_frame_seconds = percentile_seconds(&mut p99_durations, 0.99);
    let max_frame_seconds = frame_durations
        .iter()
        .copied()
        .max()
        .unwrap_or_default()
        .as_secs_f64();
    println!(
        concat!(
            "{{\"schema_version\":1,\"model\":\"{}\",\"source_samples\":{},",
            "\"processed_samples\":{},\"latency_samples\":{},",
            "\"attenuation_limit_db\":{},\"post_filter_beta\":{},\"strength\":{},",
            "\"initialization_seconds\":{},\"processing_seconds\":{},\"rtf\":{},",
            "\"frame_count\":{},\"p95_frame_seconds\":{},\"p99_frame_seconds\":{},",
            "\"max_frame_seconds\":{}}}"
        ),
        match model {
            DeepFilterModel::LowLatency => "ll",
            DeepFilterModel::Standard => "standard",
        },
        source_samples,
        input.len(),
        latency_samples,
        config.attenuation_limit_db(),
        config.post_filter_beta(),
        strength,
        initialization_seconds,
        processing_seconds,
        realtime_factor,
        frame_durations.len(),
        p95_frame_seconds,
        p99_frame_seconds,
        max_frame_seconds,
    );
    Ok(())
}
