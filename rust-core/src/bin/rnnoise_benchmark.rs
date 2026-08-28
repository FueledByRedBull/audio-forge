//! Evaluation-only benchmark for the shipped `nnnoiseless` RNNoise backend.

use nnnoiseless::DenoiseState;
use std::env;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::time::Instant;

const FRAME_SIZE: usize = 480;
const SAMPLE_RATE: f64 = 48_000.0;
const PCM_SCALE: f32 = 32_768.0;

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let position = quantile * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f64;
    sorted[lower] + fraction * (sorted[upper] - sorted[lower])
}

fn read_f32_frame(
    reader: &mut BufReader<File>,
    frame: &mut [f32; FRAME_SIZE],
) -> io::Result<usize> {
    let mut bytes = [0_u8; FRAME_SIZE * 4];
    let mut filled = 0;
    while filled < bytes.len() {
        let count = reader.read(&mut bytes[filled..])?;
        if count == 0 {
            break;
        }
        filled += count;
    }
    if filled % 4 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "input length is not a multiple of float32",
        ));
    }
    let sample_count = filled / 4;
    frame.fill(0.0);
    for (sample, chunk) in frame.iter_mut().zip(bytes[..filled].as_chunks::<4>().0) {
        *sample = f32::from_le_bytes(*chunk);
    }
    Ok(sample_count)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!(
            "usage: {} <input-f32> <output-f32> <metadata-json>",
            args.first().map_or("rnnoise_benchmark", String::as_str)
        );
        std::process::exit(2);
    }

    let mut reader = BufReader::new(File::open(&args[1])?);
    let mut writer = BufWriter::new(File::create(&args[2])?);
    let mut state = Box::new(DenoiseState::new());
    let mut frame = [0.0_f32; FRAME_SIZE];
    let mut model_input = [0.0_f32; FRAME_SIZE];
    let mut model_output = [0.0_f32; FRAME_SIZE];
    let mut frame_seconds = Vec::new();
    let mut sample_count = 0_usize;
    let started = Instant::now();

    loop {
        let read_count = read_f32_frame(&mut reader, &mut frame)?;
        if read_count == 0 {
            break;
        }
        for (target, sample) in model_input.iter_mut().zip(frame) {
            *target = if sample.is_finite() { sample } else { 0.0 }.clamp(-1.0, 1.0) * PCM_SCALE;
        }

        let frame_started = Instant::now();
        state.process_frame(&mut model_output, &model_input);
        frame_seconds.push(frame_started.elapsed().as_secs_f64());

        for sample in &model_output[..read_count] {
            writer.write_all(&(sample / PCM_SCALE).to_le_bytes())?;
        }
        sample_count += read_count;
    }
    writer.flush()?;

    let elapsed_seconds = started.elapsed().as_secs_f64();
    frame_seconds.sort_by(f64::total_cmp);
    let audio_seconds = sample_count as f64 / SAMPLE_RATE;
    let p95 = percentile(&frame_seconds, 0.95);
    let p99 = percentile(&frame_seconds, 0.99);
    let maximum = frame_seconds.last().copied().unwrap_or(0.0);
    let metadata = format!(
        concat!(
            "{{\"frames\":{},\"samples\":{},\"elapsed_seconds\":{},",
            "\"rtf\":{},\"frame_p95_seconds\":{},",
            "\"frame_p99_seconds\":{},\"frame_max_seconds\":{}}}\n"
        ),
        frame_seconds.len(),
        sample_count,
        elapsed_seconds,
        if audio_seconds > 0.0 {
            elapsed_seconds / audio_seconds
        } else {
            0.0
        },
        p95,
        p99,
        maximum
    );
    std::fs::write(&args[3], metadata)?;
    Ok(())
}
