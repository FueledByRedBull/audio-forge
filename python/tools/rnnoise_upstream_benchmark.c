/*
 * Evaluation-only wrapper around the official Xiph RNNoise C API.
 *
 * Input and output are raw mono little-endian float32 samples at 48 kHz.
 * The metadata path receives a compact JSON object with frame timing data.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

#include "rnnoise.h"

#define FRAME_SIZE 480
#define PCM_SCALE 32768.0f

static int compare_double(const void *left, const void *right) {
  const double a = *(const double *)left;
  const double b = *(const double *)right;
  return (a > b) - (a < b);
}

static double percentile(const double *sorted, size_t count, double quantile) {
  if (count == 0) return 0.0;
  const double position = quantile * (double)(count - 1);
  const size_t lower = (size_t)floor(position);
  const size_t upper = (size_t)ceil(position);
  const double fraction = position - (double)lower;
  return sorted[lower] + fraction * (sorted[upper] - sorted[lower]);
}

int main(int argc, char **argv) {
  if (argc != 4) {
    fprintf(stderr, "usage: %s <input-f32> <output-f32> <metadata-json>\n", argv[0]);
    return 2;
  }

  FILE *input = fopen(argv[1], "rb");
  FILE *output = fopen(argv[2], "wb");
  FILE *metadata = fopen(argv[3], "wb");
  if (input == NULL || output == NULL || metadata == NULL) {
    fprintf(stderr, "failed to open benchmark file\n");
    return 3;
  }

  DenoiseState *state = rnnoise_create(NULL);
  if (state == NULL) {
    fprintf(stderr, "failed to create RNNoise state\n");
    return 4;
  }

  LARGE_INTEGER frequency;
  LARGE_INTEGER started;
  LARGE_INTEGER finished;
  QueryPerformanceFrequency(&frequency);
  QueryPerformanceCounter(&started);

  size_t timing_capacity = 4096;
  size_t timing_count = 0;
  double *frame_seconds = malloc(timing_capacity * sizeof(double));
  if (frame_seconds == NULL) return 5;

  float frame[FRAME_SIZE];
  float model_frame[FRAME_SIZE];
  float denoised[FRAME_SIZE];
  size_t sample_count = 0;

  for (;;) {
    const size_t read_count = fread(frame, sizeof(float), FRAME_SIZE, input);
    if (read_count == 0) break;
    for (size_t i = read_count; i < FRAME_SIZE; ++i) frame[i] = 0.0f;
    for (size_t i = 0; i < FRAME_SIZE; ++i) {
      const float value = isfinite(frame[i]) ? frame[i] : 0.0f;
      model_frame[i] = fmaxf(-1.0f, fminf(1.0f, value)) * PCM_SCALE;
    }

    LARGE_INTEGER frame_start;
    LARGE_INTEGER frame_end;
    QueryPerformanceCounter(&frame_start);
    rnnoise_process_frame(state, denoised, model_frame);
    QueryPerformanceCounter(&frame_end);

    if (timing_count == timing_capacity) {
      timing_capacity *= 2;
      double *resized = realloc(frame_seconds, timing_capacity * sizeof(double));
      if (resized == NULL) return 6;
      frame_seconds = resized;
    }
    frame_seconds[timing_count++] =
        (double)(frame_end.QuadPart - frame_start.QuadPart) /
        (double)frequency.QuadPart;

    for (size_t i = 0; i < read_count; ++i) denoised[i] /= PCM_SCALE;
    if (fwrite(denoised, sizeof(float), read_count, output) != read_count) {
      fprintf(stderr, "failed to write benchmark output\n");
      return 7;
    }
    sample_count += read_count;
  }

  QueryPerformanceCounter(&finished);
  qsort(frame_seconds, timing_count, sizeof(double), compare_double);
  const double elapsed_seconds =
      (double)(finished.QuadPart - started.QuadPart) / (double)frequency.QuadPart;
  const double audio_seconds = (double)sample_count / 48000.0;
  const double p95 = percentile(frame_seconds, timing_count, 0.95);
  const double p99 = percentile(frame_seconds, timing_count, 0.99);
  const double maximum = timing_count == 0 ? 0.0 : frame_seconds[timing_count - 1];

  fprintf(metadata,
          "{\"frames\":%zu,\"samples\":%zu,\"elapsed_seconds\":%.12g,"
          "\"rtf\":%.12g,\"frame_p95_seconds\":%.12g,"
          "\"frame_p99_seconds\":%.12g,\"frame_max_seconds\":%.12g}\n",
          timing_count, sample_count, elapsed_seconds,
          audio_seconds > 0.0 ? elapsed_seconds / audio_seconds : 0.0, p95, p99,
          maximum);

  free(frame_seconds);
  rnnoise_destroy(state);
  fclose(input);
  fclose(output);
  fclose(metadata);
  return 0;
}
