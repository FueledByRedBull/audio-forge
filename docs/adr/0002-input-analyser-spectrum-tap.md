# The input analyser reads a spectrum from a worker fed by the DSP loop

Date: 2026-11-20

**Decision:** the Equalizer graph's input analyser is computed in the engine.
The DSP loop copies each input block, before any processing, into a 16384-sample
lock-free ring and drops the whole block when the ring is full, counting it in
`input_spectrum_dropped_blocks`. A worker thread runs a 2048-point FFT (Hann,
hop 1024) into 96 log bands from 20.7 Hz to 19.3 kHz, applies instant attack and
a 20 dB per 300 ms release, and publishes the bands as atomics with a sequence
number. The UI reads them about 30 times a second, only while the stream runs
and the Equalizer page is on screen, into one preallocated array. The graph draws
them on a fixed scale: -90 dBFS at the plot floor and -18 dBFS at the 0 dB line.

**Why:** the DSP loop is a strict realtime region, so the tap only copies samples;
the FFT, the banding and the ballistics run on the worker. Reading the processor's
own input keeps the picture exact and needs no second capture of the microphone.
A fixed scale shows real level differences between takes; auto-gain would hide
them. A worker that panics or cannot start never stops the audio: it sets
`input_spectrum_worker_failed` in the diagnostics instead. The tap never changes
the output, and a test checks that bit for bit in every path.

**Alternatives:** compute the spectrum in Python from the level meters (no
spectral data there); a second capture stream for the display (another device
handle and a different signal from the one being processed); FFT on the DSP
thread (realtime cost on every block); auto-gain on the display (hides level).
No new dependency: `realfft` was already in the engine.
