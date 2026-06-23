## What's Changed
- Fixed suppressor non-finite output poisoning the DSP/output pipeline after extended silence.
- Added suppressor output sanitization and automatic suppressor reinitialization when non-finite samples are detected.
- Reinitialized suppressor state on stop/start so in-app restart recovers from poisoned suppressor state.

## Build
- Built with features: `pyo3/extension-module`, `vad`, `deepfilter`.

## Asset
- `AudioForge-v1.7.3-win64-ultra.7z`
- SHA256: `6BAF47A969675A1AC27679CB6990A8CC72C87E8C9B88FB8B2079160EE38C9861`