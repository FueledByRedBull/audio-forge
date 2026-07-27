# AudioForge v1.8.8

## Packaging fix

- Fixed startup failures in the portable Windows executable by bundling the SciPy modules required transitively by `scipy.signal`: `scipy.integrate`, `scipy.interpolate`, and `scipy.stats`.
- Added a packaging regression guard so future builds cannot exclude those runtime dependencies accidentally.

This patch release contains the v1.8.7 feature set with the corrected portable bundle.
