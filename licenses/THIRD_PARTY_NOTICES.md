# AudioForge third-party notices

AudioForge's original source remains under the MIT license in `LICENSE`.
This source checkout uses PySide6 and the Qt modules permitted under LGPLv3,
with complete component terms retained in `QtForPython-6.11.1-NOTICES.txt`.
Dependency copyright, attribution, and applicable license terms remain in force.
The notice collection includes alternative upstream licenses and source
components that may not be present in a particular bundle; the generated
inventory identifies the bundled components.

Earlier releases containing PyQt6 remain distributed under GNU GPL version 3;
see their original notices and `GPL-3.0.txt`. This migration does not change
the terms of those previously published packages. For the current bindings,
see [Qt for Python licensing](https://doc.qt.io/qtforpython-6/licenses.html).

Each build collects versioned component identities and license texts under
`licenses/dependencies/` in the bundle. Its `inventory.json` distinguishes the
locked Python runtime and Windows Rust build graph; the latter can include
build-only dependencies. It records native/model provenance separately. Retain
the wheel `.dist-info` notices and this inventory when redistributing.

| Component | Applicable notices |
|---|---|
| CPython runtime and standard library | PSF license and bundled third-party notices |
| PySide6-Essentials, PySide6-Addons and shiboken6 | LGPLv3 and included third-party notices in `QtForPython-6.11.1-NOTICES.txt` |
| Bundled Qt6 modules and plugins | LGPLv3 and included third-party notices in the same collection; no GPL-only module is selected |
| NumPy and SciPy, including their BLAS/native payloads | Their complete retained wheel license collections |
| pywin32 | PSF license |
| PyInstaller bootloader | GPLv2 with the upstream bootloader exception; retained upstream license |
| Rust dependencies | Per-component Cargo license declarations and retained notice files |

The following separately licensed runtime components and model assets are also
included:

| Bundled files | Project | License notice | Source |
|---|---|---|---|
| `df.dll`, `DeepFilterNet3_ll_onnx.tar.gz`, `DeepFilterNet3_onnx.tar.gz` | DeepFilterNet | `DeepFilterNet-LICENSE.txt` (MIT option) | https://github.com/Rikorose/DeepFilterNet |
| `silero_vad.onnx` | Silero VAD | `Silero-VAD-LICENSE.txt` | https://github.com/snakers4/silero-vad |
| `onnxruntime.dll`, `onnxruntime_providers_shared.dll` | ONNX Runtime 1.23.2 CPU package | `ONNXRuntime-LICENSE.txt` and complete `ONNXRuntime-ThirdPartyNotices.txt` | https://github.com/microsoft/onnxruntime/commit/a83fc4d58cb48eb68890dd689f94f28288cf2278 |

AudioForge's own license is included as `LICENSE`.

The release uses the official CPU-only ONNX Runtime package and does not ship
DirectML. Its complete upstream notice file is retained beside the MIT license,
and the corresponding-source manifest records the exact ORT source commit,
submodules, and CPU FetchContent inputs used to rebuild the package.
The `.lib` import library is a build input retained in the corresponding-source
runtime archive for rebuilding; it is not a portable payload file.

The packaged Python and Qt distributions may carry Microsoft Visual C++ runtime
DLLs. When those files are present in the bundle, retain the Microsoft Visual C++
redistribution terms supplied by the originating runtime; they are not covered
by AudioForge's MIT license. Windows system UCRT files are treated as operating
system components and are not claimed as project source.

## Corresponding source and release approval

Release preparation must make the corresponding source and build instructions
for the distributed application and copyleft components available alongside the
binaries, including the exact AudioForge revision, Python/Qt/PySide source, and
required dependency modifications. An inventory or an unversioned upstream link
alone is not fulfillment of that obligation. Preserve upstream source archives
and notices for the exact component versions before publishing a final release.

`release-assets.json` records the verified upstream model blobs and exact CPU
ONNX Runtime archive. The `df.dll` source, patch, lockfile, and build recipe are
recorded separately in the corresponding-source manifest and its build
attestation; the candidate-specific DLL itself is not a source archive.
