# AudioForge third-party notices

AudioForge's original source remains under the MIT license in `LICENSE`.
The combined PyQt6 application is distributed under GNU GPL version 3; see
`GPL-3.0.txt`. This applies to both the portable application and MSI distribution.
Dependency copyright, attribution, and applicable license terms remain in force.

PyQt6 is GPLv3 or commercially licensed, not LGPL. These releases use the GPL
route, as documented by [Riverbank](https://www.riverbankcomputing.com/software/pyqt/intro).
The Qt libraries in the PyQt wheels have their own LGPLv3/GPLv3 terms and
third-party notices; see [Qt licensing](https://doc.qt.io/qt-6/licensing.html).

Each build collects versioned component identities and license texts under
`licenses/dependencies/` in the bundle. Its `inventory.json` distinguishes the
locked Python runtime and Windows Rust build graph; the latter can include
build-only dependencies. It records native/model provenance separately. Retain
the wheel `.dist-info` notices and this inventory when redistributing.

| Component | Applicable notices |
|---|---|
| CPython runtime and standard library | PSF license and bundled third-party notices |
| PyQt6 | GPL-3.0-only |
| Qt6 | Qt's LGPLv3/GPLv3 and included third-party notices |
| PyQt6-sip | BSD-2-Clause |
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
| `DirectML.dll` | Microsoft DirectML redistributable | Microsoft Software License Terms in `DirectML-LICENSE.txt`; the repository's MIT code license does not cover this DLL | https://www.nuget.org/packages/Microsoft.AI.DirectML/1.15.4 |

AudioForge's own license is included as `LICENSE`.

The inherited DirectML-linked runtime is not cleared for the selected GPLv3
distribution. Its actual redistributable terms restrict platform use and reverse
engineering. Release preparation is evaluating a CPU-only ONNX Runtime package
to remove this unused DirectML dependency; source-manifest completion must remain
blocked until the runtime and its notices are corrected.

## Corresponding source and release approval

Release preparation must make the corresponding source and build instructions
for the distributed application and copyleft components available alongside the
binaries, including the exact AudioForge revision, Python/Qt/PyQt source, and
required dependency modifications. An inventory or an unversioned upstream link
alone is not fulfillment of that obligation. Preserve upstream source archives
and notices for the exact component versions before publishing a final release.

`release-assets.json` records verified upstream model blobs and the exact
Microsoft DirectML redistributable package. The inherited `df.dll` is verified
against the existing released bytes, but its original upstream commit, compiler,
and build recipe remain unresolved. Do not describe that DLL as reproducibly
built or the source-distribution review as complete until those inputs have
been recovered or it has been replaced by a qualified, documented build.
