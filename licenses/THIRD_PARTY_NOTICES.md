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
the terms of those previously published packages. `GPL-3.0.txt` is also the
GPLv3 text that the LGPLv3 Qt components incorporate by reference, which is why
every bundle carries it. For the current bindings,
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
| NumPy, including its BLAS/native payloads | Its complete retained wheel license collection |
| Portions derived from SciPy | BSD-3-Clause; the notice is reproduced below and in `eq_fit.py` |
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

## Portions derived from SciPy

SciPy is not bundled. The Auto-EQ least-squares solver in
`python/mic_eq/analysis/auto_eq_parts/eq_fit.py` is a NumPy port of SciPy's
trust-region reflective algorithm (`scipy.optimize.least_squares`). SciPy is
distributed under this license:

```text
Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Bench test fixtures

The speech and noise recordings in `python/tests/fixtures/bench/` are test
inputs for `python/tools/bench_report.py`. They are not part of the application
bundle.

| Files | Work | Authors | License |
|---|---|---|---|
| `take-*.wav`, `noise-*.wav` | Valentini-Botinhao, Cassia (2017), "Noisy speech database for training speech enhancement algorithms and TTS models", University of Edinburgh, School of Informatics, Centre for Speech Technology Research (CSTR), https://doi.org/10.7488/ds/2117. Speech is from the CSTR VCTK Corpus (https://doi.org/10.7488/ds/1994); noise is from DEMAND | Cassia Valentini-Botinhao; DEMAND: Joachim Thiemann, Nobutaka Ito and Emmanuel Vincent (https://doi.org/10.5281/zenodo.1227121) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode) |

Changes: the takes are excerpts of clean test-set utterances of speakers p232
and p257, trimmed, faded at their edges and concatenated with silent gaps. Each
`noise-*.wav` is the noisy recording minus the clean recording of one test
utterance, unscaled. `manifest.json` lists the source archive members and the
SHA-256 of every file. The mixes are generated from these files at run time and
are not stored.

The DEMAND record lists CC BY 4.0 under its rights and CC BY-SA 3.0 in its
description. The two `noise-*.wav` files are therefore also offered under
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/legalcode).

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
