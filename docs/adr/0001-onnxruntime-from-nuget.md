# ONNX Runtime CPU binaries come from the NuGet package

Date: 2026-11-18

**Decision:** `fetch_release_assets.py` downloads the pinned ONNX Runtime 1.23.2
CPU files from the Microsoft-signed `Microsoft.ML.OnnxRuntime` 1.23.2 package on
`api.nuget.org`, and extracts each file from its `runtimes/<rid>/native/` member.
`api.nuget.org` is the only host allowed for pinned runtime archives.

**Why:** `github.com/microsoft/onnxruntime` has returned 404 since 2026-11-18,
release downloads included, so CI could not hydrate the runtime. The package is
Microsoft's official channel, its SHA-512 matches the nuget.org catalog, and the
four extracted files match the existing `release-assets.json` pins byte for byte
(as do its `LICENSE` and `ThirdPartyNotices.txt` with the bundled notices). The
shipped bytes do not change.

**Alternatives:** wait for the GitHub repository to come back (blocks every PR
until then); mirror the old archives in an AudioForge release (an extra copy to
keep and a weaker origin than Microsoft's signed package).

**Source archive (2026-11-19):** the corresponding-source bundle needs the
ONNX Runtime source tree at commit `a83fc4d58cb4`, which the NuGet package does
not carry and `codeload.github.com` no longer serves. It now downloads from the
`third-party-sources` release of this repository. That file was taken from the
published v1.23.0 source bundle and matches the existing SHA-256 pin, so the
source of truth stays the upstream commit. Rejected: Software Heritage or a repack
of another mirror (a repack is not byte-identical to the pin).
