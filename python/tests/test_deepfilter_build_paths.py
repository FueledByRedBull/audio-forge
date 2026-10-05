"""Exercise DeepFilter output ownership without fetching or building its DLL."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows DeepFilter builder")
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def receipt_writer(tmp_path):
    shell = shutil.which("pwsh")
    if shell is None:
        pytest.skip("PowerShell is required")
    project = tmp_path / "project with spaces"
    project.mkdir()
    source = (REPO / "build_deepfilter.ps1").read_text(encoding="utf-8")
    # Execute the builder's parameter defaults and path resolution, with only a
    # receipt write as the sink. The source/build/toolchain operations stay out.
    header = source[:source.index("Set-StrictMode")]
    paths = source[source.index("$outputFullPath ="):source.index("$cargoTarget =")]
    script = project / "resolve_receipt.ps1"
    script.write_text(header + '''
$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
''' + paths + '''
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $attestationFullPath) | Out-Null
Set-Content -LiteralPath $attestationFullPath -Value $outputFullPath -NoNewline
@{ output=$outputFullPath; attestation=$attestationFullPath } | ConvertTo-Json -Compress
''', encoding="utf-8")

    def write(*args: str) -> dict[str, Path]:
        result = subprocess.run(
            [shell, "-NoProfile", "-NonInteractive", "-File", str(script), *args],
            cwd=project, capture_output=True, text=True, check=False, timeout=15,
        )
        assert result.returncode == 0, result.stderr
        return {name: Path(value) for name, value in json.loads(result.stdout).items()}

    return project, write


def test_default_output_retains_canonical_receipt(receipt_writer):
    project, write = receipt_writer
    result = write()
    assert result["output"] == project / "target/deepfilter/df.dll"
    assert result["attestation"] == project / "target/deepfilter/df.dll.provenance.json"


def test_custom_outputs_do_not_overwrite_canonical_receipt(receipt_writer):
    project, write = receipt_writer
    canonical = project / "target/deepfilter/df.dll.provenance.json"
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"original canonical receipt")
    receipts = []
    for name in ("first build.dll", "second build.dll"):
        output = project / "custom output" / name
        result = write("-OutputPath", str(output))
        receipts.append(result["attestation"])
        assert result["attestation"] == Path(f"{output}.provenance.json")
        assert result["attestation"].read_text() == str(output)
    assert receipts[0] != receipts[1]
    assert canonical.read_bytes() == b"original canonical receipt"


@pytest.mark.parametrize("attestation", ["", "   "])
def test_relative_output_resolves_before_default_receipt(receipt_writer, attestation):
    project, write = receipt_writer
    result = write("-OutputPath", "relative output/df.dll", "-AttestationPath", attestation)
    assert result["output"] == project / "relative output/df.dll"
    assert result["attestation"] == project / "relative output/df.dll.provenance.json"


def test_explicit_receipt_path_remains_authoritative(receipt_writer):
    project, write = receipt_writer
    result = write(
        "-OutputPath", "custom output/df.dll", "-AttestationPath", "receipts/custom.json",
    )
    assert result["output"] == project / "custom output/df.dll"
    assert result["attestation"] == project / "receipts/custom.json"
