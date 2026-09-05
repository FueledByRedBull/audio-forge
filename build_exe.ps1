param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
# Final working build script
$ProjectRoot = $PSScriptRoot
Push-Location $ProjectRoot

try {
Write-Host "AudioForge Executable Builder" -ForegroundColor Green
Write-Host ""

# Rebuild the Rust extension so the bundle cannot use a semantically stale
# source-tree extension whose modification time happens to be newer.
$venvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "ERROR: venv Python not found: $venvPython" -ForegroundColor Red
    exit 1
}

$null = & $venvPython -m maturin develop --release --locked
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Rust extension rebuild failed." -ForegroundColor Red
    exit $LASTEXITCODE
}

$expectedSuffix = & $venvPython -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.pyd')"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$localPyd = Get-ChildItem -Path (Join-Path $ProjectRoot "python\mic_eq") -Filter "mic_eq_core*$expectedSuffix" | Select-Object -First 1
if (-not $localPyd) {
    Write-Host "ERROR: python\\mic_eq\\mic_eq_core*$expectedSuffix not found." -ForegroundColor Red
    Write-Host "Run: .\\.venv\\Scripts\\python.exe -m maturin develop --release" -ForegroundColor Yellow
    exit 1
}
Write-Host "Using local mic_eq_core: $($localPyd.FullName)" -ForegroundColor Cyan

& $venvPython (Join-Path $ProjectRoot "python\tools\verify_release_assets.py")
if ($LASTEXITCODE -ne 0) {
    Write-Host "Release asset verification failed. Packaging will not use stale dist/ contents as source assets." -ForegroundColor Red
    exit $LASTEXITCODE
}

if (Test-Path "df.dll") {
    Write-Host "DeepFilterNet support: df.dll will be bundled via AudioForge.spec" -ForegroundColor Green
} else {
    Write-Host "ERROR: df.dll not found. Release bundles require DeepFilterNet support." -ForegroundColor Red
    exit 1
}

$requiredModels = @(
    "models\DeepFilterNet3_ll_onnx.tar.gz",
    "models\DeepFilterNet3_onnx.tar.gz",
    "models\silero_vad.onnx"
)
$missingModels = @($requiredModels | Where-Object { -not (Test-Path $_) })
if ($missingModels.Count -gt 0) {
    Write-Host "Missing model assets:" -ForegroundColor Red
    $missingModels | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    exit 1
}

if (Test-Path "target\release\DirectML.dll") {
    Write-Host "DirectML runtime will be bundled via AudioForge.spec" -ForegroundColor Green
} else {
    Write-Host "ERROR: target\\release\\DirectML.dll not found. Build the Rust extension in release mode first." -ForegroundColor Red
    exit 1
}

if (Test-Path "AudioForge.ico") {
    Write-Host "Using icon: AudioForge.ico" -ForegroundColor Green
}

Write-Host "Building executable from AudioForge.spec..." -ForegroundColor Cyan

& $venvPython (Join-Path $ProjectRoot "python\tools\license_inventory.py")
if ($LASTEXITCODE -ne 0) {
    Write-Host "Dependency license collection failed." -ForegroundColor Red
    exit $LASTEXITCODE
}

$pyinstallerArgs = @("-y")
if ($Clean) {
    $pyinstallerArgs += "--clean"
}
$pyinstallerArgs += (Join-Path $ProjectRoot "AudioForge.spec")
# Do not collect unrelated native libraries from developer tools on PATH.
# In particular, an external ICU can shadow Windows' ICU and prevent Qt loading.
$basePythonDir = & $venvPython -c "import sys; from pathlib import Path; print(Path(sys._base_executable).parent)"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$buildSearchPath = $env:PATH
try {
    $env:PATH = @(
        (Split-Path -Parent $venvPython),
        $basePythonDir.Trim(),
        [Environment]::SystemDirectory,
        [Environment]::GetFolderPath('Windows')
    ) -join [IO.Path]::PathSeparator
    & $venvPython -m PyInstaller @pyinstallerArgs
    $pyinstallerExitCode = $LASTEXITCODE
} finally {
    $env:PATH = $buildSearchPath
}

if ($pyinstallerExitCode -eq 0) {
    & $venvPython (Join-Path $ProjectRoot "python\tools\prune_bundle.py") (Join-Path $ProjectRoot "dist\AudioForge")
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Bundle pruning failed!" -ForegroundColor Red
        exit $LASTEXITCODE
    }

    $version = & $venvPython -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])"
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($version)) {
        Write-Host "Failed to resolve the bundle version from pyproject.toml." -ForegroundColor Red
        exit 1
    }
    $buildInfoPath = Join-Path $ProjectRoot "dist\AudioForge\_internal\audioforge-build.json"
    @{
        schema_version = 1
        version = $version.Trim()
    } | ConvertTo-Json | Set-Content -LiteralPath $buildInfoPath -Encoding utf8

    Write-Host ""
    Write-Host "SUCCESS!" -ForegroundColor Green
    Write-Host ""

    Write-Host ""
    Write-Host "Executable: dist\AudioForge\AudioForge.exe"
    Write-Host ""
    Write-Host "The entire dist\AudioForge folder is self-contained."
    Write-Host "NOTE: DeepFilterNet assets stay bundled under the PyInstaller runtime directory."
} else {
    Write-Host "Build failed!" -ForegroundColor Red
    exit $pyinstallerExitCode
}
} finally {
    Pop-Location
}
