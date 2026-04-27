param(
    [switch]$AllowMissingModels
)

# Final working build script
$ProjectRoot = $PSScriptRoot
Push-Location $ProjectRoot

try {
Write-Host "MicEq Executable Builder" -ForegroundColor Green
Write-Host ""

# Use the locally built Rust extension from source tree.
# This avoids bundling stale site-packages binaries.
$venvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "ERROR: venv Python not found: $venvPython" -ForegroundColor Red
    exit 1
}

$expectedSuffix = & $venvPython -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.pyd')"
$localPyd = Get-ChildItem -Path (Join-Path $ProjectRoot "python\mic_eq") -Filter "mic_eq_core*$expectedSuffix" | Select-Object -First 1
if (-not $localPyd) {
    Write-Host "ERROR: python\\mic_eq\\mic_eq_core*$expectedSuffix not found." -ForegroundColor Red
    Write-Host "Run: .\\.venv\\Scripts\\python.exe -m maturin develop --release" -ForegroundColor Yellow
    exit 1
}
Write-Host "Using local mic_eq_core: $($localPyd.FullName)" -ForegroundColor Cyan

if (Test-Path "df.dll") {
    Write-Host "DeepFilterNet support: df.dll will be bundled via AudioForge.spec" -ForegroundColor Green
} else {
    Write-Host "ERROR: df.dll not found. Use -AllowMissingModels only for intentional reduced builds." -ForegroundColor Red
    if (-not $AllowMissingModels) {
        exit 1
    }
    Write-Host "DeepFilterNet support: df.dll NOT found - RNNoise only" -ForegroundColor Yellow
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
    if (-not $AllowMissingModels) {
        Write-Host "Use -AllowMissingModels only for intentional reduced builds." -ForegroundColor Yellow
        exit 1
    }
    Write-Host "Continuing with missing models because -AllowMissingModels was set." -ForegroundColor Yellow
}

if (Test-Path "target\release\DirectML.dll") {
    Write-Host "DirectML runtime will be bundled via AudioForge.spec" -ForegroundColor Green
} else {
    Write-Host "ERROR: target\\release\\DirectML.dll not found. Build the Rust extension in release mode first." -ForegroundColor Red
    exit 1
}

if (Test-Path "mic_eq.ico") {
    Write-Host "Using icon: mic_eq.ico" -ForegroundColor Green
}

Write-Host "Building executable from AudioForge.spec..." -ForegroundColor Cyan

& $venvPython -m PyInstaller --clean -y (Join-Path $ProjectRoot "AudioForge.spec")

if ($LASTEXITCODE -eq 0) {
    & $venvPython (Join-Path $ProjectRoot "python\tools\prune_bundle.py") (Join-Path $ProjectRoot "dist\AudioForge")
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Bundle pruning failed!" -ForegroundColor Red
        exit $LASTEXITCODE
    }

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
}
} finally {
    Pop-Location
}
