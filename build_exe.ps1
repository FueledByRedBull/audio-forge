# Final working build script
Write-Host "MicEq Executable Builder" -ForegroundColor Green
Write-Host ""

# Use the locally built Rust extension from source tree.
# This avoids bundling stale site-packages binaries.
$localPyd = Get-ChildItem -Path "python\mic_eq" -Filter "mic_eq_core*.pyd" | Select-Object -First 1
if (-not $localPyd) {
    Write-Host "ERROR: python\\mic_eq\\mic_eq_core*.pyd not found." -ForegroundColor Red
    Write-Host "Run: .\\.venv\\Scripts\\python.exe -m maturin develop --release" -ForegroundColor Yellow
    exit 1
}
Write-Host "Using local mic_eq_core: $($localPyd.FullName)" -ForegroundColor Cyan

# Use venv Python to ensure correct environment
$venvPython = ".\.venv\Scripts\python.exe"

if (Test-Path "df.dll") {
    Write-Host "DeepFilterNet support: df.dll will be bundled via AudioForge.spec" -ForegroundColor Green
} else {
    Write-Host "DeepFilterNet support: df.dll NOT found - RNNoise only" -ForegroundColor Yellow
}

if (Test-Path "mic_eq.ico") {
    Write-Host "Using icon: mic_eq.ico" -ForegroundColor Green
}

Write-Host "Building executable from AudioForge.spec..." -ForegroundColor Cyan

& $venvPython -m PyInstaller --clean -y .\AudioForge.spec

if ($LASTEXITCODE -eq 0) {
    & $venvPython .\python\tools\prune_bundle.py .\dist\AudioForge
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
