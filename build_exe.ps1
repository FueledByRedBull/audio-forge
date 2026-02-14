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

# Get Python standard library path
$pythonStdlib = python -c "import sys; import os; print(os.path.join(sys.prefix, 'Lib'))"

# Build arguments list
$pyinstallerArgs = @(
    "--clean"
    "-y"
    "--onedir"
    "--noconsole"
    "--name", "AudioForge"
    "--add-data", "models;models"
    "--add-data", "python/mic_eq;mic_eq"
    "--hidden-import", "json"
    "--hidden-import", "PyQt6.QtCore"
    "--hidden-import", "PyQt6.QtGui"
    "--hidden-import", "PyQt6.QtWidgets"
    "--hidden-import", "mic_eq.mic_eq_core"
    "--hidden-import", "mic_eq"
    "--hidden-import", "mic_eq.ui"
    "--paths", $pythonStdlib
    "launcher.py"
)

# Add df.dll if exists (optional - for DeepFilterNet support)
if (Test-Path "df.dll") {
    $pyinstallerArgs += "--add-binary", "df.dll;."
    Write-Host "DeepFilterNet support: df.dll will be bundled" -ForegroundColor Green
} else {
    Write-Host "DeepFilterNet support: df.dll NOT found - RNNoise only" -ForegroundColor Yellow
}

# Add icon if exists (optional - skips if not found)
if (Test-Path "mic_eq.ico") {
    $pyinstallerArgs = @("--icon", "mic_eq.ico") + $pyinstallerArgs
    $pyinstallerArgs += "--add-data", "mic_eq.ico;."
    Write-Host "Using icon: mic_eq.ico" -ForegroundColor Green
}

# Build
Write-Host "Building executable..." -ForegroundColor Cyan

# Use venv Python to ensure correct environment
$venvPython = ".\.venv\Scripts\python.exe"
& $venvPython -m PyInstaller @pyinstallerArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "SUCCESS!" -ForegroundColor Green
    Write-Host ""

    # Copy df.dll to exe directory (not _internal) for libloading to find it
    if (Test-Path "df.dll") {
        Copy-Item -Path "df.dll" -Destination "dist\AudioForge\df.dll" -Force
        Write-Host "Copied df.dll to exe directory for DeepFilterNet" -ForegroundColor Green
    }

    # Copy models directory to exe directory (code looks in ./models/)
    if (Test-Path "dist\AudioForge\_internal\models") {
        Copy-Item -Path "dist\AudioForge\_internal\models" -Destination "dist\AudioForge\models" -Recurse -Force
        Write-Host "Copied models to exe directory for DeepFilterNet" -ForegroundColor Green
    }

    Write-Host ""
    Write-Host "Executable: dist\AudioForge\AudioForge.exe"
    Write-Host ""
    Write-Host "The entire dist\AudioForge folder is self-contained."
    Write-Host "NOTE: DeepFilterNet requires df.dll - should now be bundled in the folder."
} else {
    Write-Host "Build failed!" -ForegroundColor Red
}
