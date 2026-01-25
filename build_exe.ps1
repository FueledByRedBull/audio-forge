# Final working build script
Write-Host "MicEq Executable Builder" -ForegroundColor Green
Write-Host ""

# Find the Rust extension
$pydPath = Split-Path -Parent (Get-ChildItem -Path ".venv\Lib\site-packages\mic_eq_core" -Filter "*.pyd" | Select-Object -First 1).FullName

Write-Host "Found mic_eq_core at: $pydPath" -ForegroundColor Cyan

# Build arguments list
$pyinstallerArgs = @(
    "--clean"
    "-y"
    "--onedir"
    "--noconsole"
    "--name", "MicEq"
    "--add-data", "models;models"
    "--hidden-import", "PyQt6.QtCore"
    "--hidden-import", "PyQt6.QtGui"
    "--hidden-import", "PyQt6.QtWidgets"
    "--hidden-import", "mic_eq_core"
    "--paths", $pydPath
    "launcher.py"
)

# Add icon if exists
if (Test-Path "mic_eq.ico") {
    $pyinstallerArgs = @("--icon", "mic_eq.ico") + $pyinstallerArgs
    $pyinstallerArgs += "--add-data", "mic_eq.ico;_internal"
    Write-Host "Using icon: mic_eq.ico" -ForegroundColor Green
}

# Build
Write-Host "Building executable..." -ForegroundColor Cyan

pyinstaller @pyinstallerArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "SUCCESS!" -ForegroundColor Green
    Write-Host ""

    # Copy icon to _internal for Qt to use
    if (Test-Path "mic_eq.ico") {
        Copy-Item -Path "mic_eq.ico" -Destination "dist\MicEq\_internal\mic_eq.ico" -Force
        Write-Host "Icon copied to dist\MicEq\_internal\" -ForegroundColor Green
    }

    Write-Host ""
    Write-Host "Executable: dist\MicEq\MicEq.exe"
    Write-Host ""
    Write-Host "The entire dist\MicEq folder is self-contained."
} else {
    Write-Host "Build failed!" -ForegroundColor Red
}
