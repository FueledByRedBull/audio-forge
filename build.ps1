# MicEq Build Script for Windows
# PowerShell script to build and package the application

param(
    [Parameter(Position=0)]
    [ValidateSet('develop', 'release', 'wheel', 'test', 'clean', 'all')]
    [string]$Command = 'develop'
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot

function Write-Header {
    param([string]$Message)
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host " $Message" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
}

function Test-Command {
    param([string]$Name)
    return [bool](Get-Command -Name $Name -ErrorAction SilentlyContinue)
}

function Assert-Prerequisites {
    Write-Header "Checking Prerequisites"

    $missing = @()

    if (-not (Test-Command "rustc")) {
        $missing += "Rust (rustc) - Install from https://rustup.rs"
    } else {
        $rustVersion = rustc --version
        Write-Host "  [OK] $rustVersion" -ForegroundColor Green
    }

    if (-not (Test-Command "cargo")) {
        $missing += "Cargo - Install from https://rustup.rs"
    } else {
        $cargoVersion = cargo --version
        Write-Host "  [OK] $cargoVersion" -ForegroundColor Green
    }

    if (-not (Test-Command "python")) {
        $missing += "Python - Install from https://python.org"
    } else {
        $pythonVersion = python --version
        Write-Host "  [OK] $pythonVersion" -ForegroundColor Green
    }

    if (-not (Test-Command "maturin")) {
        Write-Host "  [WARN] maturin not found - will attempt to install" -ForegroundColor Yellow
    } else {
        $maturinVersion = maturin --version
        Write-Host "  [OK] $maturinVersion" -ForegroundColor Green
    }

    if ($missing.Count -gt 0) {
        Write-Host ""
        Write-Host "Missing prerequisites:" -ForegroundColor Red
        foreach ($item in $missing) {
            Write-Host "  - $item" -ForegroundColor Red
        }
        exit 1
    }
}

function Install-Maturin {
    if (-not (Test-Command "maturin")) {
        Write-Host "Installing maturin..." -ForegroundColor Yellow
        pip install maturin
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Failed to install maturin" -ForegroundColor Red
            exit 1
        }
    }
}

function Build-Develop {
    Write-Header "Building Development Version"

    Install-Maturin

    Push-Location $ProjectRoot
    try {
        Write-Host "Building Rust core with Python bindings (debug)..." -ForegroundColor Yellow
        maturin develop
        if ($LASTEXITCODE -ne 0) {
            throw "maturin develop failed"
        }
        Write-Host "Development build complete!" -ForegroundColor Green
    }
    finally {
        Pop-Location
    }
}

function Build-Release {
    Write-Header "Building Release Version"

    Install-Maturin

    Push-Location $ProjectRoot
    try {
        Write-Host "Building Rust core with Python bindings (release)..." -ForegroundColor Yellow
        maturin develop --release
        if ($LASTEXITCODE -ne 0) {
            throw "maturin develop --release failed"
        }
        Write-Host "Release build complete!" -ForegroundColor Green
    }
    finally {
        Pop-Location
    }
}

function Build-Wheel {
    Write-Header "Building Release Wheel"

    Install-Maturin

    Push-Location $ProjectRoot
    try {
        Write-Host "Building release wheel..." -ForegroundColor Yellow
        maturin build --release
        if ($LASTEXITCODE -ne 0) {
            throw "maturin build --release failed"
        }

        $wheelDir = Join-Path $ProjectRoot "target\wheels"
        Write-Host ""
        Write-Host "Wheel built successfully!" -ForegroundColor Green
        Write-Host "Location: $wheelDir" -ForegroundColor Cyan

        # List the wheel files
        if (Test-Path $wheelDir) {
            Get-ChildItem $wheelDir -Filter "*.whl" | ForEach-Object {
                Write-Host "  - $($_.Name)" -ForegroundColor White
            }
        }
    }
    finally {
        Pop-Location
    }
}

function Run-Tests {
    Write-Header "Running Tests"

    Push-Location $ProjectRoot
    try {
        # Run Rust tests
        Write-Host "Running Rust tests..." -ForegroundColor Yellow
        Push-Location (Join-Path $ProjectRoot "rust-core")
        cargo test
        if ($LASTEXITCODE -ne 0) {
            throw "Rust tests failed"
        }
        Pop-Location
        Write-Host "Rust tests passed!" -ForegroundColor Green

        # Run Python tests if they exist
        $pythonTests = Join-Path $ProjectRoot "python\tests"
        if (Test-Path $pythonTests) {
            Write-Host ""
            Write-Host "Running Python tests..." -ForegroundColor Yellow
            python -m pytest $pythonTests -v
            if ($LASTEXITCODE -ne 0) {
                throw "Python tests failed"
            }
            Write-Host "Python tests passed!" -ForegroundColor Green
        }
    }
    finally {
        Pop-Location
    }
}

function Clean-Build {
    Write-Header "Cleaning Build Artifacts"

    Push-Location $ProjectRoot
    try {
        # Clean Rust build
        Write-Host "Cleaning Rust build artifacts..." -ForegroundColor Yellow
        cargo clean

        # Clean Python build artifacts
        $pycacheDirs = Get-ChildItem -Path $ProjectRoot -Directory -Recurse -Filter "__pycache__" -ErrorAction SilentlyContinue
        foreach ($dir in $pycacheDirs) {
            Remove-Item -Path $dir.FullName -Recurse -Force
            Write-Host "  Removed: $($dir.FullName)" -ForegroundColor Gray
        }

        # Clean .egg-info
        $eggInfoDirs = Get-ChildItem -Path $ProjectRoot -Directory -Filter "*.egg-info" -ErrorAction SilentlyContinue
        foreach ($dir in $eggInfoDirs) {
            Remove-Item -Path $dir.FullName -Recurse -Force
            Write-Host "  Removed: $($dir.FullName)" -ForegroundColor Gray
        }

        # Clean .pyd files (compiled Python modules)
        $pydFiles = Get-ChildItem -Path (Join-Path $ProjectRoot "python") -Recurse -Filter "*.pyd" -ErrorAction SilentlyContinue
        foreach ($file in $pydFiles) {
            Remove-Item -Path $file.FullName -Force
            Write-Host "  Removed: $($file.FullName)" -ForegroundColor Gray
        }

        Write-Host "Clean complete!" -ForegroundColor Green
    }
    finally {
        Pop-Location
    }
}

function Build-All {
    Write-Header "Full Build Pipeline"

    Assert-Prerequisites
    Clean-Build
    Run-Tests
    Build-Release
    Build-Wheel

    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host " Build Pipeline Complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "To run the application:" -ForegroundColor Cyan
    Write-Host "  python -m mic_eq" -ForegroundColor White
    Write-Host ""
    Write-Host "To install the wheel:" -ForegroundColor Cyan
    Write-Host "  pip install target\wheels\mic_eq-*.whl" -ForegroundColor White
    Write-Host ""
}

# Main execution
switch ($Command) {
    'develop' {
        Assert-Prerequisites
        Build-Develop
    }
    'release' {
        Assert-Prerequisites
        Build-Release
    }
    'wheel' {
        Assert-Prerequisites
        Build-Wheel
    }
    'test' {
        Assert-Prerequisites
        Run-Tests
    }
    'clean' {
        Clean-Build
    }
    'all' {
        Build-All
    }
}
