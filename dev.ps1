#Requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter(Position = 0, Mandatory = $true)]
    [ValidateSet("bootstrap", "run", "test", "doctor")]
    [string]$Command,
    [string]$VenvPath = ".venv",
    [string]$PythonPath = "",
    [string]$TestPath = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
$projectRoot = $PSScriptRoot
$script:commandExitCode = 0
$environmentNames = @(
    "PATH", "VIRTUAL_ENV", "PYO3_PYTHON", "ORT_LIB_LOCATION", "ORT_PREFER_DYNAMIC_LINK",
    "RUSTUP_TOOLCHAIN", "PYTHONDONTWRITEBYTECODE", "AUDIOFORGE_ENABLE_DEEPFILTER"
)
$previousEnvironment = @{}
foreach ($name in $environmentNames) {
    $previousEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}

function Invoke-Tool {
    param([string]$Program, [string[]]$ToolArguments, [switch]$Capture)
    if ($DryRun) {
        Write-Host ("DRY RUN " + (ConvertTo-Json -InputObject (@($Program) + $ToolArguments) -Compress))
        return
    }
    $global:LASTEXITCODE = 0
    if ($Capture) { $output = & $Program @ToolArguments }
    else { & $Program @ToolArguments }
    if ($LASTEXITCODE -ne 0) {
        $script:commandExitCode = $LASTEXITCODE
        if ($Capture) { $output | Write-Output }
        throw "$Program failed with exit code $LASTEXITCODE."
    }
    if ($Capture) { return $output }
}

function Read-PythonInfo {
    param([string]$Interpreter, [switch]$RequireVenv)
    $probe = "import json, platform, struct, sys, sysconfig; print(json.dumps(dict(version=platform.python_version(), implementation=platform.python_implementation(), bits=struct.calcsize('P')*8, target=sysconfig.get_platform(), prefix=sys.prefix, base_prefix=sys.base_prefix)))"
    $info = Invoke-Tool $Interpreter @("-B", "-c", $probe) -Capture | ConvertFrom-Json
    if ($info.version -ne $pythonVersion -or $info.implementation -ne "CPython" -or $info.bits -ne 64 -or $info.target -ne "win-amd64") {
        throw "Use existing CPython $pythonVersion x64; selected interpreter is $($info.implementation) $($info.version) $($info.bits)-bit. No interpreter will be installed."
    }
    if ($RequireVenv -and (
        [IO.Path]::GetFullPath($info.prefix) -ne $venvRoot -or $info.prefix -eq $info.base_prefix
    )) {
        throw "The selected Python is not the virtual environment at $venvRoot."
    }
    return $info
}

Push-Location $projectRoot
try {
    $manifest = Get-Content -LiteralPath "licenses/source-manifest.json" -Raw | ConvertFrom-Json
    $pythonPins = @($manifest.entries | Where-Object { $_.kind -eq "cpython-source" })
    if ($pythonPins.Count -ne 1 -or $pythonPins[0].version -notmatch '^\d+\.\d+\.\d+$') {
        throw "The source manifest must contain one exact CPython version."
    }
    $pythonVersion = $pythonPins[0].version
    $toolchain = Get-Content -LiteralPath "rust-toolchain.toml" -Raw
    if ($toolchain -notmatch '(?m)^channel\s*=\s*"(\d+\.\d+\.\d+)"\s*$') {
        throw "rust-toolchain.toml must select an exact Rust version."
    }
    $rustVersion = $Matches[1]
    $venvRoot = [IO.Path]::TrimEndingDirectorySeparator([IO.Path]::GetFullPath($VenvPath, $projectRoot))
    $venvPython = Join-Path $venvRoot "Scripts/python.exe"
    Write-Host "CPython $pythonVersion x64; Rust $rustVersion; environment $venvRoot"

    if (-not $DryRun -and $Command -ne "run") {
        $required = if ($Command -eq "test") { @("rustup") } else { @("uv", "rustup", "gh", "7z") }
        $missing = @()
        foreach ($tool in $required) {
            $found = [bool](Get-Command $tool -ErrorAction SilentlyContinue)
            # The asset hydrator also supports 7-Zip's standard installed path.
            if ($tool -eq "7z" -and (Test-Path -LiteralPath "C:/Program Files/7-Zip/7z.exe" -PathType Leaf)) {
                $found = $true
            }
            if ($found) { Write-Host "Found prerequisite: $tool" }
            else { $missing += $tool; Write-Host "Missing prerequisite: $tool" }
        }
        $hasMsvc = [bool](Get-Command cl -ErrorAction SilentlyContinue)
        $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio/Installer/vswhere.exe"
        if (-not $hasMsvc -and (Test-Path -LiteralPath $vswhere -PathType Leaf)) {
            $installation = Invoke-Tool $vswhere @("-latest", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath") -Capture
            $hasMsvc = -not [string]::IsNullOrWhiteSpace(($installation -join ""))
        }
        if ($hasMsvc) { Write-Host "Found prerequisite: MSVC C++ build tools" }
        else { $missing += "MSVC C++ build tools"; Write-Host "Missing prerequisite: MSVC C++ build tools" }
        if ($missing.Count) {
            throw "Install the missing prerequisites yourself: $($missing -join ', '). This command does not install system tools."
        }
        $rustInfo = Invoke-Tool rustup @("run", $rustVersion, "rustc", "--version") -Capture
        if (($rustInfo -join " ") -notmatch ("^rustc " + [regex]::Escape($rustVersion) + "(?: |$)")) {
            throw "The installed Rust compiler does not match rust-toolchain.toml."
        }
    }

    if ($Command -eq "bootstrap" -and -not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
        if ((Test-Path -LiteralPath $venvRoot) -and @(
            Get-ChildItem -LiteralPath $venvRoot -Force
        ).Count) {
            throw "Incomplete environment at $venvRoot; it was preserved. Choose a new -VenvPath or repair it explicitly."
        }
        if ([string]::IsNullOrWhiteSpace($PythonPath)) {
            if (-not $DryRun) { throw "Creating an environment requires -PythonPath to existing CPython $pythonVersion x64." }
            $PythonPath = "<existing CPython $pythonVersion x64 python.exe>"
        } elseif (-not $DryRun) {
            $PythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
            $null = Read-PythonInfo $PythonPath
        }
        Invoke-Tool uv @("venv", "--python", $PythonPath, "--no-python-downloads", "--no-managed-python", $venvRoot)
    }
    if (-not $DryRun) {
        if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
            throw "Environment Python is missing: $venvPython. Run bootstrap with an existing pinned -PythonPath."
        }
        $pythonInfo = Read-PythonInfo $venvPython -RequireVenv
        $env:PATH = "$(Join-Path $venvRoot 'Scripts');$($pythonInfo.base_prefix);$env:PATH"
    }
    $env:VIRTUAL_ENV = $venvRoot
    $env:PYO3_PYTHON = $venvPython
    $env:PYTHONDONTWRITEBYTECODE = "1"
    $env:RUSTUP_TOOLCHAIN = $rustVersion
    $env:ORT_LIB_LOCATION = Join-Path $projectRoot "target/onnxruntime-cpu/lib"
    $env:ORT_PREFER_DYNAMIC_LINK = "1"

    if ($Command -eq "bootstrap") {
        Invoke-Tool uv @("pip", "install", "--python", $venvPython, "--no-python-downloads", "--require-hashes", "-r", "requirements/dev.txt")
        Invoke-Tool $venvPython @("-B", "python/tools/fetch_release_assets.py")
    }
    Invoke-Tool $venvPython @("-B", "python/tools/verify_release_assets.py")
    if ($Command -in @("bootstrap", "test")) {
        Invoke-Tool $venvPython @("-m", "maturin", "develop", "--release", "--locked")
    }
    switch ($Command) {
        "run" {
            $env:AUDIOFORGE_ENABLE_DEEPFILTER = "1"
            Invoke-Tool $venvPython @("-m", "mic_eq")
        }
        "test" {
            if ([string]::IsNullOrWhiteSpace($TestPath)) {
                Invoke-Tool rustup @("run", $rustVersion, "cargo", "test", "--locked", "-p", "mic_eq_core")
                $TestPath = "python/tests"
            }
            Invoke-Tool $venvPython @("-m", "pytest", $TestPath, "-q")
        }
        "doctor" {
            Write-Host "Checking installed dependency consistency with pip check; this does not prove exact lock parity."
            Invoke-Tool $venvPython @("-B", "-m", "pip", "check")
            Invoke-Tool $venvPython @("-B", "-c", "import mic_eq; assert mic_eq.CORE_AVAILABLE, str(mic_eq._CORE_IMPORT_ERROR) if not mic_eq.CORE_AVAILABLE else ''; print('Native extension imports successfully')")
            if (-not $DryRun) {
                Write-Host "Doctor checks completed. Compiler execution and release interpreter/source provenance still require their documented build checks."
            }
        }
    }
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    if ($script:commandExitCode -eq 0) { $script:commandExitCode = 1 }
} finally {
    foreach ($name in $environmentNames) {
        $value = $previousEnvironment[$name]
        if ($null -eq $value) { $value = [NullString]::Value }
        [Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
    Pop-Location
}
exit $script:commandExitCode
