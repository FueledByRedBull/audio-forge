param(
    [string]$Payload = (Join-Path $PSScriptRoot "dist\AudioForge"),
    [string]$Output,
    [string]$WixPath,
    [string]$Version
)

$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    $publicTag = $null
    if ([string]::IsNullOrWhiteSpace($Version)) {
        $sourceVersion = & ".\.venv\Scripts\python.exe" -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])"
        if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($sourceVersion)) {
            throw "Unable to resolve the project version from pyproject.toml."
        }
        $publicTag = (& ".\.venv\Scripts\python.exe" python/tools/release_version.py tag $sourceVersion.Trim()).Trim()
        if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($publicTag)) {
            throw "Unable to resolve the canonical release tag from pyproject.toml."
        }
        $version = & ".\.venv\Scripts\python.exe" python/tools/release_version.py msi $sourceVersion.Trim()
        if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($version)) {
            throw "Unable to map the project version to a valid MSI version."
        }
    } else {
        $version = $Version
    }
    $version = $version.Trim()
    if ($version -notmatch '^\d+\.\d+\.\d+$') {
        throw "MSI version must be MAJOR.MINOR.PATCH, got '$version'."
    }
    $versionParts = $version.Split('.') | ForEach-Object {
        try { [int]::Parse($_, [Globalization.CultureInfo]::InvariantCulture) }
        catch { throw "MSI version contains a non-numeric component: '$version'." }
    }
    if ($versionParts[0] -gt 255 -or $versionParts[1] -gt 255 -or $versionParts[2] -gt 65535) {
        throw "MSI version fields exceed Windows Installer limits (major/minor <= 255, patch <= 65535): '$version'."
    }

    $payloadPath = (Resolve-Path -LiteralPath $Payload -ErrorAction Stop).Path
    if (-not (Test-Path -LiteralPath (Join-Path $payloadPath "AudioForge.exe") -PathType Leaf)) {
        throw "Portable payload is missing AudioForge.exe: $payloadPath"
    }
    if ([string]::IsNullOrWhiteSpace($Output)) {
        $fileTag = if ($publicTag) { $publicTag } else { "v$version" }
        $Output = Join-Path $PSScriptRoot "dist\AudioForge-$fileTag-win64.msi"
    }
    $outputPath = [System.IO.Path]::GetFullPath($Output)
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $outputPath) | Out-Null

    if ([string]::IsNullOrWhiteSpace($WixPath)) {
        $wixCommand = Get-Command wix.exe -ErrorAction SilentlyContinue
        if ($wixCommand) {
            $WixPath = $wixCommand.Source
        }
    }
    if ([string]::IsNullOrWhiteSpace($WixPath)) {
        $programFilesRoots = @($env:ProgramFiles, ${env:ProgramFiles(x86)}) |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
        foreach ($programFilesRoot in $programFilesRoots) {
            $installedWix = Join-Path $programFilesRoot "WiX Toolset v7.0\bin\wix.exe"
            if (Test-Path -LiteralPath $installedWix -PathType Leaf) {
                $WixPath = $installedWix
                break
            }
        }
    }

    $wixVersion = "7.0.0"
    $wixUrl = "https://github.com/wixtoolset/wix/releases/download/v$wixVersion/wix-cli-x64.msi"
    $wixSha256 = "16ff3857255e0ffa118d90e6d3a20e47499e55cda109853b5aac4700735f0293"
    $toolRoot = Join-Path $(if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { $env:TEMP }) "audioforge-wix-$wixVersion"
    $cachedMsi = Join-Path $toolRoot "wix-cli-x64.msi"

    if ([string]::IsNullOrWhiteSpace($WixPath)) {
        $cachedWix = Get-ChildItem -LiteralPath $toolRoot -Recurse -Filter wix.exe -File -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($cachedWix) {
            $WixPath = $cachedWix.FullName
        }
    }

    if ([string]::IsNullOrWhiteSpace($WixPath)) {
        New-Item -ItemType Directory -Force -Path $toolRoot | Out-Null
        if (-not (Test-Path -LiteralPath $cachedMsi -PathType Leaf)) {
            Invoke-WebRequest -Uri $wixUrl -OutFile $cachedMsi
        }
        $actualSha256 = (Get-FileHash -LiteralPath $cachedMsi -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualSha256 -ne $wixSha256) {
            throw "WiX CLI SHA-256 mismatch: expected $wixSha256, got $actualSha256."
        }

        $adminRoot = Join-Path $toolRoot ("admin-" + [guid]::NewGuid().ToString("N"))
        New-Item -ItemType Directory -Force -Path $adminRoot | Out-Null
        $extractArgs = @(
            "/a",
            ('"{0}"' -f $cachedMsi),
            ('TARGETDIR="{0}"' -f $adminRoot),
            "/qn",
            "/norestart"
        )
        $extract = Start-Process -FilePath msiexec.exe `
            -ArgumentList $extractArgs `
            -WindowStyle Hidden `
            -Wait `
            -PassThru
        if ($extract.ExitCode -ne 0) {
            throw "WiX CLI extraction failed with exit code $($extract.ExitCode)."
        }
        $WixPath = (
            Get-ChildItem -LiteralPath $adminRoot -Recurse -Filter wix.exe -File |
                Select-Object -First 1
        ).FullName
        if ([string]::IsNullOrWhiteSpace($WixPath)) {
            throw "WiX CLI extraction did not produce wix.exe."
        }
    }

    if (-not (Test-Path -LiteralPath $WixPath -PathType Leaf)) {
        throw "WiX CLI not found: $WixPath"
    }
    $wixVersionOutput = (& $WixPath --version | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $wixVersionOutput -notmatch '^7\.0\.0(?:\+.*)?$') {
        throw "Expected WiX Toolset 7.0.0, got '$wixVersionOutput'."
    }
    & $WixPath eula accept wix7
    if ($LASTEXITCODE -ne 0) {
        throw "WiX OSMF EULA acceptance failed."
    }

    & $WixPath build `
        -arch x64 `
        -b "PayloadDir=$payloadPath" `
        -d "Version=$version" `
        -pdbtype none `
        -o $outputPath `
        (Join-Path $PSScriptRoot "installer\AudioForge.wxs")
    if ($LASTEXITCODE -ne 0) {
        throw "WiX MSI build failed with exit code $LASTEXITCODE."
    }
    if (-not (Test-Path -LiteralPath $outputPath -PathType Leaf)) {
        throw "WiX reported success but did not create $outputPath."
    }
    Write-Host "MSI: $outputPath" -ForegroundColor Green
}
finally {
    Pop-Location
}
