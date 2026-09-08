<#
.SYNOPSIS
    Rebuild the pinned DeepFilter C API DLL used by the Windows bundles.

.DESCRIPTION
    The repository does not track the generated DLL. This script fetches the
    exact upstream commit named in build-support/deepfilter/provenance.json,
    applies the small compatibility/security patches recorded there, builds
    the production C API with the checked-in lockfile, and verifies the ABI.

    Source and build workspaces are kept under target/deepfilter-build. A
    mismatched existing cache is an error; the script never deletes or
    replaces a cache implicitly. The output is written to target/deepfilter by
    default and is not copied over the repository's runtime DLL unless the
    caller explicitly requests that output path.
#>

[CmdletBinding()]
param(
    [string]$OutputPath = (Join-Path $PSScriptRoot "target\deepfilter\df.dll"),
    [string]$AttestationPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToUpperInvariant()
}

function Get-CanonicalTextSha256([string]$Path) {
    $text = [System.IO.File]::ReadAllText($Path)
    $canonical = [regex]::Replace($text, "`r`n?", "`n")
    $encoding = New-Object System.Text.UTF8Encoding($false)
    $bytes = $encoding.GetBytes($canonical)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace("-", "").ToUpperInvariant()
    } finally {
        $sha.Dispose()
    }
}

function Assert-FileHash([string]$Path, [string]$Expected, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label is missing: $Path"
    }
    $actual = Get-Sha256 $Path
    if ($actual -ne $Expected.ToUpperInvariant()) {
        throw "$Label SHA-256 mismatch. Expected $Expected, got ${actual}: $Path"
    }
}

function Ensure-VerifiedArchive(
    [string]$ArchivePath,
    [string]$CacheRoot,
    [string]$ArchiveName,
    [string]$ExpectedSha,
    [string]$DownloadUri
) {
    if (Test-Path -LiteralPath $ArchivePath -PathType Leaf) {
        Assert-FileHash $ArchivePath $ExpectedSha "Cached $ArchiveName source archive"
        return
    }

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ArchivePath) | Out-Null
    $stagedArchive = "$ArchivePath.download-$PID"
    try {
        if (Test-Path -LiteralPath $stagedArchive -PathType Leaf) {
            Remove-Item -LiteralPath $stagedArchive -Force
        }
        $cachedArchive = Get-ChildItem -LiteralPath $CacheRoot -Recurse -Filter $ArchiveName -File -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($cachedArchive) {
            Copy-Item -LiteralPath $cachedArchive.FullName -Destination $stagedArchive
        } else {
            Invoke-WebRequest -Uri $DownloadUri -OutFile $stagedArchive -ErrorAction Stop
        }
        Assert-FileHash $stagedArchive $ExpectedSha "Downloaded $ArchiveName source archive"
        Move-Item -LiteralPath $stagedArchive -Destination $ArchivePath
    } finally {
        if (Test-Path -LiteralPath $stagedArchive -PathType Leaf) {
            Remove-Item -LiteralPath $stagedArchive -Force
        }
    }
    Assert-FileHash $ArchivePath $ExpectedSha "Cached $ArchiveName source archive"
}

function Invoke-Checked([string]$FilePath, [string[]]$Arguments) {
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FilePath failed with exit code $LASTEXITCODE."
    }
}

function Set-Utf8NoBom([string]$Path, [string]$Text) {
    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Text, $encoding)
}

function Get-FileVersion([string]$Path) {
    try {
        return [System.Diagnostics.FileVersionInfo]::GetVersionInfo($Path).FileVersion
    } catch {
        return ""
    }
}

function Find-Dumpbin([string]$ExpectedVersion) {
    $candidates = [System.Collections.Generic.List[string]]::new()
    $command = Get-Command dumpbin.exe -ErrorAction SilentlyContinue
    if ($command) {
        $candidates.Add($command.Source)
    }

    $roots = @()
    if ($env:ProgramFiles) { $roots += (Join-Path $env:ProgramFiles "Microsoft Visual Studio") }
    $programFilesX86 = [Environment]::GetEnvironmentVariable("ProgramFiles(x86)")
    if ($programFilesX86) { $roots += (Join-Path $programFilesX86 "Microsoft Visual Studio") }
    foreach ($root in $roots) {
        if (Test-Path -LiteralPath $root -PathType Container) {
            $candidate = Get-ChildItem -LiteralPath $root -Recurse -Filter dumpbin.exe -File -ErrorAction SilentlyContinue |
                Where-Object { $_.FullName -match "\\VC\\Tools\\MSVC\\" } |
                Sort-Object FullName
            foreach ($item in $candidate) {
                $candidates.Add($item.FullName)
            }
        }
    }

    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        $version = Get-FileVersion $candidate
        if ([string]::IsNullOrWhiteSpace($ExpectedVersion) -or $version -like "$ExpectedVersion*") {
            return $candidate
        }
    }
    if ($candidates.Count -eq 0) {
        throw "dumpbin.exe was not found. Install the MSVC build tools or put dumpbin.exe on PATH."
    }
    throw "No dumpbin.exe matched the pinned MSVC toolset $ExpectedVersion. Found: $($candidates -join ', ')"
}

function Invoke-Captured([string]$FilePath, [string[]]$Arguments) {
    $output = (& $FilePath @Arguments 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "$FilePath failed with exit code $LASTEXITCODE. $output"
    }
    return $output
}

function Assert-CleanGitCheckout([string]$GitPath, [string]$Checkout, [string]$ExpectedRepository, [string]$ExpectedCommit) {
    $remote = Invoke-Captured $GitPath @("-C", $Checkout, "remote", "get-url", "origin")
    if ($remote.Trim() -ne $ExpectedRepository) {
        throw "DeepFilter source cache origin mismatch. Expected $ExpectedRepository, got $remote."
    }
    $head = Invoke-Captured $GitPath @("-C", $Checkout, "rev-parse", "HEAD")
    if ($head.Trim() -ne $ExpectedCommit) {
        throw "DeepFilter source cache is at $($head.Trim()), expected $ExpectedCommit."
    }
    $status = (& $GitPath "-C" $Checkout "status" "--porcelain" "--untracked-files=all" 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "Could not inspect DeepFilter source cache status."
    }
    if (-not [string]::IsNullOrWhiteSpace($status)) {
        throw "DeepFilter source cache is dirty; refusing to copy unreviewed files: $Checkout`n$status"
    }
}

function Assert-WorkspaceSourceTree([string]$GitPath, [string]$SourceCache, [string]$Commit, [string]$Workspace) {
    $sourceRoot = Join-Path $SourceCache "libDF"
    $workspaceRoot = Join-Path $Workspace "libDF"
    if (-not (Test-Path -LiteralPath $workspaceRoot -PathType Container)) {
        throw "DeepFilter workspace source tree is missing: $workspaceRoot"
    }

    $expectedPaths = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    $sourceEntries = @(& $GitPath "-C" $SourceCache "ls-tree" "-r" "--name-only" $Commit "--" "libDF")
    if ($LASTEXITCODE -ne 0 -or $sourceEntries.Count -eq 0) {
        throw "Could not enumerate the pinned DeepFilter libDF source tree."
    }
    foreach ($entry in $sourceEntries) {
        $relative = ([string]$entry).Trim().Replace("/", "\")
        if ([string]::IsNullOrWhiteSpace($relative)) { continue }
        $relativeWithinLib = $relative.Substring(6)
        [void]$expectedPaths.Add($relativeWithinLib)
        $workspacePath = Join-Path $workspaceRoot $relativeWithinLib
        if (-not (Test-Path -LiteralPath $workspacePath -PathType Leaf)) {
            throw "DeepFilter workspace is missing upstream source file: $relative"
        }
        if ($relative -notin @("libDF\Cargo.toml", "libDF\src\lib.rs", "libDF\src\tract.rs")) {
            $expectedHash = Invoke-Captured $GitPath @("-C", $SourceCache, "rev-parse", "$Commit`:$($relative.Replace('\', '/'))")
            $actualHash = Invoke-Captured $GitPath @("-C", $Workspace, "hash-object", "--no-filters", "--", "libDF\$relativeWithinLib")
            if ($actualHash.Trim() -ne $expectedHash.Trim()) {
                throw "DeepFilter workspace source changed outside the recorded patch files: $relative"
            }
        }
    }

    $actualFiles = Get-ChildItem -LiteralPath $workspaceRoot -Recurse -File | ForEach-Object {
        $_.FullName.Substring($workspaceRoot.Length + 1)
    }
    foreach ($actual in $actualFiles) {
        if (-not $expectedPaths.Contains($actual)) {
            throw "DeepFilter workspace contains an unreviewed extra source file: $actual"
        }
    }
}

function Assert-PatchTree([string]$ArchiveRoot, [string]$PatchedRoot, [string]$PatchedManifestHash) {
    $expectedFiles = Get-ChildItem -LiteralPath $ArchiveRoot -Recurse -File | ForEach-Object {
        $_.FullName.Substring($ArchiveRoot.Length + 1)
    }
    $actualFiles = Get-ChildItem -LiteralPath $PatchedRoot -Recurse -File | ForEach-Object {
        $_.FullName.Substring($PatchedRoot.Length + 1)
    }
    $expectedSet = [System.Collections.Generic.HashSet[string]]::new(
        [string[]]$expectedFiles,
        [System.StringComparer]::OrdinalIgnoreCase
    )
    $actualSet = [System.Collections.Generic.HashSet[string]]::new(
        [string[]]$actualFiles,
        [System.StringComparer]::OrdinalIgnoreCase
    )
    foreach ($extra in $actualSet) {
        if (-not $expectedSet.Contains($extra)) {
            throw "DeepFilter tract-linalg patch contains an unreviewed extra file: $extra"
        }
    }
    foreach ($missing in $expectedSet) {
        if (-not $actualSet.Contains($missing)) {
            throw "DeepFilter tract-linalg patch is missing archive file: $missing"
        }
        $archiveFile = Join-Path $ArchiveRoot $missing
        $patchedFile = Join-Path $PatchedRoot $missing
        if ($missing -eq "Cargo.toml") {
            Assert-FileHash $patchedFile $PatchedManifestHash "Patched tract-linalg manifest"
        } else {
            $expectedHash = Get-Sha256 $archiveFile
            Assert-FileHash $patchedFile $expectedHash "tract-linalg patch file $missing"
        }
    }
}

$projectRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$supportRoot = Join-Path $projectRoot "build-support\deepfilter"
$provenancePath = Join-Path $supportRoot "provenance.json"
$trackedManifest = Join-Path $supportRoot "Cargo.toml"
$trackedLock = Join-Path $supportRoot "Cargo.lock"
if (-not (Test-Path -LiteralPath $provenancePath -PathType Leaf)) { throw "Missing DeepFilter provenance: $provenancePath" }
if (-not (Test-Path -LiteralPath $trackedManifest -PathType Leaf)) { throw "Missing DeepFilter build manifest: $trackedManifest" }
if (-not (Test-Path -LiteralPath $trackedLock -PathType Leaf)) { throw "Missing DeepFilter build lockfile: $trackedLock" }

$provenance = Get-Content -LiteralPath $provenancePath -Raw | ConvertFrom-Json
$manifestSha = [string]$provenance.lock.manifest_sha256
Assert-FileHash $trackedManifest $manifestSha "Tracked DeepFilter build manifest"
$commit = [string]$provenance.upstream.commit
$repository = [string]$provenance.upstream.repository
$tractVersion = [string]$provenance.tract_linalg_patch.version
$archiveSha = [string]$provenance.tract_linalg_patch.archive_sha256
$expectedRustVersion = ([string]$provenance.build.tested_rust) -replace "\s.*$", ""
$sourceRoot = Join-Path $projectRoot "target\deepfilter-build"
$sourceCache = Join-Path $sourceRoot ("source-" + $commit)
$workspace = Join-Path $sourceRoot ("workspace-" + $commit)
$archiveCache = Join-Path $sourceRoot ("tract-linalg-" + $tractVersion + ".crate")
$markerPath = Join-Path $workspace ".audioforge-deepfilter.json"
$outputFullPath = [System.IO.Path]::GetFullPath($OutputPath)
$attestationFullPath = if ([string]::IsNullOrWhiteSpace($AttestationPath)) {
    [System.IO.Path]::GetFullPath((Join-Path $projectRoot "target\deepfilter\df.dll.provenance.json"))
} else {
    [System.IO.Path]::GetFullPath($AttestationPath)
}
$cargoTarget = Join-Path $workspace "target"
$cargoHome = if ($env:CARGO_HOME) { [System.IO.Path]::GetFullPath($env:CARGO_HOME) } else { Join-Path $env:USERPROFILE ".cargo" }

New-Item -ItemType Directory -Force -Path $sourceRoot | Out-Null

$git = Get-Command git.exe -ErrorAction SilentlyContinue
if (-not $git) { throw "git.exe is required to fetch the pinned DeepFilter source." }
$gitPath = $git.Source
$cargo = Get-Command cargo.exe -ErrorAction SilentlyContinue
if (-not $cargo) { throw "cargo.exe is required to build the pinned DeepFilter source." }
$cargoPath = $cargo.Source
$rustc = Get-Command rustc.exe -ErrorAction SilentlyContinue
if (-not $rustc) { throw "rustc.exe is required to build the pinned DeepFilter source." }
$rustcPath = $rustc.Source

$rustInfo = Invoke-Captured $rustcPath @("-vV")
if ($rustInfo -notmatch "(?m)^release:\s+$([regex]::Escape($expectedRustVersion))\s*$") {
    throw "This recipe was validated with Rust $expectedRustVersion; found:`n$rustInfo"
}
if ($rustInfo -notmatch "(?m)^host:\s+$([regex]::Escape([string]$provenance.build.target))\s*$") {
    throw "Rust host target does not match the pinned DeepFilter target: $rustInfo"
}

if (-not (Test-Path -LiteralPath (Join-Path $sourceCache ".git") -PathType Container)) {
    if (Test-Path -LiteralPath $sourceCache) {
        throw "Source cache exists but is not a git checkout: $sourceCache"
    }
    Invoke-Checked $gitPath @("clone", "--config", "core.autocrlf=false", "--no-checkout", $repository, $sourceCache)
    Invoke-Checked $gitPath @("-c", "core.autocrlf=false", "-C", $sourceCache, "fetch", "--depth", "1", "origin", $commit)
    Invoke-Checked $gitPath @("-c", "core.autocrlf=false", "-C", $sourceCache, "checkout", "--detach", $commit)
}

$actualCommit = (Invoke-Captured $gitPath @("-C", $sourceCache, "rev-parse", "HEAD")).Trim()
if ($actualCommit -ne $commit) {
    throw "Source cache is at $actualCommit, expected pinned commit ${commit}: $sourceCache"
}
Assert-CleanGitCheckout $gitPath $sourceCache $repository $commit

$sourceFiles = @(
    @{ Relative = "libDF\Cargo.toml"; Hash = [string]$provenance.upstream.files.'libDF/Cargo.toml'.sha256 },
    @{ Relative = "libDF\src\capi.rs"; Hash = [string]$provenance.upstream.files.'libDF/src/capi.rs'.sha256 },
    @{ Relative = "libDF\src\lib.rs"; Hash = [string]$provenance.upstream.files.'libDF/src/lib.rs'.sha256 },
    @{ Relative = "libDF\src\tract.rs"; Hash = [string]$provenance.upstream.files.'libDF/src/tract.rs'.sha256 }
)
foreach ($file in $sourceFiles) {
    Assert-FileHash (Join-Path $sourceCache $file.Relative) $file.Hash ("Pinned upstream " + $file.Relative)
}

$workspaceHasMarker = $false
if (Test-Path -LiteralPath $workspace) {
    if (Test-Path -LiteralPath $markerPath -PathType Leaf) {
        $markerText = Get-Content -LiteralPath $markerPath -Raw
        if (-not [string]::IsNullOrWhiteSpace($markerText)) {
            $marker = $markerText | ConvertFrom-Json
            if ([string]$marker.commit -ne $commit) {
                throw "Build workspace marker does not match the pinned recipe: $markerPath"
            }
            # Schema 1 workspaces are revalidated and upgraded in place. The
            # source tree checks below make this safe without deleting caches.
            $workspaceHasMarker = ([int]$marker.schema_version -eq 2)
        } else {
            # The previous run may have stopped while writing its marker.
            # Validate its copied inputs before resuming setup.
            Assert-FileHash (Join-Path $workspace "Cargo.lock") ([string]$provenance.lock.sha256) "Existing DeepFilter lockfile"
            Assert-FileHash (Join-Path $workspace "libDF\src\capi.rs") ([string]$provenance.upstream.files.'libDF/src/capi.rs'.sha256) "Existing C API source"
            Assert-FileHash (Join-Path $workspace "models\DeepFilterNet3_onnx.tar.gz") ([string]$provenance.models.'models/DeepFilterNet3_onnx.tar.gz'.sha256) "Existing standard model archive"
            Assert-FileHash (Join-Path $workspace "models\DeepFilterNet3_ll_onnx.tar.gz") ([string]$provenance.models.'models/DeepFilterNet3_ll_onnx.tar.gz'.sha256) "Existing low-latency model archive"
        }
    } else {
        # A prior interrupted run may have prepared the workspace but not yet
        # written its marker. Accept it only when its copied inputs are exact;
        # never overwrite an unverified existing directory.
        Assert-FileHash (Join-Path $workspace "Cargo.lock") ([string]$provenance.lock.sha256) "Existing DeepFilter lockfile"
        Assert-FileHash (Join-Path $workspace "libDF\src\capi.rs") ([string]$provenance.upstream.files.'libDF/src/capi.rs'.sha256) "Existing C API source"
        Assert-FileHash (Join-Path $workspace "models\DeepFilterNet3_onnx.tar.gz") ([string]$provenance.models.'models/DeepFilterNet3_onnx.tar.gz'.sha256) "Existing standard model archive"
        Assert-FileHash (Join-Path $workspace "models\DeepFilterNet3_ll_onnx.tar.gz") ([string]$provenance.models.'models/DeepFilterNet3_ll_onnx.tar.gz'.sha256) "Existing low-latency model archive"
    }
} else {
    New-Item -ItemType Directory -Force -Path $workspace | Out-Null
    Copy-Item -LiteralPath $trackedManifest -Destination (Join-Path $workspace "Cargo.toml")
    Copy-Item -LiteralPath $trackedLock -Destination (Join-Path $workspace "Cargo.lock")
    Copy-Item -LiteralPath (Join-Path $sourceCache "README.md") -Destination (Join-Path $workspace "README.md")
    Copy-Item -LiteralPath (Join-Path $sourceCache "libDF") -Destination (Join-Path $workspace "libDF") -Recurse
    New-Item -ItemType Directory -Force -Path (Join-Path $workspace "models") | Out-Null
    Copy-Item -LiteralPath (Join-Path $projectRoot "models\DeepFilterNet3_onnx.tar.gz") -Destination (Join-Path $workspace "models\DeepFilterNet3_onnx.tar.gz")
    Copy-Item -LiteralPath (Join-Path $projectRoot "models\DeepFilterNet3_ll_onnx.tar.gz") -Destination (Join-Path $workspace "models\DeepFilterNet3_ll_onnx.tar.gz")
}

Assert-WorkspaceSourceTree $gitPath $sourceCache $commit $workspace
Assert-FileHash (Join-Path $workspace "Cargo.toml") $manifestSha "Workspace DeepFilter build manifest"
Assert-FileHash (Join-Path $workspace "Cargo.lock") ([string]$provenance.lock.sha256) "Tracked DeepFilter lockfile"
Assert-FileHash (Join-Path $workspace "libDF\src\capi.rs") ([string]$provenance.upstream.files.'libDF/src/capi.rs'.sha256) "C API source"
Assert-FileHash (Join-Path $workspace "models\DeepFilterNet3_onnx.tar.gz") ([string]$provenance.models.'models/DeepFilterNet3_onnx.tar.gz'.sha256) "Standard model archive"
Assert-FileHash (Join-Path $workspace "models\DeepFilterNet3_ll_onnx.tar.gz") ([string]$provenance.models.'models/DeepFilterNet3_ll_onnx.tar.gz'.sha256) "Low-latency model archive"

$tractPatch = Join-Path $workspace "patches\tract-linalg"
Ensure-VerifiedArchive `
    $archiveCache `
    (Join-Path $cargoHome "registry\cache") `
    ("tract-linalg-" + $tractVersion + ".crate") `
    $archiveSha `
    ([string]$provenance.tract_linalg_patch.archive_url)
if (-not (Test-Path -LiteralPath (Join-Path $tractPatch "Cargo.toml") -PathType Leaf)) {
    $extractRoot = Join-Path $sourceRoot ("tract-linalg-" + $tractVersion + "-" + $commit)
    $extracted = Join-Path $extractRoot ("tract-linalg-" + $tractVersion)
    if (-not (Test-Path -LiteralPath $extracted -PathType Container)) {
        New-Item -ItemType Directory -Force -Path $extractRoot | Out-Null
        $tar = Get-Command tar.exe -ErrorAction SilentlyContinue
        if (-not $tar) { throw "tar.exe is required to extract the pinned tract-linalg archive." }
        Invoke-Checked $tar.Source @("-xf", $archiveCache, "-C", $extractRoot)
    }
    if (-not (Test-Path -LiteralPath $extracted -PathType Container)) {
        throw "tract-linalg archive did not contain $extracted"
    }
    New-Item -ItemType Directory -Force -Path $tractPatch | Out-Null
    foreach ($item in Get-ChildItem -LiteralPath $extracted -Force) {
        Copy-Item -LiteralPath $item.FullName -Destination (Join-Path $tractPatch $item.Name) -Recurse -Force
    }

    $tractManifest = Join-Path $tractPatch "Cargo.toml"
    Assert-FileHash $tractManifest ([string]$provenance.tract_linalg_patch.original_cargo_toml_sha256) "Original tract-linalg manifest"
    $tractText = [System.IO.File]::ReadAllText($tractManifest)
    $timePattern = '(?m)^\[build-dependencies\.time\]\r?\nversion = ">=0\.3\.23, <0\.3\.42"\r?\n'
    $timeMatches = [regex]::Matches($tractText, $timePattern)
    if ($timeMatches.Count -ne 1) { throw "Expected exactly one unused tract-linalg time build dependency, found $($timeMatches.Count)." }
    $tractText = [regex]::Replace($tractText, $timePattern, "", 1)
    Set-Utf8NoBom $tractManifest $tractText
    Assert-FileHash $tractManifest ([string]$provenance.tract_linalg_patch.patched_cargo_toml_sha256) "Patched tract-linalg manifest"
    Assert-FileHash (Join-Path $tractPatch "build.rs") ([string]$provenance.tract_linalg_patch.build_rs_sha256) "tract-linalg build script"
}
Assert-FileHash (Join-Path $tractPatch "Cargo.toml") ([string]$provenance.tract_linalg_patch.patched_cargo_toml_sha256) "Patched tract-linalg manifest"
Assert-FileHash (Join-Path $tractPatch "build.rs") ([string]$provenance.tract_linalg_patch.build_rs_sha256) "tract-linalg build script"
$extractRoot = Join-Path $sourceRoot ("tract-linalg-" + $tractVersion + "-" + $commit)
$extracted = Join-Path $extractRoot ("tract-linalg-" + $tractVersion)
if (-not (Test-Path -LiteralPath $extracted -PathType Container)) {
    New-Item -ItemType Directory -Force -Path $extractRoot | Out-Null
    $tar = Get-Command tar.exe -ErrorAction SilentlyContinue
    if (-not $tar) { throw "tar.exe is required to extract the pinned tract-linalg archive." }
    Invoke-Checked $tar.Source @("-xf", $archiveCache, "-C", $extractRoot)
}
Assert-FileHash $archiveCache $archiveSha "tract-linalg source archive"
if (-not (Test-Path -LiteralPath $extracted -PathType Container)) {
    throw "tract-linalg archive did not contain $extracted"
}
Assert-PatchTree $extracted $tractPatch ([string]$provenance.tract_linalg_patch.patched_cargo_toml_sha256)

$libManifest = Join-Path $workspace "libDF\Cargo.toml"
$libSource = Join-Path $workspace "libDF\src\lib.rs"
$tractSource = Join-Path $workspace "libDF\src\tract.rs"
if ((Get-Sha256 $libManifest) -eq ([string]$provenance.upstream.files.'libDF/Cargo.toml'.sha256).ToUpperInvariant()) {
    $manifestText = [System.IO.File]::ReadAllText($libManifest)
    $ndarrayPattern = 'ndarray = \{ version = "\^0\.15", optional = true, features = \["serde"\] \}'
    if ([regex]::Matches($manifestText, $ndarrayPattern).Count -ne 1) { throw "Expected one upstream ndarray 0.15 dependency in libDF." }
    $manifestText = [regex]::Replace($manifestText, $ndarrayPattern, 'ndarray = { version = "^0.16", optional = true, features = ["serde"] }', 1)
    $devPattern = '(?ms)^\[dev-dependencies\]\r?\nrand = "0\.8"\r?\nrstest = "0\.19"\r?\nenv_logger = "0\.11"\r?\nlog = \{ version = "0\.4", features = \["std"\] \}\r?\n'
    if ([regex]::Matches($manifestText, $devPattern).Count -ne 1) { throw "Expected one upstream libDF dev-dependencies block." }
    $manifestText = [regex]::Replace($manifestText, $devPattern, "", 1)
    Set-Utf8NoBom $libManifest $manifestText
}
Assert-FileHash $libManifest ([string]$provenance.upstream.files.'libDF/Cargo.toml'.patched_sha256) "Patched libDF manifest"

if ((Get-Sha256 $libSource) -eq ([string]$provenance.upstream.files.'libDF/src/lib.rs'.sha256).ToUpperInvariant()) {
    $libSourceText = [System.IO.File]::ReadAllText($libSource)
    $unitNormPattern = '(?m)^        \*s = x\.norm\(\) \* \(1\. - alpha\) \+ \*s \* alpha;\r?$'
    $unitNormMatches = [regex]::Matches($libSourceText, $unitNormPattern)
    if ($unitNormMatches.Count -ne 2) { throw "Expected two unit normalization state updates in libDF/src/lib.rs, found $($unitNormMatches.Count)." }
    $libSourceText = [regex]::Replace($libSourceText, $unitNormPattern, '        *s = (x.norm() * (1. - alpha) + *s * alpha).max(f32::MIN_POSITIVE);')
    Set-Utf8NoBom $libSource $libSourceText
}
Assert-FileHash $libSource ([string]$provenance.upstream.files.'libDF/src/lib.rs'.patched_sha256) "Patched libDF normalization source"

if ((Get-Sha256 $tractSource) -eq ([string]$provenance.upstream.files.'libDF/src/tract.rs'.sha256).ToUpperInvariant()) {
    $tractSourceText = [System.IO.File]::ReadAllText($tractSource)
    $symbolPattern = 'm\.symbol_table\.sym\("S"\)'
    $symbolMatches = [regex]::Matches($tractSourceText, $symbolPattern)
    if ($symbolMatches.Count -ne 3) { throw "Expected three tract symbol_table calls, found $($symbolMatches.Count)." }
    $tractSourceText = [regex]::Replace($tractSourceText, $symbolPattern, 'm.symbols.sym("S")')
    Set-Utf8NoBom $tractSource $tractSourceText
}
Assert-FileHash $tractSource ([string]$provenance.upstream.files.'libDF/src/tract.rs'.patched_sha256) "Patched libDF tract source"
Assert-WorkspaceSourceTree $gitPath $sourceCache $commit $workspace

if (-not $workspaceHasMarker) {
    $markerJson = @{
        schema_version = 2
        commit = $commit
        tract_linalg_version = $tractVersion
        lock_sha256 = Get-Sha256 (Join-Path $workspace "Cargo.lock")
        lib_manifest_sha256 = Get-Sha256 $libManifest
        lib_source_sha256 = Get-Sha256 $libSource
        tract_source_sha256 = Get-Sha256 $tractSource
        tract_manifest_sha256 = Get-Sha256 (Join-Path $tractPatch "Cargo.toml")
        rustflags = @(
            "-C link-arg=/DEBUG:NONE",
            "--remap-path-prefix=<project-root>=/audioforge",
            "--remap-path-prefix=<cargo-home>=/cargo"
        )
    } | ConvertTo-Json
    Set-Utf8NoBom $markerPath $markerJson
}

$rustVersion = (Invoke-Captured $rustcPath @("--version")).Trim()
$cargoVersion = (Invoke-Captured $cargoPath @("--version")).Trim()
$expectedRust = [string]$provenance.build.tested_rust
$projectRootForRust = $projectRoot.Replace("\", "/")
$cargoHomeForRust = $cargoHome.Replace("\", "/")
$previousRustFlags = [Environment]::GetEnvironmentVariable("RUSTFLAGS", "Process")
$env:RUSTFLAGS = "-C link-arg=/DEBUG:NONE --remap-path-prefix=$projectRootForRust=/audioforge --remap-path-prefix=$cargoHomeForRust=/cargo"
$recipeRustFlags = $env:RUSTFLAGS

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $outputFullPath) | Out-Null
try {
    Invoke-Checked $cargoPath @(
        "build", "--manifest-path", (Join-Path $workspace "Cargo.toml"), "--locked",
        "--profile", "release-lto", "--package", "deep_filter", "--lib",
        "--no-default-features", "--features", "capi", "--target", ([string]$provenance.build.target),
        "--target-dir", $cargoTarget
    )
} finally {
    if ($null -eq $previousRustFlags) {
        Remove-Item Env:RUSTFLAGS -ErrorAction SilentlyContinue
    } else {
        $env:RUSTFLAGS = $previousRustFlags
    }
}

$builtDll = Join-Path $cargoTarget (([string]$provenance.build.target) + "\release-lto\df.dll")
if (-not (Test-Path -LiteralPath $builtDll -PathType Leaf)) { throw "Cargo succeeded but did not produce $builtDll" }
$dumpbin = Find-Dumpbin ""
$stagedOutput = "$outputFullPath.build-$PID"
$stagedAttestation = "$attestationFullPath.build-$PID"
if (Test-Path -LiteralPath $stagedOutput -PathType Leaf) {
    Remove-Item -LiteralPath $stagedOutput -Force
}
if (Test-Path -LiteralPath $stagedAttestation -PathType Leaf) {
    Remove-Item -LiteralPath $stagedAttestation -Force
}
Copy-Item -LiteralPath $builtDll -Destination $stagedOutput
try {
    $exportsText = Invoke-Captured $dumpbin @("/nologo", "/exports", $stagedOutput)
    $missingExports = @([string[]]$provenance.build.required_exports | Where-Object { $exportsText -notmatch ("(?m)\b" + [regex]::Escape($_) + "\b") })
    if ($missingExports.Count -gt 0) { throw "DeepFilter DLL is missing required exports: $($missingExports -join ', ')" }
    $outputSha = Get-Sha256 $stagedOutput
    $outputBytes = (Get-Item -LiteralPath $stagedOutput).Length
    $recipeFiles = [ordered]@{
        "build_deepfilter.ps1" = Get-CanonicalTextSha256 (Join-Path $projectRoot "build_deepfilter.ps1")
        "build-support/deepfilter/Cargo.toml" = Get-CanonicalTextSha256 $trackedManifest
        "build-support/deepfilter/Cargo.lock" = Get-CanonicalTextSha256 $trackedLock
        "build-support/deepfilter/provenance.json" = Get-CanonicalTextSha256 $provenancePath
        "models/DeepFilterNet3_onnx.tar.gz" = Get-Sha256 (Join-Path $projectRoot "models\DeepFilterNet3_onnx.tar.gz")
        "models/DeepFilterNet3_ll_onnx.tar.gz" = Get-Sha256 (Join-Path $projectRoot "models\DeepFilterNet3_ll_onnx.tar.gz")
    }
    $buildRecord = [ordered]@{
        schema_version = 1
        kind = "audioforge.deepfilter.build"
        output = [ordered]@{
            name = [System.IO.Path]::GetFileName($outputFullPath)
            bytes = [int64]$outputBytes
            sha256 = $outputSha
        }
        source = [ordered]@{
            repository = $repository
            commit = $commit
            workspace_validation = "Every file under libDF was matched to the pinned git blob; only the three recorded source patches were accepted."
        }
        recipe = [ordered]@{
            files = $recipeFiles
            tract_linalg_archive_sha256 = $archiveSha
            tract_linalg_patched_manifest_sha256 = [string]$provenance.tract_linalg_patch.patched_cargo_toml_sha256
            tract_linalg_build_rs_sha256 = [string]$provenance.tract_linalg_patch.build_rs_sha256
            rustflags = $recipeRustFlags
            target = [string]$provenance.build.target
            profile = [string]$provenance.build.profile
            features = @([string[]]$provenance.build.features)
            default_features = [bool]$provenance.build.default_features
        }
        toolchain = [ordered]@{
            rustc = $rustVersion
            cargo = $cargoVersion
            dumpbin = $dumpbin
        }
        abi = [ordered]@{
            required_exports = @([string[]]$provenance.build.required_exports)
        }
        reproducibility = "The output digest is attested for this build. Bit-for-bit reproducibility is not a release gate because native linker metadata may vary between approved toolchains."
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $attestationFullPath) | Out-Null
    Set-Utf8NoBom $stagedAttestation ($buildRecord | ConvertTo-Json -Depth 10)
    Move-Item -LiteralPath $stagedOutput -Destination $outputFullPath -Force
    Move-Item -LiteralPath $stagedAttestation -Destination $attestationFullPath -Force
} finally {
    if (Test-Path -LiteralPath $stagedOutput -PathType Leaf) {
        Remove-Item -LiteralPath $stagedOutput -Force
    }
    if (Test-Path -LiteralPath $stagedAttestation -PathType Leaf) {
        Remove-Item -LiteralPath $stagedAttestation -Force
    }
}

@{
    output = $outputFullPath
    bytes = (Get-Item -LiteralPath $outputFullPath).Length
    sha256 = $outputSha
    source_commit = $commit
    rustc = $rustVersion
    cargo = $cargoVersion
    expected_tested_rust = $expectedRust
    dumpbin = $dumpbin
    rustflags = $recipeRustFlags
    required_exports = @([string[]]$provenance.build.required_exports)
    attestation = $attestationFullPath
    reproducibility = "Output hash is recorded for this build; bit-reproducibility is not a release gate."
} | ConvertTo-Json -Depth 5
