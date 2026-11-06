"""Rebuild the pinned DeepFilter C API library for Linux (libdf.so).

The Linux counterpart of build_deepfilter.ps1, held to the same recipe: the
exact upstream commit named in build-support/deepfilter/provenance.json, only
the recorded source patches, the checked-in lockfile, and the required C API
exports. Source and build workspaces are kept under target/deepfilter-build; a
mismatched existing cache is an error and is never deleted or replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

from bounded_download import download

ROOT = Path(__file__).resolve().parents[2]
SUPPORT = ROOT / "build-support" / "deepfilter"
PROVENANCE = SUPPORT / "provenance.json"
TRACKED_MANIFEST = SUPPORT / "Cargo.toml"
TRACKED_LOCK = SUPPORT / "Cargo.lock"
TARGET = "x86_64-unknown-linux-gnu"
LIBRARY = "libdf.so"
PATCHED_UPSTREAM_FILES = ("libDF/Cargo.toml", "libDF/src/capi.rs", "libDF/src/lib.rs", "libDF/src/tract.rs")
MODELS = ("models/DeepFilterNet3_onnx.tar.gz", "models/DeepFilterNet3_ll_onnx.tar.gz")
# crates.io redirects its download API to static.crates.io; the archive hash is pinned.
TRACT_ARCHIVE_HOSTS = frozenset({"crates.io", "static.crates.io"})
TRACT_ARCHIVE_MAX_BYTES = 64 * 1024 * 1024


class BuildError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest().upper()


def canonical_text_sha256(path: Path) -> str:
    text = path.read_bytes().decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def assert_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise BuildError(f"{label} is missing: {path}")
    actual = sha256(path)
    if actual != expected.upper():
        raise BuildError(f"{label} SHA-256 mismatch. Expected {expected}, got {actual}: {path}")


def assert_hash_in(path: Path, expected: list[str], label: str) -> None:
    if not path.is_file():
        raise BuildError(f"{label} is missing: {path}")
    if sha256(path) not in {value.upper() for value in expected}:
        raise BuildError(f"{label} SHA-256 matches neither the pinned nor the patched source: {path}")


def run(*command: str | Path, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    completed = subprocess.run([str(part) for part in command], cwd=cwd, env=env)
    if completed.returncode != 0:
        raise BuildError(f"{command[0]} failed with exit code {completed.returncode}")


def captured(*command: str | Path, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        [str(part) for part in command], cwd=cwd, capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise BuildError(
            f"{command[0]} failed with exit code {completed.returncode}. "
            f"{(completed.stdout + completed.stderr).strip()}"
        )
    return completed.stdout.strip()


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise BuildError(f"{name} is required to build the pinned DeepFilter source")
    return path


def replace_exactly(text: str, pattern: str, replacement: str, count: int, label: str) -> str:
    found = len(re.findall(pattern, text, flags=re.M))
    if found != count:
        raise BuildError(f"Expected {count} {label}, found {found}")
    return re.sub(pattern, lambda _match: replacement, text, flags=re.M)


def write_text(path: Path, text: str) -> None:
    path.write_bytes(text.encode("utf-8"))


def assert_clean_checkout(git: str, checkout: Path, repository: str, commit: str) -> None:
    remote = captured(git, "-C", checkout, "remote", "get-url", "origin")
    if remote != repository:
        raise BuildError(f"DeepFilter source cache origin mismatch. Expected {repository}, got {remote}")
    head = captured(git, "-C", checkout, "rev-parse", "HEAD")
    if head != commit:
        raise BuildError(f"DeepFilter source cache is at {head}, expected {commit}")
    status = captured(git, "-C", checkout, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise BuildError(f"DeepFilter source cache is dirty; refusing to copy unreviewed files: {checkout}")


def assert_workspace_source_tree(git: str, source: Path, commit: str, workspace: Path) -> None:
    """Every file under libDF is the pinned blob, apart from the recorded patch files."""
    libdf = workspace / "libDF"
    if not libdf.is_dir():
        raise BuildError(f"DeepFilter workspace source tree is missing: {libdf}")
    entries = captured(git, "-C", source, "ls-tree", "-r", "--name-only", commit, "--", "libDF")
    expected = {entry for entry in entries.splitlines() if entry}
    if not expected:
        raise BuildError("Could not enumerate the pinned DeepFilter libDF source tree")
    for relative in sorted(expected):
        if not (workspace / relative).is_file():
            raise BuildError(f"DeepFilter workspace is missing upstream source file: {relative}")
        if relative in PATCHED_UPSTREAM_FILES:
            continue
        pinned = captured(git, "-C", source, "rev-parse", f"{commit}:{relative}")
        actual = captured(git, "-C", workspace, "hash-object", "--no-filters", "--", relative)
        if actual != pinned:
            raise BuildError(f"DeepFilter workspace source changed outside the recorded patch files: {relative}")
    actual_files = {
        path.relative_to(workspace).as_posix() for path in libdf.rglob("*") if path.is_file()
    }
    for extra in sorted(actual_files - expected):
        raise BuildError(f"DeepFilter workspace contains an unreviewed extra source file: {extra}")


def assert_patch_tree(archive_root: Path, patched_root: Path, patched_manifest_sha: str) -> None:
    def files(root: Path) -> set[str]:
        return {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}

    expected, actual = files(archive_root), files(patched_root)
    for extra in sorted(actual - expected):
        raise BuildError(f"DeepFilter tract-linalg patch contains an unreviewed extra file: {extra}")
    for relative in sorted(expected):
        if relative not in actual:
            raise BuildError(f"DeepFilter tract-linalg patch is missing archive file: {relative}")
        if relative == "Cargo.toml":
            assert_hash(patched_root / relative, patched_manifest_sha, "Patched tract-linalg manifest")
        else:
            assert_hash(
                patched_root / relative, sha256(archive_root / relative), f"tract-linalg patch file {relative}"
            )


def ensure_tract_archive(archive: Path, cargo_home: Path, name: str, expected_sha: str, url: str) -> None:
    if archive.is_file():
        assert_hash(archive, expected_sha, f"Cached {name} source archive")
        return
    archive.parent.mkdir(parents=True, exist_ok=True)
    staged = archive.with_name(f"{archive.name}.download-{os.getpid()}")
    try:
        cached = next((cargo_home / "registry" / "cache").rglob(name), None)
        if cached is not None:
            shutil.copyfile(cached, staged)
        else:
            # The URL is pinned in provenance.json and the archive hash is checked below.
            download(
                url,
                staged,
                hosts=TRACT_ARCHIVE_HOSTS,
                max_bytes=TRACT_ARCHIVE_MAX_BYTES,
                timeout=120,
                refusal=f"The {name} source archive must come from HTTPS on crates.io",
                too_large=f"The {name} source archive exceeded {TRACT_ARCHIVE_MAX_BYTES} bytes",
                follow_redirects=True,
            )
        assert_hash(staged, expected_sha, f"Downloaded {name} source archive")
        staged.replace(archive)
    finally:
        staged.unlink(missing_ok=True)


def extract_archive(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as source:  # nosemgrep: tarfile-extractall-traversal
        # The data filter refuses links outside the tree, absolute paths and device files.
        source.extractall(destination, filter="data")


def build(output: Path, attestation_path: Path, *, allow_untested_rust: bool) -> dict[str, Any]:
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    upstream = provenance["upstream"]
    upstream_files = upstream["files"]
    lock = provenance["lock"]
    tract = provenance["tract_linalg_patch"]
    recipe_build = provenance["build"]
    assert_hash(TRACKED_MANIFEST, lock["manifest_sha256"], "Tracked DeepFilter build manifest")
    commit, repository = upstream["commit"], upstream["repository"]
    tract_version, tract_sha = tract["version"], tract["archive_sha256"]
    tested_rust = recipe_build["tested_rust"].split()[0]

    source_root = ROOT / "target" / "deepfilter-build"
    source_cache = source_root / f"source-{commit}"
    workspace = source_root / f"workspace-{commit}"
    archive_cache = source_root / f"tract-linalg-{tract_version}.crate"
    marker = workspace / ".audioforge-deepfilter.json"
    cargo_home = Path(os.environ.get("CARGO_HOME") or Path.home() / ".cargo")
    source_root.mkdir(parents=True, exist_ok=True)

    git, cargo, rustc, nm = (require_tool(name) for name in ("git", "cargo", "rustc", "nm"))
    rust_info = captured(rustc, "-vV")
    release = re.search(r"^release:\s+(\S+)$", rust_info, re.M)
    if not allow_untested_rust and (release is None or release.group(1) != tested_rust):
        raise BuildError(
            f"This recipe was validated with Rust {tested_rust}; found:\n{rust_info}\n"
            "Pass --allow-untested-rust for a local build."
        )
    if not re.search(rf"^host:\s+{re.escape(TARGET)}$", rust_info, re.M):
        raise BuildError(f"Rust host target is not {TARGET}:\n{rust_info}")

    if not (source_cache / ".git").is_dir():
        if source_cache.exists():
            raise BuildError(f"Source cache exists but is not a git checkout: {source_cache}")
        run(git, "clone", "--config", "core.autocrlf=false", "--no-checkout", repository, source_cache)
        run(git, "-c", "core.autocrlf=false", "-C", source_cache, "fetch", "--depth", "1", "origin", commit)
        run(git, "-c", "core.autocrlf=false", "-C", source_cache, "checkout", "--detach", commit)
    assert_clean_checkout(git, source_cache, repository, commit)
    for relative in PATCHED_UPSTREAM_FILES:
        assert_hash(source_cache / relative, upstream_files[relative]["sha256"], f"Pinned upstream {relative}")

    capi_hashes = [upstream_files["libDF/src/capi.rs"]["sha256"], upstream_files["libDF/src/capi.rs"]["patched_sha256"]]
    workspace_has_marker = False
    if workspace.exists():
        marker_text = marker.read_text(encoding="utf-8").strip() if marker.is_file() else ""
        if marker_text:
            recorded = json.loads(marker_text)
            if recorded.get("commit") != commit:
                raise BuildError(f"Build workspace marker does not match the pinned recipe: {marker}")
            workspace_has_marker = recorded.get("schema_version") == 2
        else:
            # An interrupted run may have prepared the workspace without its marker;
            # accept it only when its copied inputs are exact.
            assert_hash(workspace / "Cargo.lock", lock["sha256"], "Existing DeepFilter lockfile")
            assert_hash_in(workspace / "libDF/src/capi.rs", capi_hashes, "Existing C API source")
            for model in MODELS:
                assert_hash(workspace / model, provenance["models"][model]["sha256"], f"Existing {model}")
    else:
        workspace.mkdir(parents=True)
        shutil.copyfile(TRACKED_MANIFEST, workspace / "Cargo.toml")
        shutil.copyfile(TRACKED_LOCK, workspace / "Cargo.lock")
        shutil.copyfile(source_cache / "README.md", workspace / "README.md")
        shutil.copytree(source_cache / "libDF", workspace / "libDF")
        (workspace / "models").mkdir()
        for model in MODELS:
            shutil.copyfile(ROOT / model, workspace / model)

    assert_workspace_source_tree(git, source_cache, commit, workspace)
    workspace_manifest = workspace / "Cargo.toml"
    if sha256(workspace_manifest) in {value.upper() for value in lock.get("superseded_manifest_sha256", [])}:
        shutil.copyfile(TRACKED_MANIFEST, workspace_manifest)
    if sha256(workspace_manifest) != lock["manifest_sha256"].upper():
        raise BuildError(
            f"The build manifest in {workspace} matches no known version of this recipe. "
            "Delete that folder and run again; it is recreated from the source cache."
        )
    assert_hash(workspace / "Cargo.lock", lock["sha256"], "Tracked DeepFilter lockfile")
    for model in MODELS:
        assert_hash(workspace / model, provenance["models"][model]["sha256"], model)

    tract_patch = workspace / "patches" / "tract-linalg"
    ensure_tract_archive(
        archive_cache, cargo_home, f"tract-linalg-{tract_version}.crate", tract_sha, tract["archive_url"]
    )
    extract_root = source_root / f"tract-linalg-{tract_version}-{commit}"
    extracted = extract_root / f"tract-linalg-{tract_version}"
    if not extracted.is_dir():
        extract_archive(archive_cache, extract_root)
    assert_hash(archive_cache, tract_sha, "tract-linalg source archive")
    if not extracted.is_dir():
        raise BuildError(f"tract-linalg archive did not contain {extracted}")
    tract_manifest = tract_patch / "Cargo.toml"
    if not tract_manifest.is_file():
        shutil.copytree(extracted, tract_patch, dirs_exist_ok=True)
        assert_hash(tract_manifest, tract["original_cargo_toml_sha256"], "Original tract-linalg manifest")
        text = replace_exactly(
            tract_manifest.read_text(encoding="utf-8"),
            r'^\[build-dependencies\.time\]\r?\nversion = ">=0\.3\.23, <0\.3\.42"\r?\n',
            "", 1, "unused tract-linalg time build dependency",
        )
        write_text(tract_manifest, text)
    assert_hash(tract_manifest, tract["patched_cargo_toml_sha256"], "Patched tract-linalg manifest")
    assert_hash(tract_patch / "build.rs", tract["build_rs_sha256"], "tract-linalg build script")
    assert_patch_tree(extracted, tract_patch, tract["patched_cargo_toml_sha256"])

    lib_manifest = workspace / "libDF/Cargo.toml"
    if sha256(lib_manifest) == upstream_files["libDF/Cargo.toml"]["sha256"].upper():
        text = lib_manifest.read_text(encoding="utf-8")
        text = replace_exactly(
            text,
            r'ndarray = \{ version = "\^0\.15", optional = true, features = \["serde"\] \}',
            'ndarray = { version = "^0.16", optional = true, features = ["serde"] }',
            1, "upstream ndarray 0.15 dependency in libDF",
        )
        text = replace_exactly(
            text,
            r'^\[dev-dependencies\]\r?\nrand = "0\.8"\r?\nrstest = "0\.19"\r?\nenv_logger = "0\.11"\r?\n'
            r'log = \{ version = "0\.4", features = \["std"\] \}\r?\n',
            "", 1, "upstream libDF dev-dependencies block",
        )
        write_text(lib_manifest, text)
    assert_hash(lib_manifest, upstream_files["libDF/Cargo.toml"]["patched_sha256"], "Patched libDF manifest")

    lib_source = workspace / "libDF/src/lib.rs"
    if sha256(lib_source) == upstream_files["libDF/src/lib.rs"]["sha256"].upper():
        text = replace_exactly(
            lib_source.read_text(encoding="utf-8"),
            r"^        \*s = x\.norm\(\) \* \(1\. - alpha\) \+ \*s \* alpha;\r?$",
            "        *s = (x.norm() * (1. - alpha) + *s * alpha).max(f32::MIN_POSITIVE);",
            2, "unit normalization state updates in libDF/src/lib.rs",
        )
        write_text(lib_source, text)
    assert_hash(lib_source, upstream_files["libDF/src/lib.rs"]["patched_sha256"], "Patched libDF normalization source")

    tract_source = workspace / "libDF/src/tract.rs"
    if sha256(tract_source) == upstream_files["libDF/src/tract.rs"]["sha256"].upper():
        text = tract_source.read_text(encoding="utf-8")
        found = text.count('m.symbol_table.sym("S")')
        if found != 3:
            raise BuildError(f"Expected three tract symbol_table calls, found {found}")
        write_text(tract_source, text.replace('m.symbol_table.sym("S")', 'm.symbols.sym("S")'))
    assert_hash(tract_source, upstream_files["libDF/src/tract.rs"]["patched_sha256"], "Patched libDF tract source")

    capi_source = workspace / "libDF/src/capi.rs"
    capi_patch = SUPPORT / "capi.rs"
    capi_patched = upstream_files["libDF/src/capi.rs"]["patched_sha256"]
    assert_hash(capi_patch, capi_patched, "Tracked abort-free C API source")
    if sha256(capi_source) == upstream_files["libDF/src/capi.rs"]["sha256"].upper():
        shutil.copyfile(capi_patch, capi_source)
    assert_hash(capi_source, capi_patched, "Patched libDF C API source")
    assert_workspace_source_tree(git, source_cache, commit, workspace)

    rustflags = (
        f"--remap-path-prefix={ROOT}=/audioforge --remap-path-prefix={cargo_home}=/cargo "
        "-C link-arg=-Wl,-z,relro,-z,now"
    )
    if not workspace_has_marker:
        write_text(marker, json.dumps({
            "schema_version": 2,
            "commit": commit,
            "tract_linalg_version": tract_version,
            "lock_sha256": sha256(workspace / "Cargo.lock"),
            "lib_manifest_sha256": sha256(lib_manifest),
            "lib_source_sha256": sha256(lib_source),
            "capi_source_sha256": sha256(capi_source),
            "tract_source_sha256": sha256(tract_source),
            "tract_manifest_sha256": sha256(tract_manifest),
            "rustflags": [
                "--remap-path-prefix=<project-root>=/audioforge",
                "--remap-path-prefix=<cargo-home>=/cargo",
                "-C link-arg=-Wl,-z,relro,-z,now",
            ],
        }, indent=2))

    cargo_target = workspace / "target"
    run(
        cargo, "build", "--manifest-path", workspace_manifest, "--locked",
        "--profile", recipe_build["profile"], "--package", recipe_build["crate"], "--lib",
        "--no-default-features", "--features", ",".join(recipe_build["features"]),
        "--target", TARGET, "--target-dir", cargo_target,
        env={**os.environ, "RUSTFLAGS": rustflags},
    )
    built = cargo_target / TARGET / recipe_build["profile"] / LIBRARY
    if not built.is_file():
        raise BuildError(f"Cargo succeeded but did not produce {built}")

    exports = set(captured(nm, "-D", "--defined-only", "--format=just-symbols", built).split())
    missing = [name for name in recipe_build["required_exports"] if name not in exports]
    if missing:
        raise BuildError(f"DeepFilter library is missing required exports: {', '.join(missing)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    attestation_path.parent.mkdir(parents=True, exist_ok=True)
    staged_output = output.with_name(f"{output.name}.build-{os.getpid()}")
    staged_attestation = attestation_path.with_name(f"{attestation_path.name}.build-{os.getpid()}")
    record = {
        "schema_version": 1,
        "kind": "audioforge.deepfilter.build",
        "output": {"name": output.name, "bytes": built.stat().st_size, "sha256": sha256(built)},
        "source": {
            "repository": repository,
            "commit": commit,
            "workspace_validation": (
                "Every file under libDF was matched to the pinned git blob; "
                "only the recorded source patches were accepted."
            ),
        },
        "recipe": {
            "files": {
                "python/tools/build_deepfilter.py": canonical_text_sha256(Path(__file__)),
                "python/tools/bounded_download.py": canonical_text_sha256(
                    Path(__file__).with_name("bounded_download.py")
                ),
                "build-support/deepfilter/Cargo.toml": canonical_text_sha256(TRACKED_MANIFEST),
                "build-support/deepfilter/Cargo.lock": canonical_text_sha256(TRACKED_LOCK),
                "build-support/deepfilter/capi.rs": sha256(capi_patch),
                "build-support/deepfilter/provenance.json": canonical_text_sha256(PROVENANCE),
                **{model: sha256(ROOT / model) for model in MODELS},
            },
            "tract_linalg_archive_sha256": tract_sha,
            "tract_linalg_patched_manifest_sha256": tract["patched_cargo_toml_sha256"],
            "tract_linalg_build_rs_sha256": tract["build_rs_sha256"],
            "rustflags": rustflags,
            "target": TARGET,
            "profile": recipe_build["profile"],
            "features": recipe_build["features"],
            "default_features": recipe_build["default_features"],
        },
        "toolchain": {
            "rustc": captured(rustc, "--version"),
            "cargo": captured(cargo, "--version"),
            "nm": captured(nm, "--version").splitlines()[0],
        },
        "abi": {"required_exports": recipe_build["required_exports"]},
        "reproducibility": (
            "The output digest is attested for this build. Bit-for-bit reproducibility is not a "
            "release gate because native linker metadata may vary between approved toolchains."
        ),
    }
    try:
        shutil.copyfile(built, staged_output)
        write_text(staged_attestation, json.dumps(record, indent=2))
        staged_output.replace(output)
        staged_attestation.replace(attestation_path)
    finally:
        staged_output.unlink(missing_ok=True)
        staged_attestation.unlink(missing_ok=True)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "target" / "deepfilter" / LIBRARY)
    parser.add_argument("--attestation", type=Path, help="defaults to <output>.provenance.json")
    parser.add_argument(
        "--allow-untested-rust",
        action="store_true",
        help="build with a Rust release other than the recipe's tested one (local builds only)",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    attestation = (args.attestation or output.with_name(f"{output.name}.provenance.json")).resolve()
    try:
        record = build(output, attestation, allow_untested_rust=args.allow_untested_rust)
    except (BuildError, OSError, KeyError, json.JSONDecodeError) as error:
        print(f"build_deepfilter.py failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(output), "attestation": str(attestation), **record["output"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
