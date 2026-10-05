"""Exercise the developer command without installing dependencies or building."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows developer entry point")
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def dev_project(tmp_path):
    shell = shutil.which("pwsh")
    if shell is None:
        pytest.skip("PowerShell 7 is required")
    project = tmp_path / "project with spaces"
    project.mkdir()
    shutil.copy2(REPO / "dev.ps1", project / "dev.ps1")
    (project / "licenses").mkdir()
    (project / "licenses/source-manifest.json").write_text(json.dumps({
        "entries": [{"kind": "cpython-source", "version": ".".join(map(str, sys.version_info[:3]))}],
    }))
    (project / "rust-toolchain.toml").write_text('[toolchain]\nchannel = "1.94.0"\n')
    (project / "requirements").mkdir()
    (project / "requirements/dev.txt").write_text("# The fake uv command never installs anything.\n")
    tools = project / "python/tools"
    tools.mkdir(parents=True)
    recorder = '''
import json, os, sys
from pathlib import Path
entry = dict(program="python", args=sys.orig_argv[1:], cwd=os.getcwd(),
             environment={key: os.environ.get(key) for key in
             ("VIRTUAL_ENV", "PYO3_PYTHON", "ORT_LIB_LOCATION", "RUSTUP_TOOLCHAIN",
              "AUDIOFORGE_ENABLE_DEEPFILTER", "PATH")})
with open(os.environ["DEV_TEST_LOG"], "a", encoding="utf-8") as output:
    output.write(json.dumps(entry) + "\\n")
'''
    for name in ("fetch_release_assets", "verify_release_assets"):
        (tools / f"{name}.py").write_text(
            recorder + f'raise SystemExit(23 if os.environ.get("DEV_TEST_FAIL") == "{name}" else 0)\n'
        )
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text('''
import os, sys
if "-m" in sys.orig_argv:
    module = sys.orig_argv[sys.orig_argv.index("-m") + 1]
    if module in {"maturin", "pytest", "pip", "mic_eq"}:
''' + "\n".join("        " + line for line in recorder.strip().splitlines()) + '''
        os._exit(17 if os.environ.get("DEV_TEST_FAIL") == module else 0)
''')
    create = tmp_path / "create_test_environment.py"
    create.write_text('''
import os, shutil, sys
from pathlib import Path
root = Path(sys.argv[1])
(root / "Scripts").mkdir(parents=True)
shutil.copy2(sys.executable, root / "Scripts/python.exe")
(root / "pyvenv.cfg").write_text(f"home = {sys.base_prefix}\\ninclude-system-site-packages = false\\n")
site = root / "Lib/site-packages"
site.mkdir(parents=True)
shutil.copy2(os.environ["DEV_TEST_SITECUSTOMIZE"], site / "sitecustomize.py")
(site / "mic_eq.py").write_text("CORE_AVAILABLE = True\\n")
''')
    wrapper = tmp_path / "invoke_test.ps1"
    wrapper.write_text(r'''
param([string]$Project, [string]$Command, [string]$Venv, [string]$Python, [string]$TestPath, [switch]$DryRun)
$ErrorActionPreference = "Stop"
function Record-Call([string]$Program, [object[]]$Arguments) {
    @{ program=$Program; args=@($Arguments); cwd=$PWD.Path } | ConvertTo-Json -Compress |
        Add-Content -LiteralPath $env:DEV_TEST_LOG -Encoding utf8
}
function uv {
    Record-Call "uv" $args
    if ($env:DEV_TEST_FAIL -eq "uv") { $global:LASTEXITCODE = 19; return }
    if ($args[0] -eq "venv") { & $env:DEV_TEST_PYTHON $env:DEV_TEST_CREATE $args[-1] }
    else { $global:LASTEXITCODE = 0 }
}
function rustup {
    Record-Call "rustup" $args
    $global:LASTEXITCODE = 0
    if ($args[-1] -eq "--version") { "rustc 1.94.0 (test fixture)" }
}
function gh { }
function 7z { }
function cl { }
$names = @("PATH", "VIRTUAL_ENV", "PYO3_PYTHON", "ORT_LIB_LOCATION", "ORT_PREFER_DYNAMIC_LINK",
           "RUSTUP_TOOLCHAIN", "PYTHONDONTWRITEBYTECODE", "AUDIOFORGE_ENABLE_DEEPFILTER")
$before = @{}
foreach ($name in $names) { $before[$name] = [Environment]::GetEnvironmentVariable($name, "Process") }
$originalLocation = $PWD.Path
& (Join-Path $Project "dev.ps1") $Command -VenvPath $Venv -PythonPath $Python -TestPath $TestPath -DryRun:$DryRun
$code = $LASTEXITCODE
foreach ($name in $names) {
    if ($before[$name] -cne [Environment]::GetEnvironmentVariable($name, "Process")) {
        throw "Environment was not restored: $name"
    }
}
if ($PWD.Path -ne $originalLocation) { throw "Working directory was not restored" }
exit $code
''')
    log = tmp_path / "external-calls.jsonl"
    env = {
        "DEV_TEST_LOG": str(log), "DEV_TEST_PYTHON": sys.executable,
        "DEV_TEST_CREATE": str(create), "DEV_TEST_SITECUSTOMIZE": str(sitecustomize),
        "PYTHONDONTWRITEBYTECODE": "1", "AUDIOFORGE_ENABLE_DEEPFILTER": "0",
    }
    return {"project": project, "shell": shell, "wrapper": wrapper, "env": env, "log": log,
            "venv": project / "environment with spaces", "create": create}


def run_dev(fixture, command, *, dry_run=False, test_path="", fail=""):
    env = os.environ | fixture["env"] | {"DEV_TEST_FAIL": fail}
    arguments = [fixture["shell"], "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                 "-File", str(fixture["wrapper"]), "-Project", str(fixture["project"]),
                 "-Command", command, "-Venv", str(fixture["venv"]), "-Python", sys.executable]
    if test_path:
        arguments += ["-TestPath", test_path]
    if dry_run:
        arguments.append("-DryRun")
    return subprocess.run(arguments, cwd=fixture["project"].parent, env=env,
                          capture_output=True, text=True, timeout=30)


def create_environment(fixture):
    subprocess.run([sys.executable, str(fixture["create"]), str(fixture["venv"])],
                   env=os.environ | fixture["env"], check=True)


def calls(fixture):
    return [json.loads(line) for line in fixture["log"].read_text(encoding="utf-8-sig").splitlines()]


def files(root):
    return {path.relative_to(root): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def test_bootstrap_reuses_pins_and_preserves_existing_environment(dev_project):
    first = run_dev(dev_project, "bootstrap")
    assert first.returncode == 0, first.stdout + first.stderr
    sentinel = dev_project["venv"] / "unrelated-user-file.txt"
    sentinel.write_text("preserve me")
    config = (dev_project["venv"] / "pyvenv.cfg").read_bytes()
    second = run_dev(dev_project, "bootstrap")
    assert second.returncode == 0, second.stdout + second.stderr
    commands = calls(dev_project)
    environments = [item for item in commands if item["program"] == "uv" and item["args"][0] == "venv"]
    assert len(environments) == 1
    assert environments[0]["args"][-1] == str(dev_project["venv"])
    assert "--no-python-downloads" in environments[0]["args"]
    assert "--no-managed-python" in environments[0]["args"]
    installs = [item for item in commands if item["program"] == "uv" and item["args"][0] == "pip"]
    assert len(installs) == 2
    assert all("--require-hashes" in item["args"] and "requirements/dev.txt" in item["args"] for item in installs)
    assert sentinel.read_text() == "preserve me"
    assert (dev_project["venv"] / "pyvenv.cfg").read_bytes() == config
    builds = [item for item in commands if "maturin" in item["args"]]
    assert len(builds) == 2
    assert builds[0]["args"] == ["-m", "maturin", "develop", "--release", "--locked"]
    assert builds[0]["environment"]["VIRTUAL_ENV"] == str(dev_project["venv"])
    assert builds[0]["environment"]["PYO3_PYTHON"] == str(dev_project["venv"] / "Scripts/python.exe")
    assert str(Path(sys.base_prefix)) in builds[0]["environment"]["PATH"].split(";")


def test_bootstrap_preserves_incomplete_environment(dev_project):
    dev_project["venv"].mkdir()
    (dev_project["venv"] / "keep.txt").write_text("user data")
    before = files(dev_project["project"])
    result = run_dev(dev_project, "bootstrap")
    assert result.returncode == 1
    assert "Incomplete environment" in result.stderr
    assert files(dev_project["project"]) == before
    assert not any(item["program"] == "uv" for item in calls(dev_project))


@pytest.mark.parametrize("command", ["bootstrap", "run", "test", "doctor"])
def test_dry_run_does_not_execute_or_mutate(dev_project, command):
    before = files(dev_project["project"])
    result = run_dev(dev_project, command, dry_run=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DRY RUN" in result.stdout
    assert not dev_project["log"].exists()
    assert files(dev_project["project"]) == before


def test_doctor_is_offline_and_read_only(dev_project):
    create_environment(dev_project)
    before = files(dev_project["project"])
    result = run_dev(dev_project, "doctor")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "release interpreter/source provenance" in result.stdout
    assert files(dev_project["project"]) == before
    commands = calls(dev_project)
    assert not any(item["program"] == "uv" or "maturin" in item["args"] for item in commands)
    assert any(item["args"] == ["-B", "-m", "pip", "check"] for item in commands)


@pytest.mark.parametrize(("stage", "code"), [("uv", 19), ("fetch_release_assets", 23), ("verify_release_assets", 23), ("maturin", 17)])
def test_bootstrap_propagates_failures_without_later_steps(dev_project, stage, code):
    create_environment(dev_project)
    result = run_dev(dev_project, "bootstrap", fail=stage)
    assert result.returncode == code, result.stdout + result.stderr
    last = calls(dev_project)[-1]
    assert stage in (last["program"] + " " + " ".join(last["args"]))


def test_test_path_builds_native_then_runs_only_requested_python_tests(dev_project):
    create_environment(dev_project)
    result = run_dev(dev_project, "test", test_path="python/tests/a focused test.py")
    assert result.returncode == 0, result.stdout + result.stderr
    commands = calls(dev_project)
    assert not any("cargo" in item["args"] for item in commands)
    assert commands[-2]["args"] == ["-m", "maturin", "develop", "--release", "--locked"]
    assert commands[-1]["args"] == ["-m", "pytest", "python/tests/a focused test.py", "-q"]


def test_default_test_runs_existing_rust_and_python_suites(dev_project):
    create_environment(dev_project)
    result = run_dev(dev_project, "test")
    assert result.returncode == 0, result.stdout + result.stderr
    commands = calls(dev_project)
    assert commands[-3]["args"] == ["-m", "maturin", "develop", "--release", "--locked"]
    assert commands[-2]["args"] == ["run", "1.94.0", "cargo", "test", "--locked", "-p", "mic_eq_core"]
    assert commands[-1]["args"] == ["-m", "pytest", "python/tests", "-q"]


def test_missing_environment_does_not_run_or_create_one(dev_project):
    result = run_dev(dev_project, "run")
    assert result.returncode == 1
    assert "Environment Python is missing" in result.stderr
    assert not dev_project["venv"].exists()
    assert not dev_project["log"].exists()


def test_run_enables_deepfilter_only_after_asset_verification(dev_project):
    create_environment(dev_project)
    result = run_dev(dev_project, "run")
    assert result.returncode == 0, result.stdout + result.stderr
    commands = calls(dev_project)
    assert commands[-2]["args"] == ["-B", "python/tools/verify_release_assets.py"]
    assert commands[-1]["args"] == ["-m", "mic_eq"]
    assert commands[-1]["environment"]["AUDIOFORGE_ENABLE_DEEPFILTER"] == "1"


def test_wrong_python_pin_fails_before_environment_creation(dev_project):
    manifest = dev_project["project"] / "licenses/source-manifest.json"
    manifest.write_text('{"entries":[{"kind":"cpython-source","version":"3.0.0"}]}')
    result = run_dev(dev_project, "bootstrap")
    assert result.returncode == 1
    assert "Use existing CPython 3.0.0 x64" in result.stderr
    assert not dev_project["venv"].exists()
    assert not any(item["program"] == "uv" for item in calls(dev_project))
