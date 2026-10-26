"""Executable checks for the release-promotion and dispatch gates."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest
import yaml


WORKFLOW = Path(__file__).parents[2] / ".github" / "workflows" / "release-promote.yml"


def _origin_gate_script() -> str:
    source = WORKFLOW.read_text(encoding="utf-8")
    start = source.index("          $canonicalTag =")
    end = source.index("\n      - name: Install pinned validation runtime", start)
    return textwrap.dedent(source[start:end])


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (None, None),
        (("databaseId", 999), "identity mismatch"),
        (("workflowName", "Other workflow"), "expected 'Release package'"),
        (("conclusion", "failure"), "not a successful completed"),
        (("headSha", "b" * 40), "did not build the release tag commit"),
        (("event", "schedule"), "unsupported event"),
        (("event", "push"), "unsupported event"),
        (("tag_type", "commit"), "must be an annotated tag"),
        (("head", "b" * 40), "checkout is not the release tag commit"),
        (("attempt", 0), "invalid attempt number"),
    ],
)
def test_promotion_origin_gate_executes_commit_and_run_binding(
    tmp_path: Path, mutation: tuple[str, object] | None, expected_error: str | None
) -> None:
    shell = shutil.which("pwsh")
    if shell is None:
        pytest.skip("PowerShell is required for the Windows promotion workflow")

    commit = "a" * 40
    run = {
        "databaseId": 123,
        "workflowName": "Release package",
        "event": "workflow_dispatch",
        "status": "completed",
        "conclusion": "success",
        "headSha": commit,
        "attempt": 1,
    }
    tag_type = "tag"
    head = commit
    if mutation and mutation[0] == "tag_type":
        tag_type = str(mutation[1])
    elif mutation and mutation[0] == "head":
        head = str(mutation[1])
    elif mutation:
        run[mutation[0]] = mutation[1]
    output = tmp_path / "github-output"
    setup = textwrap.dedent(
        """
        $ErrorActionPreference = 'Stop'
        function python {
          param([Parameter(ValueFromRemainingArguments=$true)][string[]]$CommandArgs)
          $global:LASTEXITCODE = 0
          if ($CommandArgs[1] -eq 'tag') { $env:TEST_TAG; return }
          if ($CommandArgs[1] -eq 'prerelease') { 'false'; return }
          throw "unexpected python command"
        }
        function git {
          param([Parameter(ValueFromRemainingArguments=$true)][string[]]$CommandArgs)
          $global:LASTEXITCODE = 0
          if ($CommandArgs[0] -eq 'fetch') { return }
          if ($CommandArgs[0] -eq 'cat-file') { $env:TEST_TAG_TYPE; return }
          if ($CommandArgs[0] -eq 'rev-parse') { $env:TEST_HEAD; return }
          $env:TEST_COMMIT
        }
        function gh {
          param([Parameter(ValueFromRemainingArguments=$true)][string[]]$CommandArgs)
          $global:LASTEXITCODE = 0
          $env:TEST_RUN_JSON
        }
        """
    )
    env = os.environ.copy()
    env.update(
        {
            "RELEASE_TAG": "v1.12.0",
            "CANDIDATE_RUN_ID": "123",
            "GITHUB_REPOSITORY": "owner/repo",
            "GITHUB_OUTPUT": str(output),
            "TEST_TAG": "v1.12.0",
            "TEST_COMMIT": commit,
            "TEST_HEAD": head,
            "TEST_TAG_TYPE": tag_type,
            "TEST_RUN_JSON": json.dumps(run),
        }
    )
    result = subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-Command", setup + _origin_gate_script()],
        env=env,
        capture_output=True,
        text=True,
    )
    diagnostics = result.stdout + result.stderr
    if expected_error is None:
        assert result.returncode == 0, diagnostics
        assert "candidate_run_id=123" in output.read_text(encoding="utf-8")
    else:
        assert result.returncode != 0, diagnostics
        assert expected_error in diagnostics


WORKFLOWS = WORKFLOW.parent
# (workflow, job): every job that runs privileged or physical work starts with
# the authorization step.
DISPATCH_JOBS = [
    ("release-promote.yml", "verify-candidate"),
    ("release-promote.yml", "publish-release"),
    ("release-hardware-qualify.yml", "qualify-hardware"),
    ("release-hardware-matrix.yml", "assemble"),
]


def _job(workflow: str, job: str) -> dict:
    return yaml.safe_load((WORKFLOWS / workflow).read_text(encoding="utf-8"))["jobs"][job]


def _shell_for(job: dict) -> str:
    """The job's own shell, so the hardware job runs under Windows PowerShell."""
    shell = shutil.which(job["defaults"]["run"]["shell"])
    if shell is None:
        pytest.skip("PowerShell is required for the dispatch authorization")
    return shell


@pytest.mark.parametrize(("workflow", "job_id"), DISPATCH_JOBS)
@pytest.mark.parametrize(
    ("ref", "role", "expected_error"),
    [
        ("refs/heads/master", "admin", None),
        ("refs/heads/master", "maintain", None),
        ("refs/heads/master", "write", "needs the admin or maintain role"),
        ("refs/heads/master", "triage", "needs the admin or maintain role"),
        ("refs/heads/master", "", "needs the admin or maintain role"),
        ("refs/heads/master", "api-error", "api unavailable"),
        ("refs/heads/feature/x", "admin", "only from refs/heads/master"),
        ("refs/heads/master-copy", "admin", "only from refs/heads/master"),
        ("refs/tags/v1.19.0", "admin", "only from refs/heads/master"),
    ],
)
def test_dispatch_authorization_needs_master_and_a_maintainer(
    tmp_path: Path, workflow: str, job_id: str, ref: str, role: str, expected_error: str | None
) -> None:
    job = _job(workflow, job_id)
    shell = _shell_for(job)
    first = job["steps"][0]
    assert first["name"] == "Authorize the dispatch"
    uri_file = tmp_path / "uri.txt"
    setup = """
$ErrorActionPreference = 'Stop'
function Invoke-RestMethod {
  param($Headers, $Uri)
  if ($env:TEST_ROLE -eq 'api-error') { throw 'api unavailable' }
  if ($Headers.Authorization -ne 'Bearer test-token') { throw 'unexpected token header' }
  Set-Content -LiteralPath $env:TEST_URI_FILE -Value $Uri
  [pscustomobject]@{ role_name = $env:TEST_ROLE }
}
"""
    result = subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-Command", setup + first["run"]],
        env={
            **os.environ,
            "GITHUB_REF": ref,
            "GITHUB_API_URL": "https://api.example.test",
            "GITHUB_REPOSITORY": "owner/repo",
            "GH_TOKEN": "test-token",
            "DISPATCHER": "some user",
            "TEST_ROLE": role,
            "TEST_URI_FILE": str(uri_file),
        },
        capture_output=True,
        text=True,
    )
    diagnostics = result.stdout + result.stderr
    if expected_error is None:
        assert result.returncode == 0, diagnostics
        assert uri_file.read_text(encoding="utf-8").strip() == (
            "https://api.example.test/repos/owner/repo/collaborators/some%20user/permission"
        )
    else:
        assert result.returncode != 0, diagnostics
        assert expected_error in diagnostics
        if "refs/heads/master" in expected_error:
            assert not uri_file.exists(), "the token must not be sent for a refused ref"


def _verified_release(tmp_path: Path, tag: str = "v1.19.0") -> tuple[dict[str, str], dict[str, Path]]:
    directory = tmp_path / "verified-release"
    directory.mkdir()
    files = {
        "archive": directory / f"AudioForge-{tag}-win64-ultra.7z",
        "installer": directory / f"AudioForge-{tag}-win64.msi",
        "source_archive": directory / f"AudioForge-{tag}-source.7z",
        "evidence_archive": directory / f"AudioForge-{tag}-evidence.7z",
    }
    for kind, path in files.items():
        path.write_bytes(f"{kind} bytes".encode())
    digests = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files.values()}
    sums = directory / f"AudioForge-{tag}-SHA256SUMS.txt"
    sums.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in digests.items()), encoding="ascii"
    )
    files["checksums"] = sums
    digests[sums.name] = hashlib.sha256(sums.read_bytes()).hexdigest()
    return digests, files


def _recheck(tmp_path: Path, digests: dict[str, str], approved: str) -> subprocess.CompletedProcess:
    job = _job("release-promote.yml", "publish-release")
    shell = _shell_for(job)
    step = next(s for s in job["steps"] if s.get("name") == "Re-check the verified assets")
    return subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-Command", "$ErrorActionPreference = 'Stop'\n" + step["run"]],
        cwd=tmp_path,
        env={
            **os.environ,
            "EXPECTED_SHA256": approved,
            "RELEASE_TAG": "v1.19.0",
            "ASSET_HASHES": json.dumps(digests),
            "GITHUB_OUTPUT": str(tmp_path / "github-output"),
        },
        capture_output=True,
        text=True,
    )


def test_publish_recheck_accepts_the_files_the_verification_job_staged(tmp_path: Path) -> None:
    digests, files = _verified_release(tmp_path)

    result = _recheck(tmp_path, digests, digests[files["archive"].name].upper())

    assert result.returncode == 0, result.stdout + result.stderr
    outputs = dict(
        line.split("=", 1)
        for line in (tmp_path / "github-output").read_text(encoding="utf-8").splitlines()
    )
    assert set(outputs) == set(files)
    assert Path(outputs["archive"]).name == files["archive"].name


@pytest.mark.parametrize(
    ("tamper", "expected_error"),
    [
        (lambda files, digests: files["installer"].write_bytes(b"swapped"), "differs from the hash recorded"),
        (lambda files, digests: files["archive"].write_bytes(b"swapped"), "differs from the hash recorded"),
        (lambda files, digests: files["evidence_archive"].unlink(), "does not hold exactly"),
        (
            lambda files, digests: (files["archive"].parent / "AudioForge-extra.7z").write_bytes(b"x"),
            "does not hold exactly",
        ),
        (lambda files, digests: digests.update({"AudioForge-extra.7z": "0" * 64}), "does not hold exactly"),
    ],
)
def test_publish_recheck_refuses_files_that_changed_between_jobs(
    tmp_path: Path, tamper, expected_error: str
) -> None:
    digests, files = _verified_release(tmp_path)
    approved = digests[files["archive"].name]
    tamper(files, digests)

    result = _recheck(tmp_path, digests, approved)

    assert result.returncode != 0
    assert expected_error in result.stdout + result.stderr


def test_publish_recheck_refuses_files_named_for_another_release(tmp_path: Path) -> None:
    digests, files = _verified_release(tmp_path, tag="v1.18.2")

    result = _recheck(tmp_path, digests, digests[files["archive"].name])

    assert result.returncode != 0
    assert "five release files of v1.19.0" in result.stdout + result.stderr


def test_publish_recheck_refuses_an_archive_other_than_the_approved_one(tmp_path: Path) -> None:
    digests, _files = _verified_release(tmp_path)

    result = _recheck(tmp_path, digests, "f" * 64)

    assert result.returncode != 0
    assert "approved by expected_archive_sha256" in result.stdout + result.stderr


def test_publish_recheck_refuses_a_malformed_approval(tmp_path: Path) -> None:
    digests, _files = _verified_release(tmp_path)

    result = _recheck(tmp_path, digests, "not-a-digest")

    assert result.returncode != 0
    assert "64 hexadecimal characters" in result.stdout + result.stderr


def test_publish_recheck_refuses_a_checksum_list_that_disagrees(tmp_path: Path) -> None:
    digests, files = _verified_release(tmp_path)
    approved = digests[files["archive"].name]
    files["checksums"].write_text(
        files["checksums"].read_text(encoding="ascii").replace(approved, "0" * 64), encoding="ascii"
    )
    digests[files["checksums"].name] = hashlib.sha256(files["checksums"].read_bytes()).hexdigest()

    result = _recheck(tmp_path, digests, approved)

    assert result.returncode != 0
    assert "SHA256SUMS does not list the verified assets" in result.stdout + result.stderr


TAG = "v1.19.0"
SEVEN_ZIP = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "7-Zip" / "7z.exe"


def _run_step(job_id: str, step_name: str, cwd: Path, env: dict[str, str], prelude: str = ""):
    """Run one step of verify-candidate in pwsh the way the runner would."""
    job = _job("release-promote.yml", job_id)
    shell = _shell_for(job)
    step = next(s for s in job["steps"] if s.get("name") == step_name)
    (cwd / "runner-temp").mkdir(exist_ok=True)
    return subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-Command",
         "$ErrorActionPreference = 'Stop'\n" + prelude + step["run"]],
        cwd=cwd,
        env={
            **os.environ,
            **env,
            "GITHUB_OUTPUT": str(cwd / "github-output"),
            "RUNNER_TEMP": str(cwd / "runner-temp"),
        },
        capture_output=True,
        text=True,
    )


def _outputs(cwd: Path) -> dict[str, str]:
    return dict(
        line.split("=", 1)
        for line in (cwd / "github-output").read_text(encoding="utf-8").splitlines()
    )


def _candidate_artifact(cwd: Path) -> tuple[Path, dict[str, str]]:
    candidate = cwd / "candidate"
    candidate.mkdir()
    archive = f"AudioForge-{TAG}-win64-ultra.7z"
    installer = f"AudioForge-{TAG}-win64.msi"
    source = f"AudioForge-{TAG}-source.7z"
    names = [
        archive, f"{archive}.sha256", f"{archive}.manifest.json", f"{archive}.metadata.json",
        source, f"{source}.sha256", f"{source}.metadata.json",
        installer, f"{installer}.sha256", f"{installer}.manifest.json", f"{installer}.metadata.json",
        f"AudioForge-{TAG}-deepfilter.provenance.json",
    ]
    for name in names:
        (candidate / name).write_bytes(f"bytes of {name}".encode())
    (cwd / "validation").mkdir()
    (cwd / "validation" / "release-qualification.json").write_text("{}", encoding="utf-8")
    return candidate, {"archive": archive, "installer": installer, "source": source}


def test_the_candidate_files_are_found_by_their_exact_names(tmp_path: Path) -> None:
    _candidate_artifact(tmp_path)
    stray = tmp_path / "candidate" / "AudioForge-v1.18.2-win64-ultra.7z"
    stray.write_bytes(b"another release")

    result = _run_step("verify-candidate", "Locate the candidate files", tmp_path, {"RELEASE_TAG": TAG})

    assert result.returncode == 0, result.stdout + result.stderr
    outputs = _outputs(tmp_path)
    assert len(outputs) == 12
    assert Path(outputs["archive"]).name == f"AudioForge-{TAG}-win64-ultra.7z"
    assert all(Path(path).parent == tmp_path / "candidate" for path in outputs.values())


@pytest.mark.parametrize(
    "missing", ["-win64-ultra.7z", "-win64-ultra.7z.metadata.json", "-source.7z.sha256", "-win64.msi", "-deepfilter.provenance.json"]
)
def test_a_candidate_without_one_of_its_files_is_refused(tmp_path: Path, missing: str) -> None:
    candidate, _names = _candidate_artifact(tmp_path)
    next(candidate.glob(f"*{missing}")).unlink()

    result = _run_step("verify-candidate", "Locate the candidate files", tmp_path, {"RELEASE_TAG": TAG})

    assert result.returncode != 0
    assert "must contain exactly one" in result.stdout + result.stderr


def _stage(tmp_path: Path, approved: str | None = None) -> subprocess.CompletedProcess:
    candidate, names = _candidate_artifact(tmp_path)
    located = _run_step("verify-candidate", "Locate the candidate files", tmp_path, {"RELEASE_TAG": TAG})
    assert located.returncode == 0, located.stdout + located.stderr
    outputs = _outputs(tmp_path)
    (tmp_path / "github-output").unlink()
    archive_hash = hashlib.sha256((candidate / names["archive"]).read_bytes()).hexdigest()
    fake_python = """
function python {
  param([Parameter(ValueFromRemainingArguments=$true)][string[]]$CommandArgs)
  $global:LASTEXITCODE = 0
  '1.19.0'
}
"""
    return _run_step(
        "verify-candidate",
        "Stage the release assets",
        tmp_path,
        {
            "RELEASE_TAG": TAG,
            "EXPECTED_SHA256": approved or archive_hash,
            **{key.upper(): value for key, value in outputs.items()},
        },
        fake_python,
    )


@pytest.mark.skipif(not SEVEN_ZIP.is_file(), reason="7-Zip is required to build the evidence archive")
def test_staged_assets_are_the_exact_five_files_and_pass_the_publish_recheck(tmp_path: Path) -> None:
    staged = _stage(tmp_path)

    assert staged.returncode == 0, staged.stdout + staged.stderr
    outputs = _outputs(tmp_path)
    recorded = json.loads(outputs["asset_hashes"])
    assert sorted(recorded) == sorted([
        f"AudioForge-{TAG}-win64-ultra.7z",
        f"AudioForge-{TAG}-win64.msi",
        f"AudioForge-{TAG}-source.7z",
        f"AudioForge-{TAG}-evidence.7z",
        f"AudioForge-{TAG}-SHA256SUMS.txt",
    ])
    assert outputs["public_version"] == "1.19.0"
    for name, digest in recorded.items():
        assert hashlib.sha256((tmp_path / "verified-release" / name).read_bytes()).hexdigest() == digest

    # The publish job accepts exactly what the verification job staged.
    rechecked = subprocess.run(
        [
            _shell_for(_job("release-promote.yml", "publish-release")), "-NoProfile", "-NonInteractive", "-Command",
            "$ErrorActionPreference = 'Stop'\n"
            + next(
                s for s in _job("release-promote.yml", "publish-release")["steps"]
                if s.get("name") == "Re-check the verified assets"
            )["run"],
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "EXPECTED_SHA256": recorded[f"AudioForge-{TAG}-win64-ultra.7z"],
            "RELEASE_TAG": TAG,
            "ASSET_HASHES": outputs["asset_hashes"],
            "GITHUB_OUTPUT": str(tmp_path / "recheck-output"),
        },
        capture_output=True,
        text=True,
    )
    assert rechecked.returncode == 0, rechecked.stdout + rechecked.stderr


@pytest.mark.skipif(not SEVEN_ZIP.is_file(), reason="7-Zip is required to build the evidence archive")
def test_nothing_is_staged_for_an_archive_other_than_the_approved_one(tmp_path: Path) -> None:
    staged = _stage(tmp_path, approved="f" * 64)

    assert staged.returncode != 0
    assert "approved by expected_archive_sha256" in staged.stdout + staged.stderr
    assert not (tmp_path / "verified-release").exists()
