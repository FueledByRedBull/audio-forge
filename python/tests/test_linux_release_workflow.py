"""The Linux half of the release workflows is held to the same rules as the Windows half.

Neither `package-linux` nor `verify-linux-candidate` could run on a hosted runner
before the first dispatch, so check_workflows.py has to fail a workflow that would
break there. Each test edits the parsed real workflow and expects that failure.
"""

from __future__ import annotations

import copy
import json

import pytest

import check_versions
import check_workflows


def _workflow(name: str = "release-package.yml") -> dict:
    return check_workflows.yaml.safe_load(
        (check_workflows.WORKFLOW_DIR / name).read_text(encoding="utf-8")
    )


def _step(job: dict, name: str) -> dict:
    return next(step for step in job["steps"] if step.get("name") == name)


def _drop(job: dict, name: str) -> None:
    job["steps"] = [step for step in job["steps"] if step.get("name") != name]


def _release_errors(mutate) -> str:
    workflow = _workflow()
    mutate(workflow["jobs"])
    errors: list[str] = []
    check_workflows._check_release_jobs(workflow, errors)
    check_workflows._check_tag_checkouts("release-package.yml", workflow, errors)
    return "; ".join(errors)


def _gate_errors(name: str, mutate) -> str:
    workflow = _workflow(name)
    mutate(workflow["jobs"])
    errors: list[str] = []
    check_workflows._check_required_gates(name, check_workflows.yaml.safe_dump(workflow), errors)
    return "; ".join(errors)


def _move_before(job: dict, name: str, before: str) -> None:
    step = _step(job, name)
    job["steps"].remove(step)
    job["steps"].insert(job["steps"].index(_step(job, before)), step)


BIND = "Bind existing release tag to checked-out commit"
STAGE = "Stage the Linux release files"
SMOKE = "Smoke-test the packaged app"


def test_the_committed_linux_release_jobs_pass():
    assert _release_errors(lambda jobs: None) == ""
    for name in ("release-package.yml", "release-promote.yml", "release-hardware-qualify-linux.yml", "ci.yml"):
        errors: list[str] = []
        source = (check_workflows.WORKFLOW_DIR / name).read_text(encoding="utf-8")
        check_workflows._check_required_gates(name, source, errors)
        assert errors == [], name


def test_the_cache_rules_cover_libdf_so_and_the_linux_build_job():
    assert "libdf.so" in check_workflows.RELEASE_UNCACHEABLE

    def restore(path):
        def mutate(jobs):
            jobs["package-linux"]["steps"].append(
                {"uses": "actions/cache@" + "0" * 40, "with": {"path": path, "key": "k"}}
            )

        return mutate

    assert "job package-linux must not restore libdf.so" in _release_errors(restore("libdf.so"))
    assert "job package-linux must not restore target/debug" in _release_errors(restore("target/debug"))
    assert "job linux-tests must not restore libdf.so" in _release_errors(
        lambda jobs: jobs["linux-tests"]["steps"].append(
            {"uses": "actions/cache@" + "0" * 40, "with": {"path": "libdf.so", "key": "k"}}
        )
    )


def test_package_linux_binds_an_existing_tag_to_its_checkout_like_package_windows():
    binding = _step(_workflow()["jobs"]["package-linux"], BIND)
    assert binding["if"] == check_workflows.RELEASE_TAG_BINDING_CONDITION

    missing = _release_errors(lambda jobs: _drop(jobs["package-linux"], BIND))
    assert "package-linux must bind an existing release tag to the checked-out commit" in missing
    assert "package-linux: checkout of" in missing and "existing release tag is the checked-out commit" in missing
    # The Windows job's own binding is still enforced separately.
    assert "package-windows must bind" not in missing

    def conditional(jobs):
        _step(jobs["package-linux"], BIND)["if"] = "github.ref_type == 'tag'"

    assert "must fetch the existing release tag and compare its commit" in _release_errors(conditional)
    assert "must bind an existing release tag" in _release_errors(
        lambda jobs: _move_before(jobs["package-linux"], "Hydrate and verify corresponding source", BIND)
    )
    assert "package-windows must bind" in _release_errors(lambda jobs: _drop(jobs["package-windows"], "Bind existing release tag to checked-out commit"))


def test_package_linux_builds_the_ref_every_other_release_job_builds():
    def other_ref(jobs):
        for step in jobs["package-linux"]["steps"]:
            if str(step.get("uses", "")).startswith("actions/checkout@"):
                step["with"]["ref"] = "${{ github.sha }}"

    assert "package-linux must check out ref ${{ inputs.release_tag || github.sha }}" in _release_errors(other_ref)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda job: _move_before(job, STAGE, SMOKE),
            "package-linux must hydrate source, fetch the runtime assets, build the bundle",
        ),
        (
            lambda job: _move_before(job, "Download and verify pinned release runtime assets", "Hydrate and verify corresponding source"),
            "package-linux must hydrate source, fetch the runtime assets, build the bundle",
        ),
        (
            lambda job: _drop(job, "Stage the packaged native module for the Linux test jobs"),
            "package-linux must hydrate source, fetch the runtime assets, build the bundle",
        ),
        (
            lambda job: _drop(job, "Record the bundle paths for review"),
            "package-linux must hydrate source, fetch the runtime assets, build the bundle",
        ),
    ],
)
def test_package_linux_steps_keep_their_order(mutate, message):
    assert message in _release_errors(lambda jobs: mutate(jobs["package-linux"]))


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        # The failure that made every Linux candidate refuse as a dirty tree.
        ('stage="$RUNNER_TEMP/linux-release"', "stage=linux-release", "must stage the release files under $RUNNER_TEMP"),
        ("--baseline \"$baseline\"", "", "against evaluation/release-bundle-path-baseline-linux.json"),
        ("--native-attestation \"$attestation\"", "", "and the libdf.so attestation"),
        ("baseline=evaluation/release-bundle-path-baseline-linux.json", "baseline=evaluation/other.json", "against evaluation/release-bundle-path-baseline-linux.json"),
        ('for archive in "$TARBALL" "$APPIMAGE"; do', 'for archive in "$TARBALL"; do', "tarball and the AppImage"),
    ],
)
def test_package_linux_stages_outside_the_tree_and_binds_both_archives(old, new, message):
    def mutate(jobs):
        step = _step(jobs["package-linux"], STAGE)
        assert old in step["run"]
        step["run"] = step["run"].replace(old, new)

    assert message in _release_errors(mutate)


def test_package_linux_uploads_the_release_files_from_outside_the_tree():
    def mutate(jobs):
        for step in jobs["package-linux"]["steps"]:
            if str(step.get("uses", "")).startswith("actions/upload-artifact@") and str(
                step["with"]["name"]
            ).startswith("audioforge-release-linux-${{"):
                step["with"]["path"] = "linux-release"

    assert "must upload the release files from the runner temp directory" in _release_errors(mutate)


def test_package_linux_pins_an_appimage_runtime_whose_source_the_manifest_holds():
    def mutate(jobs):
        env = jobs["package-linux"]["env"]
        env["APPIMAGE_RUNTIME_URL"] = env["APPIMAGE_RUNTIME_URL"].replace("20251108", "20260101")

    assert "has no appimage-runtime-type2-runtime-20260101 entry" in _release_errors(mutate)
    assert "must pin the AppImage runtime to a type2-runtime release asset" in _release_errors(
        lambda jobs: jobs["package-linux"]["env"].update(APPIMAGE_RUNTIME_URL="https://example.test/runtime")
    )


def test_package_linux_outputs_what_the_linux_test_jobs_need():
    assert "package-linux must output the release tag" in _release_errors(
        lambda jobs: jobs["package-linux"]["outputs"].pop("native_sha256")
    )


@pytest.mark.parametrize("job", check_workflows.RELEASE_LINUX_TEST_JOBS)
def test_linux_test_jobs_are_required_unconditional_and_run_on_the_linux_package(job):
    assert f"required release job {job} is missing" in _release_errors(lambda jobs: jobs.pop(job))
    assert f"{job} must need exactly package-linux" in _release_errors(
        lambda jobs: jobs[job].update(needs=["package-windows"])
    )
    assert f"job {job} must be unconditional and blocking" in _release_errors(
        lambda jobs: jobs[job].update({"if": "github.event_name == 'push'"})
    )
    assert f"job {job} must be unconditional and blocking" in _release_errors(
        lambda jobs: jobs[job].update({"continue-on-error": True})
    )
    assert f"{job} must check out ref {check_workflows.RELEASE_LINUX_PACKAGED_REF}" in _release_errors(
        lambda jobs: jobs[job]["steps"][0]["with"].update(ref="${{ github.sha }}")
    )


@pytest.mark.parametrize("job", check_workflows.RELEASE_LINUX_TEST_JOBS)
def test_linux_test_jobs_verify_the_runtime_files_they_download(job):
    def no_download(jobs):
        jobs[job]["steps"] = [
            step for step in jobs[job]["steps"]
            if not str(step.get("uses", "")).startswith("actions/download-artifact@")
        ]

    assert f"{job} must download the audioforge-release-linux-runtime artifact" in _release_errors(no_download)
    assert f"{job} must verify the downloaded runtime assets and the native module's SHA-256" in _release_errors(
        lambda jobs: _drop(jobs[job], "Verify the packaged runtime files")
    )

    def other_hash(jobs):
        step = _step(jobs[job], "Verify the packaged runtime files")
        step["env"] = {"EXPECTED_NATIVE_SHA256": "${{ needs.package-windows.outputs.native_sha256 }}"}

    assert f"{job} must verify the downloaded runtime assets and the native module's SHA-256" in _release_errors(other_hash)


def test_the_linux_rust_tests_fail_on_a_skipped_test():
    def run(old, new):
        def mutate(jobs):
            step = _step(jobs["linux-tests"], "Rust tests")
            assert old in step["run"]
            step["run"] = step["run"].replace(old, new)

        return _release_errors(mutate)

    for old, new in (
        ("set -o pipefail", ""),
        ("  exit 1\n", ""),
        ("grep -E '\\b(skipped|skipping)\\b'", "grep -E 'never'"),
        ("2>&1 | tee \"$log\"", "2>&1"),
    ):
        assert "missing required DeepFilter gate" in run(old, new), old
    assert "missing required DeepFilter gate" in _release_errors(
        lambda jobs: _step(jobs["linux-tests"], "Rust tests")["env"].pop("AUDIOFORGE_ENABLE_DEEPFILTER")
    )
    assert "missing required DeepFilter gate" in _release_errors(
        lambda jobs: _step(jobs["linux-tests"], "Rust tests").update(
            {"run": _step(jobs["linux-tests"], "Rust tests")["run"].replace("--nocapture", "")}
        )
    )


def test_the_linux_rust_tests_do_not_need_the_windows_pythonpath():
    errors: list[str] = []
    source = (check_workflows.WORKFLOW_DIR / "release-package.yml").read_text(encoding="utf-8")
    check_workflows._check_required_gates("release-package.yml", source, errors)
    assert errors == []
    assert "PYTHONPATH" not in json.dumps(_step(_workflow()["jobs"]["linux-tests"], "Rust tests"))


def test_the_linux_pytest_shards_are_the_ones_listed():
    def mutate(jobs):
        jobs["linux-pytest"]["strategy"]["matrix"]["shard"].pop(0)

    assert "linux-pytest matrix shard must be" in _release_errors(mutate)


def _ci_errors(mutate) -> str:
    return _gate_errors("ci.yml", mutate)


@pytest.mark.parametrize(
    ("step", "message"),
    [
        ("Restore DeepFilter runtime", "missing required DeepFilter gate"),
        ("Build and verify the pinned DeepFilter runtime", "missing required DeepFilter gate"),
        ("Audit the locked runtime on Linux", "missing required Linux audit gate"),
    ],
)
def test_ci_linux_tests_keep_the_deepfilter_and_audit_gates(step, message):
    assert message in _ci_errors(lambda jobs: _drop(jobs["linux-tests"], step))


def test_ci_linux_tests_fail_on_a_skipped_test_and_key_the_library_on_the_recipe():
    def tweak(old, new):
        def mutate(jobs):
            step = _step(jobs["linux-tests"], "Rust tests")
            assert old in step["run"]
            step["run"] = step["run"].replace(old, new)

        return _ci_errors(mutate)

    assert "missing required DeepFilter gate" in tweak("  exit 1\n", "")
    assert "missing required DeepFilter gate" in tweak("set -o pipefail", "")

    def narrow(jobs):
        restore = _step(jobs["linux-tests"], "Restore DeepFilter runtime")
        restore["with"]["key"] = restore["with"]["key"].replace("'python/tools/build_deepfilter.py', ", "")

    assert "missing required DeepFilter gate" in _ci_errors(narrow)


def test_promotion_requires_the_appimage_to_be_verified_against_its_own_sidecars_and_the_tarball():
    def drop_phrase(phrase):
        def mutate(jobs):
            step = _step(jobs["verify-linux-candidate"], "Verify the Linux tarball and AppImage")
            assert phrase in step["run"]
            step["run"] = step["run"].replace(phrase, "")

        return mutate

    for phrase in (
        "--baseline evaluation/release-bundle-path-baseline-linux.json",
        "--require-source-distribution",
        '--expected-producer "Release package" "$CANDIDATE_RUN_ID" "$CANDIDATE_RUN_ATTEMPT"',
        '(cd "$appimage_root" && "$appimage" --appimage-extract > /dev/null)',
        'tarball["files"] != appimage["files"]',
    ):
        assert "verify-linux-candidate" in _gate_errors("release-promote.yml", drop_phrase(phrase)), phrase


def test_promotion_matches_the_downloaded_bytes_to_the_staged_ones_before_running_them():
    name = "Compare the downloaded files with the staged ones"
    assert "missing required promotion gate" in _gate_errors(
        "release-promote.yml", lambda jobs: _drop(jobs["verify-linux-candidate"], name)
    )
    assert "must run after the gate before it" in _gate_errors(
        "release-promote.yml",
        lambda jobs: _move_before(
            jobs["verify-linux-candidate"], name, "Verify the Linux tarball and AppImage"
        )
        or _move_before(jobs["verify-linux-candidate"], "Smoke-test the Linux tarball and AppImage", name),
    )


def test_promotion_requires_the_linux_jobs_of_the_candidate_run():
    def mutate(old):
        def edit(jobs):
            step = _step(jobs["verify-candidate"], "Verify producing workflow runs and source commit")
            assert old in step["run"]
            step["run"] = step["run"].replace(old, "")

        return edit

    for old in ('"Linux Rust tests and lint"', '"Build Linux release package"'):
        assert "missing required promotion gate" in _gate_errors("release-promote.yml", mutate(old))


def test_the_linux_hardware_workflow_verifies_the_tarball_against_the_reviewed_paths():
    def mutate(jobs):
        step = _step(jobs["qualify-hardware"], "Extract and verify exact candidate")
        phrase = "--baseline evaluation/release-bundle-path-baseline-linux.json \\\n"
        assert phrase in step["run"]
        step["run"] = step["run"].replace(phrase, "")

    assert "missing required hardware gate" in _gate_errors("release-hardware-qualify-linux.yml", mutate)


def test_the_hardware_runner_is_reachable_only_by_a_maintainer_dispatching_from_master():
    workflow = _workflow("release-hardware-qualify-linux.yml")
    triggers = workflow[True]  # YAML reads the bare key `on` as True
    assert list(triggers) == ["workflow_dispatch"]
    job = workflow["jobs"]["qualify-hardware"]
    assert job["runs-on"] == ["self-hosted", "linux", "x64", "audioforge-hardware"]
    assert job["environment"] == "release"
    # A pull request trigger would hand a fork's code the runner; check_workflows rejects it.
    mutated = copy.deepcopy(workflow)
    mutated.pop(True, None)
    mutated["on"] = {"workflow_dispatch": triggers["workflow_dispatch"], "pull_request": None}
    errors: list[str] = []
    check_workflows._check_triggers_and_expressions("release-hardware-qualify-linux.yml", mutated, errors)
    assert any("only workflow_dispatch may trigger" in error for error in errors)


def test_the_appstream_release_follows_the_project_version(tmp_path, monkeypatch):
    monkeypatch.setattr(check_versions, "REPO_ROOT", tmp_path)
    metainfo = tmp_path / check_versions.METAINFO
    metainfo.parent.mkdir(parents=True)

    def check(releases, version="1.20.0"):
        metainfo.write_text(f"<component><releases>{releases}</releases></component>", encoding="utf-8")
        check_versions._check_metainfo_release(check_versions.parse_version(version))

    check('<release version="1.20.0" date="2026-11-20"/><release version="1.19.0" date="2026-10-27"/>')
    check('<release version="1.20.0" date="2026-11-20"/>', version="1.20.0rc2")
    with pytest.raises(ValueError, match="version bump updates it"):
        check('<release version="1.19.0" date="2026-10-27"/>')
    with pytest.raises(ValueError, match="version bump updates it"):
        check("")
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        check('<release version="1.20.0" date="2026-13-40"/>')
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        check('<release version="1.20.0"/>')


def test_the_committed_appstream_release_matches_the_committed_version():
    check_versions._check_metainfo_release(
        check_versions.parse_version(
            check_versions._single_match("pyproject.toml", r'^version\s*=\s*"([^"]+)"', "pyproject")
        )
    )
