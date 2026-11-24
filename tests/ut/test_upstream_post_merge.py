# SPDX-License-Identifier: Apache-2.0
"""Behavioral acceptance tests for the vLLM-SAIL Post-Merge Three-Gears CI (§2.6).

Covers all six first-release acceptance gates without requiring GPU/PPU hardware:
  Gate 1: Healthy verified-pin SHA end-to-end compatibility & dual-SHA summary
  Gate 2: Candidate 40-char SHA pass-through & SAIL native .so reuse without
          mutating csrc/upstream/manifest.toml
  Gate 3: Anti-false-green fail-closed UNVERIFIED (non-zero exit) on wheel
          mismatch or 100%-skipped pin-UT
  Gate 4: Breaking SHA red + control-first (last_green..red] bisection +
          plugin_interface_break vs sail_internal_patch_drift routing +
          incremental vs known-soft patch drift + Tier-A kernel checks
  Gate 5: Intentional failure negative test proving zero leaks across Console,
          Artifact JSON, GITHUB_STEP_SUMMARY, and Issue + fail-closed file wipe
  Gate 6: Dual git ls-remote polling (refs/heads/main + refs/tags/v*), PPU
          sliding-window coalescing, PR-CI priority yield, daily pin bump, and
          new vX.Y.0rc* routing
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml

from vllm_sail import envs

REPO_ROOT = Path(__file__).resolve().parents[2]
VERIFIED_PIN = (REPO_ROOT / ".github" / "vllm-main-verified.commit").read_text().strip()
RELEASE_TAG = (REPO_ROOT / ".github" / "vllm-release-tag.commit").read_text().strip()


@pytest.fixture
def compat_mod(monkeypatch: pytest.MonkeyPatch):
    if "vllm" not in sys.modules:
        vllm_pkg = types.ModuleType("vllm")
        vllm_pkg.__path__ = []
        monkeypatch.setitem(sys.modules, "vllm", vllm_pkg)
    if "vllm.logger" not in sys.modules:
        logger_mod = types.ModuleType("vllm.logger")
        logger_mod.init_logger = logging.getLogger
        monkeypatch.setitem(sys.modules, "vllm.logger", logger_mod)
    monkeypatch.delitem(sys.modules, "vllm_sail.compat", raising=False)
    mod = importlib.import_module("vllm_sail.compat")
    yield mod
    sys.modules.pop("vllm_sail.compat", None)


def _load_script_module(name: str, rel_path: str):
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


post_merge = _load_script_module(
    "ppu_upstream_post_merge", "scripts/ci/ppu_upstream_post_merge.py"
)
wheel_manifest = _load_script_module(
    "ppu_wheel_manifest_under_test", "scripts/ci/ppu_wheel_manifest.py"
)
ops_runner = _load_script_module(
    "ppu_ops_runner_under_test", "scripts/ci/ppu_ops_runner.py"
)


def _sha(n: int) -> str:
    return f"{n:040x}"


def _write_donor_wheels(
    wheels_dir: Path,
    *,
    vllm_commit: str,
    sail_source: Path = REPO_ROOT,
    build_image: str = "sha256:" + "a" * 64,
    pytorch_sail_arch: str = "ppu_15;ppu_10",
) -> dict:
    from tests.ut.test_ppu_wheel_manifest import make_wheels

    wheels_dir.mkdir(parents=True, exist_ok=True)
    make_wheels(wheels_dir)

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/ci/ppu_wheel_manifest.py"),
            "create",
            str(wheels_dir),
            "--sail-source",
            str(sail_source),
            "--vllm-commit",
            vllm_commit,
            "--build-image",
            build_image,
            "--pytorch-sail-arch",
            pytorch_sail_arch,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads((wheels_dir / "wheel-manifest.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Gate 1: Healthy Verified-Pin SHA End-to-End Baseline
# ---------------------------------------------------------------------------


def test_gate1_healthy_verified_sha_end_to_end(
    tmp_path, monkeypatch, caplog, compat_mod
):
    sail_head = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()

    monkeypatch.delenv("VLLM_VERSION", raising=False)
    monkeypatch.delenv("VLLM_SAIL_CANDIDATE_UPSTREAM_COMMIT", raising=False)
    monkeypatch.delenv("VLLM_PPU_CANDIDATE_UPSTREAM_COMMIT", raising=False)
    monkeypatch.setattr(
        compat_mod,
        "installed_vllm_version",
        lambda: f"0.33.0rc1.dev12+g{VERIFIED_PIN}.empty",
    )
    with caplog.at_level(logging.WARNING):
        compat_mod.check_vllm_compatibility(force=True)
    assert "candidate mode" not in caplog.text.lower()

    wheels_dir = tmp_path / "wheels"
    manifest = _write_donor_wheels(wheels_dir, vllm_commit=VERIFIED_PIN)
    verified = wheel_manifest.verify(
        wheels_dir, vllm_commit=VERIFIED_PIN, sail_commit=sail_head
    )
    assert verified["commits"] == {"vllm": VERIFIED_PIN, "vllm-sail": sail_head}
    assert manifest["commits"]["vllm"] == VERIFIED_PIN

    pub_dir = tmp_path / "public"
    summary_md = tmp_path / "step_summary.md"
    published = post_merge.publish_sanitized_channels(
        gear="probe",
        status="pass",
        sail_sha=sail_head,
        upstream_sha=VERIFIED_PIN,
        verified_pin=VERIFIED_PIN,
        build_image="download.t-head.cn/docker_release/pytorch@sha256:" + "1" * 64,
        runtime_image="download.t-head.cn/docker_release/vllm@sha256:" + "2" * 64,
        cpu_status="pass",
        ppu_status="pass",
        ppu_skip_reason="none",
        public_dir=pub_dir,
        step_summary_path=summary_md,
    )
    assert published["public_summary"]["status"] == "pass"
    assert published["public_summary"]["commits"] == {
        "vllm": VERIFIED_PIN,
        "vllm_sail": sail_head,
        "verified_pin": VERIFIED_PIN,
    }
    assert VERIFIED_PIN in summary_md.read_text(encoding="utf-8")
    assert sail_head in summary_md.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Gate 2: Candidate SHA Full-Chain Pass-Through & Native .so Reuse
# ---------------------------------------------------------------------------


def test_gate2_candidate_sha_passthrough_and_sail_native_so_reuse_without_mutating_manifest(
    tmp_path, monkeypatch, caplog, compat_mod
):
    manifest_toml = REPO_ROOT / "csrc" / "upstream" / "manifest.toml"
    manifest_sha_before = hashlib.sha256(manifest_toml.read_bytes()).hexdigest()
    candidate_sha = "a" * 40
    assert candidate_sha != VERIFIED_PIN

    # 1. Exact 40-char candidate SHA allowed with warning; manifest.toml untouched.
    monkeypatch.delenv("VLLM_VERSION", raising=False)
    monkeypatch.setenv("VLLM_SAIL_CANDIDATE_UPSTREAM_COMMIT", candidate_sha)
    assert envs.VLLM_SAIL_CANDIDATE_UPSTREAM_COMMIT == candidate_sha
    monkeypatch.setattr(
        compat_mod,
        "installed_vllm_version",
        lambda: f"0.33.1.dev12+g{candidate_sha}.empty",
    )
    with caplog.at_level(logging.WARNING):
        compat_mod.check_vllm_compatibility(force=True)
    assert "candidate" in caplog.text.lower()
    assert hashlib.sha256(manifest_toml.read_bytes()).hexdigest() == manifest_sha_before

    # Exact clean RC tag ("0.34.0rc1" without +g<sha>) also passes when vllm._version.__commit_id__ matches.
    vllm_ver_mod = types.ModuleType("vllm._version")
    vllm_ver_mod.__commit_id__ = f"g{candidate_sha}"
    monkeypatch.setitem(sys.modules, "vllm._version", vllm_ver_mod)
    monkeypatch.setattr(compat_mod, "installed_vllm_version", lambda: "0.34.0rc1+empty")
    compat_mod.check_vllm_compatibility(force=True)
    monkeypatch.delitem(sys.modules, "vllm._version", raising=False)

    # Reject wildcard, short SHA, or mismatched installed vLLM SHA.
    for bad in ("*", candidate_sha[:12], "A" * 40):
        monkeypatch.setenv("VLLM_SAIL_CANDIDATE_UPSTREAM_COMMIT", bad)
        with pytest.raises(ValueError, match="40-character lowercase hex"):
            _ = envs.VLLM_SAIL_CANDIDATE_UPSTREAM_COMMIT

    monkeypatch.setenv("VLLM_SAIL_CANDIDATE_UPSTREAM_COMMIT", candidate_sha)
    monkeypatch.setattr(
        compat_mod,
        "installed_vllm_version",
        lambda: f"0.33.1.dev12+g{'b' * 40}.empty",
    )
    with pytest.raises(RuntimeError, match="candidate"):
        compat_mod.check_vllm_compatibility(force=True)

    # 2. Execute build_ppu_wheels.sh's prepare_native_reuse function in real bash:
    #    donor wheel has vllm=VERIFIED_PIN, current build targets VLLM_REF=candidate_sha.
    #    Must set NATIVE_REUSE=0 (rebuild vLLM wheel) AND SAIL_NATIVE_REUSE=1 (reuse .so).
    donor_dir = tmp_path / "donor-wheels"
    build_img = "sha256:" + "3" * 64
    _write_donor_wheels(
        donor_dir,
        vllm_commit=VERIFIED_PIN,
        sail_source=REPO_ROOT,
        build_image=build_img,
        pytorch_sail_arch="ppu_15;ppu_10",
    )

    build_script = (REPO_ROOT / "scripts/ci/build_ppu_wheels.sh").read_text(
        encoding="utf-8"
    )
    start_idx = build_script.index("fingerprint_args=(")
    end_idx = build_script.index('if [[ "${VLLM_REF}" =~ ^[0-9a-f]{40}$ ]]')
    extracted_fn = build_script[start_idx:end_idx]

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    manifest_file = tmp_path / "build-manifest.txt"
    manifest_file.touch()
    vllm_wheel_out = tmp_path / "out-vllm-wheels"
    vllm_wheel_out.mkdir()
    harness = tmp_path / "run_reuse.sh"
    harness.write_text(
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        f'ROOT_DIR="{REPO_ROOT}"\n'
        f'WORK_DIR="{work_dir}"\n'
        f'MANIFEST="{manifest_file}"\n'
        f'VLLM_WHEEL_DIR="{vllm_wheel_out}"\n'
        f'VLLM_SAIL_BUILD_IMAGE="{build_img}"\n'
        'PYTORCH_SAIL_ARCH="ppu_15;ppu_10"\n'
        f'VLLM_REF="{candidate_sha}"\n'
        f'CANDIDATE_VLLM_COMMIT="{candidate_sha}"\n'
        "NATIVE_REUSE=0\n"
        "SAIL_NATIVE_REUSE=0\n"
        'PREBUILT_EXT_DIR=""\n'
        f"{extracted_fn}\n"
        f'prepare_native_reuse "{donor_dir}"\n'
        'printf "NATIVE_REUSE=%s\\nSAIL_NATIVE_REUSE=%s\\nPREBUILT_EXT_DIR=%s\\n" "$NATIVE_REUSE" "$SAIL_NATIVE_REUSE" "$PREBUILT_EXT_DIR"\n',
        encoding="utf-8",
    )
    reuse_env = dict(os.environ)
    reuse_env["PATH"] = f"{Path(sys.executable).parent}:{reuse_env.get('PATH', '')}"
    proc = subprocess.run(
        ["bash", str(harness)],
        env=reuse_env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "NATIVE_REUSE=0" in proc.stdout
    assert "SAIL_NATIVE_REUSE=1" in proc.stdout
    ext_line = next(
        line.split("=", 1)[1]
        for line in proc.stdout.splitlines()
        if line.startswith("PREBUILT_EXT_DIR=")
    )
    prebuilt_dir = Path(ext_line)
    for mod_name in ("_C", "_moe_C", "_upstream_C", "_upstream_moe_C"):
        assert (prebuilt_dir / f"{mod_name}.abi3.so").is_file()
    assert hashlib.sha256(manifest_toml.read_bytes()).hexdigest() == manifest_sha_before

    # 3. Verify ppu_ops_runner.clean_environment injects candidate SHA only when
    #    EXPECTED_VLLM_COMMIT != verified_pin.
    isolated = tmp_path / "isolated-ops"
    isolated.mkdir()
    cand_env = ops_runner.clean_environment(
        {"EXPECTED_VLLM_COMMIT": candidate_sha}, REPO_ROOT, isolated
    )
    assert cand_env["VLLM_SAIL_CANDIDATE_UPSTREAM_COMMIT"] == candidate_sha
    pin_env = ops_runner.clean_environment(
        {"EXPECTED_VLLM_COMMIT": VERIFIED_PIN}, REPO_ROOT, isolated
    )
    assert "VLLM_SAIL_CANDIDATE_UPSTREAM_COMMIT" not in pin_env


# ---------------------------------------------------------------------------
# Gate 3: Anti-False-Green (Mismatch & 100%-Skipped Must Exit Non-Zero)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("require_pr_artifact", ["false", "true"])
def test_gate3_anti_false_green_wheel_mismatch_fails_closed(
    tmp_path, require_pr_artifact
):
    sail_head = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    wheels_dir = tmp_path / "wheels"
    _write_donor_wheels(wheels_dir, vllm_commit=VERIFIED_PIN)
    mismatched_expected = "c" * 40

    base_env = dict(os.environ)
    base_env["PATH"] = f"{Path(sys.executable).parent}:{base_env.get('PATH', '')}"
    base_env.update(
        {
            "WHEELS_DIR": str(wheels_dir),
            "UT_RESULTS_DIR": str(tmp_path / f"ut-out-{require_pr_artifact}"),
            "REQUIRE_PR_ARTIFACT": require_pr_artifact,
            "EXPECTED_VLLM_COMMIT": mismatched_expected,
            "EXPECTED_SAIL_COMMIT": sail_head,
            "VLLM_SOURCE_ROOT": str(tmp_path),
            "MODEL_ID": "Qwen/Qwen3.8-27B",
            "HF_HOME": str(tmp_path),
        }
    )

    pin_res = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts/ci/ppu_pin_ut.sh")],
        env=base_env,
        capture_output=True,
        text=True,
    )
    assert pin_res.returncode != 0
    assert "PPU PIN UT UNVERIFIED" in (pin_res.stdout + pin_res.stderr)
    assert "PPU PIN UT SKIP" not in (pin_res.stdout + pin_res.stderr)

    smoke_res = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts/ci/ppu_e2e_smoke.sh")],
        env=base_env,
        capture_output=True,
        text=True,
    )
    assert smoke_res.returncode != 0
    assert "PPU E2E SMOKE UNVERIFIED" in (smoke_res.stdout + smoke_res.stderr)


def test_gate3_pin_ut_all_skipped_junit_fails_closed(tmp_path):
    """Verify the JUnit zero-executed-tests guard in ppu_pin_ut.sh exits non-zero."""
    pin_script = (REPO_ROOT / "scripts/ci/ppu_pin_ut.sh").read_text(encoding="utf-8")
    snippet = pin_script[pin_script.index('python - "${UT_RESULTS_DIR}/pin-ut.xml"') :]
    xml_path = tmp_path / "pin-ut.xml"
    xml_path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>'
        '<testsuites><testsuite name="pytest" tests="5" skipped="5" failures="0" errors="0"/></testsuites>\n',
        encoding="utf-8",
    )
    guard_sh = tmp_path / "check_xml.sh"
    guard_sh.write_text(
        "#!/usr/bin/env bash\nset -Eeuo pipefail\n"
        f'UT_RESULTS_DIR="{tmp_path}"\n'
        f"{snippet}\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
    res = subprocess.run(
        ["bash", str(guard_sh)], env=env, capture_output=True, text=True
    )
    assert res.returncode != 0
    assert "PPU PIN UT UNVERIFIED: zero executed tests" in res.stderr


# ---------------------------------------------------------------------------
# Gate 4: Breaking SHA Red + Bisection + Dual-Track Triage + Kernel Check
# ---------------------------------------------------------------------------


def test_gate4_bisection_control_first_and_dual_track_triage(tmp_path):
    # 1. Build a real 30-commit linear upstream history in a temp git repository.
    vllm_repo = tmp_path / "vllm-bisect-repo"
    vllm_repo.mkdir()
    subprocess.run(["git", "-C", str(vllm_repo), "init", "-q"], check=True)
    subprocess.run(
        ["git", "-C", str(vllm_repo), "config", "user.name", "Test Bot"], check=True
    )
    subprocess.run(
        ["git", "-C", str(vllm_repo), "config", "user.email", "bot@example.com"],
        check=True,
    )

    shas: list[str] = []
    for idx in range(31):  # index 0 = last_green_sha, indices 1..30 = 30 window commits
        (vllm_repo / "step.txt").write_text(f"step={idx}\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(vllm_repo), "add", "step.txt"], check=True)
        subprocess.run(
            ["git", "-C", str(vllm_repo), "commit", "-q", "-m", f"commit {idx}"],
            check=True,
        )
        sha = subprocess.check_output(
            ["git", "-C", str(vllm_repo), "rev-parse", "HEAD"], text=True
        ).strip()
        shas.append(sha)

    last_green = shas[0]
    window_commits = shas[1:]  # 30 commits
    culprit_idx = 18  # 19th commit in window (shas[19])
    culprit_sha = window_commits[culprit_idx]
    bad_set = set(window_commits[culprit_idx:])

    # Run CLI `ppu_upstream_post_merge.py bisect` end-to-end!
    bad_file = tmp_path / "bad_shas.txt"
    bad_file.write_text("\n".join(sorted(bad_set)) + "\n", encoding="utf-8")
    bisect_out = tmp_path / "bisect-result.json"
    probe_cmd = (
        f'if grep -qx "$CANDIDATE_SHA" "{bad_file}"; then '
        'printf \'{"status":"fail","stage":"plugin_registration","target_signature":"vllm.plugins.load_general_plugins","is_public_plugin_interface":true}\\n\' > "$PROBE_RESULT_JSON"; exit 1; '
        "else "
        'printf \'{"status":"pass","stage":"none","target_signature":"none"}\\n\' > "$PROBE_RESULT_JSON"; exit 0; '
        "fi"
    )
    rc = post_merge.main(
        [
            "bisect",
            "--vllm-repo",
            str(vllm_repo),
            "--last-green-sha",
            last_green,
            "--red-sha",
            window_commits[-1],
            "--probe-cmd",
            probe_cmd,
            "--output-json",
            str(bisect_out),
        ]
    )
    assert rc == 0
    bisect_data = json.loads(bisect_out.read_text(encoding="utf-8"))
    assert bisect_data["bisect_status"] == "converged"
    assert bisect_data["first_failing_sha"] == culprit_sha
    assert bisect_data["last_good_sha"] == window_commits[culprit_idx - 1]
    assert bisect_data["history"][0] == {
        "role": "baseline_control",
        "sha": last_green,
        "stage": "none",
        "status": "pass",
    }
    assert bisect_data["probe_steps_on_range"] <= 5

    # 2. Control failure on last_green_sha -> aborts immediately as infra_error!
    ctrl_fail = post_merge.bisect_first_failing_sha(
        last_green_sha=last_green,
        red_sha=window_commits[-1],
        commits=window_commits,
        probe_fn=lambda _sha: post_merge.ProbeOutcome(
            status="fail", stage="ppu_probe_smoke", target_signature="runner_oom"
        ),
        verify_baseline_first=True,
    )
    assert ctrl_fail["bisect_status"] == "baseline_control_failed"
    assert ctrl_fail["steps_executed"] == 1
    assert ctrl_fail["probe_steps_on_range"] == 0

    # 3. Dual-track classification with non-PPU hard gate:
    #    (a) WITHOUT verified non-PPU repro evidence, even a public plugin registration
    #        failure MUST downgrade to sail_internal_patch_drift!
    pub_unverified_dir = tmp_path / "pub-plugin-unverified"
    unverified_pub = post_merge.publish_sanitized_channels(
        gear="probe",
        status="fail",
        sail_sha=_sha(99),
        upstream_sha=window_commits[-1],
        verified_pin=VERIFIED_PIN,
        cpu_status="fail",
        ppu_status="skipped",
        ppu_skip_reason="cpu_fast_check_failed",
        first_failing_sha=culprit_sha,
        last_good_sha=window_commits[culprit_idx - 1],
        bisect_steps=bisect_data["steps_executed"],
        outcome=post_merge.ProbeOutcome(
            status="fail",
            stage="plugin_registration",
            target_signature="vllm.plugins.load_general_plugins",
            is_public_plugin_interface=True,
        ),
        run_url="https://github.com/vllm-project/vllm-sail/actions/runs/12345",
        public_dir=pub_unverified_dir,
    )
    assert (
        unverified_pub["public_summary"]["triage"]["classification"]
        == "sail_internal_patch_drift"
    )
    assert unverified_pub["public_summary"]["triage"]["non_ppu_repro_verified"] is False
    assert unverified_pub["issue_payload"]["should_file_sail_issue"] is True
    assert unverified_pub["issue_payload"]["should_file_upstream_issue"] is False
    assert unverified_pub["issue_payload"]["upstream_reproducer_markdown"] == ""

    #    (b) WITH verified non-PPU repro evidence (last_good exit 0, first_failing exit != 0),
    #        classifies as plugin_interface_break and renders fenced draft with inline stub plugin.
    verified_repro = post_merge.NonPpuReproEvidence(
        mode="stub_plugin_cpu",
        repro_kind="load_general_plugins",
        last_good_sha=window_commits[culprit_idx - 1],
        first_failing_sha=culprit_sha,
        last_good_exit_code=0,
        first_failing_exit_code=1,
        verified=True,
        last_good_phase="stub_check",
        first_failing_phase="stub_check",
        last_good_check_executed=True,
        first_failing_check_executed=True,
        infra_error=False,
        failure_source="stub_plugin_check",
    )
    pub_plugin_dir = tmp_path / "pub-plugin"
    plugin_pub = post_merge.publish_sanitized_channels(
        gear="probe",
        status="fail",
        sail_sha=_sha(99),
        upstream_sha=window_commits[-1],
        verified_pin=VERIFIED_PIN,
        cpu_status="fail",
        ppu_status="skipped",
        ppu_skip_reason="cpu_fast_check_failed",
        first_failing_sha=culprit_sha,
        last_good_sha=window_commits[culprit_idx - 1],
        bisect_steps=bisect_data["steps_executed"],
        outcome=post_merge.ProbeOutcome(
            status="fail",
            stage="plugin_registration",
            target_signature="vllm.plugins.load_general_plugins",
            is_public_plugin_interface=True,
            non_ppu_repro=verified_repro,
        ),
        first_failing_author="vllm-maintainer",
        run_url="https://github.com/vllm-project/vllm-sail/actions/runs/12345",
        public_dir=pub_plugin_dir,
    )
    assert (
        plugin_pub["public_summary"]["triage"]["classification"]
        == "plugin_interface_break"
    )
    assert plugin_pub["public_summary"]["triage"]["non_ppu_repro_verified"] is True
    assert (
        plugin_pub["public_summary"]["triage"]["repro_kind"] == "load_general_plugins"
    )
    assert (
        plugin_pub["public_summary"]["bisect"]["first_failing_author"]
        == "vllm-maintainer"
    )
    assert plugin_pub["issue_payload"]["should_file_sail_issue"] is False
    assert plugin_pub["issue_payload"]["should_file_upstream_issue"] is False
    repro = plugin_pub["issue_payload"]["upstream_reproducer_markdown"]
    assert "````markdown\n" in repro
    assert "VLLM_TARGET_DEVICE=empty" in repro
    assert "vllm_stub_plugin" in repro
    assert culprit_sha in repro
    assert window_commits[culprit_idx - 1] in repro
    assert "(cc @vllm-maintainer)" in repro
    outside_fence = repro.split("````markdown\n", 1)[0]
    assert "@vllm-maintainer" not in outside_fence

    pub_drift_dir = tmp_path / "pub-drift"
    drift_pub = post_merge.publish_sanitized_channels(
        gear="probe",
        status="fail",
        sail_sha=_sha(99),
        upstream_sha=window_commits[-1],
        verified_pin=VERIFIED_PIN,
        cpu_status="fail",
        ppu_status="skipped",
        ppu_skip_reason="cpu_fast_check_failed",
        first_failing_sha=culprit_sha,
        last_good_sha=window_commits[culprit_idx - 1],
        bisect_steps=5,
        outcome=post_merge.ProbeOutcome(
            status="fail",
            stage="patch_drift",
            target_signature="vllm.v1.worker.gpu_model_runner.GPUModelRunner",
            is_public_plugin_interface=False,
        ),
        public_dir=pub_drift_dir,
    )
    assert (
        drift_pub["public_summary"]["triage"]["classification"]
        == "sail_internal_patch_drift"
    )
    assert drift_pub["issue_payload"]["should_file_sail_issue"] is True
    assert (
        "[Post-Merge][sail_internal_patch_drift]["
        in drift_pub["issue_payload"]["title"]
    )
    assert drift_pub["issue_payload"]["upstream_reproducer_markdown"] == ""


def test_gate4_tier_a_kernel_manifest_compatibility_detects_missing_file_and_schema_drift(
    tmp_path,
):
    manifest_data = post_merge._load_toml(REPO_ROOT / "csrc/upstream/manifest.toml")
    fake_vllm = tmp_path / "vllm-src"
    for entry in manifest_data["entry"]:
        if entry.get("tier") == "a":
            target = fake_vllm / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("// stub upstream kernel\n", encoding="utf-8")

    # Copy vendored torch_bindings.cpp so ops.def schemas initially match 100%.
    for up_rel, sail_rel in (
        ("csrc/torch_bindings.cpp", "csrc/upstream/torch_bindings.cpp"),
        ("csrc/moe/torch_bindings.cpp", "csrc/upstream/moe/torch_bindings.cpp"),
    ):
        dst = fake_vllm / up_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / sail_rel, dst)

    assert post_merge.check_kernel_manifest_compatibility(fake_vllm, REPO_ROOT) == []

    # Mutate an ops.def signature in upstream csrc/torch_bindings.cpp -> detected!
    tb = fake_vllm / "csrc/torch_bindings.cpp"
    orig_tb = tb.read_text(encoding="utf-8")
    tb.write_text(
        orig_tb.replace(
            "rms_norm(Tensor! result,", "rms_norm(Tensor! result, int broken_arg,"
        ),
        encoding="utf-8",
    )
    schema_problems = post_merge.check_kernel_manifest_compatibility(
        fake_vllm, REPO_ROOT
    )
    assert "changed_op_schema:rms_norm" in schema_problems

    # Remove a Tier-A .cu source file -> detected!
    tb.write_text(orig_tb, encoding="utf-8")
    first_tier_a = next(
        e["path"] for e in manifest_data["entry"] if e.get("tier") == "a"
    )
    (fake_vllm / first_tier_a).unlink()
    missing_problems = post_merge.check_kernel_manifest_compatibility(
        fake_vllm, REPO_ROOT
    )
    assert f"missing_tier_a_file:{first_tier_a}" in missing_problems


# ---------------------------------------------------------------------------
# Gate 5: Intentional Failure Three-Channel Zero-Leak & Fail-Closed Wipe
# ---------------------------------------------------------------------------


def test_gate5_intentional_failure_three_channel_zero_leak_and_fail_closed_wipe(
    tmp_path,
):
    pub_dir = tmp_path / "public-neg"
    step_summary = tmp_path / "github_step_summary.md"

    build_digest = "sha256:" + "d" * 64
    runtime_digest = "sha256:" + "e" * 64
    tainted_build = f"reg.eng.t-head.cn/internal/sdk@{build_digest}"
    tainted_runtime = f"art.alibaba-inc.com/ppu/vllm@{runtime_digest}"
    tainted_sig = "/nas_aisw/models/Step-5-Preview from 10.23.45.67. token=ghp_1234567890abcdefghij123456 user=wb-998877"
    tainted_url = "https://ci.alibaba-inc.com/mnt/wl_nas/devops/run/192.168.1.8"

    res = post_merge.publish_sanitized_channels(
        gear="probe",
        status="fail",
        sail_sha=_sha(42),
        upstream_sha=_sha(43),
        verified_pin=VERIFIED_PIN,
        build_image=tainted_build,
        runtime_image=tainted_runtime,
        cpu_status="pass",
        ppu_status="fail",
        ppu_skip_reason="none",
        first_failing_sha=_sha(43),
        last_good_sha=VERIFIED_PIN,
        bisect_steps=4,
        outcome=post_merge.ProbeOutcome(
            status="fail",
            stage="ppu_probe_smoke",
            target_signature=tainted_sig,
            is_public_plugin_interface=False,
        ),
        run_url=tainted_url,
        public_dir=pub_dir,
        step_summary_path=step_summary,
    )

    channels = [
        res["console"],
        (pub_dir / "public-summary.json").read_text(encoding="utf-8"),
        (pub_dir / "issue-payload.json").read_text(encoding="utf-8"),
        step_summary.read_text(encoding="utf-8"),
    ]
    forbidden_substrings = (
        "eng.t-head.cn",
        "alibaba-inc.com",
        "/nas_aisw",
        "/mnt/wl_nas",
        "10.23.45.67",
        "192.168.1.8",
        "ghp_",
        "Step-5-Preview",
        "wb-998877",
    )
    for text in channels:
        post_merge.assert_zero_sensitive_leaks(text)
        for bad in forbidden_substrings:
            assert bad.lower() not in text.lower()

    assert res["public_summary"]["images"] == {
        "build_digest": build_digest,
        "runtime_digest": runtime_digest,
    }
    assert res["public_summary"]["run_url"] is None

    # Direct fail-closed wipe test: forced leak raises RedactionViolationError AND deletes staging files!
    for sample in (
        "host=/nas_aisw/models/Qwen",
        "mirror=https://art.eng.t-head.cn/pypi",
        "bare_domain=eng.t-head.cn",
        "bare_corp=alibaba-inc.com",
        "node ip 10.23.45.67.",
        "secret ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456",
        "model Step-5-Preview",
        "bad\x00nul",
    ):
        staged = tmp_path / "staged-leak.json"
        staged.write_text("stale", encoding="utf-8")
        with pytest.raises(post_merge.RedactionViolationError):
            post_merge.assert_zero_sensitive_leaks(sample, staging_paths=[staged])
        assert not staged.exists()

    # Exact Round-1 CLI counterexample 1: bare internal domain on `publish` CLI
    cli_pub_dir = tmp_path / "cli-bare-domain"
    cli_summary = tmp_path / "cli-summary.md"
    assert (
        post_merge.main(
            [
                "publish",
                "--gear",
                "probe",
                "--status",
                "fail",
                "--sail-sha",
                _sha(42),
                "--upstream-sha",
                _sha(43),
                "--failure-stage",
                "plugin_registration",
                "--target-signature",
                "eng.t-head.cn",
                "--public-plugin-break",
                "--public-dir",
                str(cli_pub_dir),
                "--step-summary",
                str(cli_summary),
            ]
        )
        == 0
    )
    for out_file in (
        cli_pub_dir / "public-summary.json",
        cli_pub_dir / "issue-payload.json",
        cli_summary,
    ):
        txt = out_file.read_text(encoding="utf-8")
        post_merge.assert_zero_sensitive_leaks(txt)
        assert "t-head.cn" not in txt.lower()

    # Exact Round-1 CLI counterexample 2: tainted cursor field in `route` CLI fails closed without leaking
    ls_file = tmp_path / "ls.txt"
    ls_file.write_text(f"{_sha(44)}\trefs/heads/main\n", encoding="utf-8")
    bad_cursor = tmp_path / "bad-cursor.json"
    bad_cursor.write_text(
        json.dumps({"last_ppu_green_sha": "/nas_aisw/review/Step-5-Preview"}),
        encoding="utf-8",
    )
    route_proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/ci/ppu_upstream_post_merge.py"),
            "route",
            "--ls-remote-file",
            str(ls_file),
            "--cursor-file",
            str(bad_cursor),
        ],
        capture_output=True,
        text=True,
    )
    assert route_proc.returncode != 0
    assert "/nas_aisw" not in (route_proc.stdout + route_proc.stderr)
    assert "Step-5-Preview" not in (route_proc.stdout + route_proc.stderr)


# ---------------------------------------------------------------------------
# Gate 6: Dual git ls-remote Polling, Sliding Window & RC Qualification Routing
# ---------------------------------------------------------------------------


def test_gate6_ls_remote_dual_polling_sliding_window_and_rc_routing(tmp_path):
    sha_main_1 = _sha(101)
    sha_main_2 = _sha(102)
    sha_tag_obj = _sha(200)
    sha_v033_commit = _sha(201)
    sha_v033_rc0 = _sha(202)
    sha_v034_rc0 = _sha(300)
    sha_v034_rc1 = _sha(301)

    ls_text = "\n".join(
        [
            f"{sha_main_1}\trefs/heads/main",
            f"{sha_tag_obj}\trefs/tags/v0.33.0",
            f"{sha_v033_commit}\trefs/tags/v0.33.0^{{}}",
            f"{sha_v033_rc0}\trefs/tags/v0.33.0rc0",
        ]
    )
    parsed = post_merge.parse_ls_remote(ls_text)
    assert parsed["main_sha"] == sha_main_1
    # Peeled commit SHA must override annotated tag object SHA!
    assert parsed["tags"]["v0.33.0"] == sha_v033_commit

    # 1. Docs/ROCm-only upstream changes skip PPU smoke WITHOUT advancing last_ppu_tested_sha.
    d1 = post_merge.route_three_gears(
        ls_remote_data=parsed,
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.33.0",
        schedule_cron=post_merge.PROBE_CRON,
        changed_paths_since_ppu=[
            "docs/design.md",
            "csrc/rocm/attention.cu",
            "README.md",
        ],
    )
    assert d1["selected_gear"] == "probe"
    assert d1["run_cpu"] is True
    assert d1["run_ppu"] is False
    assert d1["ppu_skip_reason"] == "allowlisted_non_ppu_changes"

    cursor1 = post_merge.advance_cursor(
        None,
        verified_pin=VERIFIED_PIN,
        selected_gear="probe",
        target_sha=sha_main_1,
        cpu_status="pass",
        ppu_status="skipped",
        ppu_skip_reason="allowlisted_non_ppu_changes",
    )
    assert cursor1["last_cpu_checked_sha"] == sha_main_1
    assert cursor1["last_ppu_tested_sha"] == VERIFIED_PIN

    # 2. Next upstream commit touches vllm/config/model.py -> cumulative diff forces PPU smoke;
    #    if SAIL PR CI is active, probe yields PPU slot without advancing last_ppu_tested_sha!
    parsed2 = {**parsed, "main_sha": sha_main_2}
    d_defer = post_merge.route_three_gears(
        ls_remote_data=parsed2,
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.33.0",
        schedule_cron=post_merge.PROBE_CRON,
        cursor=cursor1,
        changed_paths_since_ppu=["docs/design.md", "vllm/config/model.py"],
        active_pr_runs=2,
    )
    assert d_defer["run_ppu"] is False
    assert d_defer["ppu_skip_reason"] == "deferred_for_sail_pr_ci"

    # Once PR CI finishes (active_pr_runs=0), PPU smoke runs on sha_main_2!
    d_run = post_merge.route_three_gears(
        ls_remote_data=parsed2,
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.33.0",
        schedule_cron=post_merge.PROBE_CRON,
        cursor=cursor1,
        changed_paths_since_ppu=["docs/design.md", "vllm/compilation/backends.py"],
        active_pr_runs=0,
    )
    assert d_run["run_ppu"] is True
    assert d_run["ppu_skip_reason"] == "none"

    cursor2 = post_merge.advance_cursor(
        cursor1,
        verified_pin=VERIFIED_PIN,
        selected_gear="probe",
        target_sha=sha_main_2,
        cpu_status="pass",
        ppu_status="pass",
    )
    assert cursor2["last_ppu_tested_sha"] == sha_main_2
    assert cursor2["last_ppu_green_sha"] == sha_main_2

    # 3. New v0.34.0rc0 & v0.34.0rc1 tags appear -> auto-routes to newest untested v0.34.0rc1!
    parsed_rc = {
        "main_sha": sha_main_2,
        "tags": {
            **parsed["tags"],
            "v0.34.0rc0": sha_v034_rc0,
            "v0.34.0rc1": sha_v034_rc1,
        },
    }
    d_rc = post_merge.route_three_gears(
        ls_remote_data=parsed_rc,
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.33.0",
        schedule_cron=post_merge.PROBE_CRON,
        cursor=cursor2,
    )
    assert d_rc["selected_gear"] == "rc_qualify"
    assert d_rc["rc_tag"] == "v0.34.0rc1"
    assert d_rc["target_sha"] == sha_v034_rc1
    assert d_rc["run_cpu"] is True and d_rc["run_ppu"] is True

    cursor3 = post_merge.advance_cursor(
        cursor2,
        verified_pin=VERIFIED_PIN,
        selected_gear="rc_qualify",
        target_sha=sha_v034_rc1,
        rc_tag="v0.34.0rc1",
        cpu_status="pass",
        ppu_status="pass",
    )
    assert "v0.34.0rc1" in cursor3["tested_rc_tags"]
    # Verify last_probe_record survived the interleaved rc_qualify run for daily_pin_bump!
    d_bump = post_merge.route_three_gears(
        ls_remote_data=parsed_rc,
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.33.0",
        schedule_cron=post_merge.DAILY_PIN_CRON,
        cursor=cursor3,
    )
    assert d_bump["selected_gear"] == "daily_pin_bump"
    assert d_bump["target_sha"] == sha_main_2

    fake_repo = tmp_path / "sail-pin-bump"
    (fake_repo / ".github").mkdir(parents=True)
    (fake_repo / "csrc" / "upstream").mkdir(parents=True)
    pin_file = fake_repo / ".github" / "vllm-main-verified.commit"
    manifest_copy = fake_repo / "csrc" / "upstream" / "manifest.toml"
    pin_file.write_text(VERIFIED_PIN + "\n", encoding="utf-8")
    shutil.copyfile(REPO_ROOT / "csrc" / "upstream" / "manifest.toml", manifest_copy)
    bump_meta = post_merge.prepare_pin_bump(repo_root=fake_repo, target_sha=sha_main_2)
    assert bump_meta["changed"] == "true"
    assert pin_file.read_text(encoding="utf-8") == sha_main_2 + "\n"
    from vllm_sail.native import manifest as kernel_manifest_mod

    assert kernel_manifest_mod.load(manifest_copy).upstream_ref == sha_main_2


def test_gate6_red_sha_repoll_docs_successor_and_rc_skipped_never_turn_green():
    sha_red = _sha(501)
    sha_docs_after_red = _sha(502)

    # 1. CPU pass + PPU fail on sha_red -> re-polling sha_red must re-run PPU (never pass/noop)!
    cursor_red = post_merge.advance_cursor(
        None,
        verified_pin=VERIFIED_PIN,
        selected_gear="probe",
        target_sha=sha_red,
        status="fail",
        cpu_status="pass",
        ppu_status="fail",
    )
    repoll = post_merge.route_three_gears(
        ls_remote_data={"main_sha": sha_red, "tags": {}},
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.33.0",
        schedule_cron=post_merge.PROBE_CRON,
        cursor=cursor_red,
    )
    assert repoll["run_ppu"] is True
    assert repoll["ppu_skip_reason"] == "none"

    # 2. Docs-only successor on top of red PPU SHA must NOT skip PPU and must NOT be promoted!
    after_red = post_merge.route_three_gears(
        ls_remote_data={"main_sha": sha_docs_after_red, "tags": {}},
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.33.0",
        schedule_cron=post_merge.PROBE_CRON,
        cursor=cursor_red,
        changed_paths_since_ppu=["docs/design.md"],
    )
    assert after_red["run_ppu"] is True
    assert (
        post_merge.select_pin_bump_candidate(
            [
                {
                    "gear": "probe",
                    "upstream_sha": sha_docs_after_red,
                    "cpu_status": "pass",
                    "ppu_status": "skipped",
                    "ppu_skip_reason": "allowlisted_non_ppu_changes",
                    "last_ppu_tested_sha": sha_red,
                    "last_ppu_green_sha": VERIFIED_PIN,
                }
            ],
            verified_pin=VERIFIED_PIN,
        )
        is None
    )

    # 2.5 Exact Round-2 residual case: SHA A passes PPU first, then force-rerun on A fails PPU.
    #     Neither re-polling A nor docs-only successor B may skip PPU or be promoted!
    sha_a = _sha(503)
    sha_b_docs = _sha(504)
    cursor_a_green = post_merge.advance_cursor(
        None,
        verified_pin=VERIFIED_PIN,
        selected_gear="probe",
        target_sha=sha_a,
        status="pass",
        cpu_status="pass",
        ppu_status="pass",
    )
    cursor_a_force_red = post_merge.advance_cursor(
        cursor_a_green,
        verified_pin=VERIFIED_PIN,
        selected_gear="probe",
        target_sha=sha_a,
        status="fail",
        cpu_status="pass",
        ppu_status="fail",
    )
    assert cursor_a_force_red["last_ppu_status"] == "fail"
    repoll_a = post_merge.route_three_gears(
        ls_remote_data={"main_sha": sha_a, "tags": {}},
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.33.0",
        schedule_cron=post_merge.PROBE_CRON,
        cursor=cursor_a_force_red,
    )
    assert repoll_a["run_ppu"] is True
    assert repoll_a["ppu_skip_reason"] == "none"

    route_b = post_merge.route_three_gears(
        ls_remote_data={"main_sha": sha_b_docs, "tags": {}},
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.33.0",
        schedule_cron=post_merge.PROBE_CRON,
        cursor=cursor_a_force_red,
        changed_paths_since_ppu=["docs/design.md"],
    )
    assert route_b["run_ppu"] is True
    assert route_b["ppu_skip_reason"] == "none"

    # Even if a partial cursor dict omits top-level last_ppu_status, last_probe_record.ppu_status=="fail" still blocks skip!
    legacy_shape = {
        k: v
        for k, v in cursor_a_force_red.items()
        if k not in ("last_cpu_status", "last_ppu_status")
    }
    legacy_shape["last_probe_record"] = {
        k: v
        for k, v in cursor_a_force_red["last_probe_record"].items()
        if k != "last_ppu_result_status"
    }
    route_b_legacy = post_merge.route_three_gears(
        ls_remote_data={"main_sha": sha_b_docs, "tags": {}},
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.33.0",
        schedule_cron=post_merge.PROBE_CRON,
        cursor=legacy_shape,
        changed_paths_since_ppu=["docs/design.md"],
    )
    assert route_b_legacy["run_ppu"] is True
    assert (
        post_merge.select_pin_bump_candidate(
            [
                {
                    "gear": "probe",
                    "upstream_sha": sha_b_docs,
                    "cpu_status": "pass",
                    "ppu_status": "skipped",
                    "ppu_skip_reason": "allowlisted_non_ppu_changes",
                    "last_ppu_tested_sha": sha_a,
                    "last_ppu_green_sha": sha_a,
                    "last_ppu_result_status": "fail",
                }
            ],
            verified_pin=VERIFIED_PIN,
        )
        is None
    )

    # 3. Workflow outcome reducer: skipped RC gate or failed ops-select fails closed!
    rc_skipped = post_merge.evaluate_workflow_outcome(
        selected_gear="rc_qualify",
        target_sha=sha_red,
        run_cpu=True,
        run_ppu=True,
        ppu_skip_reason="none",
        build_result="success",
        cpu_result="success",
        rc_pin_result="success",
        rc_fast_result="success",
        rc_model_result="skipped",
    )
    assert rc_skipped["status"] == "fail"
    assert rc_skipped["failure_stage"] == "rc_model_sanity"

    rc_ops_fail = post_merge.evaluate_workflow_outcome(
        selected_gear="rc_qualify",
        target_sha=sha_red,
        run_cpu=True,
        run_ppu=True,
        ppu_skip_reason="none",
        ops_image="sha256:" + "f" * 64,
        build_result="success",
        cpu_result="success",
        rc_pin_result="success",
        rc_fast_result="success",
        rc_model_result="success",
        rc_ops_select_result="failure",
        rc_ops_result="skipped",
    )
    assert rc_ops_fail["status"] == "fail"
    assert rc_ops_fail["failure_stage"] == "rc_ops_tests"


def test_gate6_upstream_canary_workflow_three_gears_contract():
    wf_path = REPO_ROOT / ".github/workflows/upstream-canary.yaml"
    wf_raw = wf_path.read_text(encoding="utf-8")
    wf = yaml.safe_load(wf_raw)
    triggers = wf.get(True) or wf.get("on")

    # No workflow_call, no top-level cache-mode, no statuses:write, no legacy 60k raw log paste.
    assert "workflow_call" not in triggers
    assert "cache-mode" not in wf
    assert "slice(0, 60000)" not in wf_raw
    assert "_ppu-upstream-reuse.yaml" not in wf_raw
    assert wf["permissions"] == {"contents": "read", "actions": "read"}

    crons = [entry["cron"] for entry in triggers["schedule"]]
    assert post_merge.PROBE_CRON in crons
    assert post_merge.DAILY_PIN_CRON in crons

    jobs = wf["jobs"]
    for job in jobs.values():
        assert "statuses" not in job.get("permissions", {})

    assert "pytorch@sha256:" in jobs["cpu-fast-check"]["container"]["image"]
    assert jobs["ppu-probe-smoke"]["runs-on"] == "8-ppu-m890p"
    assert "!cancelled()" in jobs["ppu-probe-smoke"]["if"]
    assert jobs["ppu-probe-smoke"]["concurrency"] == {
        "group": "upstream-post-merge-ppu-slot",
        "cancel-in-progress": False,
    }
    assert wf["concurrency"] == {
        "group": "upstream-post-merge-state",
        "cancel-in-progress": False,
    }
    assert jobs["build"]["with"]["diagnostics_mode"] == "public"
    assert jobs["build"]["with"]["resolve_native_donors"] is True
    assert "ppu-build-and-test.yaml" not in wf_raw
    for rc_job in ("rc-pin-ut", "rc-fast-sanity", "rc-model-sanity", "rc-ops-tests"):
        assert "rc_qualify" in jobs[rc_job]["if"]


# ---------------------------------------------------------------------------
# PR #374 Review Regression Suite (All 12 Counterexamples + Controls)
# ---------------------------------------------------------------------------


def test_review_r1_container_build_candidate_export_works_under_posix_sh_and_bash():
    candidate = "b" * 40
    assert candidate != VERIFIED_PIN
    shells = ["sh", "bash"]
    if shutil.which("dash"):
        shells.append("dash")

    for wf_name in ("_ppu-build.yaml", "ppu-ops-tests.yaml"):
        jobs = yaml.safe_load((REPO_ROOT / ".github/workflows" / wf_name).read_text())[
            "jobs"
        ]
        job = next(
            j
            for j in jobs.values()
            if any(
                "export VLLM_SAIL_CANDIDATE_UPSTREAM_COMMIT" in s.get("run", "")
                for s in j.get("steps", [])
            )
        )
        step = next(
            s
            for s in job["steps"]
            if "export VLLM_SAIL_CANDIDATE_UPSTREAM_COMMIT" in s.get("run", "")
        )
        assert step.get("shell") == "bash"
        snippet = step["run"].replace(
            "bash scripts/ci/build_ppu_wheels.sh",
            'printf "%s" "${VLLM_SAIL_CANDIDATE_UPSTREAM_COMMIT:-unset}"',
        )
        for sh in shells:
            proc = subprocess.run(
                [sh, "-e", "-c", snippet],
                cwd=REPO_ROOT,
                env={**os.environ, "VLLM_REF": candidate},
                text=True,
                capture_output=True,
                check=True,
            )
            assert proc.stdout == candidate
            pin_proc = subprocess.run(
                [sh, "-e", "-c", snippet],
                cwd=REPO_ROOT,
                env={**os.environ, "VLLM_REF": VERIFIED_PIN},
                text=True,
                capture_output=True,
                check=True,
            )
            assert pin_proc.stdout == "unset"


def test_review_r2_real_patch_baseline_opaque_fingerprints_pass_redaction_and_diff(
    tmp_path,
):
    from types import SimpleNamespace

    from tools import check_patch_drift

    def _to_records(source_by_target: dict[str, str | None]):
        return [
            SimpleNamespace(target=k, original_source=v)
            for k, v in source_by_target.items()
        ]

    real_baseline = json.loads(
        (REPO_ROOT / "tools/patch_baseline.json").read_text(encoding="utf-8")
    )
    assert len(real_baseline) >= 300
    # Raw baseline contains confidential/long symbols that trip public redaction.
    with pytest.raises(post_merge.RedactionViolationError):
        post_merge.assert_zero_sensitive_leaks(
            json.dumps({"patch_fingerprints": real_baseline})
        )

    # Opaque mapping serializes cleanly through both redaction and cursor sanitization,
    # while still detecting incremental drift on any of the 376 targets.
    base_sources = {
        k: (None if v == "additive" else f"body:{k}:{v}")
        for k, v in real_baseline.items()
    }
    recorded = check_patch_drift._fingerprints(_to_records(base_sources))
    drift_clean = check_patch_drift.evaluate_drift(_to_records(base_sources), recorded)
    opaque_map = drift_clean["opaque_fingerprints"]
    assert len(opaque_map) == len(real_baseline)
    cpu_payload = json.dumps({"patch_fingerprints": opaque_map}, sort_keys=True)
    post_merge.assert_zero_sensitive_leaks(cpu_payload)

    cursor = post_merge.advance_cursor(
        None,
        verified_pin=VERIFIED_PIN,
        selected_gear="probe",
        target_sha=_sha(601),
        status="pass",
        cpu_status="pass",
        ppu_status="pass",
        patch_fingerprints=opaque_map,
    )
    cursor_file = tmp_path / "cursor.json"
    cursor_text = json.dumps(cursor, indent=2, sort_keys=True)
    post_merge.assert_zero_sensitive_leaks(cursor_text)
    cursor_file.write_text(cursor_text, encoding="utf-8")

    # Simulate a known soft drift recorded in cursor.json followed by an incremental drift on a step5 target!
    step5_key = next(k for k in real_baseline if "step5" in k.lower())
    prev_sources = {**base_sources, step5_key: "soft-drifted-body-v1"}
    prev_opaque = check_patch_drift.evaluate_drift(_to_records(prev_sources), recorded)[
        "opaque_fingerprints"
    ]
    cursor_soft = post_merge.advance_cursor(
        cursor,
        verified_pin=VERIFIED_PIN,
        selected_gear="probe",
        target_sha=_sha(602),
        status="pass",
        cpu_status="pass",
        ppu_status="pass",
        patch_fingerprints=prev_opaque,
    )
    prev_file = tmp_path / "prev-cursor.json"
    prev_file.write_text(json.dumps(cursor_soft), encoding="utf-8")
    loaded_prev = check_patch_drift._read_mapping_file(prev_file, "previous baseline")

    same_soft = check_patch_drift.evaluate_drift(
        _to_records(prev_sources), recorded, previous=loaded_prev
    )
    assert same_soft["known_soft_drifted"] == [step5_key]
    assert same_soft["incremental_drifted"] == []

    new_drift = check_patch_drift.evaluate_drift(
        _to_records({**base_sources, step5_key: "incremental-drifted-body-v2"}),
        recorded,
        previous=loaded_prev,
    )
    assert new_drift["incremental_drifted"] == [step5_key]
    safe_sig = post_merge._sanitize_patch_target_signature(step5_key)
    assert safe_sig.startswith("opaque_target:t_")
    post_merge.assert_zero_sensitive_leaks(safe_sig)


def test_review_r3_r4_r9_r10_promotion_eligibility_deferred_merge_and_cursor_discipline(
    tmp_path,
):
    sha_b = _sha(701)
    sha_c = _sha(702)

    def _cycle(state=None, gear="probe", target=sha_b, permute=None, **kw):
        decision = post_merge.route_three_gears(
            ls_remote_data={"main_sha": target, "tags": {}},
            verified_pin=VERIFIED_PIN,
            release_baseline_tag="v0.33.0",
            requested_gear=gear,
            cursor=state,
            **kw,
        )
        args = {
            "selected_gear": decision["selected_gear"],
            "target_sha": decision["target_sha"],
            "run_cpu": decision["run_cpu"],
            "run_ppu": decision["run_ppu"],
            "ppu_skip_reason": decision["ppu_skip_reason"],
        }
        jobs = {
            "build_result": "success"
            if args["run_cpu"] or args["run_ppu"]
            else "skipped",
            "cpu_result": "success" if args["run_cpu"] else "skipped",
            "ppu_smoke_result": "success"
            if args["run_ppu"] and args["selected_gear"] == "probe"
            else "skipped",
        }
        outcome = post_merge.evaluate_workflow_outcome(
            **args, **(jobs | (permute or {}))
        )
        next_state = post_merge.advance_cursor(
            state,
            verified_pin=VERIFIED_PIN,
            selected_gear=args["selected_gear"],
            target_sha=args["target_sha"],
            rc_tag=decision["rc_tag"],
            status=outcome["status"],
            cpu_status=outcome["cpu_status"],
            ppu_status=outcome["ppu_status"],
            ppu_skip_reason=outcome["ppu_skip_reason"],
        )
        return decision, outcome, next_state

    # R3: Prior passing probe on B followed by forced wheel-build failure invalidates B!
    _, _, prior = _cycle()
    assert (
        post_merge.route_three_gears(
            ls_remote_data={"main_sha": sha_b, "tags": {}},
            verified_pin=VERIFIED_PIN,
            release_baseline_tag="v0.33.0",
            requested_gear="daily_pin_bump",
            cursor=prior,
        )["target_sha"]
        == sha_b
    )
    _, rerun, failed_cursor = _cycle(
        prior,
        force=True,
        permute={
            "build_result": "failure",
            "cpu_result": "skipped",
            "ppu_smoke_result": "skipped",
        },
    )
    assert rerun["status"] == "fail"
    assert (
        post_merge.route_three_gears(
            ls_remote_data={"main_sha": sha_b, "tags": {}},
            verified_pin=VERIFIED_PIN,
            release_baseline_tag="v0.33.0",
            requested_gear="daily_pin_bump",
            cursor=failed_cursor,
        )["target_sha"]
        == ""
    )
    next_probe_after_build_fail = post_merge.route_three_gears(
        ls_remote_data={"main_sha": sha_b, "tags": {}},
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.33.0",
        requested_gear="probe",
        cursor=failed_cursor,
    )
    assert next_probe_after_build_fail["run_cpu"] is True
    assert next_probe_after_build_fail["ppu_skip_reason"] != "already_tested"

    # R4: Manual daily_pin_bump override rejects unprobed SHAs and accepts verified SHAs.
    assert (
        post_merge.route_three_gears(
            ls_remote_data={"main_sha": sha_b, "tags": {}},
            verified_pin=VERIFIED_PIN,
            release_baseline_tag="v0.33.0",
            requested_gear="daily_pin_bump",
            cursor=None,
            upstream_ref_override=sha_c,
        )["target_sha"]
        == ""
    )
    assert (
        post_merge.route_three_gears(
            ls_remote_data={"main_sha": sha_b, "tags": {}},
            verified_pin=VERIFIED_PIN,
            release_baseline_tag="v0.33.0",
            requested_gear="daily_pin_bump",
            cursor=prior,
            upstream_ref_override=sha_b,
        )["target_sha"]
        == sha_b
    )

    # R9: Split 2-run probe (CPU pass + PPU deferred, then PPU pass) promotes B!
    _, out1, st1 = _cycle(active_pr_runs=1)
    assert out1["status"] == "deferred"
    d2, out2, st2 = _cycle(st1)
    assert d2["run_cpu"] is False and d2["run_ppu"] is True
    assert out2["status"] == "pass"
    assert (
        post_merge.route_three_gears(
            ls_remote_data={"main_sha": sha_b, "tags": {}},
            verified_pin=VERIFIED_PIN,
            release_baseline_tag="v0.33.0",
            requested_gear="daily_pin_bump",
            cursor=st2,
        )["target_sha"]
        == sha_b
    )

    # R10: Read-only daily_pin_bump and noop runs never write cursor.json in `publish`.
    pub_daily = tmp_path / "pub-daily"
    assert (
        post_merge.main(
            [
                "publish",
                "--gear",
                "daily_pin_bump",
                "--status",
                "pass",
                "--sail-sha",
                _sha(709),
                "--upstream-sha",
                sha_b,
                "--public-dir",
                str(pub_daily),
            ]
        )
        == 0
    )
    assert not (pub_daily / "cursor.json").exists()


def test_review_r5_rc_red_and_green_never_starve_or_pollute_main_probe():
    sha_rc = _sha(720)
    sha_main = _sha(721)
    tags = {"v0.34.0rc1": sha_rc}

    # 1. Failed RC runs once on auto schedule; next auto poll probes advancing main!
    d_rc = post_merge.route_three_gears(
        ls_remote_data={"main_sha": sha_main, "tags": tags},
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.33.0",
        requested_gear="auto",
    )
    assert d_rc["selected_gear"] == "rc_qualify" and d_rc["target_sha"] == sha_rc
    red_rc_cursor = post_merge.advance_cursor(
        None,
        verified_pin=VERIFIED_PIN,
        selected_gear="rc_qualify",
        target_sha=sha_rc,
        rc_tag="v0.34.0rc1",
        status="fail",
        cpu_status="pass",
        ppu_status="fail",
    )
    after_red_rc = post_merge.route_three_gears(
        ls_remote_data={"main_sha": sha_main, "tags": tags},
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.33.0",
        requested_gear="auto",
        cursor=red_rc_cursor,
    )
    assert after_red_rc["selected_gear"] == "probe"
    assert after_red_rc["target_sha"] == sha_main
    assert after_red_rc["last_green_sha"] == VERIFIED_PIN

    # 2. Passing RC records RC tag without overwriting main's last_ppu_green_sha!
    green_rc_cursor = post_merge.advance_cursor(
        None,
        verified_pin=VERIFIED_PIN,
        selected_gear="rc_qualify",
        target_sha=sha_rc,
        rc_tag="v0.34.0rc1",
        status="pass",
        cpu_status="pass",
        ppu_status="pass",
    )
    assert green_rc_cursor["last_ppu_green_sha"] == VERIFIED_PIN
    assert green_rc_cursor["last_cpu_green_sha"] == VERIFIED_PIN
    after_green_rc = post_merge.route_three_gears(
        ls_remote_data={"main_sha": sha_main, "tags": tags},
        verified_pin=VERIFIED_PIN,
        release_baseline_tag="v0.33.0",
        requested_gear="auto",
        cursor=green_rc_cursor,
    )
    assert after_green_rc["selected_gear"] == "probe"
    assert after_green_rc["last_green_sha"] == VERIFIED_PIN


def test_review_r6_r12_shallow_bisect_completeness_deepening_and_precise_publish_stage(
    tmp_path,
):
    src = tmp_path / "upstream-full"
    shallow_no_origin = tmp_path / "shallow-disconnected"
    shallow_with_origin = tmp_path / "shallow-origin"
    for d in (src, shallow_no_origin, shallow_with_origin):
        d.mkdir()
        subprocess.run(["git", "-C", str(d), "init", "-q"], check=True)

    git_env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.org",
        "GIT_COMMITTER_EMAIL": "test@example.org",
    }
    commits: list[str] = []
    for i in range(6):
        (src / "state").write_text(f"{i}\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(src), "add", "state"], check=True)
        subprocess.run(
            ["git", "-C", str(src), "commit", "-qm", str(i)],
            check=True,
            env=git_env,
        )
        commits.append(
            subprocess.check_output(
                ["git", "-C", str(src), "rev-parse", "HEAD"], text=True
            ).strip()
        )

    url = src.as_uri()
    # Disconnected shallow clone (no `origin` remote): must report unresolved_history with NO false culprit!
    subprocess.run(
        [
            "git",
            "-C",
            str(shallow_no_origin),
            "fetch",
            "-q",
            "--depth=2",
            url,
            commits[-1],
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(shallow_no_origin),
            "fetch",
            "-q",
            "--depth=1",
            url,
            commits[0],
        ],
        check=True,
    )
    out_disc = tmp_path / "disc.json"
    assert (
        post_merge.main(
            [
                "bisect",
                "--vllm-repo",
                str(shallow_no_origin),
                "--last-green-sha",
                commits[0],
                "--red-sha",
                commits[-1],
                "--probe-cmd",
                f'test "$CANDIDATE_SHA" = {commits[0]}',
                "--output-json",
                str(out_disc),
            ]
        )
        == 0
    )
    disc_data = json.loads(out_disc.read_text(encoding="utf-8"))
    assert disc_data["bisect_status"] == "unresolved_history"
    assert disc_data["first_failing_sha"] == ""

    # Shallow clone WITH `origin` remote: automatically deepens until connected and finds exact commit 1!
    subprocess.run(
        ["git", "-C", str(shallow_with_origin), "remote", "add", "origin", url],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(shallow_with_origin),
            "fetch",
            "-q",
            "--depth=2",
            "origin",
            commits[-1],
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(shallow_with_origin),
            "fetch",
            "-q",
            "--depth=1",
            "origin",
            commits[0],
        ],
        check=True,
    )
    out_deep = tmp_path / "deep.json"
    assert (
        post_merge.main(
            [
                "bisect",
                "--vllm-repo",
                str(shallow_with_origin),
                "--last-green-sha",
                commits[0],
                "--red-sha",
                commits[-1],
                "--probe-cmd",
                f'test "$CANDIDATE_SHA" = {commits[0]}',
                "--output-json",
                str(out_deep),
            ]
        )
        == 0
    )
    deep_data = json.loads(out_deep.read_text(encoding="utf-8"))
    assert deep_data["bisect_status"] == "converged"
    assert deep_data["first_failing_sha"] == commits[1]

    # R6 + R12 in `publish`: unbisected failures have first_failing_sha=None; measured cpu stage wins!
    no_bisect_dir = tmp_path / "pub-no-bisect"
    assert (
        post_merge.main(
            [
                "publish",
                "--gear",
                "probe",
                "--status",
                "fail",
                "--sail-sha",
                _sha(730),
                "--upstream-sha",
                commits[-1],
                "--failure-stage",
                "wheel_build",
                "--public-dir",
                str(no_bisect_dir),
            ]
        )
        == 0
    )
    no_bisect_sum = json.loads(
        (no_bisect_dir / "public-summary.json").read_text(encoding="utf-8")
    )
    assert no_bisect_sum["bisect"]["first_failing_sha"] is None
    assert no_bisect_sum["triage"]["stage"] == "wheel_build"

    cpu_res_file = tmp_path / "cpu-res.json"
    cpu_res_file.write_text(
        json.dumps(
            {
                "stage": "patch_drift",
                "status": "fail",
                "target_signature": "vllm.config.Example",
            }
        ),
        encoding="utf-8",
    )
    detail_dir = tmp_path / "pub-cpu-detail"
    assert (
        post_merge.main(
            [
                "publish",
                "--gear",
                "probe",
                "--status",
                "fail",
                "--sail-sha",
                _sha(730),
                "--upstream-sha",
                commits[-1],
                "--failure-stage",
                "plugin_registration",
                "--cpu-result-json",
                str(cpu_res_file),
                "--public-dir",
                str(detail_dir),
            ]
        )
        == 0
    )
    detail_sum = json.loads(
        (detail_dir / "public-summary.json").read_text(encoding="utf-8")
    )
    assert detail_sum["triage"]["stage"] == "patch_drift"
    assert detail_sum["triage"]["classification"] == "sail_internal_patch_drift"


def test_review_r7_r8_moe_sum_schema_parity_and_sanitized_build_diagnostics(tmp_path):
    # R7: Every single compiled Tier-A op is extracted from vendored bindings, including `m.def("moe_sum...")`!
    manifest_data = post_merge._load_toml(REPO_ROOT / "csrc/upstream/manifest.toml")
    fake_vllm = tmp_path / "vllm-tier-a"
    for entry in manifest_data["entry"]:
        if entry.get("tier") == "a":
            target = fake_vllm / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("// stub\n", encoding="utf-8")
    for up_rel, sail_rel in (
        ("csrc/torch_bindings.cpp", "csrc/upstream/torch_bindings.cpp"),
        ("csrc/moe/torch_bindings.cpp", "csrc/upstream/moe/torch_bindings.cpp"),
    ):
        dst = fake_vllm / up_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / sail_rel, dst)

    assert post_merge.check_kernel_manifest_compatibility(fake_vllm, REPO_ROOT) == []
    moe_tb = fake_vllm / "csrc/moe/torch_bindings.cpp"
    orig_moe = moe_tb.read_text(encoding="utf-8")
    assert "moe_sum(Tensor input," in orig_moe
    moe_tb.write_text(
        orig_moe.replace(
            "moe_sum(Tensor input,", "moe_sum(int breaking_arg, Tensor input,"
        ),
        encoding="utf-8",
    )
    moe_problems = post_merge.check_kernel_manifest_compatibility(fake_vllm, REPO_ROOT)
    assert "changed_op_schema:moe_sum" in moe_problems

    # R8: `sanitize-build-manifest` strips internal registry/host paths and passes redaction gate.
    raw_manifest = tmp_path / "build-manifest.txt"
    raw_manifest.write_text(
        "\n".join(
            [
                "status=running",
                f"vllm_ref={VERIFIED_PIN}",
                "pytorch_sail_arch=ppu_15;ppu_10",
                "build_image=download.t-head.cn/docker_release/pytorch@sha256:"
                + "9" * 64,
                "reuse_mode=sail-native",
                "reuse_source=/home/runner/work/vllm-sail/artifacts/native-reuse",
                "hgcc=hgcc (T-Head PPU) /ppusw/sdk",
                "status=succeeded",
                f"vllm_commit={VERIFIED_PIN}",
                f"vllm_sail_commit={_sha(740)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert (
        post_merge.main(["sanitize-build-manifest", "--manifest", str(raw_manifest)])
        == 0
    )
    cleaned = raw_manifest.read_text(encoding="utf-8")
    post_merge.assert_zero_sensitive_leaks(cleaned)
    assert "t-head.cn" not in cleaned
    assert "/home/runner" not in cleaned
    assert "/ppusw" not in cleaned
    assert f"build_image_digest=sha256:{'9' * 64}" in cleaned
    assert "status=succeeded" in cleaned


def test_review_r2_sub1_no_pin_downgrade_fail_closed_manifest_and_rc_ops_digest(
    tmp_path, monkeypatch
):
    from scripts.ci import ppu_ops_select

    sha_a = VERIFIED_PIN
    sha_b = _sha(801)
    sha_c = _sha(802)

    # 1. Counterexample 1: B then C pass probe -> C merges as current verified_pin ->
    #    daily_pin_bump (both auto and manual override=B) must NOT downgrade C -> B!
    cur1 = post_merge.advance_cursor(
        None,
        verified_pin=sha_a,
        selected_gear="probe",
        target_sha=sha_b,
        status="pass",
        cpu_status="pass",
        ppu_status="pass",
    )
    cur2 = post_merge.advance_cursor(
        cur1,
        verified_pin=sha_a,
        selected_gear="probe",
        target_sha=sha_c,
        status="pass",
        cpu_status="pass",
        ppu_status="pass",
    )
    # Before C merges (pin == A): selects C (and manual override=B selects forward B).
    assert (
        post_merge.route_three_gears(
            ls_remote_data={"main_sha": sha_c, "tags": {}},
            verified_pin=sha_a,
            release_baseline_tag="v0.33.0",
            requested_gear="daily_pin_bump",
            cursor=cur2,
        )["target_sha"]
        == sha_c
    )
    assert (
        post_merge.route_three_gears(
            ls_remote_data={"main_sha": sha_c, "tags": {}},
            verified_pin=sha_a,
            release_baseline_tag="v0.33.0",
            requested_gear="daily_pin_bump",
            cursor=cur2,
            upstream_ref_override=sha_b,
        )["target_sha"]
        == sha_b
    )
    # After C merges (verified_pin == C): auto AND manual override=B MUST return ""!
    assert (
        post_merge.route_three_gears(
            ls_remote_data={"main_sha": sha_c, "tags": {}},
            verified_pin=sha_c,
            release_baseline_tag="v0.33.0",
            requested_gear="daily_pin_bump",
            cursor=cur2,
        )["target_sha"]
        == ""
    )
    assert (
        post_merge.route_three_gears(
            ls_remote_data={"main_sha": sha_c, "tags": {}},
            verified_pin=sha_c,
            release_baseline_tag="v0.33.0",
            requested_gear="daily_pin_bump",
            cursor=cur2,
            upstream_ref_override=sha_b,
        )["target_sha"]
        == ""
    )

    # Also verify prepare_pin_bump rejects non-descendant target_sha when --vllm-repo is provided.
    vllm_git = tmp_path / "vllm-ancestry"
    vllm_git.mkdir()
    subprocess.run(["git", "-C", str(vllm_git), "init", "-q"], check=True)
    git_env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.org",
        "GIT_COMMITTER_EMAIL": "test@example.org",
    }
    ancestry_shas: list[str] = []
    for idx in range(2):
        (vllm_git / "f.txt").write_text(f"{idx}\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(vllm_git), "add", "f.txt"], check=True)
        subprocess.run(
            ["git", "-C", str(vllm_git), "commit", "-qm", str(idx)],
            check=True,
            env=git_env,
        )
        ancestry_shas.append(
            subprocess.check_output(
                ["git", "-C", str(vllm_git), "rev-parse", "HEAD"], text=True
            ).strip()
        )
    fake_sail = tmp_path / "sail-downgrade"
    (fake_sail / ".github").mkdir(parents=True)
    (fake_sail / "csrc" / "upstream").mkdir(parents=True)
    pin_f = fake_sail / ".github" / "vllm-main-verified.commit"
    man_f = fake_sail / "csrc" / "upstream" / "manifest.toml"
    shutil.copyfile(REPO_ROOT / "csrc/upstream/manifest.toml", man_f)
    # Current pin = newer commit ancestry_shas[1]; attempting to bump to older ancestry_shas[0] fails!
    pin_f.write_text(ancestry_shas[1] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="first-parent descendant"):
        post_merge.prepare_pin_bump(
            repo_root=fake_sail,
            target_sha=ancestry_shas[0],
            vllm_repo=vllm_git,
        )
    # Forward bump ancestry_shas[0] -> ancestry_shas[1] succeeds!
    pin_f.write_text(ancestry_shas[0] + "\n", encoding="utf-8")
    ok_bump = post_merge.prepare_pin_bump(
        repo_root=fake_sail,
        target_sha=ancestry_shas[1],
        vllm_repo=vllm_git,
    )
    assert ok_bump["changed"] == "true"

    # 2. Counterexample 2: Fault injection before rewrite in sanitize_build_manifest_file
    #    wipes raw manifest (fail-closed), and _ppu-build.yaml gates public upload on sanitize success.
    fault_manifest = tmp_path / "raw-fault-manifest.txt"
    fault_manifest.write_text(
        "status=succeeded\nbuild_image=download.t-head.cn/docker_release/pytorch@sha256:"
        + "a" * 64
        + "\n",
        encoding="utf-8",
    )
    orig_read_text = Path.read_text

    def _boom(self: Path, *a, **kw):
        if self == fault_manifest:
            raise OSError("simulated read failure before rewrite")
        return orig_read_text(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", _boom)
    with pytest.raises(OSError, match="simulated read failure"):
        post_merge.sanitize_build_manifest_file(fault_manifest)
    monkeypatch.setattr(Path, "read_text", orig_read_text)
    assert not fault_manifest.exists()

    build_wf = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/_ppu-build.yaml").read_text(encoding="utf-8")
    )["jobs"]["build"]["steps"]
    sanitize_step = next(
        s for s in build_wf if s.get("name") == "Sanitize public build diagnostics"
    )
    upload_diag = next(
        s for s in build_wf if s.get("name") == "Upload build diagnostics"
    )
    upload_public_diag = next(
        s
        for s in build_wf
        if s.get("name") == "Upload sanitized public build diagnostics"
    )
    assert sanitize_step.get("id") == "sanitize_diagnostics"
    assert "inputs.diagnostics_mode != 'public'" in upload_diag["if"]
    assert r"\n" not in upload_diag["with"]["path"]
    assert "${{" not in upload_diag["with"]["path"]
    assert [
        line.strip()
        for line in upload_diag["with"]["path"].splitlines()
        if line.strip()
    ] == ["artifacts/logs/**", "artifacts/build-manifest.txt"]
    assert "inputs.diagnostics_mode == 'public'" in upload_public_diag["if"]
    assert "steps.sanitize_diagnostics.outcome == 'success'" in upload_public_diag["if"]
    assert upload_public_diag["with"]["path"].strip() == "artifacts/build-manifest.txt"

    # 3. Counterexample 3: Both raw and sanitized build-manifest.txt are accepted by
    #    ppu_ops_select.build_fingerprint (used by both rc-ops-select and ppu_ops_runner)!
    sail_sha = _sha(810)
    vllm_sha = _sha(811)
    img_digest = "7" * 64
    full_build_image = f"download.t-head.cn/docker_release/pytorch@sha256:{img_digest}"
    wheel_man = {"commits": {"vllm": vllm_sha, "vllm-sail": sail_sha}}

    raw_diag_file = tmp_path / "diag-raw.txt"
    san_diag_file = tmp_path / "diag-sanitized.txt"
    raw_diag_file.write_text(
        "\n".join(
            [
                "status=succeeded",
                f"vllm_commit={vllm_sha}",
                f"vllm_sail_commit={sail_sha}",
                f"build_image={full_build_image}",
                "pytorch_sail_arch=ppu_15;ppu_10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    shutil.copyfile(raw_diag_file, san_diag_file)
    post_merge.sanitize_build_manifest_file(san_diag_file)
    post_merge.assert_zero_sensitive_leaks(san_diag_file.read_text(encoding="utf-8"))

    fp_raw = ppu_ops_select.build_fingerprint(
        ppu_ops_select.read_diagnostics(raw_diag_file),
        wheel_man,
        tested_sha=sail_sha,
        expected_vllm_commit=vllm_sha,
        build_image=full_build_image,
    )
    fp_san = ppu_ops_select.build_fingerprint(
        ppu_ops_select.read_diagnostics(san_diag_file),
        wheel_man,
        tested_sha=sail_sha,
        expected_vllm_commit=vllm_sha,
        build_image=full_build_image,
    )
    assert fp_san == fp_raw
    # Wrong digest in sanitized manifest must still fail closed!
    wrong_image = f"download.t-head.cn/docker_release/pytorch@sha256:{'8' * 64}"
    with pytest.raises(ValueError, match="Observed build image mismatch"):
        ppu_ops_select.build_fingerprint(
            ppu_ops_select.read_diagnostics(san_diag_file),
            wheel_man,
            tested_sha=sail_sha,
            expected_vllm_commit=vllm_sha,
            build_image=wrong_image,
        )


# ---------------------------------------------------------------------------
# 2026-11-24 Maintainer Refinement: Non-PPU Repro Gate, Stub Plugins & @author
# ---------------------------------------------------------------------------


def _make_mock_vllm_git_repo(repo_dir: Path) -> dict[str, str]:
    """Create a real multi-commit git repo with a minimal stdlib vLLM plugin surface."""
    repo_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "-C", str(repo_dir), "init", "-q"], check=True)
    git_env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Upstream Maintainer",
        "GIT_COMMITTER_NAME": "Upstream Maintainer",
        "GIT_AUTHOR_EMAIL": "12345+upstream-dev@users.noreply.github.com",
        "GIT_COMMITTER_EMAIL": "12345+upstream-dev@users.noreply.github.com",
    }

    def _write_healthy_tree() -> None:
        files = {
            "vllm/__init__.py": (
                '__version__ = "0.34.0.dev0"\n'
                "from vllm.model_executor.models.registry import ModelRegistry\n"
                '__all__ = ["ModelRegistry", "__version__"]\n'
            ),
            "vllm/plugins/__init__.py": (
                "import importlib.metadata\n"
                "import os\n\n"
                "def load_plugins_by_group(group: str) -> dict:\n"
                '    raw = os.environ.get("VLLM_PLUGINS")\n'
                '    allowed = None if raw is None else {p.strip() for p in raw.split(",") if p.strip()}\n'
                "    out = {}\n"
                "    for ep in importlib.metadata.entry_points(group=group):\n"
                "        if allowed is not None and ep.name not in allowed:\n"
                "            continue\n"
                "        out[ep.name] = ep.load()\n"
                "    return out\n\n"
                "def load_general_plugins() -> None:\n"
                '    for fn in load_plugins_by_group("vllm.general_plugins").values():\n'
                "        fn()\n"
            ),
            "vllm/platforms/__init__.py": (
                "import importlib\n"
                "from vllm.platforms.interface import Platform\n"
                "from vllm.plugins import load_plugins_by_group\n\n"
                "def _resolve_platform() -> Platform:\n"
                '    plugins = load_plugins_by_group("vllm.platform_plugins")\n'
                "    for fn in plugins.values():\n"
                "        qual = fn()\n"
                '        mod_name, attr_name = qual.rsplit(".", 1)\n'
                "        cls = getattr(importlib.import_module(mod_name), attr_name)\n"
                "        return cls()\n"
                "    return Platform()\n\n"
                "current_platform = _resolve_platform()\n"
            ),
            "vllm/platforms/interface.py": (
                "from enum import Enum\n\n"
                "class PlatformEnum(Enum):\n"
                '    CUDA = "cuda"\n'
                '    OOT = "oot"\n\n'
                "class Platform:\n"
                "    _enum = PlatformEnum.OOT\n"
                '    device_name = "unspecified"\n'
                '    device_type = "cpu"\n\n'
                "    def is_out_of_tree(self) -> bool:\n"
                "        return self._enum == PlatformEnum.OOT\n"
            ),
            "vllm/model_executor/__init__.py": "",
            "vllm/model_executor/custom_op.py": (
                "op_registry = {}\n"
                "op_registry_oot = {}\n\n\n"
                "class CustomOp:\n"
                "    def __new__(cls, *args, **kwargs):\n"
                "        op_name = cls.__name__\n"
                "        target = op_registry_oot.get(op_name, cls)\n"
                "        return super().__new__(target)\n\n"
                "    @classmethod\n"
                "    def register(cls, name: str, dynamic_arg_dims=None):\n"
                "        def _dec(sub):\n"
                "            sub.name = name\n"
                "            op_registry[name] = sub\n"
                "            return sub\n"
                "        return _dec\n\n"
                "    @classmethod\n"
                "    def register_oot(cls, _decorated_op_cls=None, name: str | None = None):\n"
                "        def _dec(sub):\n"
                "            reg_name = name if name is not None else cls.__name__\n"
                "            sub.name = reg_name\n"
                "            op_registry_oot[reg_name] = sub\n"
                "            return sub\n"
                "        return _dec if _decorated_op_cls is None else _dec(_decorated_op_cls)\n"
            ),
            "vllm/model_executor/models/__init__.py": "",
            "vllm/model_executor/models/registry.py": (
                "class _Registry:\n"
                "    def __init__(self) -> None:\n"
                "        self.models = {}\n\n"
                "    def register_model(self, arch: str, target: str) -> None:\n"
                "        self.models[arch] = target\n\n"
                "    def get_supported_archs(self) -> set[str]:\n"
                "        return set(self.models)\n\n"
                "ModelRegistry = _Registry()\n"
            ),
        }
        for rel, content in files.items():
            p = repo_dir / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")

    def _commit(msg: str) -> str:
        subprocess.run(["git", "-C", str(repo_dir), "add", "-A"], check=True)
        subprocess.run(
            ["git", "-C", str(repo_dir), "commit", "-qm", msg],
            check=True,
            env=git_env,
        )
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True
        ).strip()

    _write_healthy_tree()
    good_sha = _commit("healthy baseline")
    shas = {"good": good_sha}

    # Second healthy commit (negative control: must NOT verify as a regression).
    (repo_dir / "README.md").write_text("docs touch\n", encoding="utf-8")
    shas["good_2"] = _commit("docs only change")

    # Create independent regression commits branching off `good_sha` for each of the 5 kinds.
    mutations = {
        "cpu_empty_build": (
            "vllm/__init__.py",
            'raise RuntimeError("VLLM_TARGET_DEVICE=empty build broken")\n',
        ),
        "platform_plugin": (
            "vllm/platforms/interface.py",
            "from enum import Enum\n"
            "class PlatformEnum(Enum):\n"
            '    CUDA = "cuda"\n'
            "class Platform:\n"
            "    pass\n",
        ),
        "register_oot": (
            "vllm/model_executor/custom_op.py",
            "op_registry = {}\n"
            "op_registry_oot = {}\n\n\n"
            "class CustomOp:\n"
            "    def __new__(cls, *args, **kwargs):\n"
            "        return super().__new__(cls)\n\n"
            "    @classmethod\n"
            "    def register(cls, name: str, dynamic_arg_dims=None):\n"
            "        def _dec(sub):\n"
            "            sub.name = name\n"
            "            op_registry[name] = sub\n"
            "            return sub\n"
            "        return _dec\n",
        ),
        "model_registry": (
            "vllm/model_executor/models/registry.py",
            "class _Registry:\n"
            "    def __init__(self) -> None:\n"
            "        self.models = {}\n"
            "    def get_supported_archs(self) -> set[str]:\n"
            "        return set(self.models)\n"
            "ModelRegistry = _Registry()\n",
        ),
        "load_general_plugins": (
            "vllm/plugins/__init__.py",
            "def load_plugins_by_group(group: str) -> dict:\n"
            "    return {}\n"
            "def load_general_plugins() -> None:\n"
            "    return None\n",
        ),
    }
    for kind, (rel_path, broken_body) in mutations.items():
        subprocess.run(
            ["git", "-C", str(repo_dir), "checkout", "-q", "--detach", good_sha],
            check=True,
        )
        (repo_dir / rel_path).write_text(broken_body, encoding="utf-8")
        shas[kind] = _commit(f"break {kind}")

    return shas


@pytest.mark.parametrize(
    ("stage", "target_signature", "expected_kind", "expected_mode"),
    [
        (
            "wheel_build",
            "VLLM_TARGET_DEVICE=empty",
            "cpu_empty_build",
            "cpu_empty_build",
        ),
        (
            "plugin_registration",
            "vllm.platforms.Platform",
            "platform_plugin",
            "stub_plugin_cpu",
        ),
        (
            "plugin_registration",
            "CustomOp.register_oot",
            "register_oot",
            "stub_plugin_cpu",
        ),
        (
            "plugin_registration",
            "ModelRegistry.register_model",
            "model_registry",
            "stub_plugin_cpu",
        ),
        (
            "plugin_registration",
            "vllm.plugins.load_general_plugins",
            "load_general_plugins",
            "stub_plugin_cpu",
        ),
    ],
)
def test_non_ppu_repro_hard_gate_real_stub_plugins_all_five_kinds(
    tmp_path, stage, target_signature, expected_kind, expected_mode
):
    """Execute real inline stub plugin / CPU checks on a 2-commit git repo for all 5 kinds."""
    repo = tmp_path / "mock-vllm"
    shas = _make_mock_vllm_git_repo(repo)
    repro_out = tmp_path / f"repro-{expected_kind}.json"

    # Real subprocess probe checking out $CANDIDATE_SHA and executing materialized
    # $STUB_SITE_DIR + $STUB_CHECK_SCRIPT with zero vllm-sail and zero PPU hardware.
    probe_cmd = (
        "set -euo pipefail\n"
        'wt="$(mktemp -d)"\n'
        'trap \'git -C "$VLLM_REPO" worktree remove --force "$wt" >/dev/null 2>&1 || rm -rf "$wt"\' EXIT\n'
        'git -C "$VLLM_REPO" worktree add --quiet --detach "$wt" "$CANDIDATE_SHA"\n'
        f'PYTHONPATH="$STUB_SITE_DIR:$wt" "{sys.executable}" "$STUB_CHECK_SCRIPT"\n'
    )
    rc = post_merge.main(
        [
            "verify-non-ppu-repro",
            "--vllm-repo",
            str(repo),
            "--last-good-sha",
            shas["good"],
            "--first-failing-sha",
            shas[expected_kind],
            "--stage",
            stage,
            "--target-signature",
            target_signature,
            "--probe-cmd",
            probe_cmd,
            "--output-json",
            str(repro_out),
        ]
    )
    assert rc == 0
    repro_data = json.loads(repro_out.read_text(encoding="utf-8"))
    post_merge.assert_zero_sensitive_leaks(repro_out.read_text(encoding="utf-8"))
    expected_phase = (
        "build_import" if expected_mode == "cpu_empty_build" else "stub_check"
    )
    expected_source = (
        "upstream_import_check"
        if expected_mode == "cpu_empty_build"
        else "stub_plugin_check"
    )
    assert repro_data == {
        "failure_source": expected_source,
        "first_failing_check_executed": True,
        "first_failing_exit_code": 1,
        "first_failing_phase": expected_phase,
        "first_failing_sha": shas[expected_kind],
        "infra_error": False,
        "last_good_check_executed": True,
        "last_good_exit_code": 0,
        "last_good_phase": expected_phase,
        "last_good_sha": shas["good"],
        "mode": expected_mode,
        "repro_kind": expected_kind,
        "verified": True,
    }

    # Feed verified receipt + commit author metadata into `publish` CLI end-to-end.
    commit_json = tmp_path / f"commit-{expected_kind}.json"
    commit_json.write_text(
        json.dumps(
            {
                "sha": shas[expected_kind],
                "author": {"login": "kevin-luu", "type": "User"},
            }
        ),
        encoding="utf-8",
    )
    pub_dir = tmp_path / f"pub-{expected_kind}"
    summary_path = tmp_path / f"summary-{expected_kind}.md"
    pub_rc = post_merge.main(
        [
            "publish",
            "--gear",
            "probe",
            "--status",
            "fail",
            "--sail-sha",
            _sha(901),
            "--upstream-sha",
            shas[expected_kind],
            "--cpu-status",
            "fail",
            "--ppu-status",
            "skipped",
            "--ppu-skip-reason",
            "cpu_fast_check_failed",
            "--first-failing-sha",
            shas[expected_kind],
            "--last-good-sha",
            shas["good"],
            "--failure-stage",
            stage,
            "--target-signature",
            target_signature,
            "--public-plugin-break",
            "--non-ppu-repro-json",
            str(repro_out),
            "--upstream-commit-json",
            str(commit_json),
            "--public-dir",
            str(pub_dir),
            "--step-summary",
            str(summary_path),
        ]
    )
    assert pub_rc == 0
    summary_json = json.loads(
        (pub_dir / "public-summary.json").read_text(encoding="utf-8")
    )
    issue_json = json.loads(
        (pub_dir / "issue-payload.json").read_text(encoding="utf-8")
    )
    assert summary_json["triage"]["classification"] == "plugin_interface_break"
    assert summary_json["triage"]["non_ppu_repro_verified"] is True
    assert summary_json["triage"]["non_ppu_repro_mode"] == expected_mode
    assert summary_json["triage"]["repro_kind"] == expected_kind
    assert summary_json["bisect"]["first_failing_author"] == "kevin-luu"
    assert issue_json["should_file_sail_issue"] is False
    assert issue_json["should_file_upstream_issue"] is False
    assert issue_json["first_failing_author"] == "kevin-luu"

    repro_md = issue_json["upstream_reproducer_markdown"]
    assert "````markdown\n" in repro_md
    assert "(cc @kevin-luu)" in repro_md
    assert f"(`{expected_kind}`)" in repro_md
    # Extract the embedded bash script and verify syntax with `bash -n`.
    bash_block = repro_md.split("```bash\n", 1)[1].split("\n```", 1)[0]
    script_file = tmp_path / f"repro-{expected_kind}.sh"
    script_file.write_text(bash_block + "\n", encoding="utf-8")
    syntax = subprocess.run(
        ["bash", "-n", str(script_file)], capture_output=True, text=True
    )
    assert syntax.returncode == 0, syntax.stderr
    post_merge.assert_zero_sensitive_leaks(repro_md)
    post_merge.assert_zero_sensitive_leaks(summary_path.read_text(encoding="utf-8"))


def test_non_ppu_repro_hard_gate_negative_controls_ppu_only_and_tampered_evidence(
    tmp_path,
):
    repo = tmp_path / "mock-vllm-neg"
    shas = _make_mock_vllm_git_repo(repo)
    probe_cmd = (
        "set -euo pipefail\n"
        'wt="$(mktemp -d)"\n'
        'trap \'git -C "$VLLM_REPO" worktree remove --force "$wt" >/dev/null 2>&1 || rm -rf "$wt"\' EXIT\n'
        'git -C "$VLLM_REPO" worktree add --quiet --detach "$wt" "$CANDIDATE_SHA"\n'
        f'PYTHONPATH="$STUB_SITE_DIR:$wt" "{sys.executable}" "$STUB_CHECK_SCRIPT"\n'
    )

    # 1. Real negative control A: both last_good and candidate exit 0 -> verified=False!
    no_repro = post_merge.verify_non_ppu_reproducer(
        vllm_repo=repo,
        last_good_sha=shas["good"],
        first_failing_sha=shas["good_2"],
        stage="plugin_registration",
        target_signature="vllm.plugins.load_general_plugins",
        probe_cmd=probe_cmd,
    )
    assert no_repro.last_good_exit_code == 0
    assert no_repro.first_failing_exit_code == 0
    assert no_repro.verified is False

    # 2. Real negative control B: last_good itself fails (control broken) -> verified=False!
    broken_ctrl = post_merge.verify_non_ppu_reproducer(
        vllm_repo=repo,
        last_good_sha=shas["register_oot"],
        first_failing_sha=shas["model_registry"],
        stage="plugin_registration",
        target_signature="CustomOp.register_oot",
        probe_cmd=probe_cmd,
    )
    assert broken_ctrl.last_good_exit_code != 0
    assert broken_ctrl.verified is False

    # 3. Every PPU hardware stage (ppu_probe_smoke / rc_*) WITHOUT CPU/stub repro MUST
    #    route to vllm-sail internal tracking with zero upstream payload!
    for ppu_stage in sorted(post_merge.PPU_HARDWARE_STAGES):
        out_dir = tmp_path / f"ppu-only-{ppu_stage}"
        res = post_merge.publish_sanitized_channels(
            gear="probe" if ppu_stage == "ppu_probe_smoke" else "rc_qualify",
            status="fail",
            sail_sha=_sha(910),
            upstream_sha=shas["register_oot"],
            verified_pin=VERIFIED_PIN,
            cpu_status="pass",
            ppu_status="fail",
            first_failing_sha=shas["register_oot"],
            last_good_sha=shas["good"],
            outcome=post_merge.ProbeOutcome(
                status="fail",
                stage=ppu_stage,
                target_signature="vllm.plugins.load_general_plugins",
                is_public_plugin_interface=True,
                non_ppu_repro=None,
            ),
            first_failing_author="kevin-luu",
            public_dir=out_dir,
        )
        assert (
            res["public_summary"]["triage"]["classification"]
            == "sail_internal_patch_drift"
        )
        assert res["public_summary"]["triage"]["non_ppu_repro_verified"] is False
        assert res["public_summary"]["bisect"]["first_failing_author"] is None
        assert res["issue_payload"]["should_file_sail_issue"] is True
        assert res["issue_payload"]["should_file_upstream_issue"] is False
        assert res["issue_payload"]["upstream_reproducer_markdown"] == ""
        assert "@kevin-luu" not in res["markdown"]

    # 4. Every SAIL-internal stage blocks plugin_interface_break EVEN WITH valid repro evidence!
    valid_stub = post_merge.NonPpuReproEvidence(
        mode="stub_plugin_cpu",
        repro_kind="load_general_plugins",
        last_good_sha=shas["good"],
        first_failing_sha=shas["load_general_plugins"],
        last_good_exit_code=0,
        first_failing_exit_code=1,
        verified=True,
        last_good_phase="stub_check",
        first_failing_phase="stub_check",
        last_good_check_executed=True,
        first_failing_check_executed=True,
        infra_error=False,
        failure_source="stub_plugin_check",
    )
    for internal_stage in sorted(post_merge.SAIL_INTERNAL_ONLY_STAGES):
        triage = post_merge.classify_failure(
            post_merge.ProbeOutcome(
                status="fail",
                stage=internal_stage,
                target_signature="vllm.plugins.load_general_plugins",
                is_public_plugin_interface=True,
                non_ppu_repro=valid_stub,
            ),
            last_good_sha=shas["good"],
            first_failing_sha=shas["load_general_plugins"],
            require_shas=True,
        )
        assert triage["classification"] == "sail_internal_patch_drift"
        assert triage["non_ppu_repro_verified"] is False

    # 5. Tampered / bool-trapped / stale / mismatched-stage evidence MUST fail closed!
    bad_evidences = [
        post_merge.NonPpuReproEvidence(
            "stub_plugin_cpu",
            "load_general_plugins",
            shas["good"],
            shas["load_general_plugins"],
            1,
            1,
            True,
            "stub_check",
            "stub_check",
            True,
            True,
            False,
            "stub_plugin_check",
        ),
        post_merge.NonPpuReproEvidence(
            "stub_plugin_cpu",
            "load_general_plugins",
            shas["good"],
            shas["load_general_plugins"],
            0,
            0,
            True,
            "stub_check",
            "stub_check",
            True,
            True,
            False,
            "stub_plugin_check",
        ),
        post_merge.NonPpuReproEvidence(
            "stub_plugin_cpu",
            "load_general_plugins",
            shas["good"],
            shas["load_general_plugins"],
            False,  # type: ignore[arg-type]
            True,  # type: ignore[arg-type]
            True,
            "stub_check",
            "stub_check",
            True,
            True,
            False,
            "stub_plugin_check",
        ),
        post_merge.NonPpuReproEvidence(
            "stub_plugin_cpu",
            "load_general_plugins",
            shas["good"],
            shas["good"],
            0,
            1,
            True,
            "stub_check",
            "stub_check",
            True,
            True,
            False,
            "stub_plugin_check",
        ),
        post_merge.NonPpuReproEvidence(
            "stub_plugin_cpu",
            "load_general_plugins",
            shas["good"],
            shas["register_oot"],
            0,
            1,
            True,
            "stub_check",
            "stub_check",
            True,
            True,
            False,
            "stub_plugin_check",
        ),  # mismatched first_failing_sha
        post_merge.NonPpuReproEvidence(
            "cpu_empty_build",
            "cpu_empty_build",
            shas["good"],
            shas["load_general_plugins"],
            0,
            1,
            True,
            "build_import",
            "build_install",
            True,
            False,
            False,
            "upstream_build_source",
        ),  # wrong mode for plugin_registration
        post_merge.NonPpuReproEvidence(
            "stub_plugin_cpu",
            "load_general_plugins",
            shas["good"],
            shas["load_general_plugins"],
            0,
            1,
            False,
            "stub_check",
            "stub_check",
            True,
            True,
            False,
            "stub_plugin_check",
        ),
    ]
    for idx, ev in enumerate(bad_evidences):
        triage = post_merge.classify_failure(
            post_merge.ProbeOutcome(
                status="fail",
                stage="plugin_registration",
                target_signature="vllm.plugins.load_general_plugins",
                is_public_plugin_interface=True,
                non_ppu_repro=ev,
            ),
            last_good_sha=shas["good"],
            first_failing_sha=shas["load_general_plugins"],
            require_shas=True,
        )
        assert triage["classification"] == "sail_internal_patch_drift", (
            f"case {idx} leaked"
        )
        assert triage["non_ppu_repro_verified"] is False

    # 6. Forged JSON (`verified: true` with `first_failing_exit_code: 0`) & corrupted JSON on `publish` CLI
    forged_json = tmp_path / "forged-repro.json"
    forged_json.write_text(
        json.dumps(
            {
                "mode": "stub_plugin_cpu",
                "repro_kind": "load_general_plugins",
                "last_good_sha": shas["good"],
                "first_failing_sha": shas["load_general_plugins"],
                "last_good_exit_code": 0,
                "first_failing_exit_code": 0,
                "last_good_phase": "stub_check",
                "first_failing_phase": "stub_check",
                "last_good_check_executed": True,
                "first_failing_check_executed": True,
                "infra_error": False,
                "failure_source": "stub_plugin_check",
                "verified": True,
            }
        ),
        encoding="utf-8",
    )
    corrupt_json = tmp_path / "corrupt-repro.json"
    corrupt_json.write_text("{not valid json", encoding="utf-8")
    for label, path in (("forged", forged_json), ("corrupt", corrupt_json)):
        pub_dir = tmp_path / f"pub-{label}"
        assert (
            post_merge.main(
                [
                    "publish",
                    "--gear",
                    "probe",
                    "--status",
                    "fail",
                    "--sail-sha",
                    _sha(911),
                    "--upstream-sha",
                    shas["load_general_plugins"],
                    "--first-failing-sha",
                    shas["load_general_plugins"],
                    "--last-good-sha",
                    shas["good"],
                    "--failure-stage",
                    "plugin_registration",
                    "--target-signature",
                    "vllm.plugins.load_general_plugins",
                    "--public-plugin-break",
                    "--non-ppu-repro-json",
                    str(path),
                    "--public-dir",
                    str(pub_dir),
                ]
            )
            == 0
        )
        pub_sum = json.loads(
            (pub_dir / "public-summary.json").read_text(encoding="utf-8")
        )
        pub_iss = json.loads(
            (pub_dir / "issue-payload.json").read_text(encoding="utf-8")
        )
        assert pub_sum["triage"]["classification"] == "sail_internal_patch_drift"
        assert pub_iss["should_file_sail_issue"] is True
        assert pub_iss["upstream_reproducer_markdown"] == ""


def test_upstream_reproducer_author_resolution_fenced_template_and_zero_bare_ping(
    tmp_path, monkeypatch
):
    bad_sha = _sha(920)
    good_sha = _sha(919)
    sail_sha = _sha(921)

    # 1. Handle sanitization & commit metadata parsing matrix.
    assert post_merge.sanitize_github_login("youkaichao") == "youkaichao"
    assert post_merge.sanitize_github_login("kevin-luu") == "kevin-luu"
    assert post_merge.sanitize_github_login("A1") == "A1"
    for invalid in (
        "",
        None,
        "@alice",
        "-leading",
        "trailing-",
        "double--hyphen",
        "dependabot[bot]",
        "vllm-project/maintainers",
        "alice\ncc @bob",
        "a" * 40,
        "wb-998877",
        "step-5-preview",
    ):
        assert post_merge.sanitize_github_login(invalid) == "", f"accepted {invalid!r}"

    # Valid user commit metadata
    assert (
        post_merge.parse_commit_author_login(
            {"sha": bad_sha, "author": {"login": "youkaichao", "type": "User"}},
            expected_sha=bad_sha,
        )
        == "youkaichao"
    )
    # Bot user (even with noreply email) degrades to ""
    assert (
        post_merge.parse_commit_author_login(
            {
                "sha": bad_sha,
                "author": {"login": "dependabot", "type": "Bot"},
                "commit": {
                    "author": {"email": "49699333+dependabot@users.noreply.github.com"}
                },
            },
            expected_sha=bad_sha,
        )
        == ""
    )
    # Null author + GitHub noreply email resolves login cleanly
    assert (
        post_merge.parse_commit_author_login(
            {
                "sha": bad_sha,
                "author": None,
                "commit": {
                    "author": {"email": "987654+kevin-luu@users.noreply.github.com"}
                },
            },
            expected_sha=bad_sha,
        )
        == "kevin-luu"
    )
    # Null author + corporate/private email degrades to "" (zero email leak)
    assert (
        post_merge.parse_commit_author_login(
            {
                "sha": bad_sha,
                "author": None,
                "commit": {"author": {"email": "dev@alibaba-inc.com"}},
            },
            expected_sha=bad_sha,
        )
        == ""
    )
    # Mismatched commit SHA degrades to ""
    assert (
        post_merge.parse_commit_author_login(
            {"sha": good_sha, "author": {"login": "youkaichao", "type": "User"}},
            expected_sha=bad_sha,
        )
        == ""
    )

    # 2. Remote lookup helper (mocked urlopen): valid 200 vs network failure + repo allowlist.
    class _FakeResp:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def read(self, _n: int = -1) -> bytes:
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        post_merge.urllib.request,
        "urlopen",
        lambda req, timeout=10: _FakeResp(
            json.dumps(
                {"sha": bad_sha, "author": {"login": "youkaichao", "type": "User"}}
            ).encode("utf-8")
        ),
    )
    assert (
        post_merge.resolve_upstream_commit_author(bad_sha, fetch_remote=True)
        == "youkaichao"
    )
    # Non-allowlisted repo is rejected fail-closed.
    assert (
        post_merge.resolve_upstream_commit_author(
            bad_sha, repo="internal/repo", fetch_remote=True
        )
        == ""
    )

    def _net_fail(*args, **kwargs):
        raise OSError("network unavailable")

    monkeypatch.setattr(post_merge.urllib.request, "urlopen", _net_fail)
    assert post_merge.resolve_upstream_commit_author(bad_sha, fetch_remote=True) == ""

    # 3. Fenced markdown template invariant: `@author` ONLY inside ````markdown fence, ZERO bare pings!
    repro_ev = post_merge.NonPpuReproEvidence(
        mode="stub_plugin_cpu",
        repro_kind="platform_plugin",
        last_good_sha=good_sha,
        first_failing_sha=bad_sha,
        last_good_exit_code=0,
        first_failing_exit_code=1,
        verified=True,
        last_good_phase="stub_check",
        first_failing_phase="stub_check",
        last_good_check_executed=True,
        first_failing_check_executed=True,
        infra_error=False,
        failure_source="stub_plugin_check",
    )
    with_author = post_merge.render_upstream_reproducer(
        first_failing_sha=bad_sha,
        last_good_sha=good_sha,
        sail_sha=sail_sha,
        stage="plugin_registration",
        target_signature="vllm.platforms.Platform",
        repro_kind="platform_plugin",
        first_failing_author="youkaichao",
        non_ppu_repro=repro_ev,
        run_url="https://github.com/vllm-project/vllm-sail/actions/runs/55555",
    )
    parts = with_author.split("````markdown\n", 1)
    assert len(parts) == 2
    before_fence = parts[0]
    inside_and_after = parts[1].split("\n````", 1)
    assert len(inside_and_after) == 2
    inside_fence, after_fence = inside_and_after
    assert "@" not in before_fence
    assert "@" not in after_fence
    assert f"`{bad_sha}` (cc @youkaichao)" in inside_fence
    assert "vllm.platform_plugins" in inside_fence
    assert "StubPlatform" in inside_fence
    assert f"`{good_sha}` -> `0`; `{bad_sha}` -> `1`" in inside_fence

    # Missing/sensitive author degrades cleanly to SHA-only with zero `@` mentions.
    without_author = post_merge.render_upstream_reproducer(
        first_failing_sha=bad_sha,
        last_good_sha=good_sha,
        sail_sha=sail_sha,
        stage="plugin_registration",
        target_signature="ModelRegistry.register_model",
        repro_kind="model_registry",
        first_failing_author="wb-998877",
        non_ppu_repro=None,
    )
    assert "@" not in without_author
    assert bad_sha in without_author
    assert "StubOOTForCausalLM" in without_author
    post_merge.assert_zero_sensitive_leaks(without_author)

    # Verify full GITHUB_STEP_SUMMARY output also has zero bare pings outside fenced block!
    summary_file = tmp_path / "step_summary.md"
    post_merge.publish_sanitized_channels(
        gear="probe",
        status="fail",
        sail_sha=sail_sha,
        upstream_sha=bad_sha,
        verified_pin=VERIFIED_PIN,
        cpu_status="fail",
        ppu_status="skipped",
        ppu_skip_reason="cpu_fast_check_failed",
        first_failing_sha=bad_sha,
        last_good_sha=good_sha,
        outcome=post_merge.ProbeOutcome(
            status="fail",
            stage="plugin_registration",
            target_signature="vllm.platforms.Platform",
            is_public_plugin_interface=True,
            non_ppu_repro=repro_ev,
        ),
        first_failing_author="youkaichao",
        public_dir=tmp_path / "pub-fenced",
        step_summary_path=summary_file,
    )
    full_md = summary_file.read_text(encoding="utf-8")
    md_before, md_rest = full_md.split("````markdown\n", 1)
    md_inside, md_after = md_rest.split("\n````", 1)
    assert "@" not in md_before
    assert "@" not in md_after
    assert "@youkaichao" in md_inside


def test_verify_non_ppu_reproducer_builtin_runner_workflow_and_fingerprint_compat(
    tmp_path, monkeypatch
):
    # 1. Built-in runner path (`probe_cmd=""`) uses git worktree, VLLM_TARGET_DEVICE=empty,
    #    VLLM_PLUGINS=stub, cleans up worktrees, and fails closed on TimeoutExpired.
    good_sha = _sha(930)
    bad_sha = _sha(931)
    calls: list[tuple[list[str], dict[str, str]]] = []
    removed_worktrees: list[str] = []

    def _fake_run(cmd, **kwargs):
        cmd_list = [str(c) for c in cmd]
        calls.append((cmd_list, dict(kwargs.get("env") or {})))
        if (
            cmd_list[:4] == ["git", "-C", str(tmp_path), "worktree"]
            and cmd_list[4] == "add"
        ):
            wt_path = Path(cmd_list[7])
            wt_path.mkdir(parents=True, exist_ok=True)
            (wt_path / "tools").mkdir(exist_ok=True)
            (wt_path / "tools" / "use_existing_torch.py").write_text(
                "", encoding="utf-8"
            )
            return subprocess.CompletedProcess(cmd_list, 0, "", "")
        if (
            cmd_list[:4] == ["git", "-C", str(tmp_path), "worktree"]
            and cmd_list[4] == "remove"
        ):
            removed_worktrees.append(cmd_list[6])
            return subprocess.CompletedProcess(cmd_list, 0, "", "")
        if "repro_check.py" in " ".join(cmd_list):
            sha = (kwargs.get("env") or {}).get("CANDIDATE_SHA")
            return subprocess.CompletedProcess(
                cmd_list, 0 if sha == good_sha else 1, "", ""
            )
        return subprocess.CompletedProcess(cmd_list, 0, "", "")

    monkeypatch.setattr(post_merge.subprocess, "run", _fake_run)
    ev = post_merge.verify_non_ppu_reproducer(
        vllm_repo=tmp_path,
        last_good_sha=good_sha,
        first_failing_sha=bad_sha,
        stage="plugin_registration",
        target_signature="CustomOp.register_oot",
    )
    assert ev.verified is True
    assert ev.mode == "stub_plugin_cpu"
    assert ev.repro_kind == "register_oot"
    assert ev.last_good_exit_code == 0
    assert ev.first_failing_exit_code == 1
    assert len(removed_worktrees) == 2
    pip_calls = [c for c in calls if "pip" in c[0] and "install" in c[0]]
    assert len(pip_calls) == 2
    for _, env_used in pip_calls:
        assert env_used["VLLM_TARGET_DEVICE"] == "empty"
        assert env_used["CUDA_VISIBLE_DEVICES"] == ""
        assert env_used["VLLM_PLUGINS"] == "stub"

    # Timeout during first_failing_sha -> verified=False (never false-positive regression!)
    def _timeout_on_bad(cmd, **kwargs):
        cmd_list = [str(c) for c in cmd]
        if (
            "repro_check.py" in " ".join(cmd_list)
            and (kwargs.get("env") or {}).get("CANDIDATE_SHA") == bad_sha
        ):
            raise subprocess.TimeoutExpired(cmd_list, 10)
        return _fake_run(cmd, **kwargs)

    monkeypatch.setattr(post_merge.subprocess, "run", _timeout_on_bad)
    ev_to = post_merge.verify_non_ppu_reproducer(
        vllm_repo=tmp_path,
        last_good_sha=good_sha,
        first_failing_sha=bad_sha,
        stage="plugin_registration",
        target_signature="CustomOp.register_oot",
    )
    assert ev_to.verified is False
    assert ev_to.first_failing_exit_code == 124

    # 2. Verify upstream-canary.yaml wires verify-non-ppu-repro, --non-ppu-repro-json,
    #    --fetch-upstream-author, and explicitly forbids cross-repo upstream issue creation.
    wf_raw = (REPO_ROOT / ".github/workflows/upstream-canary.yaml").read_text(
        encoding="utf-8"
    )
    assert "verify-non-ppu-repro" in wf_raw
    assert "/tmp/post-merge-cpu/non-ppu-repro.json" in wf_raw
    assert "--non-ppu-repro-json /tmp/post-merge-meta/non-ppu-repro.json" in wf_raw
    assert "--fetch-upstream-author" in wf_raw
    assert "payload.should_file_upstream_issue" in wf_raw

    # 3. Verify ppu_native_fingerprint_compat.json matches current _ppu-build.yaml /
    #    build_ppu_wheels.sh / ppu_wheel_manifest.py after merging main.
    compat = json.loads(
        (REPO_ROOT / "scripts/ci/ppu_native_fingerprint_compat.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(compat) == set(wheel_manifest.FINGERPRINT_COMPAT_PATHS)
    for rel_path, entry in compat.items():
        actual_sha = hashlib.sha256((REPO_ROOT / rel_path).read_bytes()).hexdigest()
        assert entry["current"] == actual_sha, (
            f"ppu_native_fingerprint_compat.json stale for {rel_path}"
        )


def test_review_r1_p1_install_infra_and_unexecuted_stub_cannot_verify_or_classify_upstream(
    tmp_path, monkeypatch
):
    """Exact Round-1 P1 counterexamples + normal controls across producer -> evidence -> consumer."""
    good_sha = _sha(940)
    bad_sha = _sha(941)
    sail_sha = _sha(942)

    def _make_fake_run(
        *,
        bad_pip_rc: int = 0,
        bad_pip_stderr: str = "",
        bad_check_rc: int = 1,
        bad_check_stderr: str = "",
    ):
        counters = {"bad_stub_executions": 0}

        def _fake_run(cmd, **kwargs):
            cmd_list = [str(c) for c in cmd]
            env = kwargs.get("env") or {}
            sha = env.get("CANDIDATE_SHA")
            if (
                cmd_list[:4] == ["git", "-C", str(tmp_path), "worktree"]
                and cmd_list[4] == "add"
            ):
                wt_path = Path(cmd_list[7])
                wt_path.mkdir(parents=True, exist_ok=True)
                return subprocess.CompletedProcess(cmd_list, 0, "", "")
            if (
                cmd_list[:4] == ["git", "-C", str(tmp_path), "worktree"]
                and cmd_list[4] == "remove"
            ):
                return subprocess.CompletedProcess(cmd_list, 0, "", "")
            if "pip" in cmd_list and "install" in cmd_list:
                if sha == bad_sha:
                    return subprocess.CompletedProcess(
                        cmd_list, bad_pip_rc, "", bad_pip_stderr
                    )
                return subprocess.CompletedProcess(cmd_list, 0, "", "")
            if "repro_check.py" in " ".join(cmd_list):
                if sha == bad_sha:
                    counters["bad_stub_executions"] += 1
                    return subprocess.CompletedProcess(
                        cmd_list, bad_check_rc, "", bad_check_stderr
                    )
                return subprocess.CompletedProcess(cmd_list, 0, "", "")
            return subprocess.CompletedProcess(cmd_list, 0, "", "")

        return _fake_run, counters

    # Counterexample 1 (exact Round-1 P1): stub_plugin_cpu where bad pip hits ENOSPC;
    # stub never executes -> MUST be unverified, infra_error=True, sail_internal_patch_drift, no upstream draft!
    fake_enospc, cnt1 = _make_fake_run(
        bad_pip_rc=1,
        bad_pip_stderr="ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device",
    )
    monkeypatch.setattr(post_merge.subprocess, "run", fake_enospc)
    ev_enospc = post_merge.verify_non_ppu_reproducer(
        vllm_repo=tmp_path,
        last_good_sha=good_sha,
        first_failing_sha=bad_sha,
        stage="plugin_registration",
        target_signature="vllm.plugins.load_general_plugins",
    )
    assert cnt1["bad_stub_executions"] == 0
    assert ev_enospc.verified is False
    assert ev_enospc.infra_error is True
    assert ev_enospc.first_failing_check_executed is False
    assert ev_enospc.first_failing_phase == "infra_error"
    pub_enospc = post_merge.publish_sanitized_channels(
        gear="probe",
        status="fail",
        sail_sha=sail_sha,
        upstream_sha=bad_sha,
        verified_pin=VERIFIED_PIN,
        cpu_status="fail",
        ppu_status="skipped",
        ppu_skip_reason="cpu_fast_check_failed",
        first_failing_sha=bad_sha,
        last_good_sha=good_sha,
        outcome=post_merge.ProbeOutcome(
            status="fail",
            stage="plugin_registration",
            target_signature="vllm.plugins.load_general_plugins",
            is_public_plugin_interface=True,
            non_ppu_repro=ev_enospc,
        ),
        first_failing_author="kevin-luu",
        public_dir=tmp_path / "pub-p1-enospc",
    )
    assert (
        pub_enospc["public_summary"]["triage"]["classification"]
        == "sail_internal_patch_drift"
    )
    assert pub_enospc["issue_payload"]["upstream_reproducer_markdown"] == ""
    assert pub_enospc["issue_payload"]["should_file_sail_issue"] is True

    # Counterexample 2: stub_plugin_cpu where bad pip fails with a non-infra build error
    # (stub check STILL never executed) -> MUST remain verified=False!
    fake_bad_pip, cnt2 = _make_fake_run(
        bad_pip_rc=1,
        bad_pip_stderr="SyntaxError: invalid syntax in setup.py",
    )
    monkeypatch.setattr(post_merge.subprocess, "run", fake_bad_pip)
    ev_no_stub = post_merge.verify_non_ppu_reproducer(
        vllm_repo=tmp_path,
        last_good_sha=good_sha,
        first_failing_sha=bad_sha,
        stage="plugin_registration",
        target_signature="CustomOp.register_oot",
    )
    assert cnt2["bad_stub_executions"] == 0
    assert ev_no_stub.verified is False
    assert ev_no_stub.first_failing_phase == "install"
    assert ev_no_stub.first_failing_check_executed is False

    # Counterexample 3: cpu_empty_build where bad pip fails with ENOSPC / network / SIGKILL /
    # Errno 122 (EDQUOT) / Errno 5 (EIO) / Errno 116 (ESTALE) / pip host RuntimeError /
    # 0-line metadata wrapper / opaque error -> infra_error, verified=False!
    for err_rc, err_msg in (
        (1, "OSError: [Errno 28] No space left on device"),
        (1, "OSError: [Errno 122] Disk quota exceeded"),
        (1, "OSError: [Errno 5] Input/output error"),
        (
            1,
            "Building wheel for vllm (pyproject.toml) ... error\n"
            "OSError: [Errno 122] Disk quota exceeded",
        ),
        (1, "OSError: [Errno 116] Stale file handle"),
        (
            1,
            "ERROR: Could not install packages due to an OSError: [Errno 2] host temp missing",
        ),
        (
            1,
            "Traceback (most recent call last):\n"
            '  File "/usr/lib/python3/site-packages/pip/_vendor/rich/live.py", line 113, in start\n'
            '  File "/usr/lib/python3.12/threading.py", line 992, in start\n'
            "RuntimeError: can't start new thread",
        ),
        (
            1,
            "× Preparing metadata (pyproject.toml) did not run successfully.\n"
            "│ exit code: 1\n"
            "╰─> [0 lines of output]\n"
            "note: This error originates from a subprocess, and is likely not a problem with pip.\n"
            "error: metadata-generation-failed",
        ),
        (1, "RuntimeError: runner worker pool exhausted"),
        (
            1,
            "RuntimeError: runner worker pool exhausted\n"
            "ERROR: Could not build wheels for vllm, which is required to install pyproject.toml-based projects",
        ),
        (
            1,
            "RuntimeError: runner crashed (env VLLM_TARGET_DEVICE=empty in setup.py)\n"
            "ERROR: Could not build wheels for vllm, which is required to install pyproject.toml-based projects",
        ),
        (
            1,
            "Traceback (most recent call last):\n"
            '  File "/tmp/vllm-bbbbbbbbbbbb/setup.py", line 20, in <module>\n'
            "    t.start()\n"
            '  File "/usr/lib/python3.11/threading.py", line 975, in start\n'
            "    _start_new_thread()\n"
            "RuntimeError: host thread pool exhausted",
        ),
        (
            1,
            "Traceback (most recent call last):\n"
            '  File "/tmp/vllm-bbbbbbbbbbbb/setup.py", line 30, in run\n'
            '    subprocess.check_call(["cmake", ".."])\n'
            '  File "/usr/lib/python3.11/subprocess.py", line 1950, in _execute_child\n'
            "    raise child_exception_type(errno_num, err_msg, err_filename)\n"
            "FileNotFoundError: [Errno 2] No such file or directory: 'cmake'",
        ),
        (
            1,
            "Traceback (most recent call last):\n"
            '  File "/usr/lib/python3.11/urllib/request.py", line 100, in urlopen\n'
            '    raise ValueError("bad proxy config")\n'
            "ValueError: bad proxy config\n\n"
            "The above exception was the direct cause of the following exception:\n\n"
            "Traceback (most recent call last):\n"
            '  File "/tmp/vllm-bbbbbbbbbbbb/setup.py", line 50, in <module>\n'
            '    raise RuntimeError("wheel fetch failed")\n'
            "RuntimeError: wheel fetch failed",
        ),
        (
            1,
            "Traceback (most recent call last):\n"
            '  File "/tmp/vllm-bbbbbbbbbbbb/setup.py", line 1, in <module>\n'
            "RuntimeError: VLLM_TARGET_DEVICE=empty unsupported\n"
            "Exception: runner worker pool exhausted",
        ),
        (
            1,
            "Traceback (most recent call last):\n"
            '  File "<string>", line 1, in <module>\n'
            "RuntimeError: VLLM_TARGET_DEVICE=empty unsupported",
        ),
        (
            1,
            "Traceback (most recent call last):\n"
            '  File "/opt/homebrew/lib/python3.11/site-packages/setuptools/build_meta.py", line 374, in prepare_metadata_for_build_wheel\n'
            "    self.run_setup()\n"
            "RuntimeError: corrupted setuptools installation",
        ),
        (1, "ConnectionError: Temporary failure in name resolution"),
        (
            1,
            "",
        ),  # opaque pip failure without any upstream build-backend/source error signal
        (137, "Killed"),
    ):
        fake_build_infra, _ = _make_fake_run(bad_pip_rc=err_rc, bad_pip_stderr=err_msg)
        monkeypatch.setattr(post_merge.subprocess, "run", fake_build_infra)
        ev_build_infra = post_merge.verify_non_ppu_reproducer(
            vllm_repo=tmp_path,
            last_good_sha=good_sha,
            first_failing_sha=bad_sha,
            stage="wheel_build",
            target_signature="VLLM_TARGET_DEVICE=empty",
        )
        assert ev_build_infra.verified is False, f"leaked infra stderr={err_msg!r}"
        assert ev_build_infra.infra_error is True, f"not infra stderr={err_msg!r}"
        assert ev_build_infra.first_failing_phase == "infra_error"
        assert ev_build_infra.failure_source == "infra_or_unknown"
        assert (
            post_merge.classify_failure(
                post_merge.ProbeOutcome(
                    status="fail",
                    stage="wheel_build",
                    target_signature="VLLM_TARGET_DEVICE=empty",
                    is_public_plugin_interface=True,
                    non_ppu_repro=ev_build_infra,
                ),
                last_good_sha=good_sha,
                first_failing_sha=bad_sha,
                require_shas=True,
            )["classification"]
            == "sail_internal_patch_drift"
        )

    # Normal controls 1–4: cpu_empty_build where bad pip fails with genuine structurally
    # attributed upstream build/packaging errors (setup.py RuntimeError, setup.py missing
    # requirements/common.txt [Errno 2], PEP 517 _in_process.py + setup.py + pip wheel summary,
    # and setup.py -> CMake Error at CMakeLists.txt) -> verified=True!
    for build_err_msg in (
        "Traceback (most recent call last):\n"
        '  File "/tmp/vllm-bbbbbbbbbbbb/setup.py", line 18, in <module>\n'
        '    raise RuntimeError("VLLM_TARGET_DEVICE=empty unsupported in setup.py")\n'
        "RuntimeError: VLLM_TARGET_DEVICE=empty unsupported in setup.py",
        "Preparing metadata (pyproject.toml) ... error\n"
        "Traceback (most recent call last):\n"
        '  File "/tmp/vllm-bbbbbbbbbbbb/setup.py", line 42, in get_requirements\n'
        '    with open("requirements/common.txt") as f:\n'
        "FileNotFoundError: [Errno 2] No such file or directory: 'requirements/common.txt'",
        "  error: subprocess-exited-with-error\n"
        "  × Preparing metadata (pyproject.toml) did not run successfully.\n"
        "  │ exit code: 1\n"
        "  ╰─> [10 lines of output]\n"
        "      Traceback (most recent call last):\n"
        '        File "/opt/py/lib/python3.13/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 389, in <module>\n'
        "          main()\n"
        '        File "/tmp/vllm-bbbbbbbbbbbb/setup.py", line 12, in <module>\n'
        '          raise RuntimeError("VLLM_TARGET_DEVICE=empty broken")\n'
        "      RuntimeError: VLLM_TARGET_DEVICE=empty broken\n"
        "      [end of output]\n"
        "  note: This error originates from a subprocess, and is likely not a problem with pip.\n"
        "error: metadata-generation-failed\n"
        "ERROR: Could not build wheels for vllm, which is required to install pyproject.toml-based projects",
        "CMake Error at CMakeLists.txt:42 (message):\n"
        "  Unsupported target device empty\n"
        "Traceback (most recent call last):\n"
        '  File "/tmp/vllm-bbbbbbbbbbbb/setup.py", line 210, in build_extensions\n'
        '    subprocess.check_call(["cmake", ".."])\n'
        '  File "/usr/lib/python3.11/subprocess.py", line 413, in check_call\n'
        "    raise CalledProcessError(retcode, cmd)\n"
        "subprocess.CalledProcessError: Command '['cmake', '..']' returned non-zero exit status 1.",
        # Real setuptools 80.9.0 build_meta.run_setup -> exec(code, locals()) `<string>` frames:
        "  error: subprocess-exited-with-error\n"
        "  × Preparing metadata (pyproject.toml) did not run successfully.\n"
        "  │ exit code: 1\n"
        "  ╰─> [15 lines of output]\n"
        "      Traceback (most recent call last):\n"
        '        File "/opt/homebrew/lib/python3.11/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 389, in <module>\n'
        "          main()\n"
        '        File "/opt/homebrew/lib/python3.11/site-packages/setuptools/build_meta.py", line 374, in prepare_metadata_for_build_wheel\n'
        "          self.run_setup()\n"
        '        File "/opt/homebrew/lib/python3.11/site-packages/setuptools/build_meta.py", line 317, in run_setup\n'
        "          exec(code, locals())\n"
        '        File "<string>", line 1, in <module>\n'
        "      RuntimeError: VLLM_TARGET_DEVICE=empty unsupported in setup.py\n"
        "      [end of output]\n"
        "error: metadata-generation-failed",
        "      Traceback (most recent call last):\n"
        '        File "/opt/homebrew/lib/python3.11/site-packages/setuptools/build_meta.py", line 317, in run_setup\n'
        "          exec(code, locals())\n"
        '        File "<string>", line 1, in <module>\n'
        "      FileNotFoundError: [Errno 2] No such file or directory: 'requirements/common.txt'",
        "      Traceback (most recent call last):\n"
        '        File "/opt/homebrew/lib/python3.11/site-packages/setuptools/build_meta.py", line 317, in run_setup\n'
        "          exec(code, locals())\n"
        '        File "<string>", line 1\n'
        "          def broken(:\n"
        "                     ^\n"
        "      SyntaxError: invalid syntax",
    ):
        fake_real_build_fail, _ = _make_fake_run(
            bad_pip_rc=1,
            bad_pip_stderr=build_err_msg,
        )
        monkeypatch.setattr(post_merge.subprocess, "run", fake_real_build_fail)
        ev_real_build = post_merge.verify_non_ppu_reproducer(
            vllm_repo=tmp_path,
            last_good_sha=good_sha,
            first_failing_sha=bad_sha,
            stage="wheel_build",
            target_signature="VLLM_TARGET_DEVICE=empty",
        )
        assert ev_real_build.verified is True, (
            f"rejected genuine build err={build_err_msg!r}"
        )
        assert ev_real_build.infra_error is False
        assert ev_real_build.last_good_phase == "build_import"
        assert ev_real_build.last_good_check_executed is True
        assert ev_real_build.first_failing_phase == "build_install"
        assert ev_real_build.failure_source == "upstream_build_source"
        assert (
            post_merge.classify_failure(
                post_merge.ProbeOutcome(
                    status="fail",
                    stage="wheel_build",
                    target_signature="VLLM_TARGET_DEVICE=empty",
                    is_public_plugin_interface=True,
                    non_ppu_repro=ev_real_build,
                ),
                last_good_sha=good_sha,
                first_failing_sha=bad_sha,
                require_shas=True,
            )["classification"]
            == "plugin_interface_break"
        )

    # Counterexample 4: forged serialized JSON claiming verified=True with invalid phase,
    # invalid/mismatched failure_source, or missing required fields -> fails closed!
    forged_payloads = [
        {
            "mode": "stub_plugin_cpu",
            "repro_kind": "load_general_plugins",
            "last_good_sha": good_sha,
            "first_failing_sha": bad_sha,
            "last_good_exit_code": 0,
            "first_failing_exit_code": 1,
            "last_good_phase": "stub_check",
            "first_failing_phase": "install",
            "last_good_check_executed": True,
            "first_failing_check_executed": False,
            "infra_error": False,
            "failure_source": "stub_plugin_check",
            "verified": True,
        },
        {
            "mode": "cpu_empty_build",
            "repro_kind": "cpu_empty_build",
            "last_good_sha": good_sha,
            "first_failing_sha": bad_sha,
            "last_good_exit_code": 0,
            "first_failing_exit_code": 1,
            "last_good_phase": "build_import",
            "first_failing_phase": "infra_error",
            "last_good_check_executed": True,
            "first_failing_check_executed": False,
            "infra_error": True,
            "failure_source": "infra_or_unknown",
            "verified": True,
        },
        # Forged build_install with failure_source="infra_or_unknown" must fail closed!
        {
            "mode": "cpu_empty_build",
            "repro_kind": "cpu_empty_build",
            "last_good_sha": good_sha,
            "first_failing_sha": bad_sha,
            "last_good_exit_code": 0,
            "first_failing_exit_code": 1,
            "last_good_phase": "build_import",
            "first_failing_phase": "build_install",
            "last_good_check_executed": True,
            "first_failing_check_executed": False,
            "infra_error": False,
            "failure_source": "infra_or_unknown",
            "verified": True,
        },
        # Legacy JSON missing failure_source must also fail closed!
        {
            "mode": "cpu_empty_build",
            "repro_kind": "cpu_empty_build",
            "last_good_sha": good_sha,
            "first_failing_sha": bad_sha,
            "last_good_exit_code": 0,
            "first_failing_exit_code": 1,
            "last_good_phase": "build_import",
            "first_failing_phase": "build_install",
            "last_good_check_executed": True,
            "first_failing_check_executed": False,
            "infra_error": False,
            "verified": True,
        },
        # Legacy JSON missing execution phase / check_executed fields must also fail closed!
        {
            "mode": "stub_plugin_cpu",
            "repro_kind": "load_general_plugins",
            "last_good_sha": good_sha,
            "first_failing_sha": bad_sha,
            "last_good_exit_code": 0,
            "first_failing_exit_code": 1,
            "verified": True,
        },
    ]
    for idx, payload in enumerate(forged_payloads):
        parsed = post_merge.parse_non_ppu_repro_evidence(payload)
        assert parsed is None or parsed.verified is False, f"forged case {idx} leaked"
        triage = post_merge.classify_failure(
            post_merge.ProbeOutcome(
                status="fail",
                stage="plugin_registration"
                if payload["mode"] == "stub_plugin_cpu"
                else "wheel_build",
                target_signature="vllm.plugins.load_general_plugins"
                if payload["mode"] == "stub_plugin_cpu"
                else "VLLM_TARGET_DEVICE=empty",
                is_public_plugin_interface=True,
                non_ppu_repro=parsed,
            ),
            last_good_sha=good_sha,
            first_failing_sha=bad_sha,
            require_shas=True,
        )
        assert triage["classification"] == "sail_internal_patch_drift", (
            f"forged case {idx} classified upstream"
        )

    # Counterexample 5: real subprocess `probe_cmd` in stub_plugin_cpu mode that fails
    # before invoking `$STUB_CHECK_SCRIPT` -> mechanical marker proves check did NOT execute!
    monkeypatch.undo()
    ev_probe_no_exec = post_merge.verify_non_ppu_reproducer(
        vllm_repo=tmp_path,
        last_good_sha=good_sha,
        first_failing_sha=bad_sha,
        stage="plugin_registration",
        target_signature="vllm.plugins.load_general_plugins",
        probe_cmd=(
            f'if [ "$CANDIDATE_SHA" = "{good_sha}" ]; then '
            "python3 -c \"import pathlib, os; pathlib.Path(os.environ['STUB_CHECK_SCRIPT']).exists()\" && exit 0; "
            "else echo 'pip failed before stub' >&2; exit 1; fi"
        ),
    )
    assert ev_probe_no_exec.verified is False
    assert ev_probe_no_exec.last_good_check_executed is False
    assert ev_probe_no_exec.first_failing_check_executed is False

    # Counterexample 6: real subprocess `probe_cmd` in cpu_empty_build mode where
    # last_good executes check_script cleanly, while first_failing fails inside pip/host
    # (rich/live.py thread RuntimeError, zero-line metadata wrapper, or BackendUnavailable)
    # -> producer -> JSON -> gate -> publisher must all fail closed (zero upstream draft)!
    good_vllm_dir = tmp_path / "fake_good_vllm"
    (good_vllm_dir / "vllm").mkdir(parents=True, exist_ok=True)
    (good_vllm_dir / "vllm" / "__init__.py").write_text(
        "__version__ = '0.0.1'\n", encoding="utf-8"
    )
    probe_infra_snippets = [
        (
            "Traceback (most recent call last):\n"
            '  File "/usr/lib/python3.11/site-packages/pip/_vendor/rich/live.py", line 108, in start\n'
            '  File "/usr/lib/python3.11/threading.py", line 975, in start\n'
            "RuntimeError: can't start new thread"
        ),
        (
            "  × Preparing metadata (pyproject.toml) did not run successfully.\n"
            "  │ exit code: 1\n"
            "  ╰─> [0 lines of output]\n"
            "error: metadata-generation-failed"
        ),
        (
            "ERROR: Exception:\n"
            '  File "/opt/py/lib/python3.13/site-packages/pip/_internal/operations/build/metadata.py", line 34\n'
            '  File "/opt/py/lib/python3.13/site-packages/pip/_vendor/pyproject_hooks/_impl.py", line 402\n'
            "pip._vendor.pyproject_hooks._impl.BackendUnavailable: Cannot import 'setuptools.build_meta'"
        ),
        (
            "RuntimeError: runner worker pool exhausted\n"
            "ERROR: Could not build wheels for vllm, which is required to install pyproject.toml-based projects"
        ),
        (
            "Traceback (most recent call last):\n"
            '  File "/tmp/vllm-bbbbbbbbbbbb/setup.py", line 1, in <module>\n'
            "RuntimeError: VLLM_TARGET_DEVICE=empty unsupported\n"
            "Exception: runner worker pool exhausted"
        ),
        (
            "Traceback (most recent call last):\n"
            '  File "<string>", line 1, in <module>\n'
            "RuntimeError: VLLM_TARGET_DEVICE=empty unsupported"
        ),
    ]
    sail_sha = "c" * 40
    for idx, err_blob in enumerate(probe_infra_snippets):
        err_file = tmp_path / f"probe_err_{idx}.txt"
        err_file.write_text(err_blob + "\n", encoding="utf-8")
        ev_probe_infra = post_merge.verify_non_ppu_reproducer(
            vllm_repo=tmp_path,
            last_good_sha=good_sha,
            first_failing_sha=bad_sha,
            stage="wheel_build",
            target_signature="VLLM_TARGET_DEVICE=empty",
            probe_cmd=(
                f'if [ "$CANDIDATE_SHA" = "{good_sha}" ]; then '
                f'PYTHONPATH="{good_vllm_dir}" python3 "$STUB_CHECK_SCRIPT"; '
                f'else cat "{err_file}" >&2; exit 1; fi'
            ),
        )
        assert ev_probe_infra.verified is False, f"probe infra case {idx} verified"
        assert ev_probe_infra.infra_error is True, f"probe infra case {idx} infra"
        assert ev_probe_infra.first_failing_phase == "infra_error"
        assert ev_probe_infra.failure_source == "infra_or_unknown"
        # Round-trip through JSON -> gate -> publisher
        roundtrip = post_merge.parse_non_ppu_repro_evidence(
            ev_probe_infra.to_public_dict()
        )
        assert roundtrip is not None and roundtrip.verified is False
        pub = post_merge.publish_sanitized_channels(
            gear="probe",
            status="fail",
            sail_sha=sail_sha,
            upstream_sha=bad_sha,
            verified_pin=good_sha,
            first_failing_sha=bad_sha,
            last_good_sha=good_sha,
            outcome=post_merge.ProbeOutcome(
                status="fail",
                stage="wheel_build",
                target_signature="VLLM_TARGET_DEVICE=empty",
                is_public_plugin_interface=True,
                non_ppu_repro=roundtrip,
            ),
            public_dir=tmp_path / f"pub_infra_{idx}",
        )
        assert (
            pub["public_summary"]["triage"]["classification"]
            == "sail_internal_patch_drift"
        )
        assert pub["issue_payload"]["upstream_reproducer_markdown"] == ""

    # Positive probe_cmd control: genuine PEP 517 _in_process.py + setup.py error -> verified!
    pep517_err_file = tmp_path / "probe_pep517_real.txt"
    pep517_err_file.write_text(
        "  × Preparing metadata (pyproject.toml) did not run successfully.\n"
        "  │ exit code: 1\n"
        "  ╰─> [8 lines of output]\n"
        "      Traceback (most recent call last):\n"
        '        File "/opt/py/lib/python3.13/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 389, in <module>\n'
        '        File "/tmp/vllm-bbbbbbbbbbbb/setup.py", line 10, in <module>\n'
        "      RuntimeError: VLLM_TARGET_DEVICE=empty unsupported\n"
        "      [end of output]\n"
        "error: metadata-generation-failed\n",
        encoding="utf-8",
    )
    ev_probe_real = post_merge.verify_non_ppu_reproducer(
        vllm_repo=tmp_path,
        last_good_sha=good_sha,
        first_failing_sha=bad_sha,
        stage="wheel_build",
        target_signature="VLLM_TARGET_DEVICE=empty",
        probe_cmd=(
            f'if [ "$CANDIDATE_SHA" = "{good_sha}" ]; then '
            f'PYTHONPATH="{good_vllm_dir}" python3 "$STUB_CHECK_SCRIPT"; '
            f'else cat "{pep517_err_file}" >&2; exit 1; fi'
        ),
    )
    assert ev_probe_real.verified is True
    assert ev_probe_real.infra_error is False
    assert ev_probe_real.first_failing_phase == "build_install"
    assert ev_probe_real.failure_source == "upstream_build_source"
    roundtrip_real = post_merge.parse_non_ppu_repro_evidence(
        ev_probe_real.to_public_dict()
    )
    assert roundtrip_real is not None and roundtrip_real.verified is True
    pub_real = post_merge.publish_sanitized_channels(
        gear="probe",
        status="fail",
        sail_sha=sail_sha,
        upstream_sha=bad_sha,
        verified_pin=good_sha,
        first_failing_sha=bad_sha,
        last_good_sha=good_sha,
        outcome=post_merge.ProbeOutcome(
            status="fail",
            stage="wheel_build",
            target_signature="VLLM_TARGET_DEVICE=empty",
            is_public_plugin_interface=True,
            non_ppu_repro=roundtrip_real,
        ),
        public_dir=tmp_path / "pub_real_pep517",
    )
    assert (
        pub_real["public_summary"]["triage"]["classification"]
        == "plugin_interface_break"
    )
    assert pub_real["issue_payload"]["upstream_reproducer_markdown"] != ""

    # Positive probe_cmd controls for real setuptools 80.9.0 `build_meta.run_setup` -> `<string>`
    # (RuntimeError, missing requirements/common.txt FileNotFoundError, and SyntaxError)
    for st_idx, st_stderr in enumerate(
        (
            "      Traceback (most recent call last):\n"
            '        File "/opt/homebrew/lib/python3.11/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 389, in <module>\n'
            '        File "/opt/homebrew/lib/python3.11/site-packages/setuptools/build_meta.py", line 317, in run_setup\n'
            "          exec(code, locals())\n"
            '        File "<string>", line 1, in <module>\n'
            "      RuntimeError: VLLM_TARGET_DEVICE=empty unsupported in setup.py\n",
            "      Traceback (most recent call last):\n"
            '        File "/opt/homebrew/lib/python3.11/site-packages/setuptools/build_meta.py", line 317, in run_setup\n'
            "          exec(code, locals())\n"
            '        File "<string>", line 1, in <module>\n'
            "      FileNotFoundError: [Errno 2] No such file or directory: 'requirements/common.txt'\n",
            "      Traceback (most recent call last):\n"
            '        File "/opt/homebrew/lib/python3.11/site-packages/setuptools/build_meta.py", line 317, in run_setup\n'
            "          exec(code, locals())\n"
            '        File "<string>", line 1\n'
            "          def broken(:\n"
            "                     ^\n"
            "      SyntaxError: invalid syntax\n",
        )
    ):
        st_file = tmp_path / f"probe_setuptools_{st_idx}.txt"
        st_file.write_text(st_stderr, encoding="utf-8")
        ev_st = post_merge.verify_non_ppu_reproducer(
            vllm_repo=tmp_path,
            last_good_sha=good_sha,
            first_failing_sha=bad_sha,
            stage="wheel_build",
            target_signature="VLLM_TARGET_DEVICE=empty",
            probe_cmd=(
                f'if [ "$CANDIDATE_SHA" = "{good_sha}" ]; then '
                f'PYTHONPATH="{good_vllm_dir}" python3 "$STUB_CHECK_SCRIPT"; '
                f'else cat "{st_file}" >&2; exit 1; fi'
            ),
        )
        assert ev_st.verified is True, f"setuptools case {st_idx} rejected"
        assert ev_st.infra_error is False
        assert ev_st.first_failing_phase == "build_install"
        assert ev_st.failure_source == "upstream_build_source"
        rt_st = post_merge.parse_non_ppu_repro_evidence(ev_st.to_public_dict())
        assert rt_st is not None and rt_st.verified is True
        pub_st = post_merge.publish_sanitized_channels(
            gear="probe",
            status="fail",
            sail_sha=sail_sha,
            upstream_sha=bad_sha,
            verified_pin=good_sha,
            first_failing_sha=bad_sha,
            last_good_sha=good_sha,
            outcome=post_merge.ProbeOutcome(
                status="fail",
                stage="wheel_build",
                target_signature="VLLM_TARGET_DEVICE=empty",
                is_public_plugin_interface=True,
                non_ppu_repro=rt_st,
            ),
            public_dir=tmp_path / f"pub_setuptools_{st_idx}",
        )
        assert (
            pub_st["public_summary"]["triage"]["classification"]
            == "plugin_interface_break"
        )
        assert pub_st["issue_payload"]["upstream_reproducer_markdown"] != ""


def test_review_pr374_four_inline_comments_end_to_end(tmp_path, monkeypatch):
    """Regression coverage for PR #374 review comments (3 P1s + 1 P2)."""
    import traceback

    git_env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }

    # --- 1. [P1] Rename from runtime path (`vllm/v1/worker/`) into `examples/` must NOT skip PPU ---
    wf_raw = (REPO_ROOT / ".github/workflows/upstream-canary.yaml").read_text(
        encoding="utf-8"
    )
    assert "diff --no-renames --name-only" in wf_raw

    ren_repo = tmp_path / "rename-repo"
    ren_repo.mkdir()
    subprocess.run(["git", "-C", str(ren_repo), "init", "-q"], check=True)
    worker_file = ren_repo / "vllm" / "v1" / "worker" / "gpu_probe.py"
    worker_file.parent.mkdir(parents=True, exist_ok=True)
    worker_file.write_text("def run_worker():\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(ren_repo), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(ren_repo), "commit", "-qm", "base"],
        check=True,
        env=git_env,
    )
    c_base = subprocess.check_output(
        ["git", "-C", str(ren_repo), "rev-parse", "HEAD"], text=True
    ).strip()

    dst_file = ren_repo / "examples" / "gpu_probe.py"
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(ren_repo),
            "mv",
            "vllm/v1/worker/gpu_probe.py",
            "examples/gpu_probe.py",
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(ren_repo), "commit", "-qm", "move worker to examples"],
        check=True,
        env=git_env,
    )
    c_moved = subprocess.check_output(
        ["git", "-C", str(ren_repo), "rev-parse", "HEAD"], text=True
    ).strip()

    no_rename_lines = subprocess.check_output(
        [
            "git",
            "-C",
            str(ren_repo),
            "diff",
            "--no-renames",
            "--name-only",
            c_base,
            c_moved,
        ],
        text=True,
    ).splitlines()
    assert "vllm/v1/worker/gpu_probe.py" in no_rename_lines
    assert "examples/gpu_probe.py" in no_rename_lines
    assert post_merge.should_skip_ppu_for_paths(no_rename_lines) is False

    # Defense-in-depth: even `--name-status` (`R100\tvllm/v1/worker/...\texamples/...`) requires PPU!
    name_status_lines = subprocess.check_output(
        ["git", "-C", str(ren_repo), "diff", "--name-status", c_base, c_moved],
        text=True,
    ).splitlines()
    assert post_merge.should_skip_ppu_for_paths(name_status_lines) is False

    dec_moved = post_merge.route_three_gears(
        ls_remote_data={"main_sha": c_moved, "tags": {}},
        verified_pin=c_base,
        release_baseline_tag="v0.33.0",
        requested_gear="probe",
        cursor=None,
        changed_paths_since_ppu=no_rename_lines,
        changed_files_base_sha=c_base,
    )
    assert dec_moved["run_ppu"] is True
    assert dec_moved["ppu_skip_reason"] == "none"

    # CPU-only pass without PPU execution on a renamed runtime file must NOT qualify for daily_pin_bump!
    cur_cpu_only = post_merge.advance_cursor(
        None,
        verified_pin=c_base,
        selected_gear="probe",
        target_sha=c_moved,
        status="deferred",
        cpu_status="pass",
        ppu_status="skipped",
        ppu_skip_reason="deferred_for_sail_pr_ci",
    )
    assert (
        post_merge.route_three_gears(
            ls_remote_data={"main_sha": c_moved, "tags": {}},
            verified_pin=c_base,
            release_baseline_tag="v0.33.0",
            requested_gear="daily_pin_bump",
            cursor=cur_cpu_only,
        )["target_sha"]
        == ""
    )
    # Normal control: rename strictly between two allowlisted paths still skips PPU.
    assert (
        post_merge.should_skip_ppu_for_paths(["R100\tdocs/old.md\texamples/demo.py"])
        is True
    )

    # --- 2. [P1] Initial wheel-build failure runs source-only control-first triage & bisection ---
    assert "source-build-triage" in wf_raw
    assert "if: needs.build.result == 'success'" in wf_raw

    # 2a. Untriaged/skipped fallback (exact replay from review): must be infra_error, NOT file SAIL issue!
    untriaged_out = post_merge.evaluate_workflow_outcome(
        selected_gear="probe",
        target_sha=c_moved,
        run_cpu=True,
        run_ppu=True,
        ppu_skip_reason="none",
        build_result="failure",
        cpu_result="skipped",
        ppu_smoke_result="skipped",
    )
    untriaged_dir = tmp_path / "pub-untriaged-build"
    assert (
        post_merge.main(
            [
                "publish",
                "--gear",
                "probe",
                "--status",
                untriaged_out["status"],
                "--sail-sha",
                _sha(950),
                "--upstream-sha",
                c_moved,
                "--cpu-status",
                untriaged_out["cpu_status"],
                "--ppu-status",
                untriaged_out["ppu_status"],
                "--ppu-skip-reason",
                untriaged_out["ppu_skip_reason"],
                "--failure-stage",
                untriaged_out["failure_stage"],
                "--public-dir",
                str(untriaged_dir),
            ]
        )
        == 0
    )
    untriaged_sum = json.loads(
        (untriaged_dir / "public-summary.json").read_text(encoding="utf-8")
    )
    untriaged_iss = json.loads(
        (untriaged_dir / "issue-payload.json").read_text(encoding="utf-8")
    )
    assert untriaged_sum["status"] == "infra_error"
    assert untriaged_sum["triage"]["classification"] == "infra_error"
    assert untriaged_iss["should_file_sail_issue"] is False

    # Build a 4-commit upstream repo where commits[2] breaks VLLM_TARGET_DEVICE=empty setup.py
    src_repo = tmp_path / "src-bisect-repo"
    src_repo.mkdir()
    subprocess.run(["git", "-C", str(src_repo), "init", "-q"], check=True)
    (src_repo / "vllm").mkdir()
    (src_repo / "vllm" / "__init__.py").write_text(
        "__version__ = '0.1'\n", encoding="utf-8"
    )
    (src_repo / "setup.py").write_text("# ok\n", encoding="utf-8")
    src_shas: list[str] = []
    for idx in range(5):
        (src_repo / "README.md").write_text(f"step {idx}\n", encoding="utf-8")
        if idx == 3:
            (src_repo / "setup.py").write_text(
                'raise RuntimeError("VLLM_TARGET_DEVICE=empty broken")\n',
                encoding="utf-8",
            )
        subprocess.run(["git", "-C", str(src_repo), "add", "-A"], check=True)
        subprocess.run(
            ["git", "-C", str(src_repo), "commit", "-qm", f"c{idx}"],
            check=True,
            env=git_env,
        )
        src_shas.append(
            subprocess.check_output(
                ["git", "-C", str(src_repo), "rev-parse", "HEAD"], text=True
            ).strip()
        )
    pin_file = tmp_path / "verified.commit"
    pin_file.write_text(src_shas[0] + "\n", encoding="utf-8")

    # 2b. Control-first failure -> baseline_control_failed -> infra_error, zero issues!
    ctrl_fail_dir = tmp_path / "sbt-ctrl-fail"
    assert (
        post_merge.main(
            [
                "source-build-triage",
                "--vllm-repo",
                str(src_repo),
                "--sail-root",
                str(REPO_ROOT),
                "--target-sha",
                src_shas[4],
                "--last-green-sha",
                src_shas[0],
                "--verified-pin-file",
                str(pin_file),
                "--probe-cmd",
                "echo 'Connection reset by peer' >&2; exit 1",
                "--output-dir",
                str(ctrl_fail_dir),
            ]
        )
        == 0
    )
    pub_ctrl_fail = tmp_path / "pub-sbt-ctrl-fail"
    assert (
        post_merge.main(
            [
                "publish",
                "--gear",
                "probe",
                "--status",
                "fail",
                "--sail-sha",
                _sha(951),
                "--upstream-sha",
                src_shas[4],
                "--verified-pin-file",
                str(pin_file),
                "--failure-stage",
                "wheel_build",
                "--cpu-result-json",
                str(ctrl_fail_dir / "cpu-result.json"),
                "--bisect-result-json",
                str(ctrl_fail_dir / "bisect-result.json"),
                "--public-dir",
                str(pub_ctrl_fail),
            ]
        )
        == 0
    )
    cf_sum = json.loads(
        (pub_ctrl_fail / "public-summary.json").read_text(encoding="utf-8")
    )
    cf_iss = json.loads(
        (pub_ctrl_fail / "issue-payload.json").read_text(encoding="utf-8")
    )
    assert cf_sum["status"] == "infra_error"
    assert cf_sum["triage"]["classification"] == "infra_error"
    assert cf_iss["should_file_sail_issue"] is False

    # 2c. Genuine upstream VLLM_TARGET_DEVICE=empty build break at src_shas[3] -> converges & verifies!
    sbt_probe_cmd = (
        "set -euo pipefail\n"
        'wt="$(mktemp -d)"\n'
        'trap \'git -C "$VLLM_REPO" worktree remove --force "$wt" >/dev/null 2>&1 || rm -rf "$wt"\' EXIT\n'
        'git -C "$VLLM_REPO" worktree add --quiet --detach "$wt" "$CANDIDATE_SHA"\n'
        f'"{sys.executable}" "$wt/setup.py"\n'
        f'PYTHONPATH="$wt" "{sys.executable}" "$STUB_CHECK_SCRIPT"\n'
    )
    real_sbt_dir = tmp_path / "sbt-converged"
    assert (
        post_merge.main(
            [
                "source-build-triage",
                "--vllm-repo",
                str(src_repo),
                "--sail-root",
                str(REPO_ROOT),
                "--target-sha",
                src_shas[4],
                "--last-green-sha",
                src_shas[0],
                "--verified-pin-file",
                str(pin_file),
                "--probe-cmd",
                sbt_probe_cmd,
                "--output-dir",
                str(real_sbt_dir),
            ]
        )
        == 0
    )
    pub_sbt_ok = tmp_path / "pub-sbt-converged"
    assert (
        post_merge.main(
            [
                "publish",
                "--gear",
                "probe",
                "--status",
                "fail",
                "--sail-sha",
                _sha(952),
                "--upstream-sha",
                src_shas[4],
                "--verified-pin-file",
                str(pin_file),
                "--failure-stage",
                "wheel_build",
                "--first-failing-author",
                "kevin-luu",
                "--cpu-result-json",
                str(real_sbt_dir / "cpu-result.json"),
                "--bisect-result-json",
                str(real_sbt_dir / "bisect-result.json"),
                "--non-ppu-repro-json",
                str(real_sbt_dir / "non-ppu-repro.json"),
                "--public-dir",
                str(pub_sbt_ok),
            ]
        )
        == 0
    )
    ok_sum = json.loads(
        (pub_sbt_ok / "public-summary.json").read_text(encoding="utf-8")
    )
    ok_iss = json.loads((pub_sbt_ok / "issue-payload.json").read_text(encoding="utf-8"))
    assert ok_sum["status"] == "fail"
    assert ok_sum["bisect"]["first_failing_sha"] == src_shas[3]
    assert ok_sum["bisect"]["last_good_sha"] == src_shas[2]
    assert ok_sum["bisect"]["steps"] > 0
    assert ok_sum["triage"]["classification"] == "plugin_interface_break"
    assert ok_sum["triage"]["non_ppu_repro_verified"] is True
    assert ok_iss["should_file_sail_issue"] is False
    assert "(cc @kevin-luu)" in ok_iss["upstream_reproducer_markdown"]

    # --- 3. [P1] CustomOp.register_oot against module-level `op_registry_oot` (no class attr!) ---
    inline_repo = tmp_path / "custom-op-repo"
    inline_shas = _make_mock_vllm_git_repo(inline_repo)
    custom_op_probe = (
        "set -euo pipefail\n"
        'wt="$(mktemp -d)"\n'
        'trap \'git -C "$VLLM_REPO" worktree remove --force "$wt" >/dev/null 2>&1 || rm -rf "$wt"\' EXIT\n'
        'git -C "$VLLM_REPO" worktree add --quiet --detach "$wt" "$CANDIDATE_SHA"\n'
        f'PYTHONPATH="$STUB_SITE_DIR:$wt" "{sys.executable}" -c '
        "\"from vllm.model_executor.custom_op import CustomOp; assert not hasattr(CustomOp, 'op_registry_oot')\"\n"
        f'PYTHONPATH="$STUB_SITE_DIR:$wt" "{sys.executable}" "$STUB_CHECK_SCRIPT"\n'
    )
    ev_reg_oot = post_merge.verify_non_ppu_reproducer(
        vllm_repo=inline_repo,
        last_good_sha=inline_shas["good"],
        first_failing_sha=inline_shas["register_oot"],
        stage="plugin_registration",
        target_signature="CustomOp.register_oot",
        probe_cmd=custom_op_probe,
    )
    assert ev_reg_oot.verified is True
    assert ev_reg_oot.last_good_exit_code == 0
    assert ev_reg_oot.first_failing_exit_code != 0
    assert ev_reg_oot.failure_source == "stub_plugin_check"

    # --- 4. [P2] Mixed tree-formatted `ExceptionGroup` alongside valid setup.py traceback fails closed ---
    if sys.version_info >= (3, 11):
        try:
            raise ExceptionGroup(  # noqa: F821
                "runner worker pool exhausted", [Exception("runner worker failed")]
            )
        except BaseException as eg:
            eg_text = "".join(traceback.format_exception(eg)).rstrip()
    else:
        eg_text = (
            "  + Exception Group Traceback (most recent call last):\n"
            '  |   File "<stdin>", line 1, in <module>\n'
            "  | ExceptionGroup: runner worker pool exhausted (1 sub-exception)\n"
            "  +-+---------------- 1 ----------------\n"
            "    | Exception: runner worker failed\n"
            "    +------------------------------------"
        )

    mixed_stderr = (
        "Traceback (most recent call last):\n"
        '  File "/tmp/vllm-bbbbbbbbbbbb/setup.py", line 12, in <module>\n'
        '    raise RuntimeError("VLLM_TARGET_DEVICE=empty unsupported")\n'
        "RuntimeError: VLLM_TARGET_DEVICE=empty unsupported\n" + eg_text
    )
    phase_eg, src_eg, infra_eg = post_merge._classify_install_failure(
        1, mixed_stderr, "", mode="cpu_empty_build"
    )
    assert (phase_eg, src_eg, infra_eg) == ("infra_error", "infra_or_unknown", True)

    good_vllm_dir = tmp_path / "good_vllm_eg"
    (good_vllm_dir / "vllm").mkdir(parents=True, exist_ok=True)
    (good_vllm_dir / "vllm" / "__init__.py").write_text(
        "__version__ = '0.1'\n", encoding="utf-8"
    )
    eg_err_file = tmp_path / "mixed_eg.txt"
    eg_err_file.write_text(mixed_stderr + "\n", encoding="utf-8")
    ev_eg = post_merge.verify_non_ppu_reproducer(
        vllm_repo=tmp_path,
        last_good_sha=_sha(960),
        first_failing_sha=_sha(961),
        stage="wheel_build",
        target_signature="VLLM_TARGET_DEVICE=empty",
        probe_cmd=(
            f'if [ "$CANDIDATE_SHA" = "{_sha(960)}" ]; then '
            f'PYTHONPATH="{good_vllm_dir}" "{sys.executable}" "$STUB_CHECK_SCRIPT"; '
            f'else cat "{eg_err_file}" >&2; exit 1; fi'
        ),
    )
    assert ev_eg.verified is False
    assert ev_eg.infra_error is True
    assert ev_eg.failure_source == "infra_or_unknown"


def test_review_pr374_rereview_mandatory_rust_and_target_matched_bisect(
    tmp_path: Path,
) -> None:
    """Regression test for review #6035578476 (2 P1s: r4677055119 & r4677055123)."""
    # Lightweight python wrapper implementing `-m pip install ... --target <site> <src>`
    # without requiring external `setuptools` in bare CI/host Python environments.
    py_shim = tmp_path / "py_pip_shim.py"
    py_shim.write_text(
        f"#!{sys.executable}\n"
        "import os, shutil, subprocess, sys\n"
        "from pathlib import Path\n"
        "argv = sys.argv[1:]\n"
        'if len(argv) >= 3 and argv[0] == "-m" and argv[1] == "pip" and argv[2] == "install":\n'
        '    t_idx = argv.index("--target")\n'
        "    site_dir = Path(argv[t_idx + 1])\n"
        "    src_dir = Path(argv[-1])\n"
        '    setup_py = src_dir / "setup.py"\n'
        "    if setup_py.is_file():\n"
        f'        r = subprocess.run(["{sys.executable}", str(setup_py)], cwd=str(src_dir))\n'
        "        if r.returncode != 0:\n"
        "            raise SystemExit(r.returncode)\n"
        "    site_dir.mkdir(parents=True, exist_ok=True)\n"
        '    dst = site_dir / "vllm"\n'
        "    if dst.exists():\n"
        "        shutil.rmtree(dst)\n"
        '    pkg_src = src_dir / "vllm"\n'
        '    skip_rs = os.environ.get("TEST_OMIT_INSTALLED_VLLM_RS") == "1"\n'
        "    shutil.copytree(\n"
        "        pkg_src,\n"
        "        dst,\n"
        '        ignore=(lambda _d, names: ["vllm-rs"] if skip_rs and "vllm-rs" in names else []),\n'
        "    )\n"
        "    raise SystemExit(0)\n"
        f'os.execv("{sys.executable}", ["{sys.executable}", *argv])\n',
        encoding="utf-8",
    )
    py_shim.chmod(0o755)

    def _init_sail_fixture(sail_dir: Path) -> None:
        for rel in (
            "csrc/upstream/torch_bindings.cpp",
            "csrc/upstream/moe/torch_bindings.cpp",
        ):
            p = sail_dir / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("// healthy bindings\n", encoding="utf-8")
        mf = sail_dir / "csrc/upstream/manifest.toml"
        mf.write_text(
            'compiled_tiers = ["a"]\n[[entry]]\ntier = "a"\npath = "csrc/kernel.cu"\nops = []\n',
            encoding="utf-8",
        )

    def _init_upstream_repo(repo_dir: Path):
        repo_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-q", str(repo_dir)], check=True)
        subprocess.run(
            ["git", "-C", str(repo_dir), "config", "user.email", "ci@example.com"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repo_dir), "config", "user.name", "CI"], check=True
        )
        for rel in (
            "csrc/torch_bindings.cpp",
            "csrc/moe/torch_bindings.cpp",
            "csrc/kernel.cu",
        ):
            p = repo_dir / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("// healthy\n", encoding="utf-8")
        (repo_dir / "vllm").mkdir(parents=True, exist_ok=True)
        (repo_dir / "vllm/__init__.py").write_text(
            "__version__ = '1.0'\n", encoding="utf-8"
        )

        def _commit(msg: str) -> str:
            subprocess.run(["git", "-C", str(repo_dir), "add", "-A"], check=True)
            subprocess.run(
                ["git", "-C", str(repo_dir), "commit", "-qm", msg], check=True
            )
            return subprocess.check_output(
                ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True
            ).strip()

        return _commit

    sail_dir = tmp_path / "sail"
    _init_sail_fixture(sail_dir)

    # --- 1a. [P1 r4677055119] Optional-by-default native/Rust flag (`VLLM_REQUIRE_RUST_FRONTEND=1`) ---
    opt_repo = tmp_path / "opt-repo"
    commit_opt = _init_upstream_repo(opt_repo)
    (opt_repo / "native.c").write_text("int ok(void) { return 0; }\n", encoding="utf-8")
    (opt_repo / "setup.py").write_text(
        "import os, pathlib, sys\n"
        "req = os.getenv('VLLM_REQUIRE_RUST_FRONTEND', '').strip().lower() in ('1', 'true', 'yes', 'on')\n"
        "src = pathlib.Path('native.c').read_text(encoding='utf-8')\n"
        "if '#error' in src and req:\n"
        "    sys.stderr.write('native.c:1:2: error: native probe compile regression\\n')\n"
        "    raise SystemExit(1)\n",
        encoding="utf-8",
    )
    opt_good = commit_opt("control")
    (opt_repo / "native.c").write_text(
        "#error native probe compile regression\n", encoding="utf-8"
    )
    opt_bad = commit_opt("break mandatory native build")

    opt_out = tmp_path / "opt-out"
    res_opt = post_merge.run_source_only_build_triage(
        vllm_repo=opt_repo,
        sail_root=sail_dir,
        target_sha=opt_bad,
        verified_pin=opt_good,
        last_green_sha=opt_good,
        output_dir=opt_out,
        python_exe=str(py_shim),
    )
    assert res_opt["bisect"]["bisect_status"] == "converged"
    assert res_opt["bisect"]["first_failing_sha"] == opt_bad
    assert res_opt["bisect"]["last_good_sha"] == opt_good
    assert res_opt["non_ppu_repro"]["verified"] is True

    opt_pub = tmp_path / "opt-pub"
    opt_pin = tmp_path / "opt.pin"
    opt_pin.write_text(opt_good + "\n", encoding="utf-8")
    assert (
        post_merge.main(
            [
                "publish",
                "--gear",
                "probe",
                "--status",
                "fail",
                "--sail-sha",
                _sha(970),
                "--upstream-sha",
                opt_bad,
                "--verified-pin-file",
                str(opt_pin),
                "--failure-stage",
                "wheel_build",
                "--cpu-result-json",
                str(opt_out / "cpu-result.json"),
                "--bisect-result-json",
                str(opt_out / "bisect-result.json"),
                "--non-ppu-repro-json",
                str(opt_out / "non-ppu-repro.json"),
                "--public-dir",
                str(opt_pub),
            ]
        )
        == 0
    )
    opt_sum = json.loads((opt_pub / "public-summary.json").read_text(encoding="utf-8"))
    opt_iss = json.loads((opt_pub / "issue-payload.json").read_text(encoding="utf-8"))
    assert opt_sum["triage"]["classification"] == "plugin_interface_break"
    assert opt_iss["should_file_sail_issue"] is False
    assert "VLLM_REQUIRE_RUST_FRONTEND=1" in opt_iss["upstream_reproducer_markdown"]
    assert "tools/build_rust.py --release" in opt_iss["upstream_reproducer_markdown"]

    # --- 1b & 1c. [P1 r4677055119] `tools/build_rust.py --release` rustc error & missing `vllm-rs` artifact ---
    rust_repo = tmp_path / "rust-repo"
    commit_rust = _init_upstream_repo(rust_repo)
    (rust_repo / "setup.py").write_text("", encoding="utf-8")
    (rust_repo / "tools").mkdir(parents=True, exist_ok=True)
    (rust_repo / "tools/build_rust.py").write_text(
        "from pathlib import Path\n"
        "def rust_py_extension_module_names():\n"
        "    return ['_rust_tool_parser']\n"
        "if __name__ == '__main__':\n"
        "    vdir = Path(__file__).resolve().parents[1] / 'vllm'\n"
        "    (vdir / 'vllm-rs').write_bytes(b'#!/bin/sh\\necho vllm-rs\\n')\n"
        "    (vdir / '_rust_tool_parser.abi3.so').write_bytes(b'ELF')\n",
        encoding="utf-8",
    )
    rust_good = commit_rust("control with vllm-rs and _rust_tool_parser")

    # 1b: rustc syntax error (without [E0425] code) in `rust/src/cmd/src/main.rs`
    (rust_repo / "tools/build_rust.py").write_text(
        "import sys\n"
        "def rust_py_extension_module_names():\n"
        "    return ['_rust_tool_parser']\n"
        "if __name__ == '__main__':\n"
        "    sys.stderr.write('error: expected `;`, found `}`\\n  --> rust/src/cmd/src/main.rs:19:5\\n')\n"
        "    raise SystemExit(101)\n",
        encoding="utf-8",
    )
    rust_compile_bad = commit_rust("break rust/src/cmd/src/main.rs")
    rc_out = tmp_path / "rust-compile-out"
    res_rc = post_merge.run_source_only_build_triage(
        vllm_repo=rust_repo,
        sail_root=sail_dir,
        target_sha=rust_compile_bad,
        verified_pin=rust_good,
        last_green_sha=rust_good,
        output_dir=rc_out,
        python_exe=str(py_shim),
    )
    assert res_rc["bisect"]["bisect_status"] == "converged"
    assert res_rc["bisect"]["first_failing_sha"] == rust_compile_bad
    assert res_rc["bisect"]["target_signature"] == "rust_frontend_build"
    assert res_rc["non_ppu_repro"]["verified"] is True

    # 1c: `tools/build_rust.py --release` exits 0 without producing `vllm/vllm-rs`
    subprocess.run(
        ["git", "-C", str(rust_repo), "checkout", "-q", rust_good], check=True
    )
    (rust_repo / "tools/build_rust.py").write_text(
        "def rust_py_extension_module_names():\n"
        "    return []\n"
        "if __name__ == '__main__':\n"
        "    pass\n",
        encoding="utf-8",
    )
    rust_missing_bin = commit_rust("silently drop vllm-rs artifact")
    rm_out = tmp_path / "rust-missing-out"
    res_rm = post_merge.run_source_only_build_triage(
        vllm_repo=rust_repo,
        sail_root=sail_dir,
        target_sha=rust_missing_bin,
        verified_pin=rust_good,
        last_green_sha=rust_good,
        output_dir=rm_out,
        python_exe=str(py_shim),
    )
    assert res_rm["bisect"]["bisect_status"] == "converged"
    assert res_rm["bisect"]["first_failing_sha"] == rust_missing_bin
    assert res_rm["non_ppu_repro"]["verified"] is True
    assert res_rm["non_ppu_repro"]["first_failing_phase"] == "build_import"
    assert res_rm["non_ppu_repro"]["failure_source"] == "upstream_import_check"

    # 1d: Cargo network / missing toolchain errors classify as `infra_error`, never upstream!
    for infra_err in (
        "error: failed to download from `https://rsproxy.cn/crates.io-index`",
        "error: can't find Rust compiler",
        "Traceback (most recent call last):\n"
        '  File "/tmp/vllm/tools/build_rust.py", line 13, in <module>\n'
        "    from setuptools_rust import RustExtension\n"
        "ModuleNotFoundError: No module named 'setuptools_rust'",
    ):
        ph, fsrc, is_inf = post_merge._classify_install_failure(
            1, infra_err, "", mode="cpu_empty_build"
        )
        assert (ph, fsrc, is_inf) == ("infra_error", "infra_or_unknown", True)

    # --- 2a. [P1 r4677055123] Mixed Tier-A drift (still builds wheel) + later `setup.py` break ---
    mix_repo = tmp_path / "mix-repo"
    commit_mix = _init_upstream_repo(mix_repo)
    (mix_repo / "setup.py").write_text("", encoding="utf-8")
    mix_good = commit_mix("control")
    (mix_repo / "csrc/kernel.cu").unlink()
    mix_drift = commit_mix("move Tier-A kernel file; Python wheel still builds")
    (mix_repo / "setup.py").write_text(
        'raise RuntimeError("invalid upstream build configuration")\n',
        encoding="utf-8",
    )
    mix_bad = commit_mix("break setup.py packaging")

    mix_out = tmp_path / "mix-out"
    res_mix = post_merge.run_source_only_build_triage(
        vllm_repo=mix_repo,
        sail_root=sail_dir,
        target_sha=mix_bad,
        verified_pin=mix_good,
        last_green_sha=mix_good,
        output_dir=mix_out,
        python_exe=str(py_shim),
    )
    assert res_mix["bisect"]["bisect_status"] == "converged"
    assert res_mix["bisect"]["first_failing_sha"] == mix_bad
    assert res_mix["bisect"]["last_good_sha"] == mix_drift
    assert res_mix["non_ppu_repro"]["verified"] is True

    mix_pub = tmp_path / "mix-pub"
    mix_pin = tmp_path / "mix.pin"
    mix_pin.write_text(mix_good + "\n", encoding="utf-8")
    assert (
        post_merge.main(
            [
                "publish",
                "--gear",
                "probe",
                "--status",
                "fail",
                "--sail-sha",
                _sha(971),
                "--upstream-sha",
                mix_bad,
                "--verified-pin-file",
                str(mix_pin),
                "--failure-stage",
                "wheel_build",
                "--cpu-result-json",
                str(mix_out / "cpu-result.json"),
                "--bisect-result-json",
                str(mix_out / "bisect-result.json"),
                "--non-ppu-repro-json",
                str(mix_out / "non-ppu-repro.json"),
                "--public-dir",
                str(mix_pub),
            ]
        )
        == 0
    )
    mix_sum = json.loads((mix_pub / "public-summary.json").read_text(encoding="utf-8"))
    mix_iss = json.loads((mix_pub / "issue-payload.json").read_text(encoding="utf-8"))
    assert mix_sum["triage"]["classification"] == "plugin_interface_break"
    assert mix_sum["bisect"]["first_failing_sha"] == mix_bad
    assert mix_iss["should_file_sail_issue"] is False
    assert bool(mix_iss["upstream_reproducer_markdown"]) is True

    # --- 2b. [P1 r4677055123] Independent Tier-A `kernel_port_check` triage when wheel builds at target ---
    kern_out = tmp_path / "kern-only-out"
    res_kern = post_merge.run_source_only_build_triage(
        vllm_repo=mix_repo,
        sail_root=sail_dir,
        target_sha=mix_drift,
        verified_pin=mix_good,
        last_green_sha=mix_good,
        output_dir=kern_out,
        python_exe=str(py_shim),
    )
    assert res_kern["bisect"]["bisect_status"] == "converged"
    assert res_kern["bisect"]["stage"] == "kernel_port_check"
    assert res_kern["bisect"]["first_failing_sha"] == mix_drift
    assert not (kern_out / "non-ppu-repro.json").exists()
