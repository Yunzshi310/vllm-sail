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

    # 3. Dual-track classification: plugin_interface_break vs sail_internal_patch_drift.
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
        ),
        run_url="https://github.com/vllm-project/vllm-sail/actions/runs/12345",
        public_dir=pub_plugin_dir,
    )
    assert (
        plugin_pub["public_summary"]["triage"]["classification"]
        == "plugin_interface_break"
    )
    assert plugin_pub["issue_payload"]["should_file_sail_issue"] is False
    repro = plugin_pub["issue_payload"]["upstream_reproducer_markdown"]
    assert "VLLM_TARGET_DEVICE=empty" in repro
    assert culprit_sha in repro
    assert window_commits[culprit_idx - 1] in repro

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
