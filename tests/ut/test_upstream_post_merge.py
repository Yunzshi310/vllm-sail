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
    end_idx = build_script.index('if [[ -n "${NATIVE_REUSE_DIR}"')
    extracted_fn = build_script[start_idx:end_idx]

    prebuilt_dir = tmp_path / "prebuilt-ext"
    vllm_wheel_out = tmp_path / "out-vllm-wheels"
    vllm_wheel_out.mkdir()
    harness = tmp_path / "run_reuse.sh"
    harness.write_text(
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        f'ROOT_DIR="{REPO_ROOT}"\n'
        f'NATIVE_REUSE_DIR="{donor_dir}"\n'
        f'PREBUILT_EXT_DIR="{prebuilt_dir}"\n'
        f'VLLM_WHEEL_DIR="{vllm_wheel_out}"\n'
        f'VLLM_SAIL_BUILD_IMAGE="{build_img}"\n'
        'PYTORCH_SAIL_ARCH="ppu_15;ppu_10"\n'
        f'VLLM_REF="{candidate_sha}"\n'
        "NATIVE_REUSE=0\n"
        "SAIL_NATIVE_REUSE=0\n"
        f"{extracted_fn}\n"
        "prepare_native_reuse\n"
        'printf "NATIVE_REUSE=%s\\nSAIL_NATIVE_REUSE=%s\\n" "$NATIVE_REUSE" "$SAIL_NATIVE_REUSE"\n',
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
    pin_file = fake_repo / ".github" / "vllm-main-verified.commit"
    pin_file.write_text(VERIFIED_PIN + "\n", encoding="utf-8")
    bump_meta = post_merge.prepare_pin_bump(repo_root=fake_repo, target_sha=sha_main_2)
    assert bump_meta["changed"] == "true"
    assert pin_file.read_text(encoding="utf-8") == sha_main_2 + "\n"


def test_gate6_upstream_canary_workflow_three_gears_contract():
    wf_path = REPO_ROOT / ".github/workflows/upstream-canary.yaml"
    wf_raw = wf_path.read_text(encoding="utf-8")
    wf = yaml.safe_load(wf_raw)
    triggers = wf.get(True) or wf.get("on")

    # No workflow_call, no statuses:write, no legacy 60k raw log paste, no _ppu-upstream-reuse.
    assert "workflow_call" not in triggers
    assert "slice(0, 60000)" not in wf_raw
    assert "_ppu-upstream-reuse.yaml" not in wf_raw
    assert wf["cache-mode"] == "none"
    assert wf["permissions"] == {"contents": "read", "actions": "read"}

    crons = [entry["cron"] for entry in triggers["schedule"]]
    assert post_merge.PROBE_CRON in crons
    assert post_merge.DAILY_PIN_CRON in crons

    jobs = wf["jobs"]
    for job in jobs.values():
        assert "statuses" not in job.get("permissions", {})

    assert "pytorch@sha256:" in jobs["cpu-fast-check"]["container"]["image"]
    assert jobs["ppu-probe-smoke"]["runs-on"] == "8-ppu-m890p"
    assert jobs["ppu-probe-smoke"]["concurrency"] == {
        "group": "upstream-post-merge-ppu-slot",
        "cancel-in-progress": False,
    }
    for rc_job in ("rc-pin-ut", "rc-fast-sanity", "rc-model-sanity", "rc-ops-tests"):
        assert "rc_qualify" in jobs[rc_job]["if"]
