# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for selection and trusted PR Nightly workflow routing."""

import json
import shutil
from pathlib import Path

import pytest
import yaml

from scripts.ci.ppu_ci_command import Rejected
from scripts.ci.ppu_pr_nightly import resolve_case

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("case", ["glm-5.3", "deepseek-v4-flash"])
def test_resolves_a_built_in_nightly_recipe(case):
    assert resolve_case(ROOT, case, True) == "recipe"


def test_resolves_an_exact_legacy_case_even_with_recipes_disabled():
    assert resolve_case(ROOT, "deepseek-v4-int8-dp-ep", False) == "legacy"


@pytest.mark.parametrize(
    "case,enabled,error",
    [
        ("glm-5.3", False, "disabled"),
        ("qwen3.8-27b", True, "failed model_sanity selection"),
        ("unknown-case", True, "Unknown"),
        ("all", True, "one Recipe"),
    ],
)
def test_invalid_or_disabled_case_never_reaches_build(case, enabled, error):
    with pytest.raises(Rejected, match=error):
        resolve_case(ROOT, case, enabled)


def test_case_resolution_reads_the_candidate_instead_of_the_controller_checkout(
    tmp_path,
):
    selector_dir = tmp_path / "scripts/ci"
    selector_dir.mkdir(parents=True)
    for filename in ("ppu_e2e_select.py", "ppu_recipe.py"):
        shutil.copy(ROOT / "scripts/ci" / filename, selector_dir)
    catalog = json.loads((ROOT / "scripts/ci/ppu_e2e_models.json").read_text())
    case = next(entry for entry in catalog if entry["key"] == "deepseek-v4-int8")
    (selector_dir / "ppu_e2e_models.json").write_text(
        json.dumps([{**case, "key": "new-pr-case"}])
    )
    shutil.copytree(ROOT / "tests/ppu", tmp_path / "tests/ppu")
    policy_dir = tmp_path / "tests/ppu/policy/models"
    shutil.copy(policy_dir / "glm-5.3.yaml", policy_dir / "new-pr-policy.yaml")
    assert resolve_case(tmp_path, "new-pr-case", True) == "legacy"
    assert resolve_case(tmp_path, "new-pr-policy", True) == "recipe"
    shutil.copy(policy_dir / "glm-5.3.yaml", policy_dir / "new-pr-case.yaml")
    with pytest.raises(Rejected, match="ambiguous"):
        resolve_case(tmp_path, "new-pr-case", True)


def test_pr_nightly_workflow_pins_same_head_and_artifact_and_keeps_ci_status_separate():
    workflows = ROOT / ".github/workflows"
    workflow = yaml.safe_load((workflows / "ppu-pr-nightly.yaml").read_text())
    assert set(workflow[True]) == {"workflow_dispatch"}
    assert workflow["permissions"] == {}
    assert workflow["concurrency"]["queue"] == "max"
    jobs = workflow["jobs"]
    assert jobs["validate"]["steps"][0]["with"]["ref"] == "${{ github.sha }}"
    assert "default_branch" in jobs["validate"]["if"]
    assert jobs["select"]["needs"] == "validate"
    assert jobs["select"]["permissions"] == {"contents": "read"}
    assert jobs["select"]["steps"][1]["with"] == {
        "ref": "${{ inputs.head_sha }}",
        "path": "candidate",
        "persist-credentials": False,
    }
    assert jobs["build"]["needs"] == "select"
    assert jobs["build"]["with"]["pr_head_sha"] == "${{ inputs.head_sha }}"
    for lane in ("recipe", "legacy"):
        assert jobs[lane]["needs"] == ["select", "build"]
        assert jobs[lane]["if"] == f"needs.select.outputs.lane == '{lane}'"
        call = jobs[lane]["with"]
        assert (
            call["pr_head_sha"] == call["wheel_source_sha"] == "${{ inputs.head_sha }}"
        )
        assert call["models"] == "${{ inputs.case_id }}"
        assert call["build_run_id"] == "${{ github.run_id }}"
    assert jobs["recipe"]["with"]["pr_mode"] is False
    assert jobs["legacy"]["with"]["pr_mode"] is True
    assert "needs.validate.result == 'success'" in jobs["report"]["if"]
    assert jobs["report"]["steps"][0]["with"]["ref"] == "${{ github.sha }}"
    assert all("statuses" not in job.get("permissions", {}) for job in jobs.values())
    assert all("secrets" not in job for job in jobs.values())
    command = yaml.safe_load((workflows / "run-ci-command.yml").read_text())
    assert any(
        step.get("run") == "python -m scripts.ci.ppu_pr_nightly command"
        for step in command["jobs"]["command"]["steps"]
    )
