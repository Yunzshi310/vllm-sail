#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Post-merge upstream vLLM compatibility orchestrator for vLLM SAIL.

Implements the three-gear post-merge cadence without upstream repository hooks
or cross-repository write permissions:
  1. ``probe`` (every 4-6h): SDK-container CPU fast check + sliding-window 1-PPU
     unified Recipe sanity smoke, with control-first binary search bisection on
     ``(last_green_sha..red_sha]`` when red;
  2. ``daily_pin_bump`` (1x daily): promotes the latest probe-verified upstream
     commit to ``.github/vllm-main-verified.commit`` via a normal SAIL PR;
  3. ``rc_qualify`` (on new ``vX.Y.0rc*`` tags): runs the full PPU gate suite
     (wheel build + CPU fast check + pin-UT + Fast Sanity + Model Sanity +
     optional ops qualification).

This module uses only the Python 3.10+ standard library (with ``tomli`` fallback
for Python 3.10) so CPU unit tests run without torch, vLLM, or PPU hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UPSTREAM_URL = "https://github.com/vllm-project/vllm.git"
DEFAULT_VERIFIED_PIN_FILE = REPO_ROOT / ".github" / "vllm-main-verified.commit"
DEFAULT_RELEASE_TAG_FILE = REPO_ROOT / ".github" / "vllm-release-tag.commit"
DEFAULT_KERNEL_MANIFEST = REPO_ROOT / "csrc" / "upstream" / "manifest.toml"
PROBE_CRON = "23 */5 * * *"
DAILY_PIN_CRON = "41 1 * * *"

SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
IMAGE_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
RC_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.0rc(\d+)$")
RELEASE_LINE_RE = re.compile(r"^v?(\d+)\.(\d+)(?:\.(\d+))?")
SAFE_SYMBOL_RE = re.compile(r"^[A-Za-z0-9_./:<>\[\]()!?,* -]{1,160}$")
PUBLIC_GITHUB_URL_RE = re.compile(
    r"^https://github\.com/vllm-project/(?:vllm-sail|vllm)(?:/[A-Za-z0-9_./#?=&-]*)?$"
)

VALID_GEARS = ("auto", "probe", "daily_pin_bump", "rc_qualify")
SELECTED_GEARS = ("probe", "daily_pin_bump", "rc_qualify")
VALID_STATUSES = ("pass", "fail", "skipped", "deferred", "noop", "infra_error")
VALID_CLASSIFICATIONS = (
    "none",
    "infra_error",
    "plugin_interface_break",
    "sail_internal_patch_drift",
)
VALID_STAGES = (
    "none",
    "route",
    "wheel_build",
    "compat_check",
    "plugin_registration",
    "patch_contracts",
    "patch_drift",
    "kernel_port_check",
    "ppu_probe_smoke",
    "rc_pin_ut",
    "rc_fast_sanity",
    "rc_model_sanity",
    "rc_ops_tests",
    "baseline_control",
)
VALID_PPU_SKIP_REASONS = (
    "none",
    "already_tested",
    "allowlisted_non_ppu_changes",
    "deferred_for_sail_pr_ci",
    "cpu_fast_check_failed",
    "not_applicable",
)

# Conservative skip allowlist for PPU smoke coalescing: only pure docs, examples,
# benchmarks, upstream CI metadata, or non-CUDA/non-PPU backend-exclusive files.
SAFE_PPU_SKIP_PATTERNS: tuple[str, ...] = (
    "*.md",
    "docs/*",
    "docs/**",
    "benchmarks/*",
    "benchmarks/**",
    "examples/*",
    "examples/**",
    ".buildkite/*",
    ".buildkite/**",
    ".github/*",
    ".github/**",
    "csrc/rocm/*",
    "csrc/rocm/**",
    "csrc/cpu/*",
    "csrc/cpu/**",
    "csrc/xpu/*",
    "csrc/xpu/**",
    "vllm/platforms/rocm.py",
    "vllm/platforms/tpu.py",
    "vllm/platforms/xpu.py",
    "vllm/platforms/cpu.py",
    "vllm/v1/worker/tpu_*",
    "vllm/v1/worker/xpu_*",
    "vllm/v1/worker/cpu_*",
    "vllm/v1/attention/backends/rocm_*",
    "vllm/v1/attention/backends/pallas*",
    "docker/Dockerfile.rocm*",
    "docker/Dockerfile.tpu*",
    "docker/Dockerfile.xpu*",
    "docker/Dockerfile.cpu*",
    "requirements/rocm*.txt",
    "requirements/tpu*.txt",
    "requirements/xpu*.txt",
    "requirements/cpu*.txt",
)

PUBLIC_PLUGIN_SYMBOLS: tuple[str, ...] = (
    "vllm.plugins",
    "vllm.platforms.interface",
    "vllm.platforms.Platform",
    "vllm.model_executor.models.registry",
    "ModelRegistry.register_model",
    "CustomOp.register_oot",
    "register_oot",
    "AttentionBackend",
    "ServeSubcommand",
    "VLLM_TARGET_DEVICE=empty",
)

_SENSITIVE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "internal_domain",
        re.compile(
            r"(?i)\b(?:[a-z0-9-]+\.)+(?:alibaba-inc\.com|eng\.t-head\.cn|alipay\.net|aliyun-inc\.com)\b"
        ),
    ),
    (
        "private_ipv4",
        re.compile(
            r"(?<![0-9.])(?:"
            r"10(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}"
            r"|192\.168(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){2}"
            r"|172\.(?:1[6-9]|2\d|3[01])(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){2}"
            r"|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){2}"
            r"|169\.254(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){2}"
            r")(?=\.(?:\s|$)|[^0-9.]|$)"
        ),
    ),
    (
        "host_or_nas_path",
        re.compile(
            r"(?:^|[\s'\"=(`\[])(?:"
            r"/nas_aisw(?:/|\b)"
            r"|/mnt/(?:wl_nas|[A-Za-z0-9_.-]+)"
            r"|/wl_nas(?:/|\b)"
            r"|/ppusw(?:/|\b)"
            r"|/Users/[A-Za-z0-9_.-]+"
            r"|/home/[A-Za-z0-9_.-]+"
            r"|/root/"
            r"|/private/(?:tmp|var)\b"
            r"|/var/folders/"
            r"|/tmp/[A-Za-z0-9_.-]+"
            r")"
        ),
    ),
    (
        "credential_or_token",
        re.compile(
            r"(?:\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{16,}\b"
            r"|\bgithub_pat_[A-Za-z0-9_]{20,}\b"
            r"|\bAKIA[0-9A-Z]{16}\b"
            r"|\bBearer\s+[A-Za-z0-9._~+/=-]{12,}"
            r"|\bQODER_(?:PAT|PERSONAL_ACCESS_TOKEN)\b)"
        ),
    ),
    (
        "confidential_model",
        re.compile(r"(?i)\bstep[-_]?5(?:[-_]?preview)?\b"),
    ),
    (
        "staff_id",
        re.compile(r"(?i)\b(?:wb-[a-z0-9]{4,}|emp[_-]?id\s*[:=]\s*\w+)\b"),
    ),
)


class RedactionViolationError(ValueError):
    """Raised when public output fails mechanical zero-leak validation."""


@dataclass(frozen=True)
class ProbeOutcome:
    status: str  # "pass" | "fail" | "infra_error"
    stage: str = "none"
    target_signature: str = ""
    is_public_plugin_interface: bool = False
    detail_code: str = ""


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
        import tomli as tomllib  # type: ignore[no-redef]
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _require_sha1(value: str, label: str = "commit") -> str:
    cleaned = value.strip()
    if not SHA1_RE.fullmatch(cleaned):
        raise ValueError(f"{label} must be a 40-character lowercase hex SHA")
    return cleaned


def extract_image_digest(image_ref: str) -> str:
    """Strip any registry/repository prefix and return only ``sha256:<64hex>``."""
    if not image_ref:
        return "not_recorded"
    match = IMAGE_DIGEST_RE.search(image_ref.strip())
    return match.group(0) if match else "not_recorded"


def assert_zero_sensitive_leaks(
    text: str, *, staging_paths: Sequence[Path] = ()
) -> None:
    """Fail closed if ``text`` contains control chars or sensitive patterns."""

    def _wipe() -> None:
        for path in staging_paths:
            try:
                if path.is_file() or path.is_symlink():
                    path.unlink(missing_ok=True)
            except OSError:
                pass

    for ch in text:
        code = ord(ch)
        if code == 0 or 0xD800 <= code <= 0xDFFF:
            _wipe()
            raise RedactionViolationError(
                "public payload contains forbidden NUL or UTF-16 surrogate codepoint"
            )

    for category, pattern in _SENSITIVE_PATTERNS:
        if pattern.search(text):
            _wipe()
            raise RedactionViolationError(
                f"public payload blocked by redaction gate: {category}"
            )


def _sanitize_symbol(value: str, fallback: str = "unspecified") -> str:
    cleaned = value.strip()
    if not cleaned:
        return fallback
    # Strip any accidental leading slash paths before checking symbol syntax.
    if cleaned.startswith("/") or ".." in cleaned:
        return fallback
    if not SAFE_SYMBOL_RE.fullmatch(cleaned):
        return fallback
    try:
        assert_zero_sensitive_leaks(cleaned)
    except RedactionViolationError:
        return fallback
    return cleaned


# ---------------------------------------------------------------------------
# 1. Dual polling (refs/heads/main + refs/tags/v*) & Three-Gear Router
# ---------------------------------------------------------------------------


def parse_ls_remote(output: str) -> dict[str, Any]:
    """Parse ``git ls-remote`` output, preferring peeled ``^{}`` commit SHAs."""
    main_sha: str | None = None
    raw_tags: dict[str, str] = {}
    peeled_tags: dict[str, str] = {}

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        sha, ref = parts
        if not SHA1_RE.fullmatch(sha):
            continue
        if ref == "refs/heads/main":
            main_sha = sha
        elif ref.startswith("refs/tags/"):
            tag_part = ref.removeprefix("refs/tags/")
            if tag_part.endswith("^{}"):
                peeled_tags[tag_part[:-3]] = sha
            else:
                raw_tags[tag_part] = sha

    tags = {**raw_tags, **peeled_tags}
    return {"main_sha": main_sha, "tags": dict(sorted(tags.items()))}


def select_newest_rc_tag(
    tags: dict[str, str],
    *,
    release_baseline_tag: str,
    tested_rc_tags: Sequence[str] = (),
) -> tuple[str, str] | None:
    """Return ``(rc_tag, commit_sha)`` for the newest untested ``vX.Y.0rc*``."""
    base_match = RELEASE_LINE_RE.match(release_baseline_tag.strip())
    if not base_match:
        raise ValueError(f"invalid release baseline tag: {release_baseline_tag!r}")
    base_line = (int(base_match.group(1)), int(base_match.group(2)))
    tested = set(tested_rc_tags)

    candidates: list[tuple[tuple[int, int, int], str, str]] = []
    for tag, sha in tags.items():
        match = RC_TAG_RE.fullmatch(tag)
        if not match or tag in tested:
            continue
        major, minor, rc_num = (
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
        )
        if (major, minor) > base_line:
            candidates.append(((major, minor, rc_num), tag, sha))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    _, best_tag, best_sha = candidates[-1]
    return best_tag, best_sha


def is_safe_ppu_skip_path(path: str) -> bool:
    norm = path.strip().lstrip("./")
    if not norm or ".." in norm.split("/"):
        return False
    base = norm.rsplit("/", 1)[-1]
    if base.endswith(".md"):
        return True
    for pattern in SAFE_PPU_SKIP_PATTERNS:
        if fnmatchcase(norm, pattern):
            return True
    return False


def should_skip_ppu_for_paths(changed_paths: Sequence[str] | None) -> bool:
    """Return True iff ``changed_paths`` is non-empty and 100% allowlisted."""
    if not changed_paths:
        return False
    return all(is_safe_ppu_skip_path(path) for path in changed_paths)


def select_pin_bump_candidate(
    probe_records: Sequence[dict[str, Any]],
    *,
    verified_pin: str,
) -> str | None:
    """Pick the latest healthy probe SHA eligible to advance verified_pin."""
    pin = _require_sha1(verified_pin, "verified_pin")
    for record in probe_records:
        if record.get("gear") != "probe":
            continue
        sha = str(record.get("upstream_sha", "")).strip()
        if not SHA1_RE.fullmatch(sha) or sha == pin:
            continue
        if record.get("cpu_status") != "pass":
            continue
        ppu_status = record.get("ppu_status")
        skip_reason = record.get("ppu_skip_reason", "none")
        last_ppu_green = str(record.get("last_ppu_green_sha", "")).strip()
        if ppu_status == "pass":
            return sha
        if (
            ppu_status == "skipped"
            and skip_reason == "allowlisted_non_ppu_changes"
            and SHA1_RE.fullmatch(last_ppu_green)
        ):
            return sha
    return None


def route_three_gears(
    *,
    ls_remote_data: dict[str, Any],
    verified_pin: str,
    release_baseline_tag: str,
    requested_gear: str = "auto",
    schedule_cron: str = "",
    upstream_ref_override: str = "",
    cursor: dict[str, Any] | None = None,
    changed_paths_since_ppu: Sequence[str] | None = None,
    active_pr_runs: int = 0,
    probe_history: Sequence[dict[str, Any]] = (),
    force: bool = False,
) -> dict[str, Any]:
    """Resolve gear, target SHA, and CPU/PPU execution plan."""
    if requested_gear not in VALID_GEARS:
        raise ValueError(f"unsupported gear: {requested_gear!r}")
    pin = _require_sha1(verified_pin, "verified_pin")
    state = dict(cursor or {})
    last_cpu_sha = state.get("last_cpu_checked_sha") or pin
    last_ppu_tested = state.get("last_ppu_tested_sha") or pin
    last_ppu_green = state.get("last_ppu_green_sha") or pin
    tested_rcs = list(state.get("tested_rc_tags") or [])
    tags: dict[str, str] = dict(ls_remote_data.get("tags") or {})
    main_sha = ls_remote_data.get("main_sha")

    override = upstream_ref_override.strip()
    effective_gear = requested_gear
    rc_tag = ""
    target_sha = ""

    if effective_gear == "auto":
        if schedule_cron.strip() == DAILY_PIN_CRON:
            effective_gear = "daily_pin_bump"
        elif override and RC_TAG_RE.fullmatch(override):
            effective_gear = "rc_qualify"
        else:
            rc_hit = select_newest_rc_tag(
                tags,
                release_baseline_tag=release_baseline_tag,
                tested_rc_tags=tested_rcs,
            )
            if rc_hit is not None and not override:
                effective_gear = "rc_qualify"
                rc_tag, target_sha = rc_hit
            else:
                effective_gear = "probe"

    if effective_gear == "rc_qualify":
        if not target_sha:
            if override:
                if RC_TAG_RE.fullmatch(override):
                    if override not in tags:
                        raise ValueError(f"RC tag {override!r} not found in upstream")
                    rc_tag = override
                    target_sha = tags[override]
                else:
                    target_sha = _require_sha1(override, "upstream_ref")
            else:
                rc_hit = select_newest_rc_tag(
                    tags,
                    release_baseline_tag=release_baseline_tag,
                    tested_rc_tags=tested_rcs,
                )
                if rc_hit is None:
                    return {
                        "selected_gear": "rc_qualify",
                        "target_sha": "",
                        "rc_tag": "",
                        "last_green_sha": last_ppu_green,
                        "last_ppu_tested_sha": last_ppu_tested,
                        "run_cpu": False,
                        "run_ppu": False,
                        "ppu_skip_reason": "not_applicable",
                    }
                rc_tag, target_sha = rc_hit
        return {
            "selected_gear": "rc_qualify",
            "target_sha": _require_sha1(target_sha, "rc_target_sha"),
            "rc_tag": rc_tag,
            "last_green_sha": last_ppu_green,
            "last_ppu_tested_sha": last_ppu_tested,
            "run_cpu": True,
            "run_ppu": True,
            "ppu_skip_reason": "none",
        }

    if effective_gear == "daily_pin_bump":
        if override:
            candidate = _require_sha1(override, "upstream_ref")
            bump_sha = "" if candidate == pin else candidate
        else:
            history = list(probe_history)
            if not history and state.get("last_probe_record"):
                history = [state["last_probe_record"]]
            bump_sha = select_pin_bump_candidate(history, verified_pin=pin) or ""
        return {
            "selected_gear": "daily_pin_bump",
            "target_sha": bump_sha,
            "rc_tag": "",
            "last_green_sha": last_ppu_green,
            "last_ppu_tested_sha": last_ppu_tested,
            "run_cpu": False,
            "run_ppu": False,
            "ppu_skip_reason": "not_applicable",
        }

    # Gear 1: post-merge probe
    if override:
        target_sha = _require_sha1(override, "upstream_ref")
    else:
        if not main_sha:
            raise ValueError("git ls-remote output did not contain refs/heads/main")
        target_sha = _require_sha1(main_sha, "main_sha")

    run_cpu = bool(force or target_sha != last_cpu_sha)
    if not force and target_sha == last_ppu_tested:
        run_ppu = False
        skip_reason = "already_tested"
    elif should_skip_ppu_for_paths(changed_paths_since_ppu):
        run_ppu = False
        skip_reason = "allowlisted_non_ppu_changes"
    elif active_pr_runs > 0 and not force:
        run_ppu = False
        skip_reason = "deferred_for_sail_pr_ci"
    else:
        run_ppu = True
        skip_reason = "none"

    return {
        "selected_gear": "probe",
        "target_sha": target_sha,
        "rc_tag": "",
        "last_green_sha": last_ppu_green,
        "last_ppu_tested_sha": last_ppu_tested,
        "run_cpu": run_cpu,
        "run_ppu": run_ppu,
        "ppu_skip_reason": skip_reason,
    }


def advance_cursor(
    cursor: dict[str, Any] | None,
    *,
    verified_pin: str,
    selected_gear: str,
    target_sha: str,
    rc_tag: str = "",
    cpu_status: str = "not_run",
    ppu_status: str = "not_run",
    ppu_skip_reason: str = "none",
    patch_fingerprints: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Update persistent cursor without recording hardware pass for skipped SHAs."""
    pin = _require_sha1(verified_pin, "verified_pin")
    updated: dict[str, Any] = {
        "schema": 1,
        "last_cpu_checked_sha": (cursor or {}).get("last_cpu_checked_sha") or pin,
        "last_cpu_green_sha": (cursor or {}).get("last_cpu_green_sha") or pin,
        "last_ppu_tested_sha": (cursor or {}).get("last_ppu_tested_sha") or pin,
        "last_ppu_green_sha": (cursor or {}).get("last_ppu_green_sha") or pin,
        "tested_rc_tags": list((cursor or {}).get("tested_rc_tags") or []),
        "patch_fingerprints": dict((cursor or {}).get("patch_fingerprints") or {}),
    }
    if (cursor or {}).get("last_probe_record"):
        updated["last_probe_record"] = dict((cursor or {})["last_probe_record"])
    if not target_sha or not SHA1_RE.fullmatch(target_sha):
        return updated

    if cpu_status in ("pass", "fail"):
        updated["last_cpu_checked_sha"] = target_sha
        if cpu_status == "pass":
            updated["last_cpu_green_sha"] = target_sha
            if patch_fingerprints:
                updated["patch_fingerprints"] = dict(sorted(patch_fingerprints.items()))

    # Hardware cursor advances ONLY when device smoke actually executed!
    if ppu_status in ("pass", "fail"):
        updated["last_ppu_tested_sha"] = target_sha
        if ppu_status == "pass":
            updated["last_ppu_green_sha"] = target_sha

    if (
        selected_gear == "rc_qualify"
        and rc_tag
        and cpu_status == "pass"
        and ppu_status == "pass"
    ):
        if rc_tag not in updated["tested_rc_tags"]:
            updated["tested_rc_tags"].append(rc_tag)

    if selected_gear == "probe":
        updated["last_probe_record"] = {
            "gear": "probe",
            "upstream_sha": target_sha,
            "cpu_status": cpu_status,
            "ppu_status": ppu_status,
            "ppu_skip_reason": ppu_skip_reason,
            "last_ppu_green_sha": updated["last_ppu_green_sha"],
        }
    return updated


# ---------------------------------------------------------------------------
# 2. Tier-A Kernel Manifest & Binding Schema Checker (stdlib, no GPU/sailify)
# ---------------------------------------------------------------------------

_OPS_DEF_RE = re.compile(r'ops\.def\(\s*((?:"(?:[^"\\]|\\.)*"\s*)+)\)', re.DOTALL)
_STRING_LIT_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')


def extract_ops_defs(cpp_text: str) -> dict[str, str]:
    """Extract ``{op_name: normalized_schema}`` from ``ops.def("...")`` calls."""
    schemas: dict[str, str] = {}
    for match in _OPS_DEF_RE.finditer(cpp_text):
        concatenated = "".join(_STRING_LIT_RE.findall(match.group(1)))
        normalized = " ".join(concatenated.split())
        op_name = normalized.split("(", 1)[0].strip()
        if op_name:
            schemas[op_name] = normalized
    return schemas


def check_kernel_manifest_compatibility(
    vllm_src: Path, sail_root: Path = REPO_ROOT
) -> list[str]:
    """Check Tier-A kernel source existence and C++ ``ops.def`` schema parity."""
    manifest_path = sail_root / "csrc" / "upstream" / "manifest.toml"
    manifest = _load_toml(manifest_path)
    compiled_tiers = set(manifest.get("compiled_tiers") or ["a"])
    problems: list[str] = []

    compiled_ops: set[str] = set()
    for entry in manifest.get("entry", []):
        if entry.get("tier") not in compiled_tiers:
            continue
        rel_path = entry["path"]
        if not (vllm_src / rel_path).is_file():
            problems.append(f"missing_tier_a_file:{rel_path}")
        compiled_ops.update(entry.get("ops") or [])

    binding_pairs = (
        ("csrc/torch_bindings.cpp", "csrc/upstream/torch_bindings.cpp"),
        ("csrc/moe/torch_bindings.cpp", "csrc/upstream/moe/torch_bindings.cpp"),
    )
    vendored_schemas: dict[str, str] = {}
    upstream_schemas: dict[str, str] = {}
    for up_rel, sail_rel in binding_pairs:
        sail_file = sail_root / sail_rel
        up_file = vllm_src / up_rel
        if sail_file.is_file():
            vendored_schemas.update(
                extract_ops_defs(sail_file.read_text(encoding="utf-8"))
            )
        if not up_file.is_file():
            problems.append(f"missing_binding_file:{up_rel}")
        else:
            upstream_schemas.update(
                extract_ops_defs(up_file.read_text(encoding="utf-8"))
            )

    for op_key in sorted(compiled_ops):
        base_op = op_key.split(".", 1)[0]
        if base_op not in vendored_schemas:
            continue
        if base_op not in upstream_schemas:
            problems.append(f"missing_op_schema:{base_op}")
        elif upstream_schemas[base_op] != vendored_schemas[base_op]:
            problems.append(f"changed_op_schema:{base_op}")

    return problems


# ---------------------------------------------------------------------------
# 3. Control-First Binary Search Bisection on (last_green_sha..red_sha]
# ---------------------------------------------------------------------------


def bisect_first_failing_sha(
    *,
    last_green_sha: str,
    red_sha: str,
    commits: Sequence[str],
    probe_fn: Callable[[str], ProbeOutcome],
    verify_baseline_first: bool = True,
    initial_red_outcome: ProbeOutcome | None = None,
) -> dict[str, Any]:
    """Locate the exact first failing commit in ``(last_green_sha..red_sha]``.

    ``commits`` must be ordered oldest-to-newest along ``--first-parent`` and end
    with ``red_sha``. ``red_sha`` is already known bad from the window probe, so
    binary search probes ``commits[:-1]`` in at most ``ceil(log2(len(commits)))``
    steps after optional control verification of ``last_green_sha``.
    """
    green = _require_sha1(last_green_sha, "last_green_sha")
    red = _require_sha1(red_sha, "red_sha")
    ordered = [_require_sha1(c, "commit") for c in commits]
    if not ordered or ordered[-1] != red:
        raise ValueError("commits must be non-empty and end with red_sha")

    history: list[dict[str, str]] = []
    if verify_baseline_first:
        baseline_res = probe_fn(green)
        history.append(
            {
                "sha": green,
                "role": "baseline_control",
                "status": baseline_res.status,
                "stage": _sanitize_symbol(baseline_res.stage, "baseline_control"),
            }
        )
        if baseline_res.status != "pass":
            return {
                "bisect_status": "baseline_control_failed",
                "first_failing_sha": "",
                "last_good_sha": "",
                "steps_executed": len(history),
                "probe_steps_on_range": 0,
                "winning_outcome": baseline_res,
                "history": history,
            }

    best_bad_outcome = initial_red_outcome or ProbeOutcome(
        status="fail", stage="ppu_probe_smoke", target_signature="window_head"
    )
    if len(ordered) == 1:
        return {
            "bisect_status": "converged",
            "first_failing_sha": red,
            "last_good_sha": green,
            "steps_executed": len(history),
            "probe_steps_on_range": 0,
            "winning_outcome": best_bad_outcome,
            "history": history,
        }

    lo = 0
    hi = len(ordered) - 1  # ordered[hi] == red is already known bad
    range_probes = 0
    while lo < hi:
        mid = (lo + hi) // 2
        candidate = ordered[mid]
        outcome = probe_fn(candidate)
        range_probes += 1
        history.append(
            {
                "sha": candidate,
                "role": "bisect_probe",
                "status": outcome.status,
                "stage": _sanitize_symbol(outcome.stage, "none"),
            }
        )
        if outcome.status == "infra_error":
            return {
                "bisect_status": "aborted_infra_error",
                "first_failing_sha": "",
                "last_good_sha": ordered[lo - 1] if lo > 0 else green,
                "steps_executed": len(history),
                "probe_steps_on_range": range_probes,
                "winning_outcome": outcome,
                "history": history,
            }
        if outcome.status == "pass":
            lo = mid + 1
        else:
            hi = mid
            best_bad_outcome = outcome

    first_bad = ordered[lo]
    last_good = ordered[lo - 1] if lo > 0 else green
    return {
        "bisect_status": "converged",
        "first_failing_sha": first_bad,
        "last_good_sha": last_good,
        "steps_executed": len(history),
        "probe_steps_on_range": range_probes,
        "winning_outcome": best_bad_outcome,
        "history": history,
    }


# ---------------------------------------------------------------------------
# 4. Dual-Track Root-Cause Routing & Sanitized Templates
# ---------------------------------------------------------------------------


def classify_failure(outcome: ProbeOutcome) -> dict[str, str]:
    """Classify a failure into infra_error, plugin_interface_break, or sail_internal_patch_drift."""
    stage = outcome.stage if outcome.stage in VALID_STAGES else "none"
    sig = _sanitize_symbol(outcome.target_signature, "unspecified")

    if outcome.status == "pass":
        return {
            "classification": "none",
            "stage": "none",
            "target_signature": "none",
            "bucket_key": "",
        }

    if outcome.status == "infra_error" or stage == "baseline_control":
        return {
            "classification": "infra_error",
            "stage": stage,
            "target_signature": sig,
            "bucket_key": "",
        }

    is_public = outcome.is_public_plugin_interface
    if not is_public:
        if stage == "wheel_build" and sig.startswith("vllm."):
            is_public = True
        elif stage == "wheel_build" and sig in {
            "VLLM_TARGET_DEVICE=empty",
            "vllm_empty_device_build",
            "rust_frontend_build",
        }:
            is_public = True
        elif (
            stage == "plugin_registration"
            and any(sig.startswith(pub) for pub in PUBLIC_PLUGIN_SYMBOLS)
            and "vllm_sail.patch" not in sig
        ):
            is_public = True

    if is_public:
        return {
            "classification": "plugin_interface_break",
            "stage": stage,
            "target_signature": sig,
            "bucket_key": f"upstream_plugin_api:{stage}:{sig}",
        }

    bucket_key = f"{stage}:{sig}"
    return {
        "classification": "sail_internal_patch_drift",
        "stage": stage,
        "target_signature": sig,
        "bucket_key": bucket_key,
    }


def render_upstream_reproducer(
    *,
    first_failing_sha: str,
    last_good_sha: str,
    sail_sha: str,
    stage: str,
    target_signature: str,
    run_url: str = "",
) -> str:
    """Render a sanitized, hardware-free minimal reproducer for vllm-project/vllm."""
    bad = _require_sha1(first_failing_sha, "first_failing_sha")
    good = _require_sha1(last_good_sha, "last_good_sha")
    sail = _require_sha1(sail_sha, "sail_sha")
    safe_stage = stage if stage in VALID_STAGES else "plugin_registration"
    safe_sig = _sanitize_symbol(target_signature, "vllm.plugins")
    safe_url = run_url if PUBLIC_GITHUB_URL_RE.fullmatch(run_url or "") else ""

    lines = [
        "### Out-of-tree plugin compatibility regression report",
        "",
        f"- **First failing vLLM commit:** `{bad}`",
        f"- **Last passing vLLM commit (control-verified):** `{good}`",
        f"- **Out-of-tree plugin (`vllm-project/vllm-sail`) commit:** `{sail}`",
        f"- **Failing stage:** `{safe_stage}`",
        f"- **Affected public interface:** `{safe_sig}`",
    ]
    if safe_url:
        lines.append(f"- **Public CI run:** {safe_url}")
    lines.extend(
        [
            "",
            "#### Minimal CPU-only reproducer (`VLLM_TARGET_DEVICE=empty`)",
            "",
            "```bash",
            "git clone https://github.com/vllm-project/vllm.git",
            "cd vllm",
            f"git checkout {bad}",
            "python3 tools/use_existing_torch.py",
            "VLLM_TARGET_DEVICE=empty python3 -m pip install --no-build-isolation --no-deps .",
            "python3 -c \"import vllm; from vllm.plugins import load_general_plugins; load_general_plugins(); print('loaded', vllm.__version__)\"",
            "```",
            "",
        ]
    )
    text = "\n".join(lines)
    assert_zero_sensitive_leaks(text)
    return text


def render_sail_bucket_issue(
    *,
    bucket_key: str,
    first_failing_sha: str,
    last_good_sha: str,
    sail_sha: str,
    image_digest: str,
    stage: str,
    target_signature: str,
    run_url: str = "",
) -> tuple[str, str]:
    """Render ``(title, body)`` for bucketed tracking inside ``vllm-project/vllm-sail``."""
    safe_stage = stage if stage in VALID_STAGES else "patch_drift"
    safe_sig = _sanitize_symbol(target_signature, "unspecified")
    digest_id = hashlib.sha256(bucket_key.encode("utf-8")).hexdigest()[:10]
    title = (
        f"[Post-Merge][sail_internal_patch_drift][{digest_id}] {safe_stage}: {safe_sig}"
    )
    bad = first_failing_sha if SHA1_RE.fullmatch(first_failing_sha) else "unbisected"
    good = last_good_sha if SHA1_RE.fullmatch(last_good_sha) else "unknown"
    sail = _require_sha1(sail_sha, "sail_sha")
    img = extract_image_digest(image_digest)
    safe_url = run_url if PUBLIC_GITHUB_URL_RE.fullmatch(run_url or "") else ""

    lines = [
        "## Post-Merge Internal Patch / Kernel Drift",
        "",
        f"- **Bucket ID:** `{digest_id}`",
        "- **Classification:** `sail_internal_patch_drift` (track and fix in `vllm-sail`; do not file in `vllm-project/vllm`)",
        f"- **Failure stage:** `{safe_stage}`",
        f"- **Target signature:** `{safe_sig}`",
        f"- **First failing upstream vLLM SHA:** `{bad}`",
        f"- **Last known-good upstream vLLM SHA:** `{good}`",
        f"- **Tested `vllm-sail` SHA:** `{sail}`",
        f"- **Container image digest:** `{img}`",
    ]
    if safe_url:
        lines.append(f"- **Public Actions run:** {safe_url}")
    body = "\n".join(lines) + "\n"
    assert_zero_sensitive_leaks(title)
    assert_zero_sensitive_leaks(body)
    return title, body


# ---------------------------------------------------------------------------
# 5. Three-Channel Allowlist Publisher (Console + Artifact + Summary + Issue)
# ---------------------------------------------------------------------------


def publish_sanitized_channels(
    *,
    gear: str,
    status: str,
    sail_sha: str,
    upstream_sha: str,
    verified_pin: str,
    build_image: str = "",
    runtime_image: str = "",
    rc_tag: str = "",
    cpu_status: str = "not_run",
    ppu_status: str = "not_run",
    ppu_skip_reason: str = "none",
    first_failing_sha: str = "",
    last_good_sha: str = "",
    bisect_steps: int = 0,
    outcome: ProbeOutcome | None = None,
    known_soft_drift_count: int = 0,
    run_url: str = "",
    public_dir: Path,
    step_summary_path: Path | None = None,
) -> dict[str, Any]:
    """Construct, mechanically redact-verify, and write public outputs."""
    if gear not in SELECTED_GEARS:
        raise ValueError(f"invalid selected gear: {gear!r}")
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid status: {status!r}")
    sail = _require_sha1(sail_sha, "sail_sha")
    pin = _require_sha1(verified_pin, "verified_pin")
    up_sha = _require_sha1(upstream_sha, "upstream_sha") if upstream_sha else ""
    safe_rc = rc_tag if (rc_tag and RC_TAG_RE.fullmatch(rc_tag)) else ""
    safe_skip = ppu_skip_reason if ppu_skip_reason in VALID_PPU_SKIP_REASONS else "none"
    safe_url = run_url if PUBLIC_GITHUB_URL_RE.fullmatch(run_url or "") else ""
    build_digest = extract_image_digest(build_image)
    runtime_digest = extract_image_digest(runtime_image)

    triage = (
        classify_failure(outcome)
        if outcome is not None and status in ("fail", "infra_error")
        else {
            "classification": "infra_error" if status == "infra_error" else "none",
            "stage": "none",
            "target_signature": "none",
            "bucket_key": "",
        }
    )

    safe_first_bad = (
        first_failing_sha if SHA1_RE.fullmatch(first_failing_sha) else up_sha
    )
    safe_last_good = last_good_sha if SHA1_RE.fullmatch(last_good_sha) else pin

    reproducer_md = ""
    issue_title = ""
    issue_body = ""
    if triage["classification"] == "plugin_interface_break" and safe_first_bad:
        reproducer_md = render_upstream_reproducer(
            first_failing_sha=safe_first_bad,
            last_good_sha=safe_last_good,
            sail_sha=sail,
            stage=triage["stage"],
            target_signature=triage["target_signature"],
            run_url=safe_url,
        )
    elif triage["classification"] == "sail_internal_patch_drift":
        issue_title, issue_body = render_sail_bucket_issue(
            bucket_key=triage["bucket_key"],
            first_failing_sha=safe_first_bad,
            last_good_sha=safe_last_good,
            sail_sha=sail,
            image_digest=runtime_digest
            if runtime_digest != "not_recorded"
            else build_digest,
            stage=triage["stage"],
            target_signature=triage["target_signature"],
            run_url=safe_url,
        )

    public_summary: dict[str, Any] = {
        "schema": 1,
        "gear": gear,
        "status": status,
        "commits": {
            "vllm": up_sha,
            "vllm_sail": sail,
            "verified_pin": pin,
        },
        "rc_tag": safe_rc or None,
        "images": {
            "build_digest": build_digest,
            "runtime_digest": runtime_digest,
        },
        "checks": {
            "cpu_fast_check": cpu_status
            if cpu_status in ("pass", "fail", "not_run", "infra_error")
            else "not_run",
            "ppu_smoke": ppu_status
            if ppu_status in ("pass", "fail", "skipped", "not_run", "infra_error")
            else "not_run",
            "ppu_skip_reason": safe_skip,
            "known_soft_drift_count": max(0, min(int(known_soft_drift_count), 10000)),
        },
        "bisect": {
            "first_failing_sha": safe_first_bad if status == "fail" else None,
            "last_good_sha": safe_last_good if status == "fail" else None,
            "steps": max(0, min(int(bisect_steps), 64)),
        },
        "triage": {
            "classification": triage["classification"],
            "stage": triage["stage"],
            "target_signature": triage["target_signature"],
            "bucket_key": triage["bucket_key"] or None,
        },
        "run_url": safe_url or None,
    }

    md_lines = [
        f"### Post-Merge Upstream Compatibility (`{gear}`)",
        "",
        f"- **Status:** `{status}`",
        f"- **Upstream `vllm` SHA:** `{up_sha or 'none'}`"
        + (f" (`{safe_rc}`)" if safe_rc else ""),
        f"- **`vllm-sail` SHA:** `{sail}` · **Verified Pin:** `{pin}`",
        f"- **Images (build / runtime):** `{build_digest}` / `{runtime_digest}`",
        f"- **CPU fast check:** `{public_summary['checks']['cpu_fast_check']}` "
        f"(known soft drift: `{public_summary['checks']['known_soft_drift_count']}`)",
        f"- **PPU smoke:** `{public_summary['checks']['ppu_smoke']}` "
        f"(skip reason: `{safe_skip}`)",
    ]
    if status in ("fail", "infra_error"):
        md_lines.extend(
            [
                f"- **Triage classification:** `{triage['classification']}` "
                f"(`{triage['stage']}` / `{triage['target_signature']}`)",
                f"- **First failing SHA:** `{safe_first_bad or 'n/a'}` "
                f"(bisect steps: `{public_summary['bisect']['steps']}`)",
            ]
        )
    if reproducer_md:
        md_lines.extend(["", reproducer_md.rstrip()])
    markdown_text = "\n".join(md_lines) + "\n"

    console_text = (
        f"post_merge_gear={gear} status={status} vllm_sha={up_sha or 'none'} "
        f"sail_sha={sail} classification={triage['classification']} "
        f"stage={triage['stage']} ppu_skip={safe_skip}"
    )

    public_dir.mkdir(parents=True, exist_ok=True)
    summary_json_path = public_dir / "public-summary.json"
    issue_json_path = public_dir / "issue-payload.json"
    staging = [summary_json_path, issue_json_path]
    if step_summary_path is not None:
        staging.append(step_summary_path)

    summary_json_text = (
        json.dumps(public_summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    issue_payload = {
        "should_file_sail_issue": bool(
            triage["classification"] == "sail_internal_patch_drift" and issue_title
        ),
        "title": issue_title,
        "body": issue_body,
        "upstream_reproducer_markdown": reproducer_md,
    }
    issue_json_text = (
        json.dumps(issue_payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )

    for channel_text in (
        console_text,
        summary_json_text,
        markdown_text,
        issue_json_text,
    ):
        assert_zero_sensitive_leaks(channel_text, staging_paths=staging)

    summary_json_path.write_text(summary_json_text, encoding="utf-8")
    issue_json_path.write_text(issue_json_text, encoding="utf-8")
    if step_summary_path is not None:
        with step_summary_path.open("a", encoding="utf-8") as handle:
            handle.write(markdown_text)

    return {
        "console": console_text,
        "public_summary": public_summary,
        "markdown": markdown_text,
        "issue_payload": issue_payload,
    }


def prepare_pin_bump(
    *,
    repo_root: Path,
    target_sha: str,
    verified_pin_file: Path | None = None,
) -> dict[str, str]:
    """Validate and write the daily small-step pin bump to ``vllm-main-verified.commit``."""
    target = _require_sha1(target_sha, "target_sha")
    pin_path = verified_pin_file or (
        repo_root / ".github" / "vllm-main-verified.commit"
    )
    previous = _require_sha1(pin_path.read_text(encoding="utf-8"), "previous_pin")
    if target == previous:
        return {
            "changed": "false",
            "previous_pin": previous,
            "new_pin": target,
            "title": "",
            "body": "",
        }
    pin_path.write_text(target + "\n", encoding="utf-8")
    title = f"[CI][Pin] Advance verified upstream vLLM commit to {target[:12]}"
    body = (
        "## Summary\n\n"
        f"- Advance `.github/vllm-main-verified.commit` from `{previous}` to probe-verified `{target}`.\n"
        "- Keeps `csrc/upstream/manifest.toml` unchanged unless a kernel re-port is required.\n"
        "- Validated via post-merge probe; full PR Fast Gate and Nightly Model Sanity apply before/after merge.\n"
    )
    assert_zero_sensitive_leaks(title)
    assert_zero_sensitive_leaks(body)
    return {
        "changed": "true",
        "previous_pin": previous,
        "new_pin": target,
        "title": title,
        "body": body,
    }


# ---------------------------------------------------------------------------
# 6. SDK Container CPU Fast Check Runner
# ---------------------------------------------------------------------------


def run_sdk_cpu_fast_check(
    *,
    vllm_src: Path,
    sail_root: Path,
    target_sha: str,
    verified_pin: str,
    previous_baseline: Path | None = None,
    output_json: Path,
    python_exe: str = sys.executable,
) -> int:
    """Execute SDK-container CPU compatibility checks against ``vllm_src``."""
    target = _require_sha1(target_sha, "target_sha")
    pin = _require_sha1(verified_pin, "verified_pin")
    env = dict(os.environ)
    env.pop("VLLM_VERSION", None)
    env.pop("VLLM_VERSION_OVERRIDE", None)
    if target != pin:
        env["VLLM_SAIL_CANDIDATE_UPSTREAM_COMMIT"] = target
    else:
        env.pop("VLLM_SAIL_CANDIDATE_UPSTREAM_COMMIT", None)
    env["VLLM_SOURCE_ROOT"] = str(vllm_src.resolve())

    known_soft_count = 0
    patch_fingerprints: dict[str, str] = {}

    def _write_result(outcome: ProbeOutcome) -> int:
        payload = {
            "status": outcome.status,
            "stage": outcome.stage,
            "target_signature": _sanitize_symbol(outcome.target_signature, "none"),
            "is_public_plugin_interface": outcome.is_public_plugin_interface,
            "known_soft_drift_count": known_soft_count,
            "patch_fingerprints": patch_fingerprints,
        }
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        assert_zero_sensitive_leaks(text, staging_paths=[output_json])
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(text, encoding="utf-8")
        return 0 if outcome.status == "pass" else 1

    # Step 1: check_vllm_compatibility + register() + register_out_of_tree()
    reg_code = (
        "import json\n"
        "import vllm_sail\n"
        "from vllm_sail.compat import check_vllm_compatibility\n"
        "check_vllm_compatibility(force=True)\n"
        "vllm_sail.register()\n"
        "vllm_sail.register_out_of_tree()\n"
    )
    reg = subprocess.run(
        [python_exe, "-c", reg_code],
        cwd="/tmp",
        env={k: v for k, v in env.items() if k != "PYTHONPATH"},
        capture_output=True,
        text=True,
    )
    if reg.returncode != 0:
        combined = reg.stderr + "\n" + reg.stdout
        is_public = "vllm_sail/patch/" not in combined and any(
            sym in combined for sym in PUBLIC_PLUGIN_SYMBOLS
        )
        sig = "vllm.plugins" if is_public else "vllm_sail.register_out_of_tree"
        m = re.search(r"(vllm(?:\.[A-Za-z0-9_]+)+)", combined)
        if m:
            sig = m.group(1)
        stage = (
            "compat_check"
            if "candidate mode requires" in combined
            or "requires verified vLLM commit" in combined
            else "plugin_registration"
        )
        return _write_result(
            ProbeOutcome(
                status="fail",
                stage=stage,
                target_signature=sig,
                is_public_plugin_interface=is_public,
            )
        )

    # Step 2: static upstream_source AST contract checks
    contracts = subprocess.run(
        [
            python_exe,
            "-m",
            "pytest",
            "tests/ut/test_patch_contracts.py",
            "-q",
            "-rs",
            "-m",
            "upstream_source",
            "--require-upstream-source",
        ],
        cwd=str(sail_root),
        env=env,
        capture_output=True,
        text=True,
    )
    if contracts.returncode != 0:
        m = re.search(
            r"(vllm(?:\.[A-Za-z0-9_]+)+)", contracts.stdout + "\n" + contracts.stderr
        )
        sig = m.group(1) if m else "patch_contracts"
        return _write_result(
            ProbeOutcome(
                status="fail",
                stage="patch_contracts",
                target_signature=sig,
                is_public_plugin_interface=False,
            )
        )

    # Step 3: check_patch_drift.py (incremental vs known baseline soft drift)
    drift_json = output_json.parent / "patch-drift-eval.json"
    drift_cmd = [
        python_exe,
        str(sail_root / "tools" / "check_patch_drift.py"),
        "--json-output",
        str(drift_json),
    ]
    if previous_baseline is not None and previous_baseline.is_file():
        drift_cmd.extend(
            ["--previous-baseline", str(previous_baseline), "--allow-known-soft-drift"]
        )
    drift = subprocess.run(
        drift_cmd, cwd=str(sail_root), env=env, capture_output=True, text=True
    )
    if drift_json.is_file():
        drift_data = json.loads(drift_json.read_text(encoding="utf-8"))
        known_soft_count = len(drift_data.get("known_soft_drifted") or [])
        patch_fingerprints = dict(drift_data.get("fingerprints") or {})
        drift_json.unlink(missing_ok=True)
    else:
        drift_data = {}
    if drift.returncode != 0:
        candidates = (
            list(drift_data.get("removed") or [])
            + list(drift_data.get("incremental_drifted") or [])
            + list(drift_data.get("added") or [])
        )
        sig = candidates[0] if candidates else "patch_baseline"
        return _write_result(
            ProbeOutcome(
                status="fail",
                stage="patch_drift",
                target_signature=sig,
                is_public_plugin_interface=False,
            )
        )

    # Step 4: Tier-A kernel manifest & op schema parity (+ sailify port --check if installed)
    kernel_problems = check_kernel_manifest_compatibility(vllm_src, sail_root=sail_root)
    if kernel_problems:
        return _write_result(
            ProbeOutcome(
                status="fail",
                stage="kernel_port_check",
                target_signature=kernel_problems[0],
                is_public_plugin_interface=False,
            )
        )

    has_sailify = (
        subprocess.run(
            [python_exe, "-c", "import sailify"],
            env=env,
            capture_output=True,
        ).returncode
        == 0
    )
    if has_sailify:
        port_check = subprocess.run(
            [
                python_exe,
                str(sail_root / "tools" / "port_upstream_kernels.py"),
                "--vllm-src",
                str(vllm_src),
                "--check",
            ],
            cwd=str(sail_root),
            env=env,
            capture_output=True,
            text=True,
        )
        if port_check.returncode != 0:
            m = re.search(
                r"(csrc/[A-Za-z0-9_./-]+)", port_check.stderr + "\n" + port_check.stdout
            )
            sig = m.group(1) if m else "csrc/upstream"
            return _write_result(
                ProbeOutcome(
                    status="fail",
                    stage="kernel_port_check",
                    target_signature=sig,
                    is_public_plugin_interface=False,
                )
            )

    return _write_result(ProbeOutcome(status="pass"))


# ---------------------------------------------------------------------------
# 7. CLI Entrypoint
# ---------------------------------------------------------------------------


def _write_github_outputs(path: Path | None, values: dict[str, Any]) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as stream:
        for key, raw in values.items():
            val = "true" if raw is True else ("false" if raw is False else str(raw))
            if "\n" in val or "\r" in val:
                raise ValueError(f"multiline GitHub output forbidden for {key}")
            stream.write(f"{key}={val}\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_route = sub.add_parser("route")
    p_route.add_argument("--upstream-url", default=DEFAULT_UPSTREAM_URL)
    p_route.add_argument("--ls-remote-file", type=Path, default=None)
    p_route.add_argument(
        "--verified-pin-file", type=Path, default=DEFAULT_VERIFIED_PIN_FILE
    )
    p_route.add_argument(
        "--release-tag-file", type=Path, default=DEFAULT_RELEASE_TAG_FILE
    )
    p_route.add_argument("--gear", choices=VALID_GEARS, default="auto")
    p_route.add_argument("--schedule-cron", default="")
    p_route.add_argument("--upstream-ref", default="")
    p_route.add_argument("--cursor-file", type=Path, default=None)
    p_route.add_argument("--changed-files-file", type=Path, default=None)
    p_route.add_argument("--probe-history-file", type=Path, default=None)
    p_route.add_argument("--active-pr-runs", type=int, default=0)
    p_route.add_argument("--force", action="store_true")
    p_route.add_argument("--github-output", type=Path, default=None)

    p_cpu = sub.add_parser("cpu-check")
    p_cpu.add_argument("--vllm-src", type=Path, required=True)
    p_cpu.add_argument("--sail-root", type=Path, default=REPO_ROOT)
    p_cpu.add_argument("--target-sha", required=True)
    p_cpu.add_argument(
        "--verified-pin-file", type=Path, default=DEFAULT_VERIFIED_PIN_FILE
    )
    p_cpu.add_argument("--previous-baseline", type=Path, default=None)
    p_cpu.add_argument("--output-json", type=Path, required=True)

    p_bisect = sub.add_parser("bisect")
    p_bisect.add_argument("--vllm-repo", type=Path, required=True)
    p_bisect.add_argument("--last-green-sha", required=True)
    p_bisect.add_argument("--red-sha", required=True)
    p_bisect.add_argument("--probe-cmd", required=True)
    p_bisect.add_argument("--initial-red-json", type=Path, default=None)
    p_bisect.add_argument("--default-stage", default="ppu_probe_smoke")
    p_bisect.add_argument("--output-json", type=Path, required=True)
    p_bisect.add_argument("--github-output", type=Path, default=None)

    p_bump = sub.add_parser("prepare-pin-bump")
    p_bump.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    p_bump.add_argument("--target-sha", required=True)
    p_bump.add_argument("--output-json", type=Path, default=None)
    p_bump.add_argument("--github-output", type=Path, default=None)

    p_pub = sub.add_parser("publish")
    p_pub.add_argument("--gear", choices=SELECTED_GEARS, required=True)
    p_pub.add_argument("--status", choices=VALID_STATUSES, required=True)
    p_pub.add_argument("--sail-sha", required=True)
    p_pub.add_argument("--upstream-sha", default="")
    p_pub.add_argument(
        "--verified-pin-file", type=Path, default=DEFAULT_VERIFIED_PIN_FILE
    )
    p_pub.add_argument("--build-image", default="")
    p_pub.add_argument("--runtime-image", default="")
    p_pub.add_argument("--rc-tag", default="")
    p_pub.add_argument("--cpu-status", default="not_run")
    p_pub.add_argument("--ppu-status", default="not_run")
    p_pub.add_argument("--ppu-skip-reason", default="none")
    p_pub.add_argument("--first-failing-sha", default="")
    p_pub.add_argument("--last-good-sha", default="")
    p_pub.add_argument("--bisect-steps", type=int, default=0)
    p_pub.add_argument("--failure-stage", default="none")
    p_pub.add_argument("--target-signature", default="")
    p_pub.add_argument("--public-plugin-break", action="store_true")
    p_pub.add_argument("--known-soft-drift-count", type=int, default=0)
    p_pub.add_argument("--cpu-result-json", type=Path, default=None)
    p_pub.add_argument("--bisect-result-json", type=Path, default=None)
    p_pub.add_argument("--run-url", default="")
    p_pub.add_argument("--cursor-in", type=Path, default=None)
    p_pub.add_argument("--public-dir", type=Path, required=True)
    p_pub.add_argument("--step-summary", type=Path, default=None)

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "route":
        if args.ls_remote_file is not None:
            raw_ls = args.ls_remote_file.read_text(encoding="utf-8")
        else:
            raw_ls = subprocess.check_output(
                [
                    "git",
                    "ls-remote",
                    args.upstream_url,
                    "refs/heads/main",
                    "refs/tags/v*",
                ],
                text=True,
                timeout=60,
            )
        ls_data = parse_ls_remote(raw_ls)
        verified_pin = args.verified_pin_file.read_text(encoding="utf-8").strip()
        release_tag = args.release_tag_file.read_text(encoding="utf-8").strip()
        cursor = (
            json.loads(args.cursor_file.read_text(encoding="utf-8"))
            if args.cursor_file and args.cursor_file.is_file()
            else None
        )
        changed = (
            [
                line.strip()
                for line in args.changed_files_file.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
            if args.changed_files_file and args.changed_files_file.is_file()
            else None
        )
        history = (
            json.loads(args.probe_history_file.read_text(encoding="utf-8"))
            if args.probe_history_file and args.probe_history_file.is_file()
            else ()
        )
        decision = route_three_gears(
            ls_remote_data=ls_data,
            verified_pin=verified_pin,
            release_baseline_tag=release_tag,
            requested_gear=args.gear,
            schedule_cron=args.schedule_cron,
            upstream_ref_override=args.upstream_ref,
            cursor=cursor,
            changed_paths_since_ppu=changed,
            active_pr_runs=args.active_pr_runs,
            probe_history=history,
            force=args.force,
        )
        _write_github_outputs(args.github_output, decision)
        print(json.dumps(decision, indent=2, sort_keys=True))
        return 0

    if args.command == "cpu-check":
        verified_pin = args.verified_pin_file.read_text(encoding="utf-8").strip()
        return run_sdk_cpu_fast_check(
            vllm_src=args.vllm_src,
            sail_root=args.sail_root,
            target_sha=args.target_sha,
            verified_pin=verified_pin,
            previous_baseline=args.previous_baseline,
            output_json=args.output_json,
        )

    if args.command == "bisect":
        green = _require_sha1(args.last_green_sha, "last_green_sha")
        red = _require_sha1(args.red_sha, "red_sha")
        rev = subprocess.run(
            [
                "git",
                "-C",
                str(args.vllm_repo),
                "rev-list",
                "--first-parent",
                "--reverse",
                f"{green}..{red}",
            ],
            capture_output=True,
            text=True,
        )
        commits = (
            [
                line.strip()
                for line in rev.stdout.splitlines()
                if SHA1_RE.fullmatch(line.strip())
            ]
            if rev.returncode == 0
            else []
        )
        if not commits or commits[-1] != red:
            commits = [red]

        initial_red = None
        if args.initial_red_json and args.initial_red_json.is_file():
            init_data = json.loads(args.initial_red_json.read_text(encoding="utf-8"))
            initial_red = ProbeOutcome(
                status="infra_error"
                if init_data.get("status") == "infra_error"
                else "fail",
                stage=str(init_data.get("stage") or args.default_stage),
                target_signature=str(
                    init_data.get("target_signature") or "window_head"
                ),
                is_public_plugin_interface=bool(
                    init_data.get("is_public_plugin_interface")
                ),
            )

        def _shell_probe(candidate_sha: str) -> ProbeOutcome:
            probe_json = args.output_json.parent / f"probe-{candidate_sha[:12]}.json"
            probe_json.unlink(missing_ok=True)
            child_env = dict(os.environ)
            child_env["CANDIDATE_SHA"] = candidate_sha
            child_env["PROBE_RESULT_JSON"] = str(probe_json)
            proc = subprocess.run(
                ["bash", "-c", args.probe_cmd],
                env=child_env,
                capture_output=True,
                text=True,
            )
            if probe_json.is_file():
                data = json.loads(probe_json.read_text(encoding="utf-8"))
                probe_json.unlink(missing_ok=True)
                raw_status = str(
                    data.get("status") or ("pass" if proc.returncode == 0 else "fail")
                )
                return ProbeOutcome(
                    status=raw_status
                    if raw_status in ("pass", "fail", "infra_error")
                    else "fail",
                    stage=str(data.get("stage") or args.default_stage),
                    target_signature=str(
                        data.get("target_signature") or "bisect_probe"
                    ),
                    is_public_plugin_interface=bool(
                        data.get("is_public_plugin_interface")
                    ),
                )
            if proc.returncode == 0:
                return ProbeOutcome(status="pass")
            return ProbeOutcome(
                status="infra_error" if proc.returncode == 2 else "fail",
                stage=args.default_stage,
                target_signature="bisect_probe",
            )

        res = bisect_first_failing_sha(
            last_green_sha=green,
            red_sha=red,
            commits=commits,
            probe_fn=_shell_probe,
            verify_baseline_first=True,
            initial_red_outcome=initial_red,
        )
        win: ProbeOutcome = res["winning_outcome"]
        serializable = {
            "bisect_status": res["bisect_status"],
            "first_failing_sha": res["first_failing_sha"],
            "last_good_sha": res["last_good_sha"],
            "steps_executed": res["steps_executed"],
            "probe_steps_on_range": res["probe_steps_on_range"],
            "stage": win.stage if win.stage in VALID_STAGES else args.default_stage,
            "target_signature": _sanitize_symbol(win.target_signature, "unspecified"),
            "is_public_plugin_interface": bool(win.is_public_plugin_interface),
            "history": res["history"],
        }
        text = json.dumps(serializable, indent=2, sort_keys=True) + "\n"
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        assert_zero_sensitive_leaks(text, staging_paths=[args.output_json])
        args.output_json.write_text(text, encoding="utf-8")
        _write_github_outputs(
            args.github_output,
            {
                "bisect_status": serializable["bisect_status"],
                "first_failing_sha": serializable["first_failing_sha"],
                "last_good_sha": serializable["last_good_sha"],
                "steps_executed": serializable["steps_executed"],
                "stage": serializable["stage"],
                "target_signature": serializable["target_signature"],
                "is_public_plugin_interface": serializable[
                    "is_public_plugin_interface"
                ],
            },
        )
        print(text, end="")
        return 0

    if args.command == "prepare-pin-bump":
        res = prepare_pin_bump(repo_root=args.repo_root, target_sha=args.target_sha)
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(
                json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        _write_github_outputs(
            args.github_output,
            {
                "changed": res["changed"],
                "previous_pin": res["previous_pin"],
                "new_pin": res["new_pin"],
                "title": res["title"],
            },
        )
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0

    if args.command == "publish":
        verified_pin = args.verified_pin_file.read_text(encoding="utf-8").strip()
        cpu_meta = (
            json.loads(args.cpu_result_json.read_text(encoding="utf-8"))
            if args.cpu_result_json and args.cpu_result_json.is_file()
            else {}
        )
        bisect_meta = (
            json.loads(args.bisect_result_json.read_text(encoding="utf-8"))
            if args.bisect_result_json and args.bisect_result_json.is_file()
            else {}
        )
        effective_status = args.status
        if bisect_meta.get("bisect_status") in (
            "baseline_control_failed",
            "aborted_infra_error",
        ):
            effective_status = "infra_error"
        failure_stage = (
            args.failure_stage
            if args.failure_stage != "none"
            else str(bisect_meta.get("stage") or cpu_meta.get("stage") or "none")
        )
        target_sig = args.target_signature or str(
            bisect_meta.get("target_signature")
            or cpu_meta.get("target_signature")
            or ""
        )
        public_break = bool(
            args.public_plugin_break
            or bisect_meta.get("is_public_plugin_interface")
            or cpu_meta.get("is_public_plugin_interface")
        )
        first_bad = args.first_failing_sha or str(
            bisect_meta.get("first_failing_sha") or ""
        )
        last_good = args.last_good_sha or str(bisect_meta.get("last_good_sha") or "")
        bisect_steps = args.bisect_steps or int(bisect_meta.get("steps_executed") or 0)
        soft_drift = args.known_soft_drift_count or int(
            cpu_meta.get("known_soft_drift_count") or 0
        )
        patch_fps = cpu_meta.get("patch_fingerprints")
        if not isinstance(patch_fps, dict):
            patch_fps = None

        outcome = None
        if effective_status in ("fail", "infra_error"):
            outcome = ProbeOutcome(
                status="infra_error" if effective_status == "infra_error" else "fail",
                stage=failure_stage,
                target_signature=target_sig,
                is_public_plugin_interface=public_break,
            )
        published = publish_sanitized_channels(
            gear=args.gear,
            status=effective_status,
            sail_sha=args.sail_sha,
            upstream_sha=args.upstream_sha,
            verified_pin=verified_pin,
            build_image=args.build_image,
            runtime_image=args.runtime_image,
            rc_tag=args.rc_tag,
            cpu_status=args.cpu_status,
            ppu_status=args.ppu_status,
            ppu_skip_reason=args.ppu_skip_reason,
            first_failing_sha=first_bad,
            last_good_sha=last_good,
            bisect_steps=bisect_steps,
            outcome=outcome,
            known_soft_drift_count=soft_drift,
            run_url=args.run_url,
            public_dir=args.public_dir,
            step_summary_path=args.step_summary,
        )
        prev_cursor = (
            json.loads(args.cursor_in.read_text(encoding="utf-8"))
            if args.cursor_in and args.cursor_in.is_file()
            else None
        )
        next_cursor = advance_cursor(
            prev_cursor,
            verified_pin=verified_pin,
            selected_gear=args.gear,
            target_sha=args.upstream_sha,
            rc_tag=args.rc_tag,
            cpu_status=args.cpu_status,
            ppu_status=args.ppu_status,
            ppu_skip_reason=args.ppu_skip_reason,
            patch_fingerprints=patch_fps,
        )
        cursor_path = args.public_dir / "cursor.json"
        cursor_text = json.dumps(next_cursor, indent=2, sort_keys=True) + "\n"
        assert_zero_sensitive_leaks(cursor_text, staging_paths=[cursor_path])
        cursor_path.write_text(cursor_text, encoding="utf-8")
        print(published["console"])
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
