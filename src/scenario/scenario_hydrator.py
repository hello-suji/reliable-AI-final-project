from __future__ import annotations

import copy
import re
from typing import Any, Dict, List


def hydrate_scenario_dict(
    scenario: Dict[str, Any],
    clue: Dict[str, Any],
    repo: str = "",
    context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Preserve issue-derived scenario evidence after LLM generation.

    The generator and alignment loop consume `scenario.json` and
    `scenario_validation.json`, so any clue fields dropped by the LLM should be
    recovered before those artifacts are written.
    """
    hydrated = copy.deepcopy(scenario or {})

    _fill_if_empty(hydrated, "reproduction_code", clue.get("code_examples", []))
    _fill_if_empty(hydrated, "expected_outputs", clue.get("expected_outputs", []))
    _fill_if_empty(hydrated, "actual_outputs", clue.get("actual_outputs", []))
    _fill_if_empty(hydrated, "identifiers", clue.get("identifiers", {}))
    _fill_if_empty(hydrated, "error_keywords", clue.get("error_keywords", []))

    oracle_hints = synthesize_oracle_hints(clue, hydrated, repo=repo, context=context)
    oracle_contract = synthesize_oracle_contract(clue, hydrated, repo=repo, context=context)
    hydrated["oracle_contract"] = oracle_contract
    if not hydrated.get("oracle_type"):
        hydrated["oracle_type"] = oracle_contract["oracle_type"]
    if not hydrated.get("oracle_source"):
        hydrated["oracle_source"] = oracle_contract["oracle_source"]
    if oracle_hints:
        existing = _ensure_list(hydrated.get("oracle_hints", []))
        hydrated["oracle_hints"] = _dedup(existing + oracle_hints)
        current = str(hydrated.get("oracle", "") or "").strip()
        hydrated["oracle"] = _merge_oracle_text(current, hydrated["oracle_hints"])
    else:
        hydrated.setdefault("oracle_hints", [])
        hydrated.setdefault("oracle", "")

    return hydrated


def synthesize_oracle_contract(
    clue: Dict[str, Any],
    scenario: Dict[str, Any] | None = None,
    repo: str = "",
    context: Dict[str, Any] | None = None,
) -> Dict[str, str]:
    """Classify the intended oracle so the generator can avoid fix-failing tests."""
    scenario = scenario or {}
    repo_lower = (repo or "").lower()
    text = " ".join(
        str(x)
        for x in (
            clue.get("observed_behavior", [])
            + clue.get("expected_behavior", [])
            + clue.get("repro_conditions", [])
            + [clue.get("raw_issue_text", "")]
            + _ensure_list(scenario.get("expected_failure", ""))
        )
    ).lower()
    expected = _ensure_list(scenario.get("expected_outputs") or clue.get("expected_outputs", []))
    actual = _ensure_list(scenario.get("actual_outputs") or clue.get("actual_outputs", []))

    if expected:
        oracle_type = "positive_value"
        oracle_source = "issue_expected"
        rule = "Assert the fixed value stated by the issue; do not invent a different expected value."
    elif re.search(
        r"should\s+not\s+(?:raise|error|fail|crash|warn)|"
        r"must\s+not\s+(?:raise|error|fail|crash|warn)|"
        r"does\s+not\s+(?:raise|error|fail|crash|warn)|"
        r"doesn't\s+(?:raise|error|fail|crash|warn)|"
        r"without\s+(?:raising|error|failing|crashing|warning)|"
        r"no\s+(?:exception|error|warning)|"
        r"no\s+longer\s+(?:raises|errors|fails|crashes|warns)",
        text,
    ):
        oracle_type = "success_path"
        oracle_source = "issue_expected"
        rule = "Assert the call succeeds and check a post-call state/value; do not use pytest.raises."
    elif re.search(r"should\s+raise|must\s+raise|expected\s+(?:error|exception)", text):
        oracle_type = "exception_expected"
        oracle_source = "issue_expected"
        rule = "Assert the expected exception type only; never match exception message text."
    elif re.search(r"\bwarning\b|runtimewarning|warns", text):
        oracle_type = "warning_absence" if re.search(r"no warning|without warning|should not warn", text) else "semantic_invariant"
        oracle_source = "issue_repro_code"
        rule = "Do not assert warning presence alone; verify the fixed value/state or no-warning success path."
    elif "sphinx" in repo_lower or "matplotlib" in repo_lower or "seaborn" in repo_lower:
        oracle_type = "semantic_invariant"
        oracle_source = "inferred_semantic"
        rule = "Use a small semantic invariant; avoid raw rendered text, guessed legend labels, and brittle exact strings."
    elif actual:
        oracle_type = "semantic_invariant"
        oracle_source = "actual_buggy_output"
        rule = "Use != buggy value only on the real function result and only if no positive invariant is available."
    else:
        oracle_type = "last_resort_structural"
        oracle_source = "inferred_semantic"
        rule = "Structural assertions are last resort and should not be accepted as final ALIGNED unless strengthened."

    return {
        "oracle_type": oracle_type,
        "oracle_source": oracle_source,
        "rule": rule,
    }


def hydrate_validation_report_dict(
    report: Dict[str, Any],
    clue: Dict[str, Any],
    repo: str = "",
    context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    hydrated = copy.deepcopy(report or {})
    for bucket in ("selected_scenarios", "rejected_scenarios"):
        for item in hydrated.get(bucket, []) or []:
            normalized = item.get("normalized_scenario")
            if isinstance(normalized, dict):
                item["normalized_scenario"] = hydrate_scenario_dict(
                    normalized,
                    clue,
                    repo=repo,
                    context=context,
                )
    return hydrated


def hydrate_validation_report(
    report: Any,
    clue: Dict[str, Any],
    repo: str = "",
    context: Dict[str, Any] | None = None,
) -> Any:
    for attr in ("selected_scenarios", "rejected_scenarios"):
        for item in getattr(report, attr, []) or []:
            normalized = getattr(item, "normalized_scenario", None)
            if isinstance(normalized, dict):
                item.normalized_scenario = hydrate_scenario_dict(
                    normalized,
                    clue,
                    repo=repo,
                    context=context,
                )
    return report


def synthesize_oracle_hints(
    clue: Dict[str, Any],
    scenario: Dict[str, Any] | None = None,
    repo: str = "",
    context: Dict[str, Any] | None = None,
) -> List[str]:
    scenario = scenario or {}
    repo_lower = (repo or "").lower()
    text = " ".join(
        str(x)
        for x in (
            clue.get("observed_behavior", [])
            + clue.get("expected_behavior", [])
            + clue.get("repro_conditions", [])
            + [clue.get("raw_issue_text", "")]
            + _ensure_list(scenario.get("expected_failure", ""))
        )
    ).lower()
    expected = _ensure_list(scenario.get("expected_outputs") or clue.get("expected_outputs", []))
    actual = _ensure_list(scenario.get("actual_outputs") or clue.get("actual_outputs", []))

    hints: List[str] = []
    if expected:
        hints.append(
            "Prefer a positive oracle from expected_outputs; assert the fixed behavior equals the stated correct value."
        )
        sample = " ".join(expected[:2]).lower()
        if _looks_array_like(sample) or "array" in sample:
            hints.append("For numpy/list/array outputs, use np.testing.assert_array_equal or assert_allclose, never plain assert ==.")
        if _looks_float_like(sample):
            hints.append("For floating-point outputs, use pytest.approx or a semantic numeric invariant, not exact equality.")
    elif actual:
        hints.append(
            "Only buggy actual_outputs are known; derive a positive invariant from the issue if possible, and use != buggy value only as last resort."
        )

    if re.search(r"\bnan\b|runtimewarning|invalid value|divide by zero", text):
        hints.append("For NaN/warning behavior, do not compare with float('nan'); use np.isnan, np.isfinite, or pytest.warns/no-warning checks.")

    if (
        "requests" in repo_lower
        or re.search(r"content-length|preparedrequest|\bheaders?\b|\bqop\b|requests\.", text)
    ):
        hints.append("For requests/http issues, never call external URLs; inspect PreparedRequest/request headers or use existing local helpers.")

    if "matplotlib" in repo_lower or re.search(r"\baxis\b|\binvert|hist|bins?|density|xlim|ylim", text):
        hints.append("For matplotlib axis inversion, prefer ax.yaxis_inverted()/xaxis_inverted() or semantic bin/tick invariants over raw limit equality.")

    if "django" in repo_lower:
        hints.append("For Django, do not define inline models; reuse existing test models/imports from the target test file.")

    if "warning" in text or any("warning" in str(x).lower() for x in clue.get("error_keywords", [])):
        hints.append("For warning regressions, use pytest.warns or warnings.catch_warnings around the minimal triggering call.")

    return _dedup(hints)


def _fill_if_empty(target: Dict[str, Any], key: str, value: Any) -> None:
    if target.get(key):
        return
    if value:
        target[key] = copy.deepcopy(value)


def _ensure_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, tuple):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, str):
        return [value] if value.strip() else []
    if value:
        return [str(value)]
    return []


def _dedup(values: List[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for value in values:
        norm = str(value).strip()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        result.append(norm)
    return result


def _merge_oracle_text(current: str, hints: List[str]) -> str:
    parts = _ensure_list(current)
    current_lower = current.lower()
    for hint in hints:
        hint = str(hint).strip()
        if hint and hint.lower() not in current_lower:
            parts.append(hint)
    return " ".join(_dedup(parts)).strip()


def _looks_array_like(text: str) -> bool:
    return bool(re.search(r"\barray\s*\(|\[\s*\[|dtype=|ndarray", text))


def _looks_float_like(text: str) -> bool:
    return bool(re.search(r"\b\d+\.\d+(?:e[-+]?\d+)?\b|\bapprox\b|\btolerance\b", text))
