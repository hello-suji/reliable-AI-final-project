from __future__ import annotations

from typing import Any, Dict, Iterable


_NOISY_FUNCTIONS = {
    "arange", "rand", "random", "seed", "platform", "get_backend",
    "show_versions", "main", "run", "get", "set",
}


def ensure_primary_scenario(
    validation_report: Dict[str, Any],
    clue: Dict[str, Any] | None = None,
    context: Dict[str, Any] | None = None,
    reason: str = "deterministic_repair",
) -> Dict[str, Any]:
    """Ensure validation_report has one selected normalized scenario."""
    report = validation_report if isinstance(validation_report, dict) else {}
    selected = report.setdefault("selected_scenarios", [])
    if selected and isinstance(selected[0].get("normalized_scenario"), dict):
        return report

    repaired = _build_repaired_scenario(clue or {}, context or {}, reason=reason)
    selected.insert(0, {
        "scenario_id": repaired["scenario_id"],
        "score": 0.3,
        "decision": "accept",
        "reasons": [f"force-selected repaired scenario: {reason}"],
        "normalized_scenario": repaired,
        "force_selected": True,
        "scenario_repaired": True,
        "scenario_repair_reason": reason,
        "scenario_required_fields_filled": [
            "reproduction_code",
            "expected_outputs",
            "actual_outputs",
            "identifiers",
            "target_location",
            "oracle_contract",
        ],
    })
    return report


def select_primary_scenario(validation_report: Dict[str, Any]) -> Dict[str, Any]:
    """validation_report에서 현재 사용 중인 primary 시나리오를 가져온다.

    candidate_test_file이 있는 시나리오를 우선하고, 없으면 첫 번째 선택 시나리오를 반환한다.

    Raises:
        ValueError: selected_scenarios가 비어 있거나 normalized_scenario가 없으면 발생.
    """
    selected = validation_report.get("selected_scenarios", [])
    if not selected:
        validation_report = ensure_primary_scenario(validation_report)
        selected = validation_report.get("selected_scenarios", [])

    for item in selected:
        normalized = item.get("normalized_scenario")
        if not normalized:
            continue
        target = normalized.get("target_location", {})
        if target.get("candidate_test_file"):
            return normalized

    # 첫 번째 선택 시나리오의 normalized_scenario 찾기
    first_normalized = selected[0].get("normalized_scenario")
    if not first_normalized:
        validation_report = ensure_primary_scenario(validation_report)
        return validation_report["selected_scenarios"][0]["normalized_scenario"]
    return first_normalized


def _build_repaired_scenario(
    clue: Dict[str, Any],
    context: Dict[str, Any],
    reason: str,
) -> Dict[str, Any]:
    identifiers = clue.get("identifiers", {}) if isinstance(clue.get("identifiers"), dict) else {}
    functions = identifiers.get("functions", []) or []
    classes = identifiers.get("classes", []) or []
    source_files = [
        x.get("path", "")
        for x in context.get("candidate_source_files", [])
        if isinstance(x, dict) and x.get("path")
    ]
    test_files = [
        x.get("path", "")
        for x in context.get("candidate_test_files", [])
        if isinstance(x, dict) and x.get("path")
    ]
    runner = (context.get("project_test_style") or {}).get("runner", "pytest")
    expected = clue.get("expected_outputs", []) or []
    actual = clue.get("actual_outputs", []) or []
    if expected:
        oracle_type = "positive_value"
        oracle_source = "issue_expected"
        rule = "Assert the fixed value stated by the issue."
    elif actual:
        oracle_type = "semantic_invariant"
        oracle_source = "actual_buggy_output"
        rule = "Assert a public semantic invariant and avoid exact guessed values."
    else:
        oracle_type = "last_resort_structural"
        oracle_source = "inferred_semantic"
        rule = "Use the target API and assert public state/type/range."

    target_function = _choose_target_function(clue, context, functions)
    return {
        "scenario_id": "S_REPAIRED",
        "target_location": {
            "source_file": source_files[0] if source_files else "",
            "target_function": target_function,
            "related_classes": classes[:3],
            "candidate_test_file": test_files[0] if test_files else "",
            "confidence": "low",
        },
        "setup_steps": ["Set up the issue reproduction using existing project test helpers."],
        "execution_stimulus": [
            f"Call {target_function} with the issue reproduction conditions."
            if target_function else "Execute the issue reproduction code."
        ],
        "expected_failure": (
            str((clue.get("observed_behavior") or ["Buggy behavior should be reproduced."])[0])
        ),
        "relevant_source_files": source_files[:3],
        "relevant_test_files": test_files[:3],
        "test_environment": {"required_fixtures": [], "runner": runner},
        "reproduction_code": clue.get("code_examples", []) or [],
        "expected_outputs": expected,
        "actual_outputs": actual,
        "error_keywords": clue.get("error_keywords", []) or [],
        "identifiers": identifiers,
        "oracle_contract": {
            "oracle_type": oracle_type,
            "oracle_source": oracle_source,
            "rule": rule,
        },
        "oracle_type": oracle_type,
        "oracle_source": oracle_source,
        "oracle_hints": [rule],
        "oracle": rule,
        "scenario_repaired": True,
        "scenario_repair_reason": reason,
    }


def _choose_target_function(
    clue: Dict[str, Any],
    context: Dict[str, Any],
    functions: Iterable[str],
) -> str:
    """Pick a concrete target function without inventing one."""
    function_list = [f for f in functions if f and f not in _NOISY_FUNCTIONS]
    for fault in clue.get("fault_locations", []) or []:
        fn = fault.get("function_name")
        if fn and fn not in _NOISY_FUNCTIONS:
            return fn
    if function_list:
        return function_list[0]
    for source in context.get("candidate_source_files", []) or []:
        for key in ("top_level_functions", "functions", "matched_identifiers"):
            values = source.get(key) or []
            if isinstance(values, dict):
                values = values.get("functions") or []
            for fn in values:
                if isinstance(fn, str) and fn and fn not in _NOISY_FUNCTIONS:
                    return fn
    return ""
