from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass
class ScenarioValidationResult:
    scenario_id: str
    score: float
    decision: str
    reasons: List[str]
    normalized_scenario: Dict[str, Any] | None = None
    force_selected: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScenarioValidationReport:
    selected_scenarios: List[ScenarioValidationResult]
    rejected_scenarios: List[ScenarioValidationResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_scenarios": [x.to_dict() for x in self.selected_scenarios],
            "rejected_scenarios": [x.to_dict() for x in self.rejected_scenarios],
        }


_FORCE_SELECT_MIN_SCORE = 0.0  # Always keep at least one usable scenario.


class ScenarioValidator:
    def __init__(
        self,
        accept_threshold: float = 0.65,
        duplicate_threshold: float = 0.60,
        max_selected: int = 2,
    ) -> None:
        self.accept_threshold = accept_threshold
        self.duplicate_threshold = duplicate_threshold
        self.max_selected = max_selected

    def validate(
        self,
        scenarios: List[Dict[str, Any]],
        clue: Dict[str, Any],
        context: Dict[str, Any],
    ) -> ScenarioValidationReport:
        results: List[ScenarioValidationResult] = []

        for scenario in scenarios:
            score, reasons, normalized = self._score_scenario(
                scenario=scenario,
                clue=clue,
                context=context,
            )

            decision = "accept" if score >= self.accept_threshold else "reject"

            results.append(
                ScenarioValidationResult(
                    scenario_id=scenario.get("scenario_id", "unknown"),
                    score=round(score, 4),
                    decision=decision,
                    reasons=reasons,
                    normalized_scenario=normalized,
                )
            )

        deduped = self._deduplicate(results)

        accepted = [r for r in deduped if r.decision == "accept"]
        rejected = [r for r in deduped if r.decision == "reject"]

        # Fallback: if no scenarios pass threshold, force-select the best one.
        # The generator must always receive at least one scenario; weak scenarios
        # are repaired/hydrated downstream instead of failing with t=null.
        if not accepted and rejected:
            best = max(rejected, key=lambda x: x.score)
            if best.score >= _FORCE_SELECT_MIN_SCORE:
                best.decision = "accept"
                best.force_selected = True
                best.reasons.append(
                    f"force-selected as best available (score {best.score:.2f}, below threshold {self.accept_threshold})"
                )
                # Recover normalized_scenario from the original scenario list
                if best.normalized_scenario is None:
                    for s in scenarios:
                        if s.get("scenario_id") == best.scenario_id:
                            _, _, normalized = self._score_scenario(s, clue, context)
                            best.normalized_scenario = normalized
                            break
                accepted.append(best)
                rejected = [r for r in rejected if r.scenario_id != best.scenario_id]
            else:
                # 최고 점수도 최솟값 미만 → 모든 시나리오 거부 (파이프라인이 재생성해야 함)
                best.reasons.append(
                    f"force-select skipped: best score {best.score:.2f} < minimum {_FORCE_SELECT_MIN_SCORE}"
                )

        accepted.sort(key=lambda x: x.score, reverse=True)

        overflow = accepted[self.max_selected:]
        accepted = accepted[: self.max_selected]

        for extra in overflow:
            extra.decision = "reject"
            extra.reasons.append("rejected because max_selected limit exceeded")
            extra.normalized_scenario = None
            rejected.append(extra)

        rejected.sort(key=lambda x: x.score, reverse=True)

        return ScenarioValidationReport(
            selected_scenarios=accepted,
            rejected_scenarios=rejected,
        )

    def save(self, report: ScenarioValidationReport, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

    def _score_scenario(
        self,
        scenario: Dict[str, Any],
        clue: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[float, List[str], Dict[str, Any]]:
        score = 0.0
        reasons: List[str] = []

        target = dict(scenario.get("target_location", {}) or {})
        source_file = target.get("source_file", "")
        target_function = target.get("target_function") or ""
        related_classes = set(target.get("related_classes") or [])
        candidate_test_file = target.get("candidate_test_file") or ""

        noisy_functions = {
            "arange", "rand", "random", "seed", "platform", "get_backend",
            "show_versions",
        }
        clue_funcs = {
            fn for fn in clue.get("identifiers", {}).get("functions", [])
            if fn not in noisy_functions
        }
        clue_classes = set(clue.get("identifiers", {}).get("classes", []))

        context_source_entries = context.get("candidate_source_files", [])
        context_test_entries = context.get("candidate_test_files", [])

        context_source_files = {x.get("path", "") for x in context_source_entries}
        context_test_files = {x.get("path", "") for x in context_test_entries}

        source_score_map = {x.get("path", ""): x.get("score", 0) for x in context_source_entries}
        test_score_map = {x.get("path", ""): x.get("score", 0) for x in context_test_entries}
        test_risk_map = {x.get("path", ""): x.get("collection_risk", "") for x in context_test_entries}

        if (
            not candidate_test_file
            or candidate_test_file not in context_test_files
            or test_risk_map.get(candidate_test_file)
        ):
            replacement = self._pick_viable_test_file(context, candidate_test_file)
            if replacement and replacement != candidate_test_file:
                old = candidate_test_file or "(missing)"
                candidate_test_file = replacement
                target["candidate_test_file"] = replacement
                reasons.append(
                    f"test_file_overridden_skip_or_missing:{old}->{replacement}"
                )

        preconditions = scenario.get("preconditions", [])
        setup_steps = scenario.get("setup_steps", [])
        execution_stimulus = scenario.get("execution_stimulus", [])
        expected_failure = scenario.get("expected_failure", "")
        # LLM이 str 대신 list를 반환할 수 있으므로 방어적 처리
        if isinstance(expected_failure, list):
            expected_failure = " ".join(str(x) for x in expected_failure)

        relevant_source_files = set(scenario.get("relevant_source_files", []))
        relevant_test_files = set(scenario.get("relevant_test_files", []))

        execution_text = " ".join(execution_stimulus).lower()
        expected_failure_text = expected_failure.lower()

        # 1. 필수 필드 존재 (oracle은 제거됨 — expected_failure로 대체)
        required_ok = all([
            bool(source_file),
            bool(target_function),
            len(execution_stimulus) > 0,
            bool(expected_failure),
        ])
        if required_ok:
            score += 0.16
            reasons.append("required fields are present")
        else:
            reasons.append("missing required fields")

        # 2. target function 정합성
        if target_function in clue_funcs:
            score += 0.22
            reasons.append("target function matches issue clue")
        else:
            reasons.append("target function is weakly aligned with issue clue")

        # 3. related classes 정합성
        overlap_classes = len(clue_classes & related_classes)
        if overlap_classes >= 2:
            score += 0.08
            reasons.append("related classes overlap with issue classes")
        elif overlap_classes == 1:
            score += 0.04
            reasons.append("partial class overlap with issue classes")
        else:
            reasons.append("class evidence is weak")

        # 4. source file 후보 정합성
        if source_file in context_source_files:
            score += 0.08
            reasons.append("source file is aligned with candidate source files")
            source_entry = next((x for x in context_source_entries if x.get("path") == source_file), {})
            if "graph_neighbor" in (source_entry.get("localization_signals") or []):
                score += 0.03
                reasons.append("graph_expanded_source")
        else:
            reasons.append("source file not found in candidate source files")

        # 5. source file 랭킹 반영
        src_rank_score = source_score_map.get(source_file, 0)
        if src_rank_score >= 5:
            score += 0.10
            reasons.append("source file has strong retrieval score")
        elif src_rank_score >= 3:
            score += 0.05
            reasons.append("source file has moderate retrieval score")
        else:
            reasons.append("source file retrieval evidence is weak")

        # 6. 함수-파일 의미 정합성 (일반화)
        source_name = Path(source_file).name.lower()
        func_tokens = set(target_function.lower().replace("_", " ").split())
        source_tokens = set(source_name.replace(".py", "").replace("_", " ").split())
        token_overlap = len(func_tokens & source_tokens)
        if token_overlap >= 1:
            score += 0.10
            reasons.append("source file name is semantically aligned with target function")
        elif target_function.lower() in source_name:
            score += 0.15
            reasons.append("source file name contains target function")

        # 7. relevant source files 안의 top source 우선
        if context_source_entries:
            best_source = context_source_entries[0].get("path", "")
            if source_file == best_source:
                score += 0.08
                reasons.append("scenario targets the top-ranked source file")

        # 8. test file 정합성
        overlap_tests = len(relevant_test_files & context_test_files)
        if overlap_tests >= 2:
            score += 0.08
            reasons.append("relevant test files align with candidate test files")
        elif overlap_tests == 1:
            score += 0.04
            reasons.append("partial alignment with candidate test files")
        else:
            reasons.append("test file evidence is weak")

        # 9. candidate_test_file 추가 보너스
        if candidate_test_file:
            if candidate_test_file in context_test_files:
                score += 0.08
                reasons.append("candidate test file is explicitly aligned with candidate test files")
                test_rank_score = test_score_map.get(candidate_test_file, 0)
                if test_rank_score >= 20:
                    score += 0.05
                    reasons.append("candidate test file has strong retrieval score")
            else:
                score -= 0.04
                reasons.append("candidate test file is not in retrieved candidate test files")

        # 10. execution 자극 구체성
        if target_function.lower() in execution_text:
            score += 0.08
            reasons.append("execution stimulus mentions target function")
        else:
            reasons.append("execution stimulus is too abstract")

        if "assert" in expected_failure_text:
            score += 0.05
            reasons.append("expected failure is explicit about assertion failure")

        # 11. expected_failure 구체성 (oracle 대체)
        concrete_keywords = [
            "assert", "equal", "raise", "return", "error", "exception",
            "true", "false", "none", "should", "must", "expect",
            "result", "value", "match", "correct", "fail", "pass",
            "output", "produce", "yield", "compare",
        ]
        if any(k in expected_failure_text for k in concrete_keywords):
            score += 0.10
            reasons.append("expected_failure is concrete and checkable")
        else:
            reasons.append("expected_failure is too abstract")

        # 12. setup_steps 비실행형 감점
        non_actionable = 0
        for p in setup_steps:
            p_lower = p.lower().strip()
            if p_lower.startswith("consider the following") or p_lower.startswith("if "):
                non_actionable += 1

        if non_actionable >= 2:
            score -= 0.10
            reasons.append("contains non-actionable setup steps copied from issue text")
        elif non_actionable == 1:
            score -= 0.05
            reasons.append("contains one non-actionable setup step")

        # 14. setup 부족 감점
        if len(setup_steps) < 2:
            score -= 0.05
            reasons.append("setup steps are insufficient")

        # 15. 관련 source file 목록에 핵심 파일 포함 여부
        if source_file in relevant_source_files:
            score += 0.03
            reasons.append("target source file is also included in relevant source files")

        # 16. reproduction_code가 target_function을 호출하는지
        repro_code_text = " ".join(
            block.get("code", "") if isinstance(block, dict) else str(block)
            for block in scenario.get("reproduction_code", [])
        )
        if target_function and repro_code_text and target_function in repro_code_text:
            score += 0.08
            reasons.append(f"reproduction_code references target_function:{target_function}")

        # 17. target_function 실존 검증 (context candidate_source_files의 matched_identifiers)
        source_idents_in_context: set[str] = set()
        for sf in context_source_entries:
            if sf.get("path") == source_file:
                source_idents_in_context.update(sf.get("matched_identifiers", []))

        if source_idents_in_context:
            if target_function in source_idents_in_context:
                score += 0.05
                reasons.append("target_function verified to exist in source_file")
            elif target_function:
                score -= 0.10
                reasons.append(
                    f"target_function '{target_function}' NOT found in source_file — may be hallucinated"
                )

        # 17b. top_level_functions 교차 검증 (AST 추출 결과)
        # dunder 메서드(__init__ 등)는 제외 (AST 추출에서 빠지지만 정당한 타겟)
        for sf in context_source_entries:
            if sf.get("path") == source_file:
                top_funcs = sf.get("top_level_functions") or []
                if top_funcs and target_function:
                    bare_name = target_function.split(".")[-1]
                    # dunder는 AST에서 제외되므로 검증 skip
                    if bare_name.startswith("__") and bare_name.endswith("__"):
                        break
                    in_top = any(
                        tf == target_function or tf.split(".")[-1] == bare_name
                        for tf in top_funcs
                    )
                    if not in_top:
                        score -= 0.15
                        reasons.append(
                            f"target_function '{target_function}' not in AST top_level_functions "
                            f"for {source_file}"
                        )
                        public_wrappers = [
                            fn for fn in top_funcs
                            if fn and not str(fn).startswith("_")
                            and (
                                fn.lower() in execution_text
                                or fn in repro_code_text
                                or any(tok in fn.lower() for tok in func_tokens if len(tok) >= 4)
                            )
                        ]
                        if public_wrappers:
                            reasons.append(
                                "target_public_wrapper_ok:"
                                + ",".join(public_wrappers[:3])
                            )
                    else:
                        score += 0.05
                        reasons.append("target_function confirmed in AST top_level_functions")
                break

        # 18. fault_locations 정합성 (진짜 traceback 기반 고신뢰도 후보만 강하게 반영)
        fault_locations = [
            fl for fl in clue.get("fault_locations", [])
            if fl.get("source", "traceback") == "traceback"
            and fl.get("confidence", "high") == "high"
        ]
        if fault_locations:
            fl_files: set[str] = set()
            fl_funcs: set[str] = set()
            for fl in fault_locations:
                fp = fl.get("file_path", "").replace("\\", "/")
                fn = fl.get("function_name", "")
                parts = fp.split("/")
                for k in range(1, min(5, len(parts))):
                    fl_files.add("/".join(parts[-k:]))
                if fn:
                    fl_funcs.add(fn)

            source_tail = "/".join(source_file.replace("\\", "/").split("/")[-3:])
            file_match = any(
                source_file.endswith(f) or f.endswith(source_file) or source_tail in f
                for f in fl_files
            )
            func_match = bool(target_function and target_function in fl_funcs)

            if file_match and func_match:
                score += 0.15
                reasons.append("target matches traceback fault location (file+function)")
            elif file_match:
                score += 0.08
                reasons.append("target file matches traceback fault location")
            elif func_match:
                score += 0.05
                reasons.append("target function matches traceback fault location")
            else:
                score -= 0.08
                reasons.append("target does NOT match traceback fault location — likely wrong file")

        normalized = dict(scenario)
        normalized["target_location"] = target
        # setup_steps에서 비실행형 항목 정리 (preconditions 하위호환 포함)
        all_setup = list(preconditions) + [s for s in setup_steps if s not in preconditions]
        normalized["setup_steps"] = [
            s for s in all_setup
            if not s.lower().strip().startswith("consider the following")
            and not s.lower().strip().startswith("if ")
        ]
        normalized.pop("preconditions", None)

        return max(0.0, min(score, 1.0)), reasons, normalized

    @staticmethod
    def _pick_viable_test_file(context: Dict[str, Any], current: str = "") -> str:
        repo_path = Path(context.get("repo_path", ""))
        for entry in context.get("candidate_test_files", []):
            path = entry.get("path", "")
            if not path or path == current:
                continue
            if entry.get("has_module_skip") or entry.get("collection_risk"):
                continue
            if repo_path and not (repo_path / path).exists():
                continue
            return path
        return ""

    def _deduplicate(
        self,
        results: List[ScenarioValidationResult],
    ) -> List[ScenarioValidationResult]:
        kept: List[ScenarioValidationResult] = []

        for current in sorted(results, key=lambda x: x.score, reverse=True):
            duplicate_of = None

            for prev in kept:
                sim = self._scenario_similarity(
                    current.normalized_scenario or {},
                    prev.normalized_scenario or {},
                )
                if sim >= self.duplicate_threshold:
                    duplicate_of = prev.scenario_id
                    break

            if duplicate_of is not None:
                current.decision = "reject"
                current.reasons.append(f"duplicate of higher-ranked scenario {duplicate_of}")
                current.normalized_scenario = None

            kept.append(current)

        return kept

    def _scenario_similarity(self, a: Dict[str, Any], b: Dict[str, Any]) -> float:
        if not a or not b:
            return 0.0

        a_tokens = self._collect_core_tokens(a)
        b_tokens = self._collect_core_tokens(b)

        if not a_tokens or not b_tokens:
            return 0.0

        inter = len(a_tokens & b_tokens)
        union = len(a_tokens | b_tokens)

        return inter / union if union else 0.0

    def _collect_core_tokens(self, scenario: Dict[str, Any]) -> set[str]:
        tokens: set[str] = set()

        def add_text(text: str) -> None:
            for tok in text.lower().replace("/", " ").replace("_", " ").split():
                tok = tok.strip(" ,.:;()[]{}'\"")
                if len(tok) >= 3:
                    tokens.add(tok)

        target = scenario.get("target_location", {})
        add_text(target.get("source_file", ""))
        add_text(target.get("target_function", ""))

        for x in target.get("related_classes", []):
            add_text(x)

        for item in scenario.get("execution_stimulus", []):
            add_text(str(item))

        add_text(scenario.get("expected_failure", ""))
        add_text(scenario.get("oracle", ""))

        return tokens
