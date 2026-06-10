from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

from src.benchmark.instance_loader import BenchmarkInstance, TDDInstanceLoader
from src.issue_parser.issue_clues import IssueClueExtractor
from src.context_builder.code_context import CodeContextExtractor
from src.scenario.scenario_generator import ScenarioGenerator
from src.scenario.scenario_validator import ScenarioValidator
from src.scenario.scenario_hydrator import hydrate_validation_report
from src.generator.repro_test_generator import ReproductionTestGenerator
from src.executor.alignment_runner import AlignmentRunner
from src.alignment.alignment_scorer import AlignmentScorer
from src.utils.scenario_utils import ensure_primary_scenario, select_primary_scenario

MAX_ALIGNMENT_ITERATIONS = 5


def process_instance(
    instance: BenchmarkInstance,
    output_dir: str,
    model_key: str = "qwen",
) -> Dict[str, Any]:
    """하나의 인스턴스에 대해 전체 파이프라인을 실행하고 결과를 반환한다.

    Returns:
        {"instance_id", "failure_type", "final_score", "iterations", "error"}
    """
    # ── Stage 1: Issue Clue 추출 ──
    clue_extractor = IssueClueExtractor()
    clue = clue_extractor.extract(
        instance_id=instance.instance_id,
        issue_text=instance.problem_statement,
    )

    clue_output_path = f"{output_dir}/clue.json"
    clue_extractor.save(clue, clue_output_path)
    clue_dict = clue.to_dict()

    # ── Stage 2: Code Context 추출 ──
    context_extractor = CodeContextExtractor()
    context = context_extractor.extract(
        instance=instance,
        clue=clue_dict,
    )

    context_output_path = f"{output_dir}/context.json"
    context_extractor.save(context, context_output_path)
    context_dict = context.to_dict()

    # ── Stage 2.5: fault location 추론 (traceback 없는 경우) ──
    scenario_generator = ScenarioGenerator(model_key=model_key)
    if not clue_dict.get("fault_locations"):
        inferred = scenario_generator.infer_fault_locations(clue_dict, context_dict)
        if inferred:
            clue_dict = dict(clue_dict)
            clue_dict["fault_locations"] = inferred
            with open(clue_output_path, "w", encoding="utf-8") as f:
                json.dump(clue_dict, f, ensure_ascii=False, indent=2)
            logger.info("Inferred fault locations (no traceback): %s", inferred)
            # Re-rank context after inferred localization.  The inference is
            # still patch-free and based only on issue text + pre-patch code,
            # but the candidate source/test lists should reflect it before
            # scenario generation.
            context = context_extractor.extract(
                instance=instance,
                clue=clue_dict,
            )
            context_extractor.save(context, context_output_path)
            context_dict = context.to_dict()

    # ── Stage 3: 시나리오 생성 ──
    scenario = scenario_generator.extract(
        instance=instance,
        clue=clue_dict,
        context=context_dict,
    )

    scenario_output_path = f"{output_dir}/scenario.json"
    scenario_generator.save(scenario, scenario_output_path)

    # ── Stage 4: 시나리오 검증 ──
    scenario_validator = ScenarioValidator()
    validation_report = scenario_validator.validate(
        scenarios=[s.to_dict() for s in scenario],
        clue=clue_dict,
        context=context_dict,
    )
    validation_report = hydrate_validation_report(
        validation_report,
        clue_dict,
        repo=instance.repo,
        context=context_dict,
    )

    validation_dict = ensure_primary_scenario(
        validation_report.to_dict(),
        clue=clue_dict,
        context=context_dict,
        reason="post_validation_empty_selection",
    )
    validation_output_path = f"{output_dir}/scenario_validation.json"
    Path(validation_output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(validation_output_path, "w", encoding="utf-8") as f:
        json.dump(validation_dict, f, ensure_ascii=False, indent=2)

    # force-selected 시나리오 경고 로그
    for sel in validation_dict.get("selected_scenarios", []):
        if sel.get("force_selected"):
            print(f"  ⚠ force-selected scenario: {sel.get('scenario_id')} (score={sel.get('score', 0):.2f})")

    _print_summary(instance, clue_output_path, context_output_path,
                   scenario_output_path, scenario, context)

    # ── Stage 5-6: 생성 → alignment 평가 루프 (patch-free, 최대 MAX_ALIGNMENT_ITERATIONS회) ──
    repro_test_generator = ReproductionTestGenerator(model_key=model_key)
    alignment_runner = AlignmentRunner()
    scorer = AlignmentScorer()

    # 시나리오 생성 시점에 clue 필드가 이미 포함되므로 별도 merge 불필요
    current_validation_dict = ensure_primary_scenario(
        validation_dict,
        clue=clue_dict,
        context=context_dict,
        reason="initial_generation_guard",
    )
    alignment_result = None
    iteration = 0
    runtime_error_for_next: str | None = None
    # 전체 iteration에 걸친 토큰 누적
    total_token_usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for iteration in range(1, MAX_ALIGNMENT_ITERATIONS + 1):
        print(f"\n{'='*60}")
        print(f"  Alignment Loop — iteration {iteration}/{MAX_ALIGNMENT_ITERATIONS}")
        print(f"{'='*60}")

        _clear_harness_cache(instance.instance_id, output_dir, alignment=True)

        # ── 테스트 생성 (Algorithm 1, line 1: t=null → NOT_VALID) ──
        try:
            current_validation_dict = ensure_primary_scenario(
                current_validation_dict,
                clue=clue_dict,
                context=context_dict,
                reason=f"iteration_{iteration}_empty_selection_guard",
            )
            generated_test = repro_test_generator.generate(
                instance=instance,
                clue=clue_dict,
                context=context_dict,
                validation_report=current_validation_dict,
                iteration=iteration,
                runtime_error_hint=runtime_error_for_next,
            )
        except Exception as gen_err:
            logger.warning("Test generation failed (t=null): %s", gen_err)
            failure = {
                "iteration": iteration,
                "failure_type": "NOT_VALID",
                "score_breakdown": {
                    "bug_fail_score": 0.0,
                    "issue_alignment_score": 0.0,
                    "coverage_score": 0.0,
                    "failure_type_detail": "GENERATION_FAILED",
                },
                "diagnosis": f"Test generation failed: {gen_err}",
                "feedback": {},
                "refined_scenario": select_primary_scenario(current_validation_dict),
                "should_continue": False,
                "test_results": {},
                "coverage_summary": {},
                "failure_type_detail": "GENERATION_FAILED",
            }
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            with open(f"{output_dir}/alignment_result.json", "w", encoding="utf-8") as f:
                json.dump(failure, f, ensure_ascii=False, indent=2)
            return {
                "instance_id": instance.instance_id,
                "failure_type": "NOT_VALID",
                "failure_type_detail": "GENERATION_FAILED",
                "iterations": iteration,
                "error": f"Test generation failed: {gen_err}",
            }

        generated_test_output_path = f"{output_dir}/generated_test.json"
        repro_test_generator.save(generated_test, generated_test_output_path)

        print(f"  generated test → {generated_test_output_path}")
        print(f"  scenario_id: {generated_test.scenario_id}")
        # 토큰 누적
        for k in total_token_usage:
            total_token_usage[k] += generated_test.token_usage.get(k, 0)

        # ── before-patch-only 실행 (Docker SDK 직접 실행, patch-free) ──
        align_result = alignment_runner.run(
            instance=instance,
            generated_test_json_path=generated_test_output_path,
            run_id=f"align-{instance.instance_id}-it{iteration}-{int(time.time() * 1000)}",
        )

        align_exec_path = f"{output_dir}/alignment_execution.json"
        alignment_runner.save(align_result, align_exec_path)

        print(f"  returncode: {align_result.returncode}")
        print(f"  has_failure: {align_result.has_failure}")
        print(f"  test_results: {align_result.test_results}")

        # ── Docker 빌드 실패 등 복구 불가 에러 시 즉시 중단 ──
        if align_result.error_messages and any(
            "build failed" in m or "build error" in m
            for m in align_result.error_messages
        ):
            print(f"  ✗ 복구 불가 에러, 루프 즉시 중단: "
                  f"{align_result.error_messages}")
            break

        # ── 정합성 평가 (규칙기반, patch-free) ──
        alignment_result = scorer.evaluate(
            execution_result=align_result.to_dict(),
            clue=clue_dict,
            scenario=select_primary_scenario(current_validation_dict),
            generated_test=generated_test.to_dict(),
            iteration=iteration,
            validation_report=current_validation_dict,
            context=context_dict,
        )

        alignment_output_path = f"{output_dir}/alignment_result.json"
        with open(alignment_output_path, "w", encoding="utf-8") as f:
            json.dump(alignment_result.to_dict(), f, ensure_ascii=False, indent=2)

        print(f"  failure_type: {alignment_result.failure_type}")
        breakdown = alignment_result.score_breakdown
        print(f"  bug_fail_score: {breakdown.get('bug_fail_score')}")
        print(f"  coverage_score: {breakdown.get('coverage_score')}")
        print(f"  issue_alignment_score: {breakdown.get('issue_alignment_score')}")
        print(f"  diagnosis: {alignment_result.diagnosis}")

        if not alignment_result.should_continue:
            print(f"\n  ✓ ALIGNED at iteration {iteration}")
            break

        if iteration == MAX_ALIGNMENT_ITERATIONS:
            print(f"\n  ✗ 최대 {MAX_ALIGNMENT_ITERATIONS}회 도달, 루프 종료")
            break

        print(f"  → 시나리오 보강 후 재시도 "
              f"(failure_type={alignment_result.failure_type})")

        # 다음 iteration을 위해 런타임 에러 수집
        runtime_error_for_next = None
        if align_result.error_messages:
            relevant = [
                m for m in align_result.error_messages
                if any(kw in m for kw in [
                    "TypeError", "AttributeError", "NameError",
                    "ImportError", "ModuleNotFoundError", "SyntaxError",
                    "RuntimeError", "missing 1 required", "self",
                    "OperationalError", "IntegrityError",
                    "not collected", "not found",
                ])
            ]
            if relevant:
                runtime_error_for_next = "; ".join(relevant[:2])

        # 보강된 시나리오를 validation_dict에 주입 (시나리오 ID 추적)
        # 시나리오 생성 시점에 clue 필드가 이미 포함되므로 별도 merge 불필요
        current_validation_dict = _inject_refined_scenario(
            current_validation_dict,
            alignment_result.refined_scenario,
            current_scenario_id=generated_test.scenario_id,
        )
        current_validation_dict = ensure_primary_scenario(
            current_validation_dict,
            clue=clue_dict,
            context=context_dict,
            reason=f"iteration_{iteration}_refinement_guard",
        )

    # ── 모델 로직 결과 반환 (alignment 루프까지) ──
    failure_type = alignment_result.failure_type if alignment_result else "ERROR"
    score_breakdown = alignment_result.score_breakdown if alignment_result else {}

    print(f"\n{'='*60}")
    print("  Model Pipeline Result")
    print(f"{'='*60}")
    print(f"  failure_type: {failure_type}")
    print(f"  bug_fail_score: {score_breakdown.get('bug_fail_score')}")
    print(f"  coverage_score: {score_breakdown.get('coverage_score')}")
    print(f"  issue_alignment_score: {score_breakdown.get('issue_alignment_score')}")
    print(f"  iterations: {iteration}")

    error_msg = None
    if alignment_result is None:
        error_msg = "alignment_result is None: loop did not complete"
    elif alignment_result.failure_type == "ERROR":
        error_msg = alignment_result.diagnosis

    return {
        "instance_id": instance.instance_id,
        "failure_type": failure_type,
        "failure_type_detail": getattr(alignment_result, "failure_type_detail", "") if alignment_result else "",
        "bug_fail_score": score_breakdown.get("bug_fail_score"),
        "coverage_score": score_breakdown.get("coverage_score"),
        "issue_alignment_score": score_breakdown.get("issue_alignment_score"),
        "iterations": iteration,
        "error": error_msg,
        "token_usage": total_token_usage,
    }


def main():
    loader = TDDInstanceLoader()
    instance = loader.get_by_index(0)
    output_dir = f"outputs/{instance.instance_id}"
    process_instance(instance, output_dir)


def _inject_refined_scenario(
    validation_dict: dict,
    refined_scenario: dict,
    current_scenario_id: str | None = None,
) -> dict:
    """보강된 시나리오를 validation_report에 주입하여 다음 generate()에서 사용하게 한다.
    
    Args:
        validation_dict: validation_report dict
        refined_scenario: 보강된 시나리오
        current_scenario_id: 현재 사용 중인 시나리오 ID (정확한 교체 대상 찾기 용)
    """
    import copy
    new_dict = copy.deepcopy(validation_dict)
    selected = new_dict.get("selected_scenarios", [])
    
    if selected:
        # current_scenario_id가 지정되면 해당 시나리오를 찾아서 교체
        if current_scenario_id:
            for sel in selected:
                normalized = sel.get("normalized_scenario", {})
                if normalized.get("scenario_id") == current_scenario_id:
                    sel["normalized_scenario"] = refined_scenario
                    return new_dict
        # ID 지정 없거나 찾지 못한 경우 첫 번째(=primary) 시나리오 교체
        selected[0]["normalized_scenario"] = refined_scenario
    else:
        new_dict["selected_scenarios"] = [{
            "scenario_id": refined_scenario.get("scenario_id", "S_REFINED"),
            "score": 0.3,
            "decision": "accept",
            "reasons": ["selected refined scenario because selected_scenarios was empty"],
            "normalized_scenario": refined_scenario,
            "force_selected": True,
            "scenario_repaired": True,
            "scenario_repair_reason": "inject_refined_empty_selection",
        }]
    return new_dict


def _clear_harness_cache(instance_id: str, output_dir: str, alignment: bool = False) -> None:
    """이전 실행의 하네스 캐시를 삭제한다."""
    benchmark_root = Path("benchmark/TDD-Bench-Verified")

    if alignment:
        run_id = f"align-{instance_id}"
    else:
        run_id = f"debug-{instance_id}"

    # 평가 로그 삭제
    eval_log_dir = benchmark_root / "logs" / "run_evaluation" / run_id
    if eval_log_dir.exists():
        shutil.rmtree(eval_log_dir, ignore_errors=True)

    # 리포트 파일 삭제
    for report_file in benchmark_root.glob(f"*{run_id}*.json"):
        report_file.unlink(missing_ok=True)


def _print_summary(instance, clue_path, context_path, scenario_path, scenarios, context):
    """초기 파이프라인 단계 요약 출력."""
    print("instance_id:", instance.instance_id)
    print("repo:", instance.repo)
    print("clue saved to:", clue_path)
    print("context saved to:", context_path)
    print("scenario saved to:", scenario_path)

    print(f"\nnum_scenarios: {len(scenarios)}")
    for s in scenarios:
        print("-", s.scenario_id)

    print("\ncandidate_source_files")
    for x in context.candidate_source_files:
        print("-", x["path"], "| score =", x["score"])

    print("\ncandidate_test_files")
    for x in context.candidate_test_files:
        print("-", x["path"], "| score =", x["score"])

    print("\nproject_test_style")
    print(context.project_test_style)


if __name__ == "__main__":
    main()
