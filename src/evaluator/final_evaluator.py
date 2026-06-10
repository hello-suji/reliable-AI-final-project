"""
최종 재현 테스트 평가기.

alignment 루프가 끝난 후 생성된 최종 테스트를
TDD-Bench harness (full 3-stage: INITIAL→BEFORE-PATCH→AFTER-PATCH)로 실행하여
final_score를 산출한다.

Usage:
    from src.evaluator.final_evaluator import FinalEvaluator

    evaluator = FinalEvaluator()
    result = evaluator.evaluate("astropy__astropy-12907", "outputs/astropy__astropy-12907")
    print(result)  # {"instance_id": ..., "final_score": 1.0, ...}
"""
from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from src.executor.test_runner import ReproductionTestRunner
from src.evaluator.resolve_policy import resolve_metadata
from src.utils.artifact_hash import sha256_file


class FinalEvaluator:
    """검증·개선이 끝난 재현 테스트를 full harness로 평가한다."""

    def __init__(
        self,
        benchmark_root: str = "benchmark/TDD-Bench-Verified",
        max_workers: int = 1,
    ) -> None:
        self.benchmark_root = Path(benchmark_root)
        self.runner = ReproductionTestRunner(
            benchmark_root=benchmark_root,
            max_workers=max_workers,
        )

    def evaluate(
        self,
        instance_id: str,
        output_dir: str,
        force: bool = False,
    ) -> Dict[str, Any]:
        """하나의 인스턴스에 대해 full harness 평가를 실행한다.

        Args:
            instance_id: 벤치마크 인스턴스 ID
            output_dir: generated_test.json 이 있는 디렉토리
            force: True이면 기존 캐시 삭제 후 재실행

        Returns:
            {
                "instance_id": str,
                "final_score": float,
                "before_patch": dict,   # test_name → PASSED/FAILED/ERROR
                "after_patch": dict,    # test_name → PASSED/FAILED/ERROR
                "harness_returncode": int,
                "error": Optional[str],
            }
        """
        generated_test_path = f"{output_dir}/generated_test.json"
        patch_path = Path(generated_test_path).with_suffix(".patch")
        if not Path(generated_test_path).exists():
            return {
                "instance_id": instance_id,
                "final_score": 0.0,
                "before_patch": {},
                "after_patch": {},
                "harness_returncode": -1,
                "error": f"generated_test.json 없음: {generated_test_path}",
                "patch_sha256": None,
            }
        patch_sha256 = sha256_file(patch_path) if patch_path.exists() else None

        run_id = f"debug-{instance_id}"

        if force:
            self._clear_cache(instance_id, run_id)

        execution_result = self.runner.run(
            instance_id=instance_id,
            generated_test_json_path=generated_test_path,
            run_id=run_id,
        )

        # 실행 결과 저장
        execution_output_path = f"{output_dir}/execution_result.json"
        self.runner.save(execution_result, execution_output_path)

        stdout = execution_result.harness_stdout

        # final_score 파싱
        final_score = self._parse_final_score(stdout)

        # before/after patch 결과 파싱
        before_patch = self._parse_stage_results(stdout, "Before Patch")
        after_patch = self._parse_stage_results(stdout, "After Patch")
        error = None
        if execution_result.harness_returncode != 0 and not (before_patch or after_patch):
            stderr = (execution_result.harness_stderr or "").strip()
            error = stderr[-4000:] if stderr else "final evaluation harness failed before stage results"

        result = {
            "instance_id": instance_id,
            "final_score": final_score,
            "before_patch": before_patch,
            "after_patch": after_patch,
            "harness_returncode": execution_result.harness_returncode,
            "error": error,
            "patch_sha256": patch_sha256,
        }
        result.update(self._resolve_metadata(result))

        # 평가 결과 저장
        eval_result_path = f"{output_dir}/final_evaluation.json"
        Path(eval_result_path).parent.mkdir(parents=True, exist_ok=True)
        with open(eval_result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return result

    @staticmethod
    def ensure_environment() -> None:
        """Fail before mutating final_evaluation artifacts if Docker is unavailable."""
        try:
            proc = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=20,
            )
        except FileNotFoundError as e:
            raise RuntimeError("Docker executable not found; final evaluation cannot run") from e
        except Exception as e:
            raise RuntimeError(f"Docker preflight failed: {e}") from e
        if proc.returncode != 0:
            msg = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"Docker is not available for final evaluation: {msg}")

    @staticmethod
    def _resolve_metadata(result: Dict[str, Any]) -> Dict[str, Any]:
        return resolve_metadata(result)

    def _clear_cache(self, instance_id: str, run_id: str) -> None:
        import subprocess
        # Docker 컨테이너 이름 충돌 방지: 기존 debug 컨테이너 제거
        container_name = f"sweb.eval.{instance_id}.{run_id}"
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
        )
        eval_log_dir = self.benchmark_root / "logs" / "run_evaluation" / run_id
        if eval_log_dir.exists():
            shutil.rmtree(eval_log_dir, ignore_errors=True)
        for report_file in self.benchmark_root.glob(f"*{run_id}*.json"):
            report_file.unlink(missing_ok=True)

    @staticmethod
    def _parse_final_score(stdout: str) -> float:
        """harness stdout에서 Final Report의 final_score를 파싱."""
        report_match = re.search(
            r"-+Final Report-+\n(\{.*?\})\n-{10,}",
            stdout, re.DOTALL,
        )
        if report_match:
            try:
                outer = ast.literal_eval(report_match.group(1).strip())
                inner = next(iter(outer.values())) if outer else {}
                return float(inner.get("final_score", 0.0))
            except (ValueError, SyntaxError, StopIteration):
                pass
        return 0.0

    @staticmethod
    def _parse_stage_results(stdout: str, stage_name: str) -> Dict[str, str]:
        """stdout에서 특정 stage (Before Patch / After Patch) 결과를 파싱."""
        pattern = rf"-+{re.escape(stage_name)}-+\s*\n(.+?)\n-+"
        match = re.search(pattern, stdout, re.DOTALL)
        if not match:
            return {}
        text = match.group(1).strip()
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            results: Dict[str, str] = {}
            for m in re.finditer(r"'([^']+)':\s*'(PASSED|FAILED|ERROR|SKIP)'", text):
                results[m.group(1)] = m.group(2)
            return results
