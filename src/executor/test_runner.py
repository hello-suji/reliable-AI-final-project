from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.artifact_hash import sha256_text


@dataclass
class HarnessExecutionResult:
    instance_id: str
    benchmark_root: str
    predictions_path: str
    run_id: str
    harness_command: List[str]
    harness_returncode: int
    harness_stdout: str
    harness_stderr: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class ReproductionTestRunner:
    def __init__(
        self,
        benchmark_root: str = "benchmark/TDD-Bench-Verified",
        max_workers: int = 1,
    ) -> None:
        self.benchmark_root = Path(benchmark_root).resolve()
        self.max_workers = max_workers

    def run(
        self,
        instance_id: str,
        generated_test_json_path: str,
        run_id: Optional[str] = None,
    ) -> HarnessExecutionResult:
        generated_path = Path(generated_test_json_path).resolve()
        if not generated_path.exists():
            raise FileNotFoundError(f"generated_test.json 파일이 없습니다: {generated_path}")

        patch_path = generated_path.with_suffix(".patch").resolve()
        if not patch_path.exists():
            raise FileNotFoundError(f"generated_test.patch 파일이 없습니다: {patch_path}")

        if not self.benchmark_root.exists():
            raise FileNotFoundError(f"TDD-Bench-Verified 경로가 없습니다: {self.benchmark_root}")

        patch_text = patch_path.read_text(encoding="utf-8")
        patch_sha256 = sha256_text(patch_text)

        predictions_path = generated_path.with_name("predictions.json")
        predictions = [
            {
                "instance_id": instance_id,
                "model_patch": patch_text,
                "patch_sha256": patch_sha256,
            }
        ]
        with open(predictions_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)

        run_id = run_id or f"debug-{instance_id}"

        command = [
            "python",
            "-m",
            "tddbench.harness.run_evaluation",
            "--dataset_name",
            "TDD_Bench.json",
            "--predictions_path",
            str(predictions_path),
            "--max_workers",
            str(self.max_workers),
            "--instance_ids",
            instance_id,
            "--run_id",
            run_id,
        ]

        try:
            result = subprocess.run(
                command,
                cwd=str(self.benchmark_root),
                capture_output=True,
                text=True,
                timeout=1800,
            )
        except subprocess.TimeoutExpired:
            return HarnessExecutionResult(
                instance_id=instance_id,
                benchmark_root=str(self.benchmark_root),
                predictions_path=str(predictions_path),
                run_id=run_id,
                harness_command=command,
                harness_returncode=-1,
                harness_stdout="",
                harness_stderr="Harness timed out after 1800 seconds",
            )
        except Exception as e:
            return HarnessExecutionResult(
                instance_id=instance_id,
                benchmark_root=str(self.benchmark_root),
                predictions_path=str(predictions_path),
                run_id=run_id,
                harness_command=command,
                harness_returncode=-1,
                harness_stdout="",
                harness_stderr=f"subprocess error: {e}",
            )

        return HarnessExecutionResult(
            instance_id=instance_id,
            benchmark_root=str(self.benchmark_root),
            predictions_path=str(predictions_path),
            run_id=run_id,
            harness_command=command,
            harness_returncode=result.returncode,
            harness_stdout=result.stdout,
            harness_stderr=result.stderr,
        )

    def save(self, result: HarnessExecutionResult, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
