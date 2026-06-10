"""
결과 수립기.

개별 인스턴스의 alignment_result + final_evaluation 결과를 수집하여
배치 단위 통계와 최종 리포트를 생성한다.

Usage:
    # 단일 인스턴스 결과 수집
    collector = ResultCollector()
    collector.collect("outputs/astropy__astropy-12907")

    # 전체 배치 결과 집계
    collector = ResultCollector("outputs")
    report = collector.aggregate()
    collector.save_report(report, "outputs/final_report.json")

    # CLI
    python -m src.evaluator.result_collector                        # outputs 전체
    python -m src.evaluator.result_collector --output_root outputs  # 명시적 경로
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.evaluator.resolve_policy import resolve_metadata
from src.utils.artifact_hash import sha256_file


class ResultCollector:
    """배치 결과를 수집하고 통계를 산출한다."""

    def __init__(self, output_root: str = "outputs") -> None:
        self.output_root = Path(output_root)

    def collect(self, instance_dir: str) -> Optional[Dict[str, Any]]:
        """단일 인스턴스 디렉토리에서 결과를 수집한다.

        Returns:
            {
                "instance_id": str,
                "failure_type": str,
                "iterations": int,
                "final_score": float,     # final_evaluation.json이 fresh일 때만 기록
                "before_patch": dict,
                "after_patch": dict,
            }
            또는 결과가 없으면 None
        """
        d = Path(instance_dir)
        instance_id = d.name

        # alignment 결과
        alignment_path = d / "alignment_result.json"
        if not alignment_path.exists():
            return None

        with open(alignment_path, "r", encoding="utf-8") as f:
            alignment = json.load(f)

        entry: Dict[str, Any] = {
            "instance_id": instance_id,
            "failure_type": self._normalize_failure_type(alignment.get("failure_type", "UNKNOWN")),
            "iterations": alignment.get("iteration", 0),
            "score_breakdown": alignment.get("score_breakdown", {}),
            "before_patch": {},
            "after_patch": {},
        }

        # final evaluation 결과 (분리된 평가기가 생성한 파일)
        final_eval_path = d / "final_evaluation.json"
        if final_eval_path.exists():
            with open(final_eval_path, "r", encoding="utf-8") as f:
                final_eval = json.load(f)
            if not self._is_final_eval_fresh(d, final_eval):
                if entry["failure_type"] == "ALIGNED":
                    entry["resolved"] = False
                return entry
            entry["final_score"] = final_eval.get("final_score", 0.0)
            entry["before_patch"] = final_eval.get("before_patch", {})
            entry["after_patch"] = final_eval.get("after_patch", {})
            metadata = resolve_metadata({**final_eval, **entry})
            entry.update(metadata)
            entry.pop("strict_resolved", None)
            entry.pop("relaxed_resolved", None)

        return entry

    @staticmethod
    def _normalize_failure_type(value: Any) -> str:
        if value == "NO_FAIL":
            return "NOT_FAILED"
        if value == "NOT_COLLECTED":
            return "NOT_VALID"
        return str(value or "UNKNOWN")

    @staticmethod
    def _is_final_eval_fresh(instance_dir: Path, final_eval: Dict[str, Any]) -> bool:
        patch_path = instance_dir / "generated_test.patch"
        if not patch_path.exists():
            return False
        try:
            current = sha256_file(patch_path)
        except Exception:
            return False
        infra_failed = (
            final_eval.get("harness_returncode") not in (None, 0)
            and not final_eval.get("before_patch")
            and not final_eval.get("after_patch")
        )
        return bool(final_eval.get("patch_sha256") == current and not infra_failed)

    @classmethod
    def _is_resolved(cls, entry: Dict[str, Any]) -> bool:
        return bool(resolve_metadata(entry).get("resolved"))

    def aggregate(self, instance_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """전체 또는 지정된 인스턴스의 결과를 집계한다.

        Returns:
            {
                "total": int,
                "aligned": int,
                "resolved": int,
                "failure_type_counts": {...},
                "aligned_rate": str,
                "resolve_rate": str,
                "avg_iterations": float,
                "per_instance": [...]
            }
        """
        results: List[Dict[str, Any]] = []

        if instance_ids is not None:
            dirs = [self.output_root / iid for iid in instance_ids]
        else:
            dirs = sorted(
                d for d in self.output_root.iterdir()
                if d.is_dir() and (d / "alignment_result.json").exists()
            )

        for d in dirs:
            entry = self.collect(str(d))
            if entry:
                results.append(entry)

        total = len(results)

        # failure type 집계
        ft_counts: Dict[str, int] = {}
        for r in results:
            ft = r["failure_type"]
            ft_counts[ft] = ft_counts.get(ft, 0) + 1

        aligned = ft_counts.get("ALIGNED", 0)
        resolved = sum(1 for r in results if r.get("resolved") or self._is_resolved(r))
        final_eval_count = sum(1 for r in results if r.get("final_score") is not None)

        total_iterations = sum(r["iterations"] for r in results)
        report = {
            "total": total,
            "aligned": aligned,
            "resolved": resolved,
            "final_eval_count": final_eval_count,
            "failure_type_counts": ft_counts,
            "aligned_rate": f"{aligned / total * 100:.1f}%" if total else "0.0%",
            "resolve_rate": f"{resolved / total * 100:.1f}%" if total else "0.0%",
            "avg_iterations": round(total_iterations / total, 2) if total else 0.0,
            "per_instance": results,
        }

        return report

    @staticmethod
    def save_report(report: Dict[str, Any], output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    @staticmethod
    def print_summary(report: Dict[str, Any]) -> None:
        print(f"\n{'='*60}")
        print("  Final Report")
        print(f"{'='*60}")
        print(f"  total:              {report['total']}")
        print(f"  aligned:            {report['aligned']}")
        print(f"  resolved:           {report['resolved']}")
        print(f"  final_eval_count:   {report['final_eval_count']}")
        print(f"  aligned_rate:       {report['aligned_rate']}")
        print(f"  resolve_rate:       {report['resolve_rate']}")
        print(f"  avg_iterations:     {report['avg_iterations']}")
        print()
        ft = report["failure_type_counts"]
        for k, v in sorted(ft.items()):
            print(f"  {k:20s} {v}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="결과 수립기")
    parser.add_argument(
        "--output_root", type=str, default="outputs",
        help="인스턴스 결과 디렉토리 루트 (default: outputs)",
    )
    parser.add_argument(
        "--report_path", type=str, default="outputs/final_report.json",
        help="리포트 저장 경로 (default: outputs/final_report.json)",
    )
    parser.add_argument(
        "--instance_ids", type=str, default=None,
        help="대상 인스턴스 ID (comma-separated, default: 전체)",
    )
    args = parser.parse_args()

    ids = None
    if args.instance_ids:
        ids = [x.strip() for x in args.instance_ids.split(",") if x.strip()]

    collector = ResultCollector(args.output_root)
    report = collector.aggregate(instance_ids=ids)
    collector.save_report(report, args.report_path)
    collector.print_summary(report)
    print(f"\n  report → {args.report_path}")


if __name__ == "__main__":
    main()
