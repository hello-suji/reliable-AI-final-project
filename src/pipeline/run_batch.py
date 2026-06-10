"""전체 인스턴스 배치 실행기 (모델 파이프라인 전용).

alignment 루프까지 실행한다. 최종 harness 평가는 별도로
src.evaluator.final_evaluator / result_collector로 수행한다.

Usage:
    python -m src.pipeline.run_batch                             # 전체 449개 (순차)
    python -m src.pipeline.run_batch --workers 4                 # 4개 병렬
    python -m src.pipeline.run_batch --start 0 --end 10          # 인덱스 0~9
    python -m src.pipeline.run_batch --instance_ids astropy__astropy-12907,django__django-10880
    python -m src.pipeline.run_batch --force                     # 완료된 것도 재실행
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import traceback
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.benchmark.instance_loader import TDDInstanceLoader
from src.evaluator.resolve_policy import resolve_metadata
from src.models.config import load_model_config
from src.pipeline.run_single import MAX_ALIGNMENT_ITERATIONS, process_instance
from src.utils.artifact_hash import sha256_file


KST = timezone(timedelta(hours=9))


def _now_kst_iso() -> str:
    return datetime.now(KST).isoformat()


def _is_completed(output_dir: str) -> bool:
    """이미 alignment_result.json이 존재하면 완료로 판단."""
    return Path(f"{output_dir}/alignment_result.json").exists()


def _preflight_model_endpoint(model_key: str, timeout: int = 5) -> None:
    """로컬 OpenAI-compatible 서버가 살아있는지 배치 시작 전에 확인한다."""
    config = load_model_config(model_key)
    if config.provider != "local" or not config.base_url:
        return

    url = config.base_url.rstrip("/") + "/models"
    request = Request(url, method="GET")
    try:
        with urlopen(request, timeout=timeout):
            return
    except HTTPError as e:
        # /models 미지원이어도 HTTP 응답이 왔으면 서버 자체는 살아있다.
        if e.code < 500:
            return
        raise RuntimeError(
            f"Model endpoint responded with HTTP {e.code}: {url}"
        ) from e
    except URLError as e:
        raise RuntimeError(
            f"Model endpoint is not reachable for '{model_key}': {url}\n"
            f"Start the local OpenAI-compatible server or update configs/models.yaml."
        ) from e


def _load_existing_summary(summary_path: str) -> Dict[str, Any]:
    """기존 batch_summary.json을 로드한다. 없으면 빈 구조 반환."""
    p = Path(summary_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "started_at": _now_kst_iso(),
        "finished_at": None,
        "per_instance": [],
    }


def _save_summary(summary: Dict[str, Any], summary_path: str) -> None:
    """집계 정보를 계산하고 batch_summary.json을 저장.
    메모리의 summary dict도 동일하게 업데이트한다 (호출자가 summary[key]로 읽을 수 있도록).
    """
    results = summary["per_instance"]
    _normalize_summary_entries(results)
    total = len(results)

    counts: Dict[str, int] = {}
    for r in results:
        ft = r.get("failure_type", "UNKNOWN")
        counts[ft] = counts.get(ft, 0) + 1
    aligned = counts.get("ALIGNED", 0)
    resolved = sum(1 for r in results if r.get("resolved"))
    final_eval_count = sum(
        1
        for r in results
        if r.get("final_score") is not None
    )
    skipped = sum(1 for r in results if r.get("skipped", False))

    # ── 배치 평균 통계 계산 ──
    # 토큰 평균 (token_usage가 있는 케이스만)
    token_entries = [r["token_usage"] for r in results if r.get("token_usage") and r["token_usage"].get("total_tokens", 0) > 0]
    avg_prompt_tokens = round(sum(e["prompt_tokens"] for e in token_entries) / len(token_entries)) if token_entries else None
    avg_completion_tokens = round(sum(e["completion_tokens"] for e in token_entries) / len(token_entries)) if token_entries else None
    avg_total_tokens = round(sum(e["total_tokens"] for e in token_entries) / len(token_entries)) if token_entries else None

    # 커버리지 평균 (coverage가 있는 케이스만)
    # patch line coverage: gold patch 변경 소스 라인 중 테스트가 커버한 라인 비율.
    patch_cov_entries = [
        r["patch_line_coverage_percent"]
        for r in results
        if r.get("patch_line_coverage_percent") is not None
    ]
    avg_patch_cov = round(sum(patch_cov_entries) / len(patch_cov_entries), 1) if patch_cov_entries else None
    iteration_entries = [
        r.get("iterations")
        for r in results
        if isinstance(r.get("iterations"), (int, float))
    ]
    avg_iterations = round(sum(iteration_entries) / len(iteration_entries), 2) if iteration_entries else None

    # 메모리 dict 업데이트 (호출자가 summary['total'] 등으로 읽을 수 있게)
    summary["total"] = total
    summary["aligned"] = aligned
    summary["resolved"] = resolved
    summary["final_eval_count"] = final_eval_count
    summary["not_failed"] = counts.get("NOT_FAILED", 0)
    summary.pop("no_fail", None)
    summary["error"] = counts.get("ERROR", 0)
    summary["no_coverage"] = counts.get("NO_COVERAGE", 0)
    summary["weak_alignment"] = counts.get("WEAK_ALIGNMENT", 0)
    summary["skipped"] = skipped
    summary["aligned_rate"] = f"{aligned / total * 100:.1f}%" if total else "0.0%"
    summary["resolve_rate"] = f"{resolved / total * 100:.1f}%" if total else "0.0%"
    summary["failure_type_counts"] = counts
    summary["artifact_extra_instance_count"] = 0
    summary.pop("artifact_extra_instances", None)
    summary.pop("resolved_loss_reason_counts", None)
    summary.pop("partial_flip_after_failed", None)
    summary["avg_prompt_tokens"] = avg_prompt_tokens
    summary["avg_completion_tokens"] = avg_completion_tokens
    summary["avg_total_tokens"] = avg_total_tokens
    summary.pop("max_total_tokens", None)
    summary.pop("p50_total_tokens", None)
    summary.pop("p90_total_tokens", None)
    summary.pop("token_outlier_instances", None)
    summary["avg_patch_line_coverage_percent"] = avg_patch_cov
    summary.pop("avg_file_coverage", None)
    summary["avg_iterations"] = avg_iterations

    # 집계 통계가 맨 위에 오도록 순서를 명시적으로 구성하여 파일 저장
    ordered: Dict[str, Any] = {
        "total": total,
        "aligned": aligned,
        "resolved": resolved,
        "final_eval_count": final_eval_count,
        "not_failed": counts.get("NOT_FAILED", 0),
        "error": counts.get("ERROR", 0),
        "no_coverage": counts.get("NO_COVERAGE", 0),
        "weak_alignment": counts.get("WEAK_ALIGNMENT", 0),
        "skipped": skipped,
        "aligned_rate": f"{aligned / total * 100:.1f}%" if total else "0.0%",
        "resolve_rate": f"{resolved / total * 100:.1f}%" if total else "0.0%",
        "avg_iterations": avg_iterations,
        "avg_prompt_tokens": avg_prompt_tokens,
        "avg_completion_tokens": avg_completion_tokens,
        "avg_total_tokens": avg_total_tokens,
        "avg_patch_line_coverage_percent": avg_patch_cov,
        "failure_type_counts": counts,
        "artifact_extra_instance_count": 0,
        "started_at": summary.get("started_at"),
        "finished_at": summary.get("finished_at"),
        "per_instance": [_compact_summary_entry(r) for r in results],
    }
    # None 값 필드 제거
    ordered = {k: v for k, v in ordered.items() if v is not None}
    summary.pop("artifact_extra_instances", None)
    # per_instance는 None이어도 유지
    ordered["per_instance"] = [_compact_summary_entry(r) for r in results]

    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)


def _normalize_summary_entries(results: List[Dict[str, Any]]) -> None:
    """Keep batch summary rows compact and normalize legacy coverage fields."""
    for entry in results:
        entry["failure_type"] = _normalize_failure_type(entry.get("failure_type"))
        entry.pop("patch_sha256", None)
        entry.pop("strict_resolved", None)
        entry.pop("relaxed_resolved", None)
        entry.pop("relaxed_alignment_detail", None)
        entry.pop("original_failure_type", None)
        if (
            "patch_line_coverage" in entry
            and "patch_line_coverage_percent" not in entry
            and isinstance(entry.get("patch_line_coverage"), (int, float))
        ):
            entry["patch_line_coverage_percent"] = entry.pop("patch_line_coverage")
        entry.pop("patch_line_covered_lines", None)
        entry.pop("patch_line_total_lines", None)


def _compact_summary_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Return the per-instance row shape written to batch_summary.json."""
    compact = dict(entry)
    compact.pop("avg_file_coverage", None)
    compact.pop("patch_line_covered_lines", None)
    compact.pop("patch_line_total_lines", None)
    compact.pop("resolved_reason", None)
    compact.pop("resolved_loss_reason", None)
    compact.pop("diagnostic_flip_to_pass", None)
    compact.pop("has_final_eval", None)
    compact.pop("relaxed_alignment_detail", None)
    return compact


def _normalize_failure_type(value: Any) -> str:
    if value == "NO_FAIL":
        return "NOT_FAILED"
    if value == "NOT_COLLECTED":
        return "NOT_VALID"
    return str(value or "UNKNOWN")


def _compute_patch_line_coverage(patch: str, coverage_data: Dict) -> Dict[str, Any]:
    """gold patch가 변경한 소스 라인 중 테스트가 커버한 비율.

    다른 논문들의 평가 기준에 맞춘 patch-level coverage.
    test 파일은 제외하고 소스 파일 변경분만 계산.
    """
    import re as _re
    if not patch or not coverage_data:
        return {"patch_line_coverage_percent": 0.0}

    current_file: Optional[str] = None
    patch_lines_by_file: Dict[str, List[int]] = {}

    for line in patch.splitlines():
        if line.startswith("diff --git"):
            m = _re.match(r"diff --git a/(\S+)", line)
            if m:
                f = m.group(1)
                if "test" not in f.lower():
                    current_file = f
                    patch_lines_by_file.setdefault(f, [])
                else:
                    current_file = None
        elif line.startswith("@@") and current_file is not None:
            m = _re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
            if m:
                start = int(m.group(1))
                count = int(m.group(2) or 1)
                patch_lines_by_file[current_file].extend(range(start, start + count))

    total = 0
    covered = 0
    for patch_file, line_nums in patch_lines_by_file.items():
        if not line_nums:
            continue
        matched = next(
            (cf for cf in coverage_data
             if cf.endswith(patch_file) or patch_file.endswith(cf)),
            None,
        )
        total += len(line_nums)
        if matched:
            missing = set(coverage_data[matched].get("missing_lines", []))
            covered += sum(1 for ln in line_nums if ln not in missing)

    return {"patch_line_coverage_percent": round(covered / total * 100, 1) if total > 0 else 0.0}


def _attach_coverage(result: Dict[str, Any], output_dir: str) -> None:
    """실행 완료 후 alignment_execution.json에서 커버리지 정보를 result에 추가한다."""
    iid = result["instance_id"]
    exec_path = Path(f"{output_dir}/alignment_execution.json")
    if not exec_path.exists():
        return
    try:
        ae = json.load(exec_path.open(encoding="utf-8"))
        coverage_data = ae.get("coverage_data", {})
        if not coverage_data:
            return

        # Keep file coverage available for the batch-level average only.
        file_covers = [v.get("cover", 0.0) for v in coverage_data.values() if isinstance(v, dict)]
        result["avg_file_coverage"] = round(sum(file_covers) / len(file_covers), 1) if file_covers else 0.0

        # Patch line coverage is an analysis metric only after ALIGNED.
        # Never feed gold patch information into generation or alignment.
        if result.get("failure_type") != "ALIGNED":
            return

        # Patch line coverage: gold patch 변경 라인 중 커버된 비율
        try:
            from src.benchmark.instance_loader import TDDInstanceLoader
            loader = TDDInstanceLoader()
            all_ids = loader.list_instance_ids()
            if iid in all_ids:
                inst = loader.get_by_index(all_ids.index(iid))
                result.update(_compute_patch_line_coverage(inst.patch, coverage_data))
        except Exception:
            pass
    except Exception:
        pass


def _current_patch_sha(output_dir: str) -> str | None:
    patch_path = Path(output_dir) / "generated_test.patch"
    if not patch_path.exists():
        return None
    try:
        return sha256_file(patch_path)
    except Exception:
        return None


def _is_final_eval_fresh(output_dir: str, final_eval: Dict[str, Any]) -> bool:
    current = _current_patch_sha(output_dir)
    recorded = final_eval.get("patch_sha256")
    infra_failed = (
        final_eval.get("harness_returncode") not in (None, 0)
        and not final_eval.get("before_patch")
        and not final_eval.get("after_patch")
    )
    return bool(current and recorded and current == recorded and not infra_failed)


def _apply_final_eval(entry: Dict[str, Any], final_eval: Dict[str, Any]) -> bool:
    """final_evaluation.json 내용을 batch summary entry에 반영한다."""
    final_score = final_eval.get("final_score", 0.0)
    entry["final_score"] = final_score
    metadata = resolve_metadata(final_eval)
    is_resolved = bool(metadata.get("resolved"))
    entry.update(metadata)
    entry.pop("strict_resolved", None)
    entry.pop("relaxed_resolved", None)
    if final_eval.get("error"):
        entry["final_eval_error"] = final_eval.get("error")
    return is_resolved


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _token_usage_from_artifacts(output_dir: Path, previous: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if previous.get("token_usage"):
        return previous.get("token_usage")
    generated = _read_json(output_dir / "generated_test.json") or {}
    token_usage = generated.get("token_usage")
    return token_usage if isinstance(token_usage, dict) else None


def _entry_from_artifacts(
    iid: str,
    output_dir: str | Path,
    previous: Optional[Dict[str, Any]] = None,
    skipped: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    """Rebuild one summary row from per-instance artifacts.

    batch_summary.json is derived state. This keeps resume/skip/final-eval paths
    from trusting stale rows after alignment_result.json or generated_test.patch
    changed.
    """
    previous = previous or {}
    out = Path(output_dir)
    alignment = _read_json(out / "alignment_result.json")
    if not alignment:
        return None

    failure_type = _normalize_failure_type(alignment.get("failure_type", "UNKNOWN"))
    entry: Dict[str, Any] = {
        "instance_id": iid,
        "failure_type": failure_type,
        "failure_type_detail": (
            alignment.get("failure_type_detail")
            or (alignment.get("score_breakdown") or {}).get("failure_type_detail", "")
        ),
        "iterations": alignment.get("iterations", alignment.get("iteration", 0)),
        "error": None,
        "elapsed_sec": previous.get("elapsed_sec", 0),
        "skipped": previous.get("skipped", False) if skipped is None else skipped,
        "strict_gate_pass": failure_type == "ALIGNED",
    }
    breakdown = alignment.get("score_breakdown") or {}
    for key in ("bug_fail_score", "coverage_score", "issue_alignment_score"):
        if key in breakdown:
            entry[key] = breakdown[key]

    if failure_type in {"ERROR", "NOT_VALID"}:
        entry["error"] = alignment.get("diagnosis")

    token_usage = _token_usage_from_artifacts(out, previous)
    if token_usage:
        entry["token_usage"] = token_usage

    _attach_coverage(entry, str(out))

    if failure_type != "ALIGNED":
        return entry

    final_eval = _read_json(out / "final_evaluation.json")
    if final_eval and _is_final_eval_fresh(str(out), final_eval):
        _apply_final_eval(entry, final_eval)
    elif final_eval:
        entry["resolved"] = False
    else:
        entry["resolved"] = False

    return entry


def _upsert_summary_entry(summary: Dict[str, Any], entry: Dict[str, Any]) -> None:
    for i, existing in enumerate(summary["per_instance"]):
        if existing.get("instance_id") == entry.get("instance_id"):
            summary["per_instance"][i] = entry
            return
    summary["per_instance"].append(entry)


def _sync_summary_from_artifacts(
    summary: Dict[str, Any],
    model_output_root: str | Path,
    instance_ids: Optional[List[str]] = None,
) -> None:
    """Refresh summary rows from alignment/final-eval artifacts.

    The summary file is never allowed to be the source of truth for status.
    Existing rows keep their order and timing fields, but status/final-eval
    fields are rebuilt from current artifacts and patch hashes.
    """
    root = Path(model_output_root)
    previous_by_id = {
        r.get("instance_id"): r
        for r in summary.get("per_instance", [])
        if r.get("instance_id")
    }
    ordered_ids: List[str] = []
    seen = set()

    def add_id(iid: str) -> None:
        if iid and iid not in seen:
            seen.add(iid)
            ordered_ids.append(iid)

    requested = list(instance_ids or [])
    requested_set = set(requested)
    if requested:
        for iid in requested:
            add_id(iid)
    else:
        for row in summary.get("per_instance", []):
            add_id(row.get("instance_id", ""))

    if root.exists():
        for d in sorted(root.iterdir()):
            if d.is_dir() and (d / "alignment_result.json").exists():
                if not requested_set:
                    add_id(d.name)
    summary.pop("artifact_extra_instances", None)
    summary["artifact_extra_instance_count"] = 0

    refreshed: List[Dict[str, Any]] = []
    for iid in ordered_ids:
        previous = previous_by_id.get(iid, {})
        entry = _entry_from_artifacts(iid, root / iid, previous=previous)
        refreshed.append(entry if entry is not None else previous)
    summary["per_instance"] = [r for r in refreshed if r]


def _run_one(
    iid: str,
    output_dir: str,
    model_key: str,
    loader: TDDInstanceLoader,
) -> Dict[str, Any]:
    """단일 인스턴스를 실행하고 result dict를 반환한다."""
    t0 = time.time()
    try:
        instance = loader.get_instance(iid)
        result = process_instance(instance, output_dir, model_key=model_key)
        result["elapsed_sec"] = round(time.time() - t0, 1)
        result["skipped"] = False
    except Exception as e:
        tb = traceback.format_exc()
        print(f"\n  ✗ ERROR [{iid}]: {e}\n{tb}")
        result = {
            "instance_id": iid,
            "failure_type": "ERROR",
            "iterations": 0,
            "elapsed_sec": round(time.time() - t0, 1),
            "skipped": False,
            "error": str(e),
        }
    # 커버리지 정보 추가
    _attach_coverage(result, output_dir)
    return result


def run_batch(
    instance_ids: List[str],
    force: bool = False,
    model_key: str = "qwen",
    output_root: str = "outputs",
    workers: int = 1,
    smallbatch: bool = False,
    batch_run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """인스턴스 목록을 실행한다. workers > 1 이면 ThreadPoolExecutor로 병렬 실행."""
    if smallbatch:
        batch_run_id = batch_run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        model_output_root = f"{output_root}/smallbatch/{batch_run_id}"
    else:
        model_output_root = f"{output_root}/{model_key}"
    summary_path = f"{model_output_root}/batch_summary.json"
    loader = TDDInstanceLoader()
    summary = _load_existing_summary(summary_path)
    _sync_summary_from_artifacts(summary, model_output_root, instance_ids)

    # 배치를 다시 실행할 때마다 실행 시작 시각을 새로 기록
    summary["started_at"] = _now_kst_iso()
    summary["finished_at"] = None

    summary_lock = threading.Lock()
    _save_summary(summary, summary_path)

    total = len(instance_ids)

    # ── skip 처리 & 실행 대상 분리 ──
    to_run: List[str] = []
    for idx, iid in enumerate(instance_ids, 1):
        output_dir = f"{model_output_root}/{iid}"
        if not force and _is_completed(output_dir):
            print(f"[{idx}/{total}] {iid} — SKIP")
            previous = next(
                (r for r in summary["per_instance"] if r.get("instance_id") == iid),
                {},
            )
            entry = _entry_from_artifacts(iid, output_dir, previous=previous, skipped=True)
            if entry is None:
                entry = {
                    "instance_id": iid,
                    "failure_type": "UNKNOWN",
                    "iterations": 0,
                    "elapsed_sec": 0,
                    "skipped": True,
                    "error": None,
                }
            with summary_lock:
                _upsert_summary_entry(summary, entry)
                _save_summary(summary, summary_path)
        else:
            to_run.append(iid)

    run_total = len(to_run)
    if run_total > 0:
        _preflight_model_endpoint(model_key)

    def _handle_result(result: Dict[str, Any], idx: int) -> None:
        iid = result["instance_id"]
        with summary_lock:
            replaced = False
            for i, r in enumerate(summary["per_instance"]):
                if r["instance_id"] == iid:
                    summary["per_instance"][i] = result
                    replaced = True
                    break
            if not replaced:
                summary["per_instance"].append(result)
            _save_summary(summary, summary_path)
        ft = result.get("failure_type", "?")
        secs = result.get("elapsed_sec", 0)
        print(f"  [{idx}/{run_total}] {iid} → {ft} ({secs}s)")

    # 1차 실행: alignment까지
    if workers <= 1:
        # 순차 실행
        for idx, iid in enumerate(to_run, 1):
            output_dir = f"{model_output_root}/{iid}"
            print(f"\n{'#'*60}\n  [{idx}/{run_total}] {iid}\n{'#'*60}")
            result = _run_one(iid, output_dir, model_key, loader)
            _handle_result(result, idx)
    else:
        # 병렬 실행
        print(f"\n병렬 실행: {run_total}개 인스턴스, workers={workers}")
        counter = {"n": 0}
        counter_lock = threading.Lock()

        def _task(iid: str) -> Dict[str, Any]:
            output_dir = f"{model_output_root}/{iid}"
            return _run_one(iid, output_dir, model_key, loader)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_task, iid): iid for iid in to_run}
            for future in as_completed(futures):
                with counter_lock:
                    counter["n"] += 1
                    idx = counter["n"]
                result = future.result()
                _handle_result(result, idx)

    # 2차 실행: ALIGNED만 final evaluation
    _sync_summary_from_artifacts(summary, model_output_root, instance_ids)
    _save_summary(summary, summary_path)
    from src.evaluator.final_evaluator import FinalEvaluator
    final_evaluator = FinalEvaluator()

    aligned_entries = [e for e in summary["per_instance"] if e.get("failure_type") == "ALIGNED"]
    aligned_total = len(aligned_entries)
    resolved = sum(1 for e in summary["per_instance"] if e.get("resolved"))
    needs_final_eval = []
    for entry in aligned_entries:
        output_dir = f"{model_output_root}/{entry['instance_id']}"
        existing = _read_json(Path(output_dir) / "final_evaluation.json")
        if not existing or not _is_final_eval_fresh(output_dir, existing):
            needs_final_eval.append(entry["instance_id"])
    if needs_final_eval:
        final_evaluator.ensure_environment()

    print(f"\n{'='*60}")
    print(f"  Final Evaluation — {aligned_total}건 ALIGNED 평가 시작")
    print(f"{'='*60}")

    eval_count = 0
    for i, entry in enumerate(aligned_entries, 1):
        iid = entry["instance_id"]
        output_dir = f"{model_output_root}/{iid}"
        final_eval_path = Path(f"{output_dir}/final_evaluation.json")

        existing_final_eval = None
        if final_eval_path.exists():
            try:
                with open(final_eval_path, "r", encoding="utf-8") as f:
                    existing_final_eval = json.load(f)
            except Exception as e:
                entry["final_eval_error"] = f"failed to load final_evaluation.json: {e}"

        if existing_final_eval and _is_final_eval_fresh(output_dir, existing_final_eval):
            _apply_final_eval(entry, existing_final_eval)
            status = "RESOLVED" if entry.get("resolved") else "not_resolved"
            print(f"  [{i}/{aligned_total}] {iid} — LOAD ({status})")
            eval_count += 1
            resolved = sum(1 for e in summary["per_instance"] if e.get("resolved"))
            summary["resolved"] = resolved
            summary["final_eval_count"] = eval_count
            summary["resolve_rate"] = f"{resolved / summary['total'] * 100:.1f}%" if summary.get("total") else "0.0%"
            _save_summary(summary, summary_path)
            continue
        if existing_final_eval:
            entry["resolved"] = False
            entry.pop("final_score", None)
            print(f"  [{i}/{aligned_total}] {iid} — STALE final eval, rerun")

        t0 = time.time()
        print(f"  [{i}/{aligned_total}] {iid} ...", end=" ", flush=True)
        try:
            final_eval = final_evaluator.evaluate(iid, output_dir, force=True)
            elapsed = round(time.time() - t0, 1)
            is_resolved = _apply_final_eval(entry, final_eval)
            final_score = entry.get("final_score", 0.0)
            label = "RESOLVED" if is_resolved else "not_resolved"
            print(f"{label}  (score={final_score:.2f}, {elapsed}s)")
        except Exception as e:
            elapsed = round(time.time() - t0, 1)
            entry["final_score"] = 0.0
            entry["final_eval_error"] = str(e)
            entry["resolved"] = False
            print(f"ERROR: {e}  ({elapsed}s)")

        eval_count += 1
        # 매 인스턴스마다 즉시 저장
        resolved = sum(1 for e in summary["per_instance"] if e.get("resolved"))
        summary["resolved"] = resolved
        summary["final_eval_count"] = eval_count
        summary["resolve_rate"] = f"{resolved / summary['total'] * 100:.1f}%" if summary.get("total") else "0.0%"
        _save_summary(summary, summary_path)

    # 최종 정리 및 저장
    summary["resolved"] = resolved
    summary["final_eval_count"] = eval_count
    summary["resolve_rate"] = f"{resolved / summary['total'] * 100:.1f}%" if summary.get("total") else "0.0%"
    summary["finished_at"] = _now_kst_iso()
    _save_summary(summary, summary_path)

    # ── 최종 통계 출력 ──
    print(f"\n{'='*60}")
    print("  Batch Complete")
    print(f"{'='*60}")
    print(f"  total:                  {summary['total']}")
    print(f"  aligned:                {summary['aligned']}  ({summary['aligned_rate']})")
    print(f"  resolved:               {summary['resolved']} ({summary['resolve_rate']})")
    print(f"  not_failed:             {summary['not_failed']}")
    print(f"  error:                  {summary['error']}")
    print(f"  no_coverage:            {summary['no_coverage']}")
    print(f"  weak_alignment:         {summary['weak_alignment']}")
    print(f"  skipped:                {summary['skipped']}")

    pc = summary.get('avg_patch_line_coverage_percent')
    pt = summary.get('avg_prompt_tokens')
    ct = summary.get('avg_completion_tokens')
    tt = summary.get('avg_total_tokens')
    n_tok = sum(1 for r in summary["per_instance"] if r.get("token_usage", {}).get("total_tokens", 0) > 0)
    n_patch_cov = sum(
        1 for r in summary["per_instance"]
        if r.get("patch_line_coverage_percent") is not None
    )
    print(f"  avg_patch_line_coverage:{f'{pc}%' if pc is not None else 'N/A'}  (n={n_patch_cov})")
    print(f"  avg_prompt_tokens:      {pt if pt is not None else 'N/A'}  (n={n_tok})")
    print(f"  avg_completion_tokens:  {ct if ct is not None else 'N/A'}")
    print(f"  avg_total_tokens:       {tt if tt is not None else 'N/A'}")
    print(f"  summary → {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="전체 인스턴스 배치 실행기")
    parser.add_argument("--start", type=int, default=0, help="시작 인덱스 (default: 0)")
    parser.add_argument("--end", type=int, default=None, help="끝 인덱스, exclusive (default: 전체)")
    parser.add_argument(
        "--instance_ids",
        type=str,
        default=None,
        help="실행할 인스턴스 ID (comma-separated)",
    )
    parser.add_argument("--force", action="store_true", help="완료된 인스턴스도 재실행")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen",
        help="사용할 모델 키 (configs/models.yaml 기준, default: qwen)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="병렬 실행 worker 수 (default: 1, 순차)",
    )
    parser.add_argument(
        "--smallbatch",
        action="store_true",
        help="outputs/smallbatch/<실행시간>/<instance_id> 아래에 별도 저장",
    )
    parser.add_argument(
        "--standard-output",
        action="store_true",
        help="부분 실행도 기존 outputs/<model>/<instance_id> 경로에 저장",
    )
    args = parser.parse_args()

    loader = TDDInstanceLoader()
    all_ids = loader.list_instance_ids()

    if args.instance_ids:
        ids = [x.strip() for x in args.instance_ids.split(",") if x.strip()]
    else:
        ids = all_ids[args.start : args.end]

    explicit_subset = bool(args.instance_ids) or args.start != 0 or args.end is not None
    use_smallbatch = args.smallbatch or (explicit_subset and not args.standard_output)
    batch_run_id = datetime.now().strftime("%Y%m%d_%H%M%S") if use_smallbatch else None

    print(f"실행 대상: {len(ids)}개 인스턴스 (model={args.model})")
    if use_smallbatch:
        print(f"smallbatch output → outputs/smallbatch/{batch_run_id}")
    if len(ids) <= 10:
        for iid in ids:
            print(f"  - {iid}")

    run_batch(
        ids,
        force=args.force,
        model_key=args.model,
        workers=args.workers,
        smallbatch=use_smallbatch,
        batch_run_id=batch_run_id,
    )


if __name__ == "__main__":
    main()
