"""Patch-free alignment runner — Docker SDK 직접 실행.

harness를 호출하지 않고 Docker SDK(docker-py)를 통해 직접 컨테이너에서
before-patch 테스트를 실행한다.  이미 빌드된 instance image를 재사용한다.

흐름:
  1) instance image에서 컨테이너 생성·시작
  2) eval.sh 생성(test_patch 적용 + pytest + coverage)
  3) /bin/bash /eval.sh 실행
  4) stdout 파싱 → test_results, coverage_data
  5) 컨테이너 정리
"""
from __future__ import annotations

import ast as _ast
import json
import os
import re
import signal
import tarfile
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import docker

from src.benchmark.instance_loader import BenchmarkInstance


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class AlignmentExecutionResult:
    instance_id: str
    run_id: str
    returncode: int          # 0=성공, 1+=실패
    raw_output: str          # 컨테이너 stdout 전체

    # before-patch 테스트 결과 (test_name → PASSED/FAILED/ERROR)
    test_results: Dict[str, str] = field(default_factory=dict)
    has_failure: bool = False
    has_error: bool = False
    # 커버리지 (file → {"stmts", "miss", "cover", "missing", "missing_lines"})
    coverage_data: Dict[str, Dict] = field(default_factory=dict)
    # contributing test functions
    contributing_functions: List[str] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Docker 유틸 (harness 의존 없음)
# ---------------------------------------------------------------------------

def _copy_to_container(container, src: Path, dst: Path) -> None:
    """로컬 파일을 컨테이너에 복사한다."""
    tar_path = src.with_suffix(".tar")
    with tarfile.open(tar_path, "w") as tar:
        tar.add(str(src), arcname=dst.name)
    try:
        with open(tar_path, "rb") as f:
            data = f.read()
        container.exec_run(f"mkdir -p {dst.parent}")
        container.put_archive(str(dst.parent), data)
    finally:
        tar_path.unlink(missing_ok=True)


def _exec_with_timeout(container, cmd: str, timeout: int = 600):
    exec_result = ""
    exec_id = None
    exception = None
    timed_out = False
    stream = None

    def _run():
        nonlocal exec_result, exec_id, exception, stream
        try:
            exec_id = container.client.api.exec_create(container.id, cmd)["Id"]
            stream = container.client.api.exec_start(exec_id, stream=True)
            for chunk in stream:
                exec_result += chunk.decode("utf-8", errors="replace")
        except Exception as e:
            exception = e
        finally:
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass

    t = threading.Thread(target=_run, daemon=True)
    start_time = time.time()
    t.start()
    t.join(timeout)
    elapsed = time.time() - start_time
    timed_out = t.is_alive()

    if exception is not None:
        raise exception

    return exec_result, timed_out, elapsed


def _cleanup_container(client, container) -> None:
    """컨테이너를 정지·제거한다."""
    if container is None:
        return
    cid = container.id
    try:
        container.stop(timeout=10)
    except Exception as e:
        print(f"[cleanup] container.stop failed: {e}")
        try:
            info = client.api.inspect_container(cid)
            pid = info["State"].get("Pid", 0)
            if pid > 0:
                os.kill(pid, signal.SIGKILL)
        except Exception:
            pass
    try:
        container.remove(force=True)
    except Exception as e:
        print(f"[cleanup] container.remove failed: {e}")


# ---------------------------------------------------------------------------
# eval.sh 생성
# ---------------------------------------------------------------------------

def _build_eval_script(
    test_patch: str,
    base_commit: str,
    repo: str,
    version: str,
) -> str:
    """컨테이너 안에서 실행할 eval.sh 스크립트를 생성한다.

    harness의 make_eval_script_list() 로직을 재현하되
    instance-specific constants를 직접 참조한다.
    """
    from tddbench.harness.constants import MAP_REPO_VERSION_TO_SPECS
    from tddbench.harness.utils import get_test_directives

    specs = MAP_REPO_VERSION_TO_SPECS[repo][version]
    HEREDOC_DELIMITER = "EOF_114329324912"
    DIFF_MODIFIED_FILE_REGEX = r"--- a/(.*)"

    test_files = re.findall(DIFF_MODIFIED_FILE_REGEX, test_patch)
    reset_tests = f"git checkout {base_commit} {' '.join(test_files)}"
    apply_patch = (
        f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{test_patch}\n{HEREDOC_DELIMITER}"
    )

    # instance dict 형태 — get_test_directives 용
    inst_dict = {"repo": repo, "version": version, "test_patch": test_patch}
    test_command = " ".join([
        specs["test_cmd"],
        *get_test_directives(inst_dict),
    ])

    lines = [
        "#!/bin/bash",
        "set -uxo pipefail",
        "source /opt/miniconda3/bin/activate",
        "conda activate testbed",
        "cd /testbed",
    ]
    if "eval_commands" in specs:
        lines += specs["eval_commands"]
    lines += [
        "git config --global --add safe.directory /testbed",
        "cd /testbed",
        "git status",
        f"git diff {base_commit}",
        "source /opt/miniconda3/bin/activate",
        "conda activate testbed",
    ]
    if "install" in specs:
        lines.append(specs["install"])

    # sympy: pytest 미설치 환경이므로 coverage run -m pytest 전에 사전 설치
    if repo == "sympy/sympy":
        lines.append("pip install pytest -q --disable-pip-version-check")

    # sympy: ./bin/test → python -m pytest (coverage 수집 가능)
    if repo == "sympy/sympy" and "./bin/test" in test_command:
        test_command = re.sub(
            r"\./bin/test(?:\s+-C)?(?:\s+--verbose)?",
            "python -m pytest -x --no-header -rN",
            test_command,
        )

    # Django는 test_cmd가 coverage run을 포함하지 않으므로 래핑
    if "django" in repo.lower():
        if test_command.lstrip().startswith("coverage"):
            # test_cmd가 이미 coverage run을 포함하는 경우 중복 추가 방지
            test_command_cov = test_command
        else:
            test_command_cov = re.sub(
                r"^python(?:3)?\s+(-m\s+)?",
                lambda m: f"coverage run {m.group(1) or ''}",
                test_command,
                count=1,
            )
            if test_command_cov == test_command:
                # regex 미매칭 시 coverage run 직접 prepend
                test_command_cov = f"coverage run {test_command}"
    elif repo == "sympy/sympy":
        # sympy는 pytest로 교체됐으므로 coverage run으로 래핑
        test_command_cov = re.sub(
            r"^python\s+-m\s+pytest",
            "coverage run -m pytest",
            test_command,
            count=1,
        )
        if test_command_cov == test_command:
            test_command_cov = f"coverage run -m pytest -x --no-header -rN"
    else:
        test_command_cov = test_command

    if repo.strip() == "sphinx-doc/sphinx":
        lines += [
            reset_tests,
            apply_patch,
            "python3 -m pip install coverage",
            "pip install pytest-cov",
            'export PYTEST_ADDOPTS="--cov=sphinx --cov-report=term-missing"',
            test_command,
            "coverage report -m",
            reset_tests,
        ]
    else:
        lines += [
            reset_tests,
            apply_patch,
            "python3 -m pip install coverage",
            test_command_cov,
            "coverage report --show-missing",
            reset_tests,
        ]

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 로그 파서 (harness의 MAP_REPO_TO_PARSER 재사용)
# ---------------------------------------------------------------------------

def _parse_test_output(test_output: str, repo: str) -> Dict[str, str]:
    """pytest/django/sympy 등 프레임워크별 파서로 테스트 결과를 파싱한다.

    Falls back to a simple pytest regex parser if repo-specific parser
    returns empty results.
    """
    unittest_result = _unittest_line_parse(test_output)
    if unittest_result:
        return unittest_result

    from tddbench.harness.log_parsers import MAP_REPO_TO_PARSER
    parser = MAP_REPO_TO_PARSER.get(repo)
    result: Dict[str, str] = {}
    if parser is not None:
        result = parser(test_output)
    if result:
        sanitized = _sanitize_test_results(result)
        if sanitized:
            return sanitized
        # sanitization이 모든 항목을 제거한 경우 (harness parser가 오탐한 경우)
        # fallback 파서로 계속 진행
    # sympy bin/test 전용 fallback: "test_xxx F      [FAIL]" 또는 "test_xxx ok   [OK]" 패턴
    if repo == "sympy/sympy":
        sympy_result = _sympy_fallback_parse(test_output)
        if sympy_result:
            return sympy_result
    # Fallback: simple pytest PASSED/FAILED/ERROR extraction
    return _fallback_pytest_parse(test_output)


# Valid pytest node ID must contain "::" and end with a Python identifier
_VALID_NODE_RE = re.compile(r".+\.py::[\w\[\]\-]+")

# Django/unittest test ID: "test_name (module.ClassName)" or "test_name"
_DJANGO_NODE_RE = re.compile(r"^\w[\w.]* \([\w.]+\)$")

# SymPy bin/test ID: plain "test_xxx" function name (no module path, no "::")
_SYMPY_NODE_RE = re.compile(r"^test_\w+$")

# SymPy harness parser ID: "path/to/test.py:test_name" (single colon, no "::")
_SYMPY_PATH_NODE_RE = re.compile(r".+\.py:test_\w+$")


def _unittest_line_parse(output: str) -> Dict[str, str]:
    """Parse unittest/Django runner result lines.

    Django's harness parser can occasionally infer FAILED from surrounding
    traceback/diff text even when the canonical unittest result line says
    ``... ok`` or ``... ERROR``. The explicit per-test line is authoritative.
    """
    status_map = {"ok": "PASSED", "FAIL": "FAILED", "ERROR": "ERROR"}
    results: Dict[str, str] = {}
    for line in output.splitlines():
        m = re.match(r"^(test[\w.\-]+\s+\([^)]+\))\s+\.\.\.\s+(ok|FAIL|ERROR)\s*$", line.strip())
        if m:
            results[m.group(1)] = status_map[m.group(2)]
    return results


def _sanitize_test_results(results: Dict[str, str]) -> Dict[str, str]:
    """Remove entries with invalid test node IDs (e.g. 'not', '[2]').

    Accepts:
    - pytest-style IDs: path.py::Class::method
    - Django/unittest-style IDs: test_name (module.ClassName)
    - SymPy bin/test IDs: test_name (plain function name)
    - SymPy harness-parser IDs: path/to/test.py:test_name (single colon)
    Some harness parsers mis-parse non-standard output lines into garbage.
    """
    return {
        k: v for k, v in results.items()
        if (_VALID_NODE_RE.match(k) or _DJANGO_NODE_RE.match(k)
            or _SYMPY_NODE_RE.match(k) or _SYMPY_PATH_NODE_RE.match(k))
    }


def _extract_error_details(raw_output: str) -> List[str]:
    """raw Docker output에서 구체적 에러 메시지를 추출한다."""
    patterns = [
        r"(ImportError:\s*.+)",
        r"(ModuleNotFoundError:\s*.+)",
        r"(SyntaxError:\s*.+)",
        r"(AttributeError:\s*.+)",
        r"(E\s+ImportError:\s*.+)",
        r"(E\s+ModuleNotFoundError:\s*.+)",
    ]
    seen: set = set()
    errors: List[str] = []
    for pat in patterns:
        for m in re.finditer(pat, raw_output):
            msg = m.group(1).strip().lstrip("E").strip()[:300]
            if msg not in seen:
                errors.append(msg)
                seen.add(msg)
    return errors[:5]


def _detect_test_not_collected(raw_output: str) -> Optional[str]:
    """Detect 'test not collected' patterns from raw Docker output.

    Returns a diagnostic message if detected, otherwise None.
    """
    # NameError (e.g. 'unittest' not defined) — more specific cause
    name_error = re.search(r"NameError: name '(\w+)' is not defined", raw_output)
    if name_error:
        name = name_error.group(1)
        return (
            f"Test not collected (NameError: '{name}' is not defined). "
            f"Missing import: add 'import {name}' to the test file."
        )

    # Django app_label RuntimeError — `from tests.xxx import Y` causes this
    if re.search(r"RuntimeError: Model class tests\.", raw_output):
        return (
            "Test not collected (RuntimeError: Django model imported via `from tests.xxx import Y`). "
            "CRITICAL: NEVER use `from tests.xxx import Y` — it triggers RuntimeError in Django's test runner. "
            "Import app modules directly without the `tests.` prefix "
            "(e.g. `from modeladmin.models import X` not `from tests.modeladmin.models import X`). "
            "Check the existing test file's import block for the correct paths."
        )

    # ImportError / ModuleNotFoundError during collection — import 오류이므로 진단 분리
    import_error = re.search(
        r"(?:E\s+)?(ImportError|ModuleNotFoundError):\s*(.+)", raw_output
    )
    if import_error:
        kind = import_error.group(1)
        detail = import_error.group(2).strip()[:200]
        return (
            f"Test not collected ({kind}: {detail}). "
            "Fix the import in the generated test."
        )

    patterns = [
        (r"ERROR:\s*not found:\s*(\S+)", "not found"),
        (r"no tests ran", "no tests ran"),
        (r"collected 0 items", "collected 0 items"),
    ]
    for pattern, label in patterns:
        m = re.search(pattern, raw_output, re.IGNORECASE)
        if m:
            detail = m.group(0)[:200]
            return (
                f"Test not collected ({label}): {detail}. "
                "Possible cause: module-level pytest.importorskip, "
                "pytest.mark.skip, or missing dependency."
            )
    return None


def _sympy_fallback_parse(output: str) -> Dict[str, str]:
    """sympy bin/test 출력 파싱.

    지원하는 출력 형식:
    1) bin/test 스타일:
        test_foo F                    [FAIL]
        test_bar ok                   [OK]
        test_baz E                    [FAIL]  (ERROR)
    2) pytest FAILURES 섹션 스타일 (sympy가 pytest로 실행될 때):
        ___ path/to/test.py:test_name ___
       → test_name = FAILED
    3) sympy bin/test 요약 스타일:
        tests finished: N passed, M failed
       → 통과/실패 카운트로 test_name 보완
    """
    results: Dict[str, str] = {}

    # Format 1: bin/test line-by-line
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped.startswith("test_"):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        test_name = parts[0]
        last = parts[-1]
        second = parts[1] if len(parts) >= 2 else ""
        if last in ("[FAIL]", "[ERROR]") or second in ("F", "E"):
            results[test_name] = "FAILED"
        elif last == "[OK]" or second == "ok":
            results[test_name] = "PASSED"

    # Format 2: pytest FAILURES section header
    # 두 형식 모두 지원:
    #   "___ path/to/file.py:test_name ___"  (경로+콜론 prefix)
    #   "___ test_name ___"                  (pytest -x --no-header 형식, 콜론 없음)
    for m in re.finditer(r"_{3,}\s+(?:\S+:)?(test_\w+)\s+_{3,}", output):
        test_name = m.group(1)
        results[test_name] = "FAILED"

    return results


def _fallback_pytest_parse(output: str) -> Dict[str, str]:
    """Extract test results from raw pytest output using common patterns."""
    results: Dict[str, str] = {}
    # Pattern: "PASSED tests/foo.py::TestBar::test_baz"  or
    #          "tests/foo.py::test_baz PASSED"
    for m in re.finditer(
        r"(PASSED|FAILED|ERROR)\s+([\w/.\-]+::\S+)", output
    ):
        results[m.group(2)] = m.group(1)
    for m in re.finditer(
        r"([\w/.\-]+::\S+)\s+(PASSED|FAILED|ERROR)", output
    ):
        results[m.group(1)] = m.group(2)

    # Fallback: -rN 플래그 사용 시 PASSED 형식이 없고 요약만 있는 경우
    # e.g. "======================== 1 passed, 97 warnings in 0.35s ========================"
    # e.g. "coverage run -m pytest ... path/to/test.py::test_name"
    if not results:
        summary_m = re.search(r"(\d+) passed", output)
        if summary_m and int(summary_m.group(1)) > 0:
            # coverage run 커맨드 라인에서 test node id 추출
            cmd_m = re.search(r"coverage run.*?pytest.*?([\w/.\-]+\.py::test_\w+)", output)
            if cmd_m:
                results[cmd_m.group(1)] = "PASSED"
            else:
                # coverage run 없이 pytest 직접 실행 시
                node_m = re.search(r"pytest.*?([\w/.\-]+\.py::test_\w+)", output)
                if node_m:
                    results[node_m.group(1)] = "PASSED"

    return results


# ---------------------------------------------------------------------------
# 커버리지 파싱
# ---------------------------------------------------------------------------

def _parse_coverage_text(coverage_text: str) -> Dict[str, Dict]:
    """``coverage report --show-missing`` 텍스트를 파싱한다.

    Handles both ``--show-missing`` and ``-m`` formatting and is tolerant of
    slightly different column layouts produced by different coverage versions.
    """
    data: Dict[str, Dict] = {}
    header_seen = False
    for line in coverage_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("---"):
            header_seen = True
            continue
        if line.startswith("Name"):
            header_seen = True
            continue
        if line.startswith("TOTAL"):
            continue
        if not header_seen:
            continue

        parts = line.split()
        if len(parts) < 4:
            continue
        filename = parts[0]
        if not filename.endswith(".py"):
            continue

        # Find the coverage percentage column (ends with '%')
        cover_idx = -1
        cover = 0.0
        for idx, p in enumerate(parts[1:], 1):
            if p.endswith("%"):
                try:
                    cover = float(p.rstrip("%"))
                    cover_idx = idx
                except ValueError:
                    continue
                break

        if cover_idx < 0:
            # Some versions omit % — try column 3 or 4 as a number
            for idx in (3, 4):
                if idx < len(parts):
                    try:
                        cover = float(parts[idx])
                        cover_idx = idx
                        break
                    except ValueError:
                        continue
            if cover_idx < 0:
                continue

        try:
            stmts = int(parts[1])
            miss = int(parts[2])
        except (ValueError, IndexError):
            continue

        missing_str = " ".join(parts[cover_idx + 1:]) if cover_idx + 1 < len(parts) else ""
        data[filename] = {
            "stmts": stmts,
            "miss": miss,
            "cover": cover,
            "missing": missing_str,
            "missing_lines": _parse_missing_lines(missing_str),
        }
    return data


def _parse_missing_lines(missing_str: str) -> List[int]:
    """'23, 45-50, 60' → [23, 45, 46, ..., 50, 60]"""
    lines: List[int] = []
    if not missing_str:
        return lines
    for part in missing_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "->" in part:
            continue
        if "-" in part:
            try:
                s, e = part.split("-", 1)
                lines.extend(range(int(s.strip()), int(e.strip()) + 1))
            except ValueError:
                continue
        else:
            try:
                lines.append(int(part))
            except ValueError:
                continue
    return lines


# ---------------------------------------------------------------------------
# contributing functions 추출
# ---------------------------------------------------------------------------

def _get_contributing_functions(test_patch: str) -> Dict[str, List[str]]:
    """test_patch에서 추가/수정된 테스트 함수 목록을 추출한다.

    Returns: {filename: [func_name, ...]}
    """
    funcs: Dict[str, List[str]] = {}
    segments = test_patch.split("+++ b")
    for seg in segments[1:]:
        filename = seg.split("\n")[0].strip()
        if filename.startswith("/"):
            filename = filename[1:]
        for part in seg.split("def test")[1:]:
            fname = "test" + part.split("(")[0].strip()
            flines = part.split("\n")
            for ln in flines:
                if ln.strip().startswith("+"):
                    cleaned = ln.replace("+", "").replace("-", "")
                    if cleaned.strip() == "":
                        continue
                    funcs.setdefault(filename, [])
                    if fname not in funcs[filename]:
                        funcs[filename].append(fname)
                    break
    return funcs


def _resolve_fun2test(container, contributing: Dict[str, List[str]], timeout: int) -> List[str]:
    """contributing functions를 Class::method 형식의 pytest 노드 ID로 변환한다."""
    fun2test: List[str] = []
    for test_file, func_names in contributing.items():
        # 컨테이너에서 테스트 파일 읽기
        output, _, _ = _exec_with_timeout(container, f"cat {test_file}", timeout)
        if not output.strip():
            for fn in func_names:
                fun2test.append(f"{test_file}::{fn}")
            continue

        # class method 매핑
        class_func = _get_class_functions(output)
        outer_func = _get_outer_functions(output)

        for fn in func_names:
            if fn in class_func:
                fun2test.append(f"{test_file}::{class_func[fn]}::{fn}")
            elif fn in outer_func:
                fun2test.append(f"{test_file}::{fn}")
            else:
                fun2test.append(f"{test_file}::{fn}")
    return fun2test


def _get_class_functions(text: str) -> Dict[str, str]:
    """test method → class name 매핑을 반환한다."""
    try:
        tree = _ast.parse(text)
    except SyntaxError:
        return {}
    mapping: Dict[str, str] = {}
    for node in _ast.walk(tree):
        if isinstance(node, _ast.ClassDef):
            for item in node.body:
                if isinstance(item, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                    mapping[item.name] = node.name
    return mapping


def _get_outer_functions(text: str) -> List[str]:
    """모듈 레벨 test 함수 이름을 반환한다."""
    try:
        tree = _ast.parse(text)
    except SyntaxError:
        return []
    return [
        node.name
        for node in _ast.iter_child_nodes(tree)
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef))
        and "test" in node.name
    ]


def _modify_eval_script(script: str, repo: str, fun2test: List[str]) -> str:
    """eval.sh의 pytest/tox/sympy 테스트 명령어를 fun2test만 실행하도록 수정한다."""
    if not fun2test:
        return script

    out_lines: List[str] = []
    for ln in script.split("\n"):
        if (
            "coverage run" in ln
            or "tox --current-env -epy39 -v --" in ln
            or "./bin/test" in ln          # sympy bin/test (교체 전 원본 스크립트 매칭)
            or "python -m pytest" in ln   # plain pytest repos (astropy, matplotlib, sympy 교체 후)
            or (ln.strip().startswith("pytest ") and "#" not in ln)
            or ("runtests.py" in ln and repo == "django/django")
        ):
            parts = ln.split(" ")
            # 마지막 파일/테스트 인자 제거
            fcount = 0
            for i in range(len(parts) - 1, 0, -1):
                if ".py" in parts[i] or "." in parts[i]:
                    fcount += 1
                else:
                    break
            base_cmd = " ".join(parts[:len(parts) - fcount])

            if repo == "django/django":
                cases = []
                for item in fun2test:
                    item = item.removeprefix("tests/")
                    item = item.replace(".py", "").replace("::", ".").replace("/", ".")
                    cases.append(item)
                out_lines.append(f"{base_cmd} {' '.join(cases)}")
            elif repo == "sympy/sympy":
                # sympy는 ./bin/test → coverage run -m pytest로 교체
                for item in fun2test:
                    out_lines.append(f"coverage run -m pytest -x --no-header -rN {item}")
            else:
                out_lines.append(f"{base_cmd} {' '.join(fun2test)}")
        else:
            out_lines.append(ln)
    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# stdout split helper
# ---------------------------------------------------------------------------

_COVERAGE_SPLIT_PATTERNS = [
    "+ coverage report",          # set -x echo
    "+ python3 -m coverage",      # alternative invocation
    "Name    Stmts   Miss",       # coverage table header (direct)
    "Name                 Stmts",  # wider column variant
]


def _split_output(test_output: str) -> tuple:
    """Split container output into (test_text, coverage_text)."""
    for pattern in _COVERAGE_SPLIT_PATTERNS:
        if pattern in test_output:
            parts = test_output.split(pattern, 1)
            return parts[0], pattern + parts[1]
    # No coverage output found; return full output as test text
    return test_output, ""


# ---------------------------------------------------------------------------
# AlignmentRunner 메인 클래스
# ---------------------------------------------------------------------------

class AlignmentRunner:
    """Docker SDK 기반 patch-free alignment runner.

    harness를 호출하지 않고 직접 Docker 컨테이너에서 테스트를 실행한다.
    이미 빌드된 ``sweb.eval.{arch}.{instance_id}:latest`` 이미지를 재사용한다.
    """

    def __init__(self, timeout: int = 600) -> None:
        self.timeout = timeout
        self._client = docker.from_env()

    # ------------------------------------------------------------------ #
    #  public API
    # ------------------------------------------------------------------ #

    def run(
        self,
        instance: BenchmarkInstance,
        generated_test_json_path: str,
        run_id: Optional[str] = None,
    ) -> AlignmentExecutionResult:
        generated_path = Path(generated_test_json_path).resolve()
        if not generated_path.exists():
            raise FileNotFoundError(f"generated_test.json 없음: {generated_path}")

        patch_path = generated_path.with_name("generated_test.patch")
        if not patch_path.exists():
            raise FileNotFoundError(f"generated_test.patch 없음: {patch_path}")

        gen_test_patch = patch_path.read_text(encoding="utf-8")
        run_id = run_id or f"align-{instance.instance_id}-{uuid.uuid4().hex[:8]}"

        # ── 1) instance image 확인 (없으면 자동 빌드) ──
        from tddbench.harness.test_spec import make_test_spec
        spec = make_test_spec(instance.raw)
        image_name = spec.instance_image_key

        try:
            self._client.images.get(image_name)
        except docker.errors.ImageNotFound:
            print(f"  [auto-build] Docker image not found: {image_name}")
            print(f"  [auto-build] Building instance image …")
            try:
                from tddbench.harness.docker_build import build_instance_images
                successful, failed = build_instance_images(
                    self._client, [instance.raw],
                    force_rebuild=False, max_workers=1,
                )
                if failed:
                    return self._error_result(
                        instance.instance_id, run_id,
                        f"Docker image build failed: {image_name}",
                    )
                # env 이미지 실패 시 instance 빌드가 조용히 스킵될 수 있음
                try:
                    self._client.images.get(image_name)
                except docker.errors.ImageNotFound:
                    return self._error_result(
                        instance.instance_id, run_id,
                        f"Docker image build failed: {image_name} (env image dependency likely failed)",
                    )
                print(f"  [auto-build] Image built successfully: {image_name}")
            except Exception as build_err:
                return self._error_result(
                    instance.instance_id, run_id,
                    f"Docker image build error: {build_err}",
                )

        container = None
        try:
            # ── 2) 컨테이너 생성·시작 ──
            container_name = f"sweb.align.{instance.instance_id}.{run_id}"
            # 기존 동명 컨테이너 제거
            try:
                old = self._client.containers.get(container_name)
                old.remove(force=True)
            except docker.errors.NotFound:
                pass
            except docker.errors.APIError as e:
                if getattr(e, "status_code", None) != 409:
                    raise
                container_name = f"{container_name}.{uuid.uuid4().hex[:8]}"

            container = self._client.containers.create(
                image_name,
                name=container_name,
                detach=True,
                tty=True,
            )
            container.start()

            # ── 3) INITIAL phase: contributing functions 추출 ──
            contributing = _get_contributing_functions(gen_test_patch)

            # eval.sh의 pip install coverage 이전까지만 실행 (환경 세팅)
            full_eval = _build_eval_script(
                gen_test_patch, instance.base_commit,
                instance.repo, instance.version,
            )
            setup_part = full_eval.split("python3 -m pip install coverage")[0].strip() + "\n"

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".sh", delete=False, dir="/tmp",
            ) as tmp:
                tmp.write(setup_part)
                tmp_path = Path(tmp.name)

            _copy_to_container(container, tmp_path, Path("/setup.sh"))
            tmp_path.unlink(missing_ok=True)
            _exec_with_timeout(container, "/bin/bash /setup.sh", self.timeout)


            # fun2test 결정
            fun2test = _resolve_fun2test(container, contributing, self.timeout)
            _exec_with_timeout(container, "git clean -fd", self.timeout)

            # ── 4) BEFORE-PATCH phase: 테스트 실행 ──
            # repo를 명확히 넘기도록 수정
            modified_eval = _modify_eval_script(full_eval, instance.repo, fun2test)

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".sh", delete=False, dir="/tmp",
            ) as tmp:
                tmp.write(modified_eval)
                tmp_path = Path(tmp.name)

            _copy_to_container(container, tmp_path, Path("/eval.sh"))
            tmp_path.unlink(missing_ok=True)

            test_output, timed_out, elapsed = _exec_with_timeout(
                container, "/bin/bash /eval.sh", self.timeout,
            )

            if timed_out:
                return self._error_result(
                    instance.instance_id, run_id,
                    f"Timeout after {self.timeout}s",
                    raw_output=test_output,
                )

            # ── 5) 결과 파싱 ──
            test_text, coverage_text = _split_output(test_output)

            test_results = _parse_test_output(test_text, instance.repo)
            coverage_data = _parse_coverage_text(coverage_text)

            has_failure = any(v in ("FAILED", "ERROR") for v in test_results.values())
            has_error = any(v == "ERROR" for v in test_results.values())

            error_msgs: List[str] = []
            if not test_results:
                error_msgs.append("No test results parsed from output")
                error_msgs.extend(_extract_error_details(test_output))

            # ── 5b) "test not collected" 감지 (module-level skip 등) ──
            not_collected_msg = _detect_test_not_collected(test_output)
            if not_collected_msg:
                has_error = True
                error_msgs.append(not_collected_msg)
                # Clear malformed results — they are not real test outcomes
                if test_results and not any(
                    v in ("PASSED", "FAILED") for v in test_results.values()
                ):
                    test_results = {}
                    has_failure = False

            return AlignmentExecutionResult(
                instance_id=instance.instance_id,
                run_id=run_id,
                returncode=0,
                raw_output=test_output,
                test_results=test_results,
                has_failure=has_failure,
                has_error=has_error,
                coverage_data=coverage_data,
                contributing_functions=fun2test,
                error_messages=error_msgs,
            )

        except Exception as e:
            return self._error_result(
                instance.instance_id, run_id,
                str(e),
                raw_output="",
            )
        finally:
            _cleanup_container(self._client, container)

    # ------------------------------------------------------------------ #

    def save(self, result: AlignmentExecutionResult, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------ #
    #  helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _error_result(
        instance_id: str,
        run_id: str,
        msg: str,
        raw_output: str = "",
    ) -> AlignmentExecutionResult:
        return AlignmentExecutionResult(
            instance_id=instance_id,
            run_id=run_id,
            returncode=1,
            raw_output=raw_output,
            has_error=True,
            error_messages=[msg],
        )
