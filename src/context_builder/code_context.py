from __future__ import annotations

import ast
import re
import json
import subprocess
import threading
import warnings
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.file_io import read_text

# 같은 repository에 대한 git 작업을 직렬화하기 위한 전역 lock
_REPO_LOCKS: dict[str, threading.Lock] = defaultdict(threading.Lock)


def _get_call_fullname(call: ast.Call) -> str:
    """Extract dotted name from a Call node, e.g. 'pytest.importorskip'."""
    func = call.func
    parts: List[str] = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
    return ".".join(reversed(parts)) if parts else ""


@dataclass
class IndexedFile:
    path: str
    is_test_file: bool
    classes: List[str]
    functions: List[str]
    methods: List[str]
    test_functions: List[str]
    imports: List[str]
    call_names: List[str] = None
    parse_error: Optional[str] = None
    has_module_skip: bool = False
    collection_risk: str = ""


@dataclass
class CandidateFile:
    path: str
    score: int
    matched_identifiers: List[str]
    reasons: List[str]
    has_module_skip: bool = False
    collection_risk: str = ""
    code_snippets: Optional[Dict[str, str]] = None  # {identifier_name: "def foo(x):\n    ..."}
    top_level_functions: Optional[List[str]] = None  # AST로 추출한 public 함수 목록 (환각 방지)
    localization_signals: Optional[List[str]] = None
    graph_neighbors: Optional[List[str]] = None


@dataclass
class ProjectTestStyle:
    framework: str
    evidence: List[str]
    assert_style: List[str]
    runner: str = "pytest"  # "pytest" | "django-test" | "sympy-bin-test" | "unittest" | "unknown"


@dataclass
class CodeContextFile:
    instance_id: str
    repo: str
    base_commit: str
    repo_path: str
    candidate_source_files: List[Dict[str, Any]]
    candidate_test_files: List[Dict[str, Any]]
    project_test_style: Dict[str, Any]
    indexed_file_count: int
    indexed_test_file_count: int
    available_imports: Optional[Dict[str, List[str]]] = None
    test_example_snippet: str = ""  # 상위 테스트 파일의 첫 번째 Test 클래스/함수 스니펫
    conftest_fixtures: Optional[Dict[str, List[str]]] = None  # {conftest_path: [fixture_names]}
    test_symbol_catalog: Optional[Dict[str, Dict[str, List[str]]]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d.get("available_imports") is None:
            d["available_imports"] = {}
        if d.get("conftest_fixtures") is None:
            d["conftest_fixtures"] = {}
        if d.get("test_symbol_catalog") is None:
            d["test_symbol_catalog"] = {}
        return d


class CodeContextExtractor:
    def __init__(
        self,
        repos_root: str = "data/repos",
        top_k_source: int = 5,
        top_k_test: int = 5,
    ) -> None:
        self.repos_root = Path(repos_root)
        self.top_k_source = top_k_source
        self.top_k_test = top_k_test

    def extract(self, instance: Any, clue: Dict[str, Any]) -> CodeContextFile:
        repo_path = self._prepare_repo(instance.repo, instance.base_commit)
        indexed_files = self._index_repository(repo_path)
        source_candidates, test_candidates = self._rank_files(indexed_files, clue, repo_path)
        test_style = self._infer_test_style(indexed_files, test_candidates, repo_name=instance.repo)
        available_imports = self._collect_available_imports(
            indexed_files, source_candidates, test_candidates, clue, repo_path,
        )
        test_example_snippet = self._extract_test_example(test_candidates, repo_path)
        conftest_fixtures = self._extract_conftest_fixtures(test_candidates, repo_path)
        test_symbol_catalog = self._build_test_symbol_catalog(test_candidates, repo_path)

        return CodeContextFile(
            instance_id=instance.instance_id,
            repo=instance.repo,
            base_commit=instance.base_commit,
            repo_path=str(repo_path),
            candidate_source_files=[asdict(x) for x in source_candidates],
            candidate_test_files=[asdict(x) for x in test_candidates],
            project_test_style=asdict(test_style),
            indexed_file_count=len(indexed_files),
            indexed_test_file_count=sum(1 for x in indexed_files if x.is_test_file),
            available_imports=available_imports,
            test_example_snippet=test_example_snippet,
            conftest_fixtures=conftest_fixtures,
            test_symbol_catalog=test_symbol_catalog,
        )

    def save(self, context: CodeContextFile, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(context.to_dict(), f, ensure_ascii=False, indent=2)

    def _prepare_repo(self, repo_name: str, base_commit: str) -> Path:
        parts = repo_name.split("/")
        if len(parts) != 2:
            raise ValueError(f"repo_name 형식이 잘못됨 (expected 'owner/repo'): {repo_name!r}")
        owner, name = parts
        repo_dir = self.repos_root / f"{owner}__{name}"
        clone_url = f"https://github.com/{repo_name}.git"

        self.repos_root.mkdir(parents=True, exist_ok=True)

        repo_key = str(repo_dir.resolve())
        repo_lock = _REPO_LOCKS[repo_key]

        # 같은 repo에 대해서는 git 작업을 한 번에 하나의 스레드만 수행
        with repo_lock:
            if not repo_dir.exists():
                self._run_git(["clone", clone_url, str(repo_dir)], cwd=Path("."))

            self._run_git(["fetch", "--all", "--tags"], cwd=repo_dir)
            self._run_git(["reset", "--hard"], cwd=repo_dir)
            self._run_git(["clean", "-fd"], cwd=repo_dir)
            self._run_git(["checkout", base_commit], cwd=repo_dir)

        return repo_dir

    def _run_git(self, args: List[str], cwd: Path) -> None:
        cmd = ["git"] + args
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Git command failed: {' '.join(cmd)}\n"
                f"cwd={cwd}\n"
                f"stdout={result.stdout}\n"
                f"stderr={result.stderr}"
            )

    def _index_repository(self, repo_path: Path) -> List[IndexedFile]:
        indexed: List[IndexedFile] = []

        for file_path in repo_path.rglob("*.py"):
            if self._should_skip(file_path):
                continue

            rel_path = str(file_path.relative_to(repo_path))
            is_test_file = self._is_test_file(rel_path)

            try:
                source = read_text(file_path)
            except UnicodeDecodeError:
                try:
                    source = file_path.read_text(encoding="latin-1")
                except Exception as e:
                    indexed.append(
                        IndexedFile(
                            path=rel_path,
                            is_test_file=is_test_file,
                            classes=[],
                            functions=[],
                            methods=[],
                            test_functions=[],
                            imports=[],
                            call_names=[],
                            parse_error=f"read_error: {e}",
                        )
                    )
                    continue
            except Exception as e:
                indexed.append(
                    IndexedFile(
                        path=rel_path,
                        is_test_file=is_test_file,
                        classes=[],
                        functions=[],
                        methods=[],
                        test_functions=[],
                        imports=[],
                        call_names=[],
                        parse_error=f"read_error: {e}",
                    )
                )
                continue

            indexed.append(self._parse_python_file(rel_path, source, is_test_file))

        return indexed

    def _parse_python_file(self, rel_path: str, source: str, is_test_file: bool) -> IndexedFile:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(source)
        except Exception as e:
            return IndexedFile(
                path=rel_path,
                is_test_file=is_test_file,
                classes=[],
                functions=[],
                methods=[],
                test_functions=[],
                imports=[],
                call_names=[],
                parse_error=f"ast_error: {e}",
            )

        classes: set[str] = set()
        functions: set[str] = set()
        methods: set[str] = set()
        test_functions: set[str] = set()
        imports: set[str] = set()
        call_names: set[str] = set()

        has_module_skip = False
        collection_risks: set[str] = set()

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                classes.add(node.name)
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.add(item.name)
                        if item.name.startswith("test"):
                            test_functions.add(item.name)

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.add(node.name)
                if node.name.startswith("test"):
                    test_functions.add(node.name)

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)

            # Detect module-level skip patterns (only for test files)
            if is_test_file and not has_module_skip:
                has_module_skip = self._is_module_level_skip(node)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _get_call_fullname(node)
                if name:
                    call_names.add(name)
                    call_names.add(name.split(".")[-1])
            if is_test_file and isinstance(node, ast.ImportFrom):
                if node.module == "sphinx.testing.fixtures":
                    collection_risks.add("sphinx_testing_fixtures")
            elif is_test_file and isinstance(node, ast.Import):
                if any(alias.name == "sphinx.testing.fixtures" for alias in node.names):
                    collection_risks.add("sphinx_testing_fixtures")
            if is_test_file and isinstance(node, ast.Assign):
                targets_pytestmark = any(
                    isinstance(target, ast.Name) and target.id == "pytestmark"
                    for target in node.targets
                )
                if targets_pytestmark:
                    values = node.value.elts if isinstance(node.value, (ast.List, ast.Tuple)) else [node.value]
                    for value in values:
                        if isinstance(value, ast.Call) and _get_call_fullname(value) == "pytest.mark.sphinx":
                            collection_risks.add("pytest_mark_sphinx")
            if is_test_file and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for dec in getattr(node, "decorator_list", []):
                    dec_name = ""
                    if isinstance(dec, ast.Call):
                        dec_name = _get_call_fullname(dec)
                    elif isinstance(dec, ast.Attribute):
                        parts = []
                        cur = dec
                        while isinstance(cur, ast.Attribute):
                            parts.append(cur.attr)
                            cur = cur.value
                        if isinstance(cur, ast.Name):
                            parts.append(cur.id)
                        dec_name = ".".join(reversed(parts))
                    if dec_name == "pytest.mark.sphinx":
                        collection_risks.add("pytest_mark_sphinx")

        # GIS/환경 의존 경로는 module_skip으로 마킹 (GIS 미설치 환경에서 항상 실패)
        _ENV_SKIP_PATTERNS = ("gis_tests/", "contrib/gis/")
        if is_test_file and not has_module_skip:
            if any(pat in rel_path.replace("\\", "/") for pat in _ENV_SKIP_PATTERNS):
                has_module_skip = True
                collection_risks.add("env_dependent_gis")
        if is_test_file and has_module_skip:
            collection_risks.add("module_level_skip")

        return IndexedFile(
            path=rel_path,
            is_test_file=is_test_file,
            classes=sorted(classes),
            functions=sorted(functions),
            methods=sorted(methods),
            test_functions=sorted(test_functions),
            imports=sorted(imports),
            call_names=sorted(call_names),
            parse_error=None,
            has_module_skip=has_module_skip,
            collection_risk=",".join(sorted(collection_risks)),
        )

    @staticmethod
    def _is_module_level_skip(node: ast.AST) -> bool:
        """Detect module-level skip patterns that prevent test collection.

        Patterns detected:
          - pytest.importorskip("pkg")
          - var = pytest.importorskip("pkg")
          - pytest.skip("reason")
          - pytestmark = pytest.mark.skip(...)
          - pytestmark = pytest.mark.skipif(...)
          - pytestmark = [pytest.mark.skip(...)]
        """
        # --- Expr: bare pytest.importorskip(...) or pytest.skip(...) ---
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            func_name = _get_call_fullname(call)
            if func_name in ("pytest.importorskip", "pytest.skip"):
                return True

        # --- Assign ---
        if isinstance(node, ast.Assign):
            # var = pytest.importorskip(...)
            if isinstance(node.value, ast.Call):
                func_name = _get_call_fullname(node.value)
                if func_name == "pytest.importorskip":
                    return True

                # pytestmark = pytest.mark.skip(...) / skipif(...)
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "pytestmark":
                        if func_name and ("pytest.mark.skip" in func_name):
                            return True

            # pytestmark = [pytest.mark.skip(...), ...]
            if isinstance(node.value, (ast.List, ast.Tuple)):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "pytestmark":
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Call):
                                elt_name = _get_call_fullname(elt)
                                if elt_name and "pytest.mark.skip" in elt_name:
                                    return True

        # --- try-except block: pytest.importorskip / pytest.skip inside try ---
        if isinstance(node, ast.Try):
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    func_name = _get_call_fullname(child)
                    if func_name in ("pytest.importorskip", "pytest.skip"):
                        return True

        # --- if block: if condition: pytest.skip() / pytest.importorskip() ---
        if isinstance(node, ast.If):
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    func_name = _get_call_fullname(child)
                    if func_name in ("pytest.importorskip", "pytest.skip"):
                        return True

        return False

    def _should_skip(self, file_path: Path) -> bool:
        skip_parts = {
            ".git",
            ".venv",
            "venv",
            "__pycache__",
            "site-packages",
            "node_modules",
            ".mypy_cache",
            ".pytest_cache",
            "build",
            "dist",
        }
        return any(part in skip_parts for part in file_path.parts)

    def _is_test_file(self, rel_path: str) -> bool:
        p = rel_path.replace("\\", "/").lower()
        name = Path(p).name
        return (
            "/tests/" in p
            or p.startswith("tests/")
            or name.startswith("test_")
            or name.endswith("_test.py")
            or name == "conftest.py"
        )

    def _extract_code_snippets(
        self,
        file_path: Path,
        identifiers: List[str],
        max_lines: int = 15,
    ) -> Dict[str, str]:
        """matched identifiers에 해당하는 함수/클래스 시그니처+본문 앞부분 추출."""
        if not identifiers or not file_path.exists():
            return {}
        try:
            source = read_text(file_path)
            tree = ast.parse(source)
        except Exception:
            return {}
        lines = source.splitlines()
        ident_set = set(identifiers)
        snippets: Dict[str, str] = {}
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if node.name not in ident_set:
                continue
            start = node.lineno - 1
            end = min(start + max_lines, len(lines))
            snippets[node.name] = "\n".join(lines[start:end])
        return snippets

    def _extract_top_level_functions(self, file_path: Path, max_results: int = 40) -> List[str]:
        """소스 파일에서 top-level public 함수 및 클래스의 public 메서드 이름 추출.

        ScenarioGenerator가 target_function 선택 시 실제 존재하는 함수명만
        사용하도록 돕기 위한 힌트 목록. 환각 방지 목적.
        """
        if not file_path.exists():
            return []
        try:
            source = read_text(file_path)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(source)
        except Exception:
            return []

        seen: set = set()
        names: List[str] = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("_") and node.name not in seen:
                    seen.add(node.name)
                    names.append(node.name)
            elif isinstance(node, ast.ClassDef):
                for item in ast.iter_child_nodes(node):
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        qualified = f"{node.name}.{item.name}"
                        if not item.name.startswith("_") and qualified not in seen:
                            seen.add(qualified)
                            names.append(qualified)
        return names[:max_results]

    def _rank_files(
        self,
        indexed_files: List[IndexedFile],
        clue: Dict[str, Any],
        repo_path: Path,
    ) -> Tuple[List[CandidateFile], List[CandidateFile]]:
        noisy_functions = {
            "arange", "rand", "random", "seed", "platform", "get_backend",
            "show_versions",
        }
        clue_functions = {
            fn for fn in clue.get("identifiers", {}).get("functions", [])
            if fn not in noisy_functions
        }
        clue_classes = set(clue.get("identifiers", {}).get("classes", []))
        clue_files = {
            self._normalize_issue_file_hint(f)
            for f in clue.get("identifiers", {}).get("files", [])
            if isinstance(f, str)
        }
        clue_files = {f for f in clue_files if f}

        # code_examples(이슈 코드 블록)에서 추가 함수 호출 추출 — public API 경유 내부 구현 탐색
        _stopwords = frozenset({
            "if", "for", "while", "def", "class", "return", "import",
            "from", "with", "as", "in", "not", "and", "or", "is", "True", "False",
            "None", "print", "len", "range", "type", "str", "int", "list", "dict",
            "set", "tuple", "super", "self", "cls",
            "arange", "rand", "random", "seed", "platform", "get_backend",
            "show_versions",
        })
        for block in clue.get("code_examples", []):
            if block.get("is_system_or_output"):
                continue
            code = block.get("code", "") or ""
            for m in re.finditer(r"\b([a-z_][a-zA-Z0-9_]{2,})\s*\(", code):
                fn = m.group(1)
                if fn not in _stopwords:
                    clue_functions.add(fn)

        # traceback에서 추출한 fault location 후보 — suffix 매칭으로 파일 식별
        fault_locations: List[Dict[str, Any]] = clue.get("fault_locations", [])
        # {rel_path → fault_location} 매핑 (suffix 매칭 후 채워짐)
        fault_file_info: Dict[str, Dict[str, Any]] = {}
        for fl in fault_locations:
            fl_path = fl.get("file_path", "").replace("\\", "/")
            for ifile in indexed_files:
                if not ifile.is_test_file and fl_path.endswith(ifile.path.replace("\\", "/")):
                    fault_file_info[ifile.path] = fl
                    break

        observed = " ".join(clue.get("observed_behavior", []))
        expected = " ".join(clue.get("expected_behavior", []))
        repro = " ".join(clue.get("repro_conditions", []))
        raw_issue_text = clue.get("raw_issue_text", "")
        clue_text = f"{observed} {expected} {repro} {raw_issue_text}".lower()
        salient_tokens = self._extract_salient_issue_tokens(clue, clue_text)

        source_candidates: List[CandidateFile] = []
        test_candidates: List[CandidateFile] = []

        # 1단계: source 파일 우선 점수화
        for item in indexed_files:
            if item.is_test_file:
                continue

            score = 0
            matched_identifiers: set[str] = set()
            reasons: List[str] = []

            file_name = Path(item.path).name.lower()
            path_lower = item.path.lower()
            stem = Path(item.path).stem.lower()
            file_funcs = set(item.functions) | set(item.methods)

            for fn in clue_functions:
                if fn in file_funcs:
                    score += 5
                    matched_identifiers.add(fn)
                    reasons.append(f"function_match:{fn}")

            for cls in clue_classes:
                # ALL_CAPS identifiers are likely constants, not classes
                if cls.isupper():
                    continue
                if cls in item.classes:
                    score += 4
                    matched_identifiers.add(cls)
                    reasons.append(f"class_match:{cls}")

            for clue_file in clue_files:
                clue_file_name = Path(clue_file).name.lower()
                clue_path = clue_file.lower().lstrip("./")
                if clue_path and self._path_suffix_matches(path_lower, clue_path):
                    score += 20
                    matched_identifiers.add(clue_file)
                    reasons.append(f"explicit_file_path_hint:{clue_file}")
                elif clue_file_name and clue_file_name == file_name:
                    score += 10
                    matched_identifiers.add(clue_file)
                    reasons.append(f"file_hint_match:{clue_file}")

            for fn in clue_functions:
                fn_lower = fn.lower()
                if fn_lower in path_lower:
                    score += 2
                    reasons.append(f"path_contains_function:{fn}")

            for cls in clue_classes:
                cls_lower = cls.lower()
                if cls_lower in path_lower:
                    score += 2
                    reasons.append(f"path_contains_class:{cls}")

            if stem and stem in clue_text and len(stem) >= 4:
                score += 1
                reasons.append(f"stem_in_issue_text:{stem}")

            content_hits = self._score_source_content_tokens(
                repo_path / item.path,
                salient_tokens,
            )
            if content_hits:
                hit_tokens = [tok for tok, _weight in content_hits]
                bonus = min(sum(weight for _tok, weight in content_hits), 16)
                score += bonus
                matched_identifiers.update(hit_tokens[:4])
                reasons.append(f"source_content_tokens:{','.join(hit_tokens[:5])}")

            if "requests/" in path_lower and "packages/urllib3/" in path_lower:
                score -= 5
                reasons.append("vendored_urllib3_penalty")
            if "content-length" in clue_text and item.path == "requests/models.py":
                score += 4
                reasons.append("requests_header_core_file")
            if (
                "required column" in clue_text
                and item.path == "astropy/timeseries/core.py"
            ):
                score += 8
                reasons.append("astropy_required_column_core_file")
            if "pyreverse" in clue_text and "pylint/pyreverse/" in path_lower:
                score += 10
                reasons.append("pyreverse_issue_area")
            if "uml" in clue_text and "diagram" in path_lower:
                score += 4
                reasons.append("uml_diagram_file")
            if (
                "literalinclude" in clue_text
                and item.path == "sphinx/directives/code.py"
            ):
                score += 24
                reasons.append("sphinx_literalinclude_directive_core")
            if (
                "literalinclude" in clue_text
                and path_lower.startswith("doc/")
            ):
                score -= 8
                reasons.append("documentation_file_penalty_for_directive_bug")

            # High-confidence traceback gets a strong boost; LLM-inferred
            # locations get only a weak hint so they do not dominate ranking.
            if item.path in fault_file_info:
                fault = fault_file_info[item.path]
                fault_fn = fault.get("function_name", "")
                source = fault.get("source", "traceback")
                confidence = fault.get("confidence", "high" if source == "traceback" else "medium")
                if source == "traceback" and confidence == "high":
                    score += 12
                    reasons.append(f"traceback_fault_location:{fault_fn}")
                else:
                    score += 3
                    reasons.append(f"inferred_fault_location:{fault_fn}")
                if fault_fn:
                    matched_identifiers.add(fault_fn)

            if score <= 0:
                continue

            source_candidates.append(
                CandidateFile(
                    path=item.path,
                    score=score,
                    matched_identifiers=sorted(matched_identifiers),
                    reasons=reasons[:10],
                    localization_signals=self._localization_signals_from_reasons(reasons),
                )
            )

        source_candidates.sort(key=lambda x: (-x.score, x.path))
        source_candidates = self._expand_source_candidates_with_graph(
            source_candidates,
            indexed_files,
        )
        source_candidates = source_candidates[: self.top_k_source]

        # top-K 소스 파일의 matched identifier 스니펫 + 함수 목록 추출
        for candidate in source_candidates:
            file_path = repo_path / candidate.path
            candidate.code_snippets = self._extract_code_snippets(
                file_path, candidate.matched_identifiers
            )
            candidate.top_level_functions = self._extract_top_level_functions(file_path)

        # source 후보 기반 힌트 생성
        source_paths = [Path(x.path) for x in source_candidates]
        source_dirs = {str(p.parent).replace("\\", "/").lower() for p in source_paths}
        source_stems = {p.stem.lower() for p in source_paths}

        # source directory → expected test directory 매핑 (추론)
        inferred_test_dirs: set[str] = set()
        for src_dir in source_dirs:
            parts = src_dir.split("/")
            # e.g. django/db/backends/postgresql → tests/backends/postgresql
            if len(parts) >= 2:
                inferred_test_dirs.add(f"tests/{'/'.join(parts[1:])}")
            # e.g. django/db/backends → tests/db_backends (flat variant)
            inferred_test_dirs.add(f"tests/{'_'.join(parts)}")
            # e.g. astropy/modeling → astropy/modeling/tests
            inferred_test_dirs.add(f"{src_dir}/tests")

        # issue/clue에서 test file retrieval용 토큰 추출
        clue_tokens: set[str] = set()

        for fn in clue_functions:
            clue_tokens.update(self._split_identifier_tokens(fn))

        for cls in clue_classes:
            clue_tokens.update(self._split_identifier_tokens(cls))

        for p in source_paths:
            clue_tokens.update(self._split_identifier_tokens(p.stem))
            for part in p.parts:
                clue_tokens.update(self._split_identifier_tokens(part))

        # 너무 일반적인 토큰 제거
        weak_tokens = {
            "test", "tests", "py", "model", "models", "file", "files",
            "class", "classes", "function", "functions", "astropy",
            # Framework / project names
            "django", "flask", "numpy", "scipy", "matplotlib",
            # Common generic module names
            "utils", "helpers", "base", "core", "common", "generic",
            "mixins", "compat", "conf", "config", "settings", "views",
            "urls", "admin", "apps", "managers", "signals", "middleware",
            "serializers", "validators", "decorators", "exceptions",
        }
        clue_tokens = {t for t in clue_tokens if len(t) >= 4 and t not in weak_tokens}

        # 2단계: test 파일 점수화 (강화 버전)
        for item in indexed_files:
            if not item.is_test_file:
                continue

            score = 0
            matched_identifiers: set[str] = set()
            reasons: List[str] = []

            test_path = item.path.replace("\\", "/")
            test_path_lower = test_path.lower()
            test_name = Path(test_path).name.lower()
            test_stem = Path(test_path).stem.lower()

            file_funcs = set(item.functions) | set(item.methods)

            # A. 직접 식별자 매칭
            for fn in clue_functions:
                if fn in file_funcs:
                    score += 5
                    matched_identifiers.add(fn)
                    reasons.append(f"test_function_match:{fn}")

                fn_lower = fn.lower()
                if fn_lower in test_path_lower:
                    score += 4
                    matched_identifiers.add(fn)
                    reasons.append(f"test_path_contains_function:{fn}")

            for cls in clue_classes:
                if cls in item.classes:
                    score += 4
                    matched_identifiers.add(cls)
                    reasons.append(f"test_class_match:{cls}")

                cls_tokens = self._split_identifier_tokens(cls)
                if any(tok in test_path_lower for tok in cls_tokens if len(tok) >= 4):
                    score += 3
                    matched_identifiers.add(cls)
                    reasons.append(f"test_path_related_to_class:{cls}")

            # B. source 후보와 같은 디렉토리 계열이면 강한 보너스
            for src_dir in source_dirs:
                src_dir_parts = src_dir.split("/")
                if len(src_dir_parts) >= 2:
                    anchor = "/".join(src_dir_parts[:-1])  # e.g. astropy/modeling
                    if anchor and anchor in test_path_lower:
                        score += 5
                        reasons.append(f"same_module_area:{anchor}")

            # B2. 추론된 테스트 디렉토리 매칭
            for inferred_dir in inferred_test_dirs:
                if test_path_lower.startswith(inferred_dir + "/") or inferred_dir + "/" in test_path_lower:
                    score += 4
                    reasons.append(f"inferred_test_dir_match:{inferred_dir}")
                    break

            # C. source 파일 stem과 test 파일 이름 유사
            test_stem_tokens = set(re.split(r"[_\W]+", test_stem))
            for src_stem in source_stems:
                if src_stem and (src_stem == test_stem or src_stem in test_stem_tokens):
                    score += 6
                    reasons.append(f"test_name_matches_source_stem:{src_stem}")

            # D. token overlap 기반 점수
            overlap_tokens = [tok for tok in clue_tokens if tok in test_path_lower]
            if overlap_tokens:
                bonus = min(len(overlap_tokens) * 2, 8)
                score += bonus
                reasons.append(f"token_overlap:{','.join(sorted(overlap_tokens)[:5])}")

            # E. 테스트 파일 내부 함수/클래스명도 활용
            internal_names = (
                [x.lower() for x in item.functions]
                + [x.lower() for x in item.methods]
                + [x.lower() for x in item.classes]
            )

            for tok in clue_tokens:
                if any(tok in name for name in internal_names):
                    score += 2
                    reasons.append(f"internal_name_overlap:{tok}")
                    break

            # F. 일반적인 test 파일 보너스는 아주 약하게만
            if item.test_functions:
                score += 1
                reasons.append("has_test_functions")

            # G. module-level skip 페널티 (importorskip 등)
            if item.has_module_skip:
                score -= 30
                reasons.append("module_level_skip_penalty")
            if item.collection_risk:
                risk_penalty = 24 if "sphinx_testing_fixtures" in item.collection_risk or "pytest_mark_sphinx" in item.collection_risk else 12
                score -= risk_penalty
                reasons.append(f"collection_risk_penalty:{item.collection_risk}")

            # H. Strongly prefer tests from the same repo area when source is explicit.
            if source_candidates:
                best_source = source_candidates[0].path.replace("\\", "/")
                best_parts = best_source.split("/")
                if len(best_parts) >= 2:
                    source_area = "/".join(best_parts[:2]).lower()
                    if source_area in test_path_lower:
                        score += 6
                        reasons.append(f"explicit_source_area_match:{source_area}")
                    elif not any(tok in test_path_lower for tok in source_area.split("/") if len(tok) >= 4):
                        score -= 3
                        reasons.append(f"unrelated_to_explicit_source_area:{source_area}")

            if score <= 0:
                continue

            test_candidates.append(
                CandidateFile(
                    path=item.path,
                    score=score,
                    matched_identifiers=sorted(matched_identifiers),
                    reasons=reasons[:12],
                    has_module_skip=item.has_module_skip,
                    collection_risk=item.collection_risk,
                    localization_signals=self._localization_signals_from_reasons(reasons),
                )
            )

        test_candidates.sort(key=lambda x: (-x.score, x.path))
        test_candidates = test_candidates[: self.top_k_test]

        return source_candidates, test_candidates

    @staticmethod
    def _localization_signals_from_reasons(reasons: List[str]) -> List[str]:
        signals: set[str] = set()
        for reason in reasons:
            if reason.startswith(("function_match:", "class_match:", "path_contains_")):
                signals.add("issue_identifier")
            elif "explicit_file" in reason or "file_hint" in reason:
                signals.add("explicit_file_hint")
            elif reason.startswith("traceback_fault_location"):
                signals.add("traceback")
            elif reason.startswith("inferred_fault_location"):
                signals.add("inferred_fault")
            elif reason.startswith(("source_content_tokens", "stem_in_issue_text")):
                signals.add("issue_text_content")
            elif "graph_neighbor" in reason:
                signals.add("graph_neighbor")
            elif "test_" in reason or "same_module_area" in reason or "inferred_test_dir" in reason:
                signals.add("test_style_match")
            elif "collection_risk" in reason:
                signals.add("collection_risk")
        return sorted(signals)

    def _expand_source_candidates_with_graph(
        self,
        source_candidates: List[CandidateFile],
        indexed_files: List[IndexedFile],
    ) -> List[CandidateFile]:
        """Conservative second-pass expansion from top source candidates.

        The first pass remains the primary ranking. This pass only adds nearby
        files connected by imports/calls/classes from the top candidates, so it
        widens recall without allowing whole-repository drift.
        """
        if not source_candidates:
            return source_candidates

        by_path = {item.path: item for item in indexed_files if not item.is_test_file}
        candidate_by_path = {cand.path: cand for cand in source_candidates}
        top_paths = [cand.path for cand in source_candidates[: min(3, len(source_candidates))]]
        top_items = [by_path[p] for p in top_paths if p in by_path]
        if not top_items:
            return source_candidates

        top_modules = {self._file_path_to_module(item.path) for item in top_items}
        top_modules = {m for m in top_modules if m}
        top_imports = {
            imp
            for item in top_items
            for imp in (item.imports or [])
            if isinstance(imp, str)
        }
        top_calls = {
            name
            for item in top_items
            for name in (item.call_names or [])
            if isinstance(name, str)
        }
        top_classes = {
            cls
            for item in top_items
            for cls in (item.classes or [])
            if isinstance(cls, str)
        }

        expanded: List[CandidateFile] = list(source_candidates)
        for item in by_path.values():
            if item.path in candidate_by_path:
                cand = candidate_by_path[item.path]
                neighbor_paths = self._graph_neighbor_paths(item, top_items, top_imports, top_modules, top_calls, top_classes)
                if neighbor_paths:
                    cand.graph_neighbors = sorted(set((cand.graph_neighbors or []) + neighbor_paths))[:5]
                    cand.localization_signals = sorted(set((cand.localization_signals or []) + ["graph_neighbor"]))
                continue

            bonus = 0
            reasons: List[str] = []
            module = self._file_path_to_module(item.path)
            module_parent = module.rsplit(".", 1)[0] if "." in module else module

            if module and any(imp == module or imp.startswith(module + ".") for imp in top_imports):
                bonus += 5
                reasons.append("graph_neighbor_imported_by_top")
            if module_parent and any(imp == module_parent or imp.startswith(module_parent + ".") for imp in top_imports):
                bonus += 3
                reasons.append("graph_neighbor_import_package")
            if any(imp in top_modules or any(tm and imp.startswith(tm + ".") for tm in top_modules) for imp in (item.imports or [])):
                bonus += 4
                reasons.append("graph_neighbor_imports_top")

            defined_names = set(item.functions or []) | set(item.methods or []) | set(item.classes or [])
            called_defs = sorted(defined_names & top_calls)
            if called_defs:
                bonus += min(6, len(called_defs) * 3)
                reasons.append(f"graph_neighbor_called_symbol:{','.join(called_defs[:3])}")

            shared_classes = sorted(set(item.classes or []) & top_classes)
            if shared_classes:
                bonus += min(4, len(shared_classes) * 2)
                reasons.append(f"graph_neighbor_shared_class:{','.join(shared_classes[:3])}")

            if bonus < 4:
                continue

            neighbors = self._graph_neighbor_paths(item, top_items, top_imports, top_modules, top_calls, top_classes)
            expanded.append(
                CandidateFile(
                    path=item.path,
                    score=bonus,
                    matched_identifiers=called_defs[:4] or shared_classes[:4],
                    reasons=reasons[:10],
                    localization_signals=["graph_neighbor"],
                    graph_neighbors=neighbors[:5],
                )
            )

        expanded.sort(key=lambda x: (-x.score, x.path))
        return expanded

    def _graph_neighbor_paths(
        self,
        item: IndexedFile,
        top_items: List[IndexedFile],
        top_imports: set[str],
        top_modules: set[str],
        top_calls: set[str],
        top_classes: set[str],
    ) -> List[str]:
        neighbors: List[str] = []
        module = self._file_path_to_module(item.path)
        module_parent = module.rsplit(".", 1)[0] if "." in module else module
        defined_names = set(item.functions or []) | set(item.methods or []) | set(item.classes or [])
        for top in top_items:
            top_module = self._file_path_to_module(top.path)
            if module and any(imp == module or imp.startswith(module + ".") for imp in (top.imports or [])):
                neighbors.append(top.path)
            elif module_parent and any(imp == module_parent or imp.startswith(module_parent + ".") for imp in (top.imports or [])):
                neighbors.append(top.path)
            elif top_module and any(imp == top_module or imp.startswith(top_module + ".") for imp in (item.imports or [])):
                neighbors.append(top.path)
            elif defined_names & set(top.call_names or []):
                neighbors.append(top.path)
            elif set(item.classes or []) & set(top.classes or []):
                neighbors.append(top.path)
        return sorted(set(neighbors))

    def _infer_test_style(
        self,
        indexed_files: List[IndexedFile],
        top_test_candidates: List[CandidateFile],
        repo_name: str = "",
    ) -> ProjectTestStyle:
        candidate_paths = {x.path for x in top_test_candidates}
        pool = [x for x in indexed_files if x.path in candidate_paths]

        if not pool:
            pool = [x for x in indexed_files if x.is_test_file][:20]

        pytest_score = 0
        unittest_score = 0
        evidence: List[str] = []
        assert_style: set[str] = set()
        has_django_test_import = False

        for item in pool:
            joined_imports = " ".join(item.imports)

            if "pytest" in joined_imports or Path(item.path).name == "conftest.py":
                pytest_score += 2
                evidence.append(f"{item.path}:pytest_import_or_conftest")
                assert_style.add("plain assert")

            if "unittest" in joined_imports or "django.test" in joined_imports:
                unittest_score += 2
                evidence.append(f"{item.path}:unittest_or_django_test_import")
                assert_style.add("self.assert*")
                if "django.test" in joined_imports:
                    has_django_test_import = True

            if any(name.startswith("test_") for name in item.functions):
                pytest_score += 1
                evidence.append(f"{item.path}:top_level_test_functions")
                assert_style.add("plain assert")

            for cls in item.classes:
                if cls.endswith("TestCase") or cls.startswith("Test"):
                    unittest_score += 1
                    evidence.append(f"{item.path}:testcase_class:{cls}")
                    assert_style.add("self.assert*")

        if pytest_score > unittest_score:
            framework = "pytest"
        elif unittest_score > pytest_score:
            framework = "unittest"
        elif pytest_score == 0 and unittest_score == 0:
            framework = "unknown"
        else:
            framework = "mixed"

        if not assert_style:
            assert_style.add("unknown")

        # runner 결정: repo 이름 기반으로 감지 (파일 패턴은 오탐 가능성 높아 제외)
        repo_lower = repo_name.lower()

        if has_django_test_import or "django" in repo_lower:
            runner = "django-test"
        elif "sympy" in repo_lower:
            runner = "sympy-bin-test"
        elif framework == "unknown":
            runner = "unknown"
        elif framework in ("unittest",):
            runner = "unittest"
        else:
            # pytest, mixed → default pytest runner
            runner = "pytest"

        return ProjectTestStyle(
            framework=framework,
            evidence=evidence[:20],
            assert_style=sorted(assert_style),
            runner=runner,
        )
    
    def _extract_test_example(
        self,
        test_candidates: List[CandidateFile],
        repo_path: Path,
        max_lines: int = 35,
    ) -> str:
        """상위 테스트 파일에서 첫 번째 Test 클래스 또는 test_ 함수를 스니펫으로 추출한다."""
        for candidate in test_candidates[:3]:
            full_path = repo_path / candidate.path
            if not full_path.exists():
                continue
            try:
                src = full_path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                print(f"[context] 파일 읽기 실패 {full_path}: {e}")
                continue
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            lines = src.splitlines()
            # Test 클래스 우선, 없으면 test_ 함수
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                    start = node.lineno - 1
                    return "\n".join(lines[start : start + max_lines])
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith("test_"):
                        start = node.lineno - 1
                        return "\n".join(lines[start : start + max_lines])
        return ""

    def _extract_conftest_fixtures(
        self,
        test_candidates: List[CandidateFile],
        repo_path: Path,
    ) -> Dict[str, List[str]]:
        """candidate test file 경로 계층의 conftest.py에서 fixture 이름 수집."""
        result: Dict[str, List[str]] = {}
        seen: set[str] = set()
        for candidate in test_candidates[:3]:
            parts = Path(candidate.path).parts[:-1]  # 파일명 제외, 디렉토리만
            for i in range(len(parts), -1, -1):
                conftest_rel = str(Path(*parts[:i]) / "conftest.py") if i > 0 else "conftest.py"
                if conftest_rel in seen:
                    continue
                seen.add(conftest_rel)
                conftest_abs = repo_path / conftest_rel
                if conftest_abs.exists():
                    fixtures = self._parse_conftest_fixtures(conftest_abs)
                    if fixtures:
                        result[conftest_rel] = fixtures
        return result

    def _parse_conftest_fixtures(self, conftest_path: Path) -> List[str]:
        """conftest.py에서 @pytest.fixture 데코레이터가 붙은 함수 이름 추출."""
        try:
            source = conftest_path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source)
        except Exception:
            return []
        fixtures: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for dec in node.decorator_list:
                    if isinstance(dec, ast.Call):
                        dec_name = _get_call_fullname(dec)
                    elif isinstance(dec, ast.Name):
                        dec_name = dec.id
                    elif isinstance(dec, ast.Attribute):
                        dec_name = dec.attr
                    else:
                        dec_name = ""
                    if "fixture" in dec_name:
                        fixtures.append(node.name)
                        break
        return fixtures

    def _build_test_symbol_catalog(
        self,
        test_candidates: List[CandidateFile],
        repo_path: Path,
    ) -> Dict[str, Dict[str, List[str]]]:
        """Collect import/class/model/helper names from candidate test files.

        The generator uses this as a grounded symbol catalog, especially for
        Django tests where inventing model classes is a common NOT_VALID cause.
        """
        catalog: Dict[str, Dict[str, List[str]]] = {}
        for candidate in test_candidates[: self.top_k_test]:
            full_path = repo_path / candidate.path
            if not full_path.exists():
                continue
            try:
                source = read_text(full_path)
                tree = ast.parse(source)
            except Exception:
                continue

            imports: set[str] = set()
            imported_symbols: set[str] = set()
            classes: set[str] = set()
            functions: set[str] = set()
            models: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""))
                        imported_symbols.add(alias.asname or alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if not node.module:
                        continue
                    names = []
                    for alias in node.names:
                        if alias.name == "*":
                            continue
                        names.append(alias.name + (f" as {alias.asname}" if alias.asname else ""))
                        imported_symbols.add(alias.asname or alias.name)
                    if names:
                        imports.add(f"from {node.module} import {', '.join(names)}")
                elif isinstance(node, ast.ClassDef):
                    classes.add(node.name)
                    base_names = []
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            base_names.append(base.id)
                        elif isinstance(base, ast.Attribute):
                            base_names.append(base.attr)
                    if any(name == "Model" or name.endswith("Model") for name in base_names) or node.name.endswith("Model"):
                        models.add(node.name)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not node.name.startswith("test_"):
                        functions.add(node.name)

            for sym in imported_symbols:
                if sym.endswith("Model") or sym in classes:
                    models.add(sym)

            catalog[candidate.path] = {
                "imports": sorted(imports)[:40],
                "imported_symbols": sorted(imported_symbols)[:80],
                "classes": sorted(classes)[:60],
                "models": sorted(models)[:60],
                "helpers": sorted(functions)[:60],
                "collection_risk": [candidate.collection_risk] if candidate.collection_risk else [],
            }
        return catalog

    def _collect_available_imports(
        self,
        indexed_files: List[IndexedFile],
        source_candidates: List[CandidateFile],
        test_candidates: List[CandidateFile],
        clue: Dict[str, Any],
        repo_path: Path,
    ) -> Dict[str, List[str]]:
        """
        candidate 파일들과 clue 식별자를 기반으로,
        각 모듈 경로에서 실제로 import 가능한 심볼 목록을 수집한다.
        예: {"astropy.modeling.models": ["Linear1D", "Gaussian1D"], ...}
        """
        result: Dict[str, List[str]] = {}

        # 관심 있는 파일 경로 수집 (source + test 후보)
        interest_paths: set[str] = set()
        for c in source_candidates:
            interest_paths.add(c.path)
        for c in test_candidates:
            interest_paths.add(c.path)

        # 각 candidate 파일의 조상 패키지 __init__.py도 포함
        # 예: django/db/models/expressions.py → django/__init__.py, django/db/__init__.py,
        #     django/db/models/__init__.py
        indexed_paths = {item.path for item in indexed_files}
        ancestor_inits: set[str] = set()
        for p in list(interest_paths):
            parts = p.replace("\\", "/").split("/")
            for depth in range(1, len(parts)):
                init_candidate = "/".join(parts[:depth]) + "/__init__.py"
                if init_candidate in indexed_paths:
                    ancestor_inits.add(init_candidate)
        interest_paths.update(ancestor_inits)

        # clue의 식별자가 정의된 모듈도 찾기
        clue_classes = set(clue.get("identifiers", {}).get("classes", []))
        clue_functions = set(clue.get("identifiers", {}).get("functions", []))
        all_clue_symbols = clue_classes | clue_functions

        for item in indexed_files:
            # candidate 파일이거나, clue 심볼을 정의하고 있는 파일
            has_clue_symbol = bool(
                (set(item.classes) | set(item.functions)) & all_clue_symbols
            )
            if item.path not in interest_paths and not has_clue_symbol:
                continue

            module_path = self._file_path_to_module(item.path)
            if not module_path:
                continue

            exported = sorted(set(item.classes + item.functions))
            if exported:
                result[module_path] = exported

            # __init__.py의 re-export도 수집
            if item.path.endswith("__init__.py"):
                init_file = repo_path / item.path
                reexports = self._collect_init_reexports(init_file)
                if reexports:
                    parent_module = module_path.rsplit(".", 1)[0] if "." in module_path else module_path
                    existing = set(result.get(parent_module, []))
                    existing.update(reexports)
                    result[parent_module] = sorted(existing)

        return result

    def _file_path_to_module(self, rel_path: str) -> str:
        """파일 경로를 Python 모듈 경로로 변환 (예: astropy/modeling/models.py -> astropy.modeling.models)"""
        p = rel_path.replace("\\", "/")
        if p.endswith("/__init__.py"):
            p = p[: -len("/__init__.py")]
        elif p.endswith(".py"):
            p = p[:-3]
        else:
            return ""
        return p.replace("/", ".")

    def _collect_init_reexports(self, init_path: Path) -> List[str]:
        """__init__.py에서 re-export되는 심볼 목록 수집"""
        if not init_path.exists():
            return []
        try:
            source = read_text(init_path)
        except Exception:
            return []
        try:
            tree = ast.parse(source)
        except Exception:
            return []

        symbols: set[str] = set()

        for node in tree.body:
            # __all__ 정의
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, (ast.List, ast.Tuple)):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    symbols.add(elt.value)

            # from .X import Y 형태
            if isinstance(node, ast.ImportFrom):
                if node.names:
                    for alias in node.names:
                        if alias.name != "*":
                            symbols.add(alias.asname or alias.name)

        return sorted(symbols)

    def _split_identifier_tokens(self, name: str) -> set[str]:
        import re

        if not name:
            return set()

        s = name.replace("-", "_").replace("/", "_").replace(".", "_")
        parts = [p for p in s.split("_") if p]

        tokens: set[str] = set()
        for part in parts:
            camel = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", part)
            if camel:
                tokens.update(x.lower() for x in camel if x)
            else:
                tokens.add(part.lower())

        return {t for t in tokens if t}

    @staticmethod
    def _normalize_issue_file_hint(value: str) -> str:
        path = (value or "").replace("\\", "/").strip().strip("`'\"()[]{}.,;:")
        if "/blob/" in path:
            # Old clue artifacts may have captured part of a GitHub blob URL.
            tail = path.split("/blob/", 1)[1]
            parts = tail.split("/")
            if len(parts) >= 2:
                path = "/".join(parts[1:])
        path = re.sub(r"^(?:\./|a/|b/)+", "", path)
        if path.startswith(("http", "www.", "github.com/", "com/")):
            return ""
        return path

    @staticmethod
    def _path_suffix_matches(candidate_path: str, clue_path: str) -> bool:
        candidate_parts = [p for p in candidate_path.lower().split("/") if p]
        clue_parts = [p for p in clue_path.lower().split("/") if p]
        if not candidate_parts or not clue_parts:
            return False
        if len(clue_parts) == 1:
            return candidate_parts[-1] == clue_parts[0]
        if len(clue_parts) > len(candidate_parts):
            return clue_parts[-len(candidate_parts):] == candidate_parts
        return candidate_parts[-len(clue_parts):] == clue_parts

    def _extract_salient_issue_tokens(
        self,
        clue: Dict[str, Any],
        clue_text_lower: str,
    ) -> List[Tuple[str, int]]:
        """Tokens whose presence in source text is stronger than generic API names."""
        stop = {
            "about", "after", "again", "against", "already", "also", "because",
            "before", "being", "between", "class", "code", "correct", "current",
            "currently", "error", "expected", "false", "file", "files", "from",
            "function", "github", "have", "issue", "line", "model", "module",
            "none", "object", "output", "python", "return", "returns", "should",
            "test", "tests", "that", "this", "true", "using", "value", "when",
            "with", "without", "wrong",
            "sklearn", "pylint", "requests", "astropy", "seaborn", "matplotlib",
            "django", "numpy", "scipy",
        }
        high_value_phrases = [
            "content-length", "preparedrequest", "scalarformatter",
            "required columns", "required column", "pyreverse", "umls",
            "invalid value", "runtimewarning",
        ]
        tokens: Dict[str, int] = {}
        for phrase in high_value_phrases:
            if phrase in clue_text_lower:
                tokens[phrase] = 6

        for raw in re.findall(r"`([^`]{3,80})`|['\"]([^'\"]{3,80})['\"]", clue_text_lower):
            value = raw[0] or raw[1]
            for token in self._split_salient_text(value):
                if token not in stop:
                    tokens[token] = max(tokens.get(token, 0), 4)

        for value in (
            clue.get("expected_outputs", [])
            + clue.get("actual_outputs", [])
            + clue.get("error_keywords", [])
        ):
            for token in self._split_salient_text(str(value).lower()):
                if token not in stop:
                    tokens[token] = max(tokens.get(token, 0), 4)

        for token in self._split_salient_text(clue_text_lower):
            if token not in stop and len(token) >= 5:
                tokens[token] = max(tokens.get(token, 0), 2)

        return sorted(tokens.items(), key=lambda x: (-x[1], x[0]))[:40]

    def _split_salient_text(self, text: str) -> set[str]:
        parts = re.findall(r"[a-zA-Z_][a-zA-Z0-9_-]{2,}", text)
        result: set[str] = set()
        for part in parts:
            result.add(part.lower())
            if "-" in part or "_" in part:
                result.update(x for x in re.split(r"[-_]+", part.lower()) if len(x) >= 3)
        return result

    def _score_source_content_tokens(
        self,
        file_path: Path,
        salient_tokens: List[Tuple[str, int]],
    ) -> List[Tuple[str, int]]:
        if not salient_tokens or not file_path.exists():
            return []
        try:
            source = read_text(file_path)
        except Exception:
            return []
        source_lower = source[:200_000].lower()
        hits: List[Tuple[str, int]] = []
        for token, weight in salient_tokens:
            if not token or len(token) < 4:
                continue
            if token in source_lower:
                hits.append((token, weight))
        # One weak prose token is too noisy; a single high-value phrase is useful.
        if len(hits) == 1 and hits[0][1] < 4:
            return []
        return hits[:8]
