from __future__ import annotations

import ast
import copy
import difflib
import json
import logging
import re
import subprocess
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.models.client import LLMClient
from src.models.config import load_model_config
from src.utils.artifact_hash import sha256_text
from src.utils.file_io import read_text
from src.utils.scenario_utils import select_primary_scenario

logger = logging.getLogger(__name__)

_PROMPT_RAW_ISSUE_CHARS = 500
_PROMPT_CODE_EXAMPLES_MAX = 2
_PROMPT_CODE_CHARS = 350
_PROMPT_INTERACTIVE_CHARS = 180
_PROMPT_OUTPUTS_MAX = 2
_PROMPT_OUTPUT_CHARS = 180
_PROMPT_TEST_EXAMPLE_CHARS = 300
_PROMPT_IMPORT_MODULES = 8
_PROMPT_IMPORT_SYMBOLS = 8
_PROMPT_EXISTING_IMPORTS_CHARS = 600
_PROMPT_EXISTING_SYMBOLS = 25
_PROMPT_ORACLE_TEXT_CHARS = 450
_PROMPT_ORACLE_HINTS_MAX = 4
_PROMPT_ORACLE_HINT_CHARS = 180
_PROMPT_CONFTEST_PATHS = 4
_PROMPT_CONFTEST_FIXTURES = 8
_RETRY_PREVIOUS_RESPONSE_CHARS = 600
_RETRY_ERROR_ITEMS_MAX = 6
_RETRY_ERROR_CHARS = 180
_RETRY_TASK_SUMMARY_CHARS = 1200


@dataclass
class GeneratedReproductionTest:
    instance_id: str
    scenario_id: str
    model_name: str
    repo_path: str
    target_test_file: str
    target_test_file_abspath: str
    target_source_file: str
    insert_mode: str
    insertion_hint: str
    imports: List[str]
    test_code: str
    original_test_file_content: str
    modified_test_file_content: str
    test_patch: str
    raw_response: str
    prompt: str
    patch_sha256: str = ""
    repair_attempted: bool = False
    repair_actions: List[str] = None
    repair_failed_reason: str = ""
    repair_retry_count: int = 0
    retry_required_oracle_risks: List[str] = None
    semantic_risk_flags: List[str] = None
    prompt_profile: Dict[str, Any] = None
    # LLM 토큰 사용량 (API 응답 기준, 누적)
    token_usage: Dict[str, int] = None

    def __post_init__(self):
        if self.token_usage is None:
            self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if self.repair_actions is None:
            self.repair_actions = []
        if self.retry_required_oracle_risks is None:
            self.retry_required_oracle_risks = []
        if self.semantic_risk_flags is None:
            self.semantic_risk_flags = []
        if self.prompt_profile is None:
            self.prompt_profile = {}
        if not self.patch_sha256 and self.test_patch is not None:
            self.patch_sha256 = sha256_text(self.test_patch)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    fixed_imports: Optional[List[str]] = None


def _fix_django_imports(imports: List[str]) -> List[str]:
    """unittest.TestCase → django.test.TestCase 교정, relative import 제거."""
    result = []
    has_django_test = any("from django.test import" in imp for imp in imports)
    for imp in imports:
        if imp.strip().startswith("from ."):
            continue  # relative import 제거
        if "from unittest import" in imp or imp.strip() in ("import unittest", "import unittest.TestCase"):
            if not has_django_test:
                result.append("from django.test import TestCase, SimpleTestCase")
                has_django_test = True
            continue
        result.append(imp)
    return result


def _fix_sphinx_test_code(test_code: str) -> str:
    """sphinx 생성 테스트에서 sphinx.testing.fixtures 의존 패턴 제거.

    @pytest.mark.sphinx 데코레이터는 sphinx.testing.fixtures 플러그인을 로드하는데,
    이 플러그인이 docutils.utils.roman (Python 3.9+에서 제거)을 require → ImportError.
    """
    # @pytest.mark.sphinx(...) 데코레이터 제거 (멀티라인 포함)
    test_code = re.sub(r'@pytest\.mark\.sphinx\([^)]*\)\s*\n', '', test_code)
    test_code = re.sub(
        r'^\s*(?:pytest_plugins\s*=\s*["\']sphinx\.testing\.fixtures["\']|'
        r'from\s+sphinx\.testing\.fixtures\s+import\s+.*|'
        r'import\s+sphinx\.testing\.fixtures.*)\s*$',
        '',
        test_code,
        flags=re.MULTILINE,
    )
    # sphinx fixture 파라미터가 있는 함수 시그니처 교정: def test_xxx(app, ...): → def test_xxx():
    # app, status, warning은 sphinx.testing.fixtures 전용 fixture
    test_code = re.sub(
        r'def (test_\w+)\(\s*(?:app|status|warning)(?:\s*,\s*(?:app|status|warning))*\s*\)\s*:',
        r'def \1():',
        test_code,
    )
    return test_code


def _clip_prompt_text(
    text: Any,
    limit: int,
    section: str = "",
    prompt_profile: Optional[Dict[str, Any]] = None,
) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    if prompt_profile is not None and section:
        prompt_profile.setdefault("truncated_sections", []).append(section)
    return value[:limit].rstrip() + "\n... (truncated)"


def _mark_prompt_section(
    prompt_profile: Optional[Dict[str, Any]],
    name: str,
) -> None:
    if prompt_profile is not None:
        prompt_profile.setdefault("sections_included", []).append(name)


def _compact_list(values: Any, max_items: int, char_limit: int) -> List[str]:
    compacted: List[str] = []
    if not isinstance(values, list):
        return compacted
    for value in values:
        if len(compacted) >= max_items:
            break
        text = str(value).strip()
        if text:
            compacted.append(text[:char_limit])
    return compacted


def _dedup_text_items(items: List[str], limit: Optional[int] = None) -> List[str]:
    result: List[str] = []
    seen = set()
    for item in items:
        text = str(item or "").strip()
        norm = re.sub(r"\s+", " ", text.lower())
        if not text or norm in seen:
            continue
        seen.add(norm)
        result.append(text)
        if limit is not None and len(result) >= limit:
            break
    return result


def _compact_validation_errors(error_message: str) -> str:
    raw_parts = re.split(r";|\n", str(error_message or ""))
    parts = _dedup_text_items(
        [p[:_RETRY_ERROR_CHARS] for p in raw_parts if p.strip()],
        limit=_RETRY_ERROR_ITEMS_MAX,
    )
    return "\n".join(f"- {p}" for p in parts) if parts else "- Unknown validation error"


def _extract_retry_task_summary(original_prompt: str) -> str:
    lines: List[str] = []
    keep_prefixes = (
        "Repository:",
        "Instance ID:",
        "Base Commit:",
        "Observed behavior:",
        "Expected behavior:",
        "Reproduction conditions:",
        "- functions:",
        "- classes:",
        "- error/exception keywords:",
        "Candidate source files:",
        "Candidate test files:",
        "oracle_type:",
        "oracle_source:",
        "rule:",
    )
    for line in str(original_prompt or "").splitlines():
        stripped = line.strip()
        if stripped.startswith(keep_prefixes) or stripped in {
            "[Issue Clue]",
            "[Code Context]",
            "[Oracle Contract — follow this before writing assertions]",
        }:
            lines.append(line)
        if len("\n".join(lines)) >= _RETRY_TASK_SUMMARY_CHARS:
            break
    summary = "\n".join(lines).strip()
    if not summary:
        summary = str(original_prompt or "")[:_RETRY_TASK_SUMMARY_CHARS].strip()
    return summary[:_RETRY_TASK_SUMMARY_CHARS]


def _prioritized_available_imports(
    available_imports: Dict[str, Any],
    clue: Dict[str, Any],
    context: Dict[str, Any],
    scenario: Dict[str, Any],
) -> List[tuple[str, List[str]]]:
    identifiers = (scenario.get("identifiers") or clue.get("identifiers") or {})
    target_location = scenario.get("target_location", {}) or {}
    source_paths = [
        str(target_location.get("source_file") or ""),
        *[str(x.get("path", "")) for x in context.get("candidate_source_files", [])[:3]],
        *[str(x.get("path", "")) for x in context.get("candidate_test_files", [])[:2]],
    ]
    target_terms = {
        str(target_location.get("target_function") or "").lower(),
        *[str(x).lower() for x in identifiers.get("functions", [])[:8]],
        *[str(x).lower() for x in identifiers.get("classes", [])[:8]],
    }
    target_terms = {x for x in target_terms if x}
    path_terms = set()
    for path in source_paths:
        for part in re.split(r"[/.\\_-]+", path.lower()):
            if len(part) >= 3:
                path_terms.add(part)

    ranked: List[tuple[int, str, List[str]]] = []
    for module, symbols in sorted((available_imports or {}).items()):
        if not isinstance(symbols, list):
            symbols = []
        module_l = str(module).lower()
        score = 0
        if any(term and term in module_l for term in path_terms):
            score += 4
        if any(term and term in module_l for term in target_terms):
            score += 3
        symbol_l = {str(s).lower() for s in symbols}
        score += min(4, len(symbol_l & target_terms))
        ranked.append((score, str(module), [str(s) for s in symbols]))

    ranked.sort(key=lambda x: (-x[0], x[1]))
    selected = [item for item in ranked if item[0] > 0][:_PROMPT_IMPORT_MODULES]
    if len(selected) < _PROMPT_IMPORT_MODULES:
        selected.extend(
            item for item in ranked
            if item not in selected
        )
    return [(module, symbols) for _, module, symbols in selected[:_PROMPT_IMPORT_MODULES]]


def _choose_viable_test_file(
    context: Dict[str, Any],
    preferred: str = "",
) -> tuple[str, str]:
    """Return an existing, collection-safe test file and a short reason."""
    repo_path = Path(context.get("repo_path", ""))
    candidates = context.get("candidate_test_files", []) or []

    def viable(path: str, entry: Optional[Dict[str, Any]] = None) -> bool:
        if not path:
            return False
        if repo_path and not (repo_path / path).exists():
            return False
        if entry and (entry.get("has_module_skip") or entry.get("collection_risk")):
            return False
        return True

    preferred_entry = next((c for c in candidates if c.get("path") == preferred), None)
    if viable(preferred, preferred_entry):
        return preferred, ""

    for entry in candidates:
        path = entry.get("path", "")
        if viable(path, entry):
            if not preferred:
                return path, "missing_preferred_test_file"
            if preferred_entry and (preferred_entry.get("has_module_skip") or preferred_entry.get("collection_risk")):
                return path, "preferred_test_file_collection_risk"
            return path, "preferred_test_file_missing_or_unusable"

    return preferred, ""


def _detect_blocking_oracle_risks(code: str, clue: Optional[Dict[str, Any]] = None) -> List[str]:
    """Reject high-risk oracles before spending an alignment iteration."""
    lower = code.lower()
    errors: List[str] = []

    def _has_issue_expected_signal() -> bool:
        expected_outputs = (clue or {}).get("expected_outputs", [])
        norm_code = re.sub(r"\s+", "", lower)
        for out in expected_outputs[:3]:
            norm_expected = re.sub(r"\s+", "", str(out).lower())
            if norm_expected and norm_expected[:80] in norm_code:
                return True
        return False
    has_oracle = bool(re.search(
        r"^\s*(assert\b|self\.assert|with\s+.*raises|pytest\.raises|"
        r".*assert_(?:allclose|array|equal|raises))",
        code,
        re.MULTILINE,
    ))
    if not has_oracle:
        errors.append(
            "CRITICAL: no explicit oracle remains. Add an assertion for the post-fix behavior."
        )
    if re.search(
        r"^(?:\s*)(?:self\.)?assertTrue\s*\(\s*(?:True|1)\s*(?:,\s*[^)]*)?\)\s*(?:#.*)?$"
        r"|^\s*assert\s+(?:True|1)\s*(?:#.*)?$",
        code,
        re.MULTILINE | re.IGNORECASE,
    ):
        errors.append(
            "CRITICAL: trivial oracle detected. Assert the post-fix return value or state change, not True."
        )
    if re.search(r"requests\.(get|post|put|delete|request)\s*\(\s*['\"]https?://", code):
        errors.append(
            "CRITICAL: real network calls are not allowed. Use PreparedRequest, mocks, or local helpers."
        )
    if re.search(r"class\s+\w+\s*\([^)]*models\.Model[^)]*\)", code):
        errors.append(
            "CRITICAL: do not define Django models inside generated tests. Reuse existing test models/imports."
        )
    if re.search(r"float\s*\(\s*['\"]nan['\"]\s*\)|\bnp\.nan\b", lower) and re.search(r"!=|==|assertnot", lower):
        errors.append(
            "CRITICAL: do not compare NaN directly. Use np.isnan(...) or warning behavior."
        )
    if re.search(r"^\s*assert\s+.+==\s*np\.array\s*\(", code, re.MULTILINE):
        errors.append(
            "CRITICAL: do not compare numpy arrays with plain assert ==. Use np.testing.assert_array_equal or assert_allclose."
        )
    if re.search(r"get_[xy]lim\(\)\s*\[\s*[01]\s*\]\s*==", code):
        errors.append(
            "CRITICAL: raw Matplotlib limit equality is brittle. Use ax.xaxis_inverted()/ax.yaxis_inverted() or semantic tick/bin assertions."
        )
    if re.search(
        r"assert\s+str\s*\(\s*[\w.]+\s*\)\s*!=\s*['\"]|"
        r"\w+(?:\.value)?\.args\[\d+\]\s*!=\s*['\"]|"
        r"assert\s+['\"].+['\"]\s+not\s+in\s+str\s*\(|"
        r"self\.assert(?:NotIn|NotRegex)\s*\([^,\n]+,\s*str\s*\(|"
        r"self\.assertNotEqual\s*\(\s*str\s*\(",
        code,
        re.IGNORECASE,
    ):
        errors.append(
            "CRITICAL: do not assert exception message absence/change. Assert the success path or exception type."
        )
    if re.search(
        r"(?:expected|baseline|correct|desired)_(?:matrix|array|result|values?)\s*=.*\n"
        r"(?s:.*?)(?:assert_array_equal|assert_allclose|assert_equal)\s*\([^,\n]+,\s*"
        r"(?:expected|baseline|correct|desired)_(?:matrix|array|result|values?)",
        code,
        re.IGNORECASE,
    ) and not _has_issue_expected_signal():
        errors.append(
            "CRITICAL: guessed expected arrays are brittle. Use issue-stated expected output or a semantic invariant."
        )
    if re.search(
        r"assert(?:In|NotIn)\s*\(\s*['\"][^'\"]{80,}['\"]|"
        r"assert(?:In|NotIn)\s*\(\s*['\"][^'\"]*(?:\\PYG|\\sphinx|<[^>]+>)[^'\"]*['\"]",
        code,
        re.IGNORECASE,
    ):
        errors.append(
            "CRITICAL: raw rendered HTML/LaTeX/Sphinx string oracle is brittle. Use a small semantic marker/invariant."
        )
    retry_risks = _detect_retry_required_oracle_risks(code, clue=clue)
    for risk in retry_risks:
        errors.append(f"CRITICAL: retry required: {risk}")
    return errors


def _issue_says_success_path(clue: Optional[Dict[str, Any]]) -> bool:
    clue = clue or {}
    text = " ".join(
        str(x)
        for x in (
            clue.get("observed_behavior", [])
            + clue.get("expected_behavior", [])
            + clue.get("repro_conditions", [])
            + [clue.get("raw_issue_text", "")]
        )
    ).lower()
    return bool(re.search(
        r"should\s+not\s+(?:raise|error|fail|crash|warn)|"
        r"must\s+not\s+(?:raise|error|fail|crash|warn)|"
        r"does\s+not\s+(?:raise|error|fail|crash|warn)|"
        r"doesn't\s+(?:raise|error|fail|crash|warn)|"
        r"without\s+(?:raising|error|failing|crashing|warning)|"
        r"no\s+(?:exception|error|warning)|"
        r"no\s+longer\s+(?:raises|errors|fails|crashes|warns)",
        text,
    ))


def _issue_says_exception_expected(clue: Optional[Dict[str, Any]]) -> bool:
    clue = clue or {}
    text = " ".join(
        str(x)
        for x in (
            clue.get("observed_behavior", [])
            + clue.get("expected_behavior", [])
            + clue.get("repro_conditions", [])
            + [clue.get("raw_issue_text", "")]
        )
    ).lower()
    return bool(re.search(
        r"should\s+raise|must\s+raise|expected\s+(?:error|exception)|"
        r"should\s+(?:error|fail)\b|raises?\s+(?:a\s+)?(?:typeerror|valueerror|attributeerror|runtimeerror)",
        text,
    ))


def _has_issue_expected_signal(code: str, clue: Optional[Dict[str, Any]]) -> bool:
    expected_outputs = (clue or {}).get("expected_outputs", [])
    norm_code = re.sub(r"\s+", "", code.lower())
    for out in expected_outputs[:3]:
        norm_expected = re.sub(r"\s+", "", str(out).lower())
        if norm_expected and norm_expected[:80] in norm_code:
            return True
    return False


def _detect_retry_required_oracle_risks(
    code: str,
    clue: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Detect oracle patterns that should trigger repair/retry, not final eval.

    This is intentionally patch-free: it uses only generated code and issue clue.
    """
    risks: List[str] = []

    def add(flag: str) -> None:
        if flag not in risks:
            risks.append(flag)

    if re.search(r"@image_comparison", code):
        add("image_comparison_decorator")

    has_raises = bool(re.search(r"pytest\.raises|assertRaises|assert_raises|with\s+.*raises", code))
    has_body_assertion = bool(re.search(
        r"^\s*(assert\s+(?!.*raises)|self\.assert(?:Equal|True|False|Is|In|NotIn|Almost)|np\.testing\.assert)",
        code,
        re.MULTILINE,
    ))
    if has_raises and (
        _issue_says_success_path(clue)
        or (not _issue_says_exception_expected(clue) and re.search(r"post[- ]fix|should\s+accept|fit\s+success|succeed", code, re.IGNORECASE))
    ):
        add("fix_disappearing_exception_oracle")
    if has_raises and not has_body_assertion:
        add("raises_only_no_body_assertion")

    if re.search(
        r"len\s*\(\s*w\s*\)\s*==|"
        r"issubclass\s*\([^)]*(?:Warning|RuntimeWarning)|"
        r"\.category\s*,\s*(?:Warning|RuntimeWarning)|"
        r"assertWarns|pytest\.warns",
        code,
        re.IGNORECASE | re.DOTALL,
    ):
        add("warning_presence_oracle")

    if re.search(
        r"(?:self\.)?assert(?:In|NotIn)\s*\(\s*['\"][^'\"]{80,}['\"]|"
        r"(?:self\.)?assert(?:In|NotIn)\s*\(\s*['\"][^'\"]*(?:\\PYG|\\sphinx|<[^>]+>|latex|html)[^'\"]*['\"]",
        code,
        re.IGNORECASE,
    ):
        add("raw_rendered_output_exact_match")

    # Private attribute reads are often fragile, but treating every read as a
    # hard retry gate causes many otherwise executable regression tests to die
    # at generation time.  Direct private-state assignments are removed by
    # _fix_private_attr_access(); reads are left to alignment/final eval.

    if re.search(
        r"(?:expected|baseline|correct|desired)_(?:matrix|array|result|values?)\s*=.*\n"
        r"(?s:.*?)(?:assert_array_equal|assert_allclose|assert_equal)\s*\([^,\n]+,\s*"
        r"(?:expected|baseline|correct|desired)_(?:matrix|array|result|values?)",
        code,
        re.IGNORECASE,
    ) and not _has_issue_expected_signal(code, clue):
        add("guessed_expected_array")

    if re.search(
        r"(?:expected|baseline|correct|desired)_(?:value|output|result)\s*=.*\n"
        r"(?s:.*?)(?:assert\s+[^=\n]+==\s*|self\.assertEqual\s*\([^,\n]+,\s*)"
        r"(?:expected|baseline|correct|desired)_(?:value|output|result)",
        code,
        re.IGNORECASE,
    ) and not _has_issue_expected_signal(code, clue):
        add("guessed_expected_value")

    if re.search(
        r"(?:expected|baseline|correct|desired|known)_[A-Za-z0-9_]*\s*!=|"
        r"assert\s+repr\s*\(\s*(?:expected|baseline|correct|desired|known)_[A-Za-z0-9_]*\s*\)\s*!=",
        code,
        re.IGNORECASE,
    ):
        add("constant_negative_oracle")

    return risks


def _identifier_terms(clue: Optional[Dict[str, Any]]) -> set[str]:
    clue = clue or {}
    identifiers = clue.get("identifiers", {}) if isinstance(clue.get("identifiers"), dict) else {}
    terms: set[str] = set()
    for key in ("functions", "classes", "exceptions", "files"):
        for value in identifiers.get(key, []) or []:
            text = str(value).lower()
            if len(text) >= 3:
                terms.add(text)
                terms.update(t for t in re.split(r"[_\W]+", text) if len(t) >= 4)
    return terms


def _target_terms(context: Optional[Dict[str, Any]], scenario: Optional[Dict[str, Any]] = None) -> set[str]:
    terms: set[str] = set()
    scenario = scenario or {}
    target = scenario.get("target_location", {}) if isinstance(scenario.get("target_location"), dict) else {}
    paths = [
        target.get("source_file", ""),
        target.get("target_function", ""),
    ]
    for item in (context or {}).get("candidate_source_files", [])[:3]:
        if isinstance(item, dict):
            paths.append(item.get("path", ""))
            paths.extend(item.get("matched_identifiers", []) or [])
    for path in paths:
        text = str(path).lower()
        terms.update(t for t in re.split(r"[/_.\W]+", text) if len(t) >= 4)
    return terms


def _detect_semantic_risk_flags(
    code: str,
    clue: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
    scenario: Optional[Dict[str, Any]] = None,
    original_content: str = "",
) -> List[str]:
    """Detect issue/context-inconsistent generated tests without using patches."""
    flags: List[str] = []

    def add(flag: str) -> None:
        if flag not in flags:
            flags.append(flag)

    lower = code.lower()
    clue_terms = _identifier_terms(clue)
    target_terms = _target_terms(context, scenario)
    allowed_terms = clue_terms | target_terms

    unrelated_sklearn = {
        "countvectorizer",
        "tfidfvectorizer",
        "latentdirichletallocation",
        "lda",
        "kmeans",
        "randomforestclassifier",
        "svc",
    }
    if "sklearn" in lower:
        for symbol in unrelated_sklearn:
            if symbol in lower and symbol not in allowed_terms:
                add("unrelated_api_invention")
                break

    if re.search(r"@unittest\.skip|unittest\.skip\s*\(", code) and re.search(r"\bxxx\b", code):
        add("inline_skipped_class_reproduction")

    if "httpdigestauth" in lower and "www-authenticate" not in lower and "chal" not in lower:
        add("requests_digest_without_challenge")

    if "author.objects" in lower and ".annotate(" in lower and ".order_by(" not in lower:
        add("django_query_warning_unhandled")

    if re.search(r"\b(?:xxx|MockModel|Dummy[A-Za-z0-9_]*|FooModel|BarModel)\b", code):
        add("placeholder_symbol")

    target = (scenario or {}).get("target_location", {}) if isinstance((scenario or {}).get("target_location"), dict) else {}
    target_function = str(target.get("target_function") or "")
    # target_function_not_called gets an explicit rewrite chance in generate().
    # Do not hard-block it here, or valid public-wrapper reproductions become
    # NOT_VALID before alignment can judge them.

    runner = ((context or {}).get("project_test_style") or {}).get("runner", "")
    if runner == "django-test":
        if re.search(r"class\s+\w+\s*\([^)]*models\.Model[^)]*\)", code):
            add("django_inline_model")
        if re.search(r"\b(?:models\.Model|MockModel)\.objects\b|MockModel\._meta\b|\bself\.apps\b|\bapp_label\s*=", code):
            add("django_invalid_model_api")
        known_symbols = _imported_or_existing_symbols(code + "\n" + original_content)
        for model_name in re.findall(r"\b([A-Z][A-Za-z0-9_]+)\.objects\.", code):
            if model_name not in known_symbols:
                add(f"unknown_django_model={model_name}")
                break

    if "sphinx.testing.fixtures" in lower or "pytest_plugins" in lower and "sphinx.testing" in lower:
        add("sphinx_testing_fixture_import")

    return flags


def _imported_or_existing_symbols(code: str) -> set[str]:
    symbols: set[str] = set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return symbols
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                symbols.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    symbols.add(alias.asname or alias.name)
        elif isinstance(node, ast.ClassDef):
            symbols.add(node.name)
    return symbols


def _remove_warning_presence_assertions(code: str) -> str:
    """Remove warning-count/type assertions when another value oracle remains."""
    patterns = [
        r"^\s*assert\s+len\s*\(\s*w\s*\)\s*==\s*\d+\s*$",
        r"^\s*self\.assertEqual\s*\(\s*len\s*\(\s*w\s*\)\s*,\s*\d+\s*\)\s*$",
        r"^\s*assert\s+issubclass\s*\([^)]*(?:Warning|RuntimeWarning)[^)]*\)\s*$",
        r"^\s*self\.assertTrue\s*\(\s*issubclass\s*\([^)]*(?:Warning|RuntimeWarning)[^)]*\)\s*\)\s*$",
    ]
    new_code = code
    for pat in patterns:
        new_code = re.sub(
            pat,
            lambda m: " " * (len(m.group(0)) - len(m.group(0).lstrip())) + "# [removed: warning presence oracle — assert fixed value/state instead]",
            new_code,
            flags=re.MULTILINE | re.IGNORECASE,
        )
    return new_code


def _remove_image_comparison_decorators(code: str) -> str:
    return re.sub(
        r"^\s*@image_comparison\s*\([^)]*\)\s*\n",
        "",
        code,
        flags=re.MULTILINE | re.DOTALL,
    )


def _apply_oracle_repairs(
    code: str,
    clue: Optional[Dict[str, Any]] = None,
) -> tuple[str, List[str]]:
    """Apply deterministic, patch-free oracle repairs to generated code."""
    actions: List[str] = []
    repaired = code

    def changed(name: str, new_code: str) -> None:
        nonlocal repaired
        if new_code != repaired:
            repaired = new_code
            actions.append(name)

    changed("remove_image_comparison_decorator", _remove_image_comparison_decorators(repaired))
    changed("remove_trivial_assertions", _remove_trivial_assertions(repaired))
    changed("remove_exception_message_matching", _fix_exception_message_matching(repaired))
    changed("remove_private_attribute_assignment", _fix_private_attr_access(repaired))
    changed("remove_warning_presence_assertions", _remove_warning_presence_assertions(repaired))
    changed("prune_to_best_generated_test", _prune_to_best_generated_test(repaired, clue))
    changed("fill_empty_exception_blocks", _fill_empty_exception_blocks(repaired))

    return repaired, actions


def _fill_empty_exception_blocks(code: str) -> str:
    """Insert `pass` into exception/raises blocks emptied by static repairs."""
    lines = code.splitlines()
    if not lines:
        return code
    block_header = re.compile(
        r"^(?:except\b.*|with\s+(?:pytest\.raises|self\.assertRaises|assert_raises)\b.*|try)\s*:\s*(?:#.*)?$"
    )
    result: List[str] = []
    for idx, line in enumerate(lines):
        result.append(line)
        stripped = line.strip()
        if not block_header.match(stripped):
            continue
        indent = len(line) - len(line.lstrip())
        next_meaningful = ""
        next_indent = -1
        for nxt in lines[idx + 1:]:
            nxt_stripped = nxt.strip()
            if not nxt_stripped or nxt_stripped.startswith("#"):
                continue
            next_meaningful = nxt_stripped
            next_indent = len(nxt) - len(nxt.lstrip())
            break
        if not next_meaningful or next_indent <= indent:
            result.append(" " * (indent + 4) + "pass")
    return "\n".join(result)


def _has_explicit_oracle(code: str) -> bool:
    return bool(re.search(
        r"^\s*(assert\b|self\.assert|with\s+.*raises|pytest\.raises|"
        r".*assert_(?:allclose|array|equal|raises))",
        code or "",
        re.MULTILINE,
    ))


def _inject_tier2_assertion(code: str, actual_output: str) -> str:
    """probe로 수집한 buggy_value를 코드 레벨 Tier 2 assertion으로 주입.

    LLM의 assertion 품질과 무관하게 `assert repr(result) != BUGGY_REPR` 구문을
    결정적으로 삽입한다.
    """
    if not actual_output or not code.strip():
        return code

    lines = code.splitlines()

    # except/with 블록 범위 계산 (이 안에서는 result_var가 미정의일 수 있음)
    except_ranges: list = []
    current_except_start: int = -1
    current_except_indent: int = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if re.match(r'except\b', stripped) or re.match(r'with\s+.*raises', stripped):
            current_except_start = i
            current_except_indent = indent
        elif current_except_start >= 0 and stripped and indent <= current_except_indent:
            except_ranges.append((current_except_start, i - 1))
            current_except_start = -1
    if current_except_start >= 0:
        except_ranges.append((current_except_start, len(lines) - 1))

    def _in_except_block(idx: int) -> bool:
        return any(s <= idx <= e for s, e in except_ranges)

    # 마지막 assert 줄 찾기 — except 블록 바깥에 있는 것만 유효
    last_assert_idx = -1
    for i, line in enumerate(lines):
        if re.match(r'\s*(assert|self\.assert)', line.strip()) and not _in_except_block(i):
            last_assert_idx = i

    if last_assert_idx == -1:
        return code

    # assert 직전까지에서 마지막 함수 호출 결과 변수 찾기 (except 블록 밖에서만)
    # expected_*, buggy_*, baseline_* 같은 상수 변수는 제외 (함수 결과가 아님)
    _SKIP_VAR_PREFIXES = ("expected", "buggy", "baseline", "correct", "desired", "known")
    result_var = None
    for i, line in enumerate(lines[:last_assert_idx]):
        if _in_except_block(i):
            continue
        stripped = line.strip()
        m = re.match(r'^(\w+)\s*=\s*\S+\(', stripped)
        if m and not stripped.startswith(("import ", "from ", "def ", "class ")):
            var_name = m.group(1)
            if not any(var_name.lower().startswith(p) for p in _SKIP_VAR_PREFIXES):
                result_var = var_name

    if result_var is None:
        return code

    indent = len(lines[last_assert_idx]) - len(lines[last_assert_idx].lstrip())
    buggy_repr = repr(actual_output)
    tier2 = [
        f"{' ' * indent}# [Tier 2: probe-verified buggy repr — must differ after fix]",
        f"{' ' * indent}assert repr({result_var}) != {buggy_repr}",
    ]
    lines = lines[:last_assert_idx + 1] + tier2 + lines[last_assert_idx + 1:]
    return "\n".join(lines)


def _fix_exception_message_matching(code: str) -> str:
    """예외 메시지 exact/containment matching 제거.

    패치 전후로 에러 메시지 포맷이 달라지면 before AND after 모두 실패한다.
    메시지 비교를 제거하고 예외 타입 체크만 남긴다.
    """
    # 1. assert "msg" in str(exc) / assert "msg" not in str(exc)
    code = re.sub(
        r'^(\s*)assert\s+["\'].+["\']\s+(?:not\s+)?in\s+str\s*\(.+\)\s*$',
        r'\1# [removed: exception message matching — version-dependent]',
        code,
        flags=re.MULTILINE,
    )
    # 2. assert str(exc) ==/!= "exact message"  /  assert str(cm.exception) ==/!= "..."
    code = re.sub(
        r'^(\s*)assert\s+str\s*\(\s*[\w.]+\s*\)\s*(?:==|!=)\s*["\'].+["\']\s*$',
        r'\1# [removed: exception message exact match — version-dependent]',
        code,
        flags=re.MULTILINE,
    )
    # 3. assert exc.value.args[0] ==/!= "exact message" / assert exc.args[0] ==/!= "..."
    code = re.sub(
        r'^(\s*)assert\s+\w+(?:\.value)?\.args\[\d+\]\s*(?:==|!=)\s*["\'].+["\']\s*$',
        r'\1# [removed: exception args exact match — version-dependent]',
        code,
        flags=re.MULTILINE,
    )
    # 4. self.assertEqual/NotEqual(str(exc), "exact message")
    code = re.sub(
        r'^(\s*)self\.assert(?:Equal|NotEqual)\s*\(\s*str\s*\(\s*[\w.]+\s*\)\s*,\s*["\'].+["\']\s*\)\s*$',
        r'\1# [removed: exception message assertEqual — version-dependent]',
        code,
        flags=re.MULTILINE,
    )
    # 5. self.assertIn/NotIn("msg", str(exc))
    code = re.sub(
        r'^(\s*)self\.assert(?:In|NotIn|Regex|NotRegex)\s*\(\s*[^,\n]+,\s*str\s*\(\s*[\w.]+\s*\).*\)\s*$',
        r'\1# [removed: exception message containment — version-dependent]',
        code,
        flags=re.MULTILINE,
    )
    return code


def _remove_trivial_assertions(code: str) -> str:
    """Remove assertions that do not check repository behavior."""
    code = re.sub(
        r'^\s*(?:self\.)?assertTrue\s*\(\s*(?:True|1)\s*(?:,\s*[^)]*)?\)\s*(?:#.*)?$',
        '# [removed: trivial oracle — assert post-fix behavior instead]',
        code,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    code = re.sub(
        r'^\s*assert\s+(?:True|1)\s*(?:#.*)?$',
        '# [removed: trivial oracle — assert post-fix behavior instead]',
        code,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    return code


def _fix_private_attr_access(code: str) -> str:
    """obj._private_attr = value 직접 설정 패턴 제거.

    이 패턴은 내부 상태를 우회 설정하므로 실제 API 동작을 트리거하지 않아
    테스트가 패치 전/후 모두 실패하게 만든다.
    """
    return re.sub(
        r'^(\s*)\w[\w.]*\._\w+\s*=\s*.+$',
        r'\1# [removed: private attribute assignment — use public API instead]',
        code,
        flags=re.MULTILINE,
    )


def _has_private_attr_read(code: str) -> bool:
    """Detect private attribute reads that should get a rewrite chance."""
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if re.search(r"\._[A-Za-z]\w*\s*=", line):
            continue
        if re.search(r"\._[A-Za-z]\w*", line):
            return True
    return False


def _fix_django_test_code(test_code: str) -> str:
    """test_code 내 unittest.TestCase / unittest.SimpleTestCase 상속 교정.

    추가로:
    - TestCase 상속 없는 class TestXxx → class TestXxx(TestCase)로 교정
    - TestCase 사용하는데 import 없으면 상단에 주입
    - relative import (from . import / from .. import) 제거
    """
    # unittest.TestCase → TestCase 교정
    test_code = re.sub(r'\(unittest\.TestCase\)', '(TestCase)', test_code)
    test_code = re.sub(r'\(unittest\.SimpleTestCase\)', '(SimpleTestCase)', test_code)

    # class TestXxx: (상속 없음) → class TestXxx(TestCase):
    test_code = re.sub(
        r'^(class\s+Test\w+)\s*:',
        r'\1(TestCase):',
        test_code,
        flags=re.MULTILINE,
    )

    # relative import 제거 (from . import ... / from .. import ...)
    filtered_lines = []
    for line in test_code.splitlines():
        if re.match(r'\s*from\s+\.', line):
            logger.warning("Removed relative import from append_block: %s", line.strip())
            continue
        filtered_lines.append(line)
    test_code = "\n".join(filtered_lines)

    # TestCase가 코드에 쓰이는데 import가 없으면 상단에 주입
    uses_testcase = bool(re.search(r'\(TestCase\)|\(SimpleTestCase\)', test_code))
    has_import = bool(re.search(r'from django\.test import', test_code))
    if uses_testcase and not has_import:
        test_code = "from django.test import TestCase, SimpleTestCase\n" + test_code

    return test_code


# stdlib / 공통 모듈 자동 주입 목록: (사용 패턴, import 구문)
_AUTO_INJECT_IMPORTS: List[tuple] = [
    ("unittest.",        "import unittest"),
    ("uuid.",            "import uuid"),
    ("datetime.",        "import datetime"),
    ("os.path",          "import os"),
    ("os.",              "import os"),
    ("sys.",             "import sys"),
    ("json.",            "import json"),
    ("re.",              "import re"),
    ("io.",              "import io"),
    ("math.",            "import math"),
    ("copy.",            "import copy"),
    ("collections.",     "import collections"),
    ("itertools.",       "import itertools"),
    ("functools.",       "import functools"),
    ("pathlib.",         "from pathlib import Path"),
    ("tempfile.",        "import tempfile"),
    ("textwrap.",        "import textwrap"),
    ("inspect.",         "import inspect"),
    ("typing.",          "from typing import Any, Dict, List, Optional, Tuple"),
]


# 관용 alias → import 매핑: 이 alias가 block에서 사용되는데 import가 없으면 실패 처리
_COMMON_ALIAS_IMPORTS: List[tuple] = [
    ("np.",    "import numpy as np"),
    ("pd.",    "import pandas as pd"),
    ("plt.",   "import matplotlib.pyplot as plt"),
    ("scipy.", "import scipy"),
    ("sk.",    "import sklearn"),
    ("tf.",    "import tensorflow as tf"),
    ("torch.", "import torch"),
]


def _detect_missing_common_aliases(append_block: str, existing_file_content: str = "") -> List[str]:
    """append_block에서 관용 alias(np., pd. 등)가 import 없이 쓰이는 경우를 감지한다.

    기존 파일의 imports도 함께 확인하여 이미 import된 것은 제외한다.
    Returns: 누락된 import 구문 목록
    """
    combined_imports = existing_file_content + "\n" + append_block
    missing = []
    for alias, import_stmt in _COMMON_ALIAS_IMPORTS:
        if alias not in append_block:
            continue
        module = import_stmt.split()[1]  # "numpy", "pandas", etc.
        already = (
            f"import {module}" in combined_imports
            or f"as {alias.rstrip('.')}" in combined_imports
        )
        if not already:
            missing.append(import_stmt)
    return missing


def _ensure_repro_suffix(append_block: str) -> str:
    """생성된 테스트 함수/메서드명에 _repro 접미사를 보장한다.

    기존 레포에 같은 이름의 테스트가 있으면 하네스가 잘못된 테스트를 실행하기 때문에
    반드시 유일한 이름을 사용해야 한다. _repro로 끝나지 않는 test_* 이름은 모두 교체한다.
    """
    def add_suffix(m: re.Match) -> str:
        name = m.group(1)
        if name.endswith("_repro"):
            return m.group(0)
        return m.group(0).replace(name, name + "_repro", 1)

    # def test_xxx(...): 패턴
    result = re.sub(r'\bdef (test_\w+)(?=\s*\()', add_suffix, append_block)
    return result


def _count_generated_tests(code: str) -> int:
    """Count test functions/methods introduced by an append block."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            count += 1
    return count


def _prune_to_best_generated_test(code: str, clue: Optional[Dict[str, Any]] = None) -> str:
    """Keep the strongest generated test when the model emits several.

    This is a pre-patch/static repair only: it uses issue clue text and the
    generated code, never final-eval or post-patch results.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
        and getattr(node, "end_lineno", None)
    ]
    if len(candidates) <= 1:
        return code

    clue = clue or {}
    identifiers = clue.get("identifiers", {}) if isinstance(clue.get("identifiers"), dict) else {}
    terms: List[str] = []
    for key in ("functions", "classes", "exceptions", "files"):
        terms.extend(str(x).lower() for x in identifiers.get(key, []) if x)
    terms.extend(str(x).lower()[:80] for x in clue.get("expected_outputs", [])[:3] if x)
    terms.extend(str(x).lower()[:80] for x in clue.get("actual_outputs", [])[:3] if x)
    terms = [t for t in terms if len(t) >= 3]

    def score(node: ast.AST) -> float:
        segment = ast.get_source_segment(code, node) or ""
        lower = segment.lower()
        value = 0.0
        value += min(sum(1 for term in terms if term in lower), 8) * 2.0
        value += len(re.findall(r"\bassert\b|self\.assert|pytest\.raises|assert_(?:allclose|array|equal)", segment)) * 1.5
        if re.search(r"assert(?:True)?\s*\(\s*(?:True|1)\s*\)|^\s*assert\s+(?:True|1)\b", segment, re.MULTILINE):
            value -= 8.0
        if re.search(r"str\s*\(|args\[\d+\]|assert(?:In|NotIn)\s*\(\s*['\"]", segment):
            value -= 1.5
        if re.search(r"is\s+not\s+None|assertIsInstance|len\s*\([^)]*\)\s*>", segment):
            value -= 0.5
        # Earlier tests are usually the primary reproduction, all else equal.
        value -= getattr(node, "lineno", 0) * 0.001
        return value

    best = max(candidates, key=score)
    remove_ranges = {
        line_no
        for node in candidates
        if node is not best
        for line_no in range(node.lineno, node.end_lineno + 1)
    }
    lines = code.splitlines()
    pruned = "\n".join(
        line for idx, line in enumerate(lines, start=1)
        if idx not in remove_ranges
    ).rstrip() + "\n"
    try:
        ast.parse(pruned)
    except SyntaxError:
        return code
    return pruned if _count_generated_tests(pruned) == 1 else code


def _truncate_scenario_for_prompt(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """시나리오 JSON을 프롬프트에 포함하기 전에 긴 필드를 잘라낸다.

    actual_outputs / expected_outputs / reproduction_code 등이 길면 토큰 초과 원인이 된다.
    """
    result = dict(scenario)
    for field in ("actual_outputs", "expected_outputs"):
        items = result.get(field)
        if isinstance(items, list):
            truncated = []
            for item in items[:2]:
                s = item if isinstance(item, str) else str(item)
                truncated.append(s[:300] + "…" if len(s) > 300 else s)
            result[field] = truncated
    # reproduction_code: code 필드만 300자로 제한
    repro = result.get("reproduction_code")
    if isinstance(repro, list):
        result["reproduction_code"] = [
            {**b, "code": b["code"][:300] + "…"} if isinstance(b.get("code"), str) and len(b["code"]) > 300 else b
            for b in repro[:2]
        ]
    contract = result.get("oracle_contract")
    if isinstance(contract, dict):
        result["oracle_contract"] = {
            k: contract.get(k, "")
            for k in ("oracle_type", "oracle_source", "rule")
            if contract.get(k)
        }
    return result


def _fix_append_block_imports(
    parsed: Dict[str, Any],
    repo_path: Optional[str] = None,
) -> Dict[str, Any]:
    """append_block에서 사용 중인데 import가 누락된 stdlib 모듈을 자동 주입한다.

    - append_block이 없으면 no-op.
    - `from tests.` import 패턴은 repo에 실제 파일이 없을 때만 제거한다.
      실제 파일이 있는 경우 (e.g. tests/base/models.py) 그대로 유지한다.
    """
    block = parsed.get("append_block", "")
    if not block:
        return parsed

    # append_block의 기존 import 라인 수집
    existing = set()
    lines = block.splitlines()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            existing.add(stripped)

    # `from tests.` import 제거 — 단, repo에 실제 파일이 존재하면 유지
    filtered_lines = []
    removed_tests_import = False
    for line in lines:
        if re.match(r'\s*from tests[\. ]', line) or re.match(r'\s*import tests[\. ]', line):
            # repo_path가 있으면 파일 존재 여부 확인
            keep = False
            if repo_path:
                m = re.match(r'\s*from (tests[\.\w]+)', line)
                if m:
                    module_str = m.group(1)  # e.g. "tests.base.models"
                    rel_path = module_str.replace(".", "/") + ".py"
                    if (Path(repo_path) / rel_path).exists():
                        keep = True
            if keep:
                filtered_lines.append(line)
                continue
            removed_tests_import = True
            logger.warning("Removed invalid 'from tests.' import: %s", line.strip())
            continue
        filtered_lines.append(line)
    if removed_tests_import:
        block = "\n".join(filtered_lines)

    # stdlib 누락 import 감지 및 주입
    to_inject = []
    for pattern, import_stmt in _AUTO_INJECT_IMPORTS:
        if pattern in block:
            # 이미 import 있는지 확인 (import_stmt의 모듈명으로)
            module = import_stmt.split()[-1].split(".")[0]
            already = any(
                (f"import {module}" in ex or f"from {module}" in ex)
                for ex in existing
            )
            if not already and import_stmt not in to_inject:
                to_inject.append(import_stmt)

    for alias, import_stmt in _COMMON_ALIAS_IMPORTS:
        if alias in block:
            module = import_stmt.split()[1]
            alias_name = alias.rstrip(".")
            already = any(
                (f"import {module}" in ex or f" as {alias_name}" in ex)
                for ex in existing
            )
            if not already and import_stmt not in to_inject:
                to_inject.append(import_stmt)

    if to_inject or removed_tests_import:
        new_block = "\n".join(to_inject) + ("\n" if to_inject else "") + block
        parsed = dict(parsed)
        parsed["append_block"] = new_block
        parsed["test_code"] = new_block
        if to_inject:
            logger.info("Auto-injected imports into append_block: %s", to_inject)

    return parsed


def _fix_append_block_imports_against_file(
    parsed: Dict[str, Any],
    file_content: str,
) -> Dict[str, Any]:
    """실제 삽입될 파일의 기존 imports를 기준으로 append_block의 누락 import를 추가 주입한다.

    LLM이 hint와 다른 파일을 선택했을 때, 해당 파일에 없는 모듈이 append_block에서
    사용되면 NameError가 발생한다. 이를 방지한다.
    """
    block = parsed.get("append_block", "")
    if not block:
        return parsed

    # 실제 파일의 기존 import 수집
    file_imports: set = set()
    for line in file_content.splitlines():
        stripped = line.strip()
        if (stripped.startswith("import ") or stripped.startswith("from ")) and not line.startswith((" ", "\t")):
            file_imports.add(stripped)

    # append_block의 import 수집
    block_imports: set = set()
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            block_imports.add(stripped)

    # 파일 + block에 있는 imports 합집합
    all_available = file_imports | block_imports

    # stdlib 자동 주입: block에서 사용되는데 어디에도 없는 것
    to_inject = []
    for pattern, import_stmt in _AUTO_INJECT_IMPORTS:
        if pattern in block:
            module = import_stmt.split()[-1].split(".")[0]
            already = any(
                (f"import {module}" in ex or f"from {module}" in ex)
                for ex in all_available
            )
            if not already and import_stmt not in to_inject:
                to_inject.append(import_stmt)

    for alias, import_stmt in _COMMON_ALIAS_IMPORTS:
        if alias in block:
            module = import_stmt.split()[1]
            alias_name = alias.rstrip(".")
            already = any(
                (f"import {module}" in ex or f" as {alias_name}" in ex)
                for ex in all_available
            )
            if not already and import_stmt not in to_inject:
                to_inject.append(import_stmt)

    if to_inject:
        new_block = "\n".join(to_inject) + "\n" + block
        parsed = dict(parsed)
        parsed["append_block"] = new_block
        parsed["test_code"] = new_block
        logger.info("File-aware import injection into append_block: %s", to_inject)

    return parsed


class ReproductionTestGenerator:

    SYSTEM_PROMPT = "You are a careful software test generation assistant. Return JSON only."

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        max_retries: int = 3,
        model_key: str = "qwen",
    ) -> None:
        self.client = client or LLMClient(load_model_config(model_key))
        self.max_retries = max_retries

    def generate(
        self,
        instance: Any,
        clue: Dict[str, Any],
        context: Dict[str, Any],
        validation_report: Dict[str, Any],
        iteration: int = 1,
        runtime_error_hint: Optional[str] = None,
    ) -> GeneratedReproductionTest:
        scenario = copy.deepcopy(select_primary_scenario(validation_report))

        repo_path = context.get("repo_path", "")
        if not repo_path:
            raise ValueError("context.json is missing repo_path.")

        runner = context.get("project_test_style", {}).get("runner", "pytest")

        target_location = scenario.get("target_location", {})
        target_test_file_hint = target_location.get("candidate_test_file") or (
            scenario.get("relevant_test_files", [""])[0] if scenario.get("relevant_test_files") else ""
        )
        chosen_test_file, choice_reason = _choose_viable_test_file(context, target_test_file_hint)
        if chosen_test_file and chosen_test_file != target_test_file_hint:
            target_test_file_hint = chosen_test_file
            target_location = dict(target_location)
            target_location["candidate_test_file"] = chosen_test_file
            scenario["target_location"] = target_location
            rel_tests = list(scenario.get("relevant_test_files") or [])
            if chosen_test_file not in rel_tests:
                rel_tests.insert(0, chosen_test_file)
            scenario["relevant_test_files"] = rel_tests
            setup_steps = scenario.get("setup_steps")
            if not isinstance(setup_steps, list):
                setup_steps = []
            setup_steps.append(
                f"[pre-generation guard] test_file_overridden_skip_or_missing:{choice_reason}"
            )
            scenario["setup_steps"] = setup_steps

        # 대상 테스트 파일의 기존 import 블록 + 실제 test 메서드 예시 추출 (프롬프트용)
        existing_test_imports = ""
        target_test_example = ""
        if target_test_file_hint:
            test_file_abs = Path(repo_path) / target_test_file_hint
            if test_file_abs.exists():
                existing_test_imports = self._extract_import_block(test_file_abs)
                target_test_example = self._extract_test_examples_from_file(str(test_file_abs))

        prompt_profile: Dict[str, Any] = {
            "budget_mode": "compact",
            "sections_included": [],
            "truncated_sections": [],
            "retry_prompt_chars": [],
            "target_test_file_choice_reason": choice_reason,
        }
        prompt = self._build_prompt(
            instance=instance,
            clue=clue,
            context=context,
            scenario=scenario,
            existing_test_imports=existing_test_imports,
            target_test_example=target_test_example,
            runtime_error_hint=runtime_error_hint,
            prompt_profile=prompt_profile,
        )
        prompt_profile["prompt_chars"] = len(prompt)
        prompt_profile["initial_prompt_chars"] = len(prompt)

        last_error_msg = ""
        last_raw_response = ""
        last_parsed = None
        last_validation_errors: List[str] = []
        repair_actions_accum: List[str] = []
        repair_retry_count = 0
        retry_required_oracle_risks: List[str] = []
        semantic_risk_flags: List[str] = []
        # 이 generate() 호출에서 누적된 토큰 사용량
        accumulated_tokens: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        iter_offset = (iteration - 1) * 0.1
        max_attempts = self.max_retries + 2
        validation_passed = False
        for attempt in range(max_attempts):
            current_temperature = self.client.config.temperature + iter_offset + (attempt * 0.03)

            if attempt == 0:
                current_prompt = prompt
            else:
                current_prompt = self._build_fix_prompt(
                    original_prompt=prompt,
                    previous_response=last_raw_response,
                    error_message=last_error_msg,
                    attempt=attempt,
                )
                prompt_profile.setdefault("retry_prompt_chars", []).append(len(current_prompt))

            try:
                raw_response = self.client.generate(
                    current_prompt,
                    system_prompt=self.SYSTEM_PROMPT,
                    temperature=current_temperature,
                )
                # 토큰 누적
                for k in accumulated_tokens:
                    accumulated_tokens[k] += self.client.last_usage.get(k, 0)
            except Exception as e:
                last_error_msg = f"LLM call failed: {e}"
                last_raw_response = ""
                logger.warning("[attempt %d/%d] %s", attempt + 1, max_attempts, last_error_msg)
                continue

            last_raw_response = raw_response

            try:
                parsed = self._parse_model_output(raw_response, scenario, context)
            except (ValueError, TypeError, AttributeError) as e:
                last_error_msg = f"Model output parsing failed: {e}"
                logger.warning("[attempt %d/%d] %s", attempt + 1, max_attempts, last_error_msg)
                continue

            for key in ("append_block", "test_code"):
                if parsed.get(key):
                    parsed[key], repair_actions = _apply_oracle_repairs(parsed[key], clue)
                    repair_actions_accum.extend(f"{key}:{a}" for a in repair_actions)
                    parsed[key] = _fill_empty_exception_blocks(parsed[key])

            if "sphinx" in getattr(instance, "repo", "").lower():
                for key in ("append_block", "test_code"):
                    if parsed.get(key):
                        parsed[key] = _fix_sphinx_test_code(parsed[key])

            # stdlib 누락 import 자동 주입 (검증 전에 수행해야 재시도 시 반영됨)
            parsed = _fix_append_block_imports(parsed, repo_path=repo_path)

            target = scenario.get("target_location", {}) if isinstance(scenario.get("target_location"), dict) else {}
            target_function = str(target.get("target_function") or "")
            target_missing = (
                target_function
                and not target_function.startswith("_")
                and len(target_function) >= 3
                and target_function not in {"path", "main", "run", "get", "set"}
                and not any(
                    parsed.get(key) and self._check_target_function_presence(target_function, parsed[key])
                    for key in ("append_block", "test_code")
                )
            )
            if target_missing and attempt < max_attempts - 1:
                last_parsed = parsed
                last_validation_errors = [
                    "CRITICAL: semantic risk: target_function_public_api_rewrite"
                ]
                repair_retry_count += 1
                last_error_msg = (
                    "CRITICAL: semantic risk: target_function_public_api_rewrite. "
                    "Rewrite the test to call the target function from the scenario, "
                    "or a public wrapper that visibly exercises that target behavior."
                )
                logger.warning("[attempt %d/%d] validation failed: %s", attempt + 1, max_attempts, last_error_msg)
                continue

            private_read_keys = [
                key
                for key in ("append_block", "test_code")
                if parsed.get(key) and _has_private_attr_read(parsed[key])
            ]
            if private_read_keys and attempt < max_attempts - 1:
                last_parsed = parsed
                last_validation_errors = [
                    "CRITICAL: semantic risk: private_attribute_public_api_rewrite"
                ]
                repair_retry_count += 1
                last_error_msg = (
                    "CRITICAL: semantic risk: private_attribute_public_api_rewrite. "
                    "Rewrite assertions to inspect public API return values, public state, "
                    "artist/axis/legend accessors, or issue-visible behavior instead of _private attributes."
                )
                logger.warning("[attempt %d/%d] validation failed: %s", attempt + 1, max_attempts, last_error_msg)
                continue

            # 사전 검증
            validation = self._validate_generated_code(
                parsed=parsed,
                repo_path=repo_path,
                context=context,
                clue=clue,
                scenario=scenario,
            )

            if validation.fixed_imports is not None:
                parsed["imports"] = validation.fixed_imports

            if validation.is_valid:
                last_parsed = parsed
                last_validation_errors = []
                last_error_msg = ""
                validation_passed = True
                logger.info("[attempt %d/%d] validation passed", attempt + 1, max_attempts)
                break
            else:
                last_parsed = parsed
                last_validation_errors = list(validation.errors)
                if any("retry required" in e or "semantic risk" in e for e in validation.errors):
                    repair_retry_count += 1
                last_error_msg = "; ".join(validation.errors)
                logger.warning("[attempt %d/%d] validation failed: %s", attempt + 1, max_attempts, last_error_msg)

        # Django runner: unittest.TestCase → django.test.TestCase 자동 교정, relative import 제거
        if last_parsed is not None and runner == "django-test":
            last_parsed["imports"] = _fix_django_imports(last_parsed.get("imports", []))
            last_parsed["test_code"] = _fix_django_test_code(last_parsed.get("test_code", ""))
            # append_block 방식에도 동일하게 적용
            if last_parsed.get("append_block"):
                last_parsed["append_block"] = _fix_django_test_code(last_parsed["append_block"])

        # sphinx: pytest.mark.sphinx 데코레이터 제거 (sphinx.testing.fixtures 의존성 회피)
        if last_parsed is not None and "sphinx" in getattr(instance, "repo", "").lower():
            if last_parsed.get("append_block"):
                last_parsed["append_block"] = _fix_sphinx_test_code(last_parsed["append_block"])
            if last_parsed.get("test_code"):
                last_parsed["test_code"] = _fix_sphinx_test_code(last_parsed["test_code"])

        # private attribute 직접 설정 제거 (fragile, public API 로직 우회)
        if last_parsed is not None:
            if last_parsed.get("append_block"):
                last_parsed["append_block"] = _fix_private_attr_access(last_parsed["append_block"])
            if last_parsed.get("test_code"):
                last_parsed["test_code"] = _fix_private_attr_access(last_parsed["test_code"])

        # 예외 메시지 exact matching 제거 (버전 의존, after_patch에서 항상 실패)
        if last_parsed is not None:
            for key in ("append_block", "test_code"):
                if last_parsed.get(key):
                    last_parsed[key], repair_actions = _apply_oracle_repairs(last_parsed[key], clue)
                    repair_actions_accum.extend(f"final_{key}:{a}" for a in repair_actions)
                    last_parsed[key] = _fill_empty_exception_blocks(last_parsed[key])

        # Tier 2 assertion 주입 (probe actual_outputs가 있을 때만, 코드 레벨 강제)
        if last_parsed is not None:
            actual_outs = (scenario or {}).get("actual_outputs") or []
            if actual_outs:
                for key in ("append_block", "test_code"):
                    if last_parsed.get(key):
                        last_parsed[key] = _inject_tier2_assertion(
                            last_parsed[key], actual_outs[0]
                        )

        # 모든 retry 소진 후에도 정적/의미 검증을 통과하지 못한 결과는 사용하지 않는다.
        # Invalid tests flowing into alignment create misleading ALIGNED rows and poor final eval.
        if last_parsed is None or not validation_passed:
            raise ValueError(
                f"Failed to generate a valid test after {max_attempts} attempts. "
                f"Last error: {last_error_msg}"
            )

        final_validation = self._validate_generated_code(
            parsed=last_parsed,
            repo_path=repo_path,
            context=context,
            clue=clue,
            scenario=scenario,
        )
        if final_validation.fixed_imports is not None:
            last_parsed["imports"] = final_validation.fixed_imports
        if not final_validation.is_valid:
            raise ValueError(
                "Generated test became invalid after automatic repair: "
                + "; ".join(final_validation.errors)
            )

        parsed = last_parsed
        repair_failed_reason = ""
        for key in ("append_block", "test_code"):
            if parsed.get(key):
                remaining_retry_risks = _detect_retry_required_oracle_risks(parsed[key], clue=clue)
                retry_required_oracle_risks = sorted(set(remaining_retry_risks))
                if remaining_retry_risks:
                    repair_failed_reason = "retry_required_oracle_risk=" + ",".join(remaining_retry_risks)
                    break
        for key in ("append_block", "test_code"):
            if parsed.get(key):
                semantic_risk_flags = sorted(set(_detect_semantic_risk_flags(parsed[key], clue, context, scenario)))
                if semantic_risk_flags and not repair_failed_reason:
                    repair_failed_reason = "semantic_risk=" + ",".join(semantic_risk_flags)
                break
        if not repair_failed_reason and last_validation_errors:
            retry_errors = [e for e in last_validation_errors if "retry required" in e]
            if retry_errors:
                repair_failed_reason = "; ".join(retry_errors[:3])

        retry_chars = prompt_profile.get("retry_prompt_chars", [])
        if retry_chars:
            prompt_profile["max_retry_prompt_chars"] = max(retry_chars)
            prompt_profile["avg_retry_prompt_chars"] = round(sum(retry_chars) / len(retry_chars))
            prompt_profile["retry_to_initial_char_ratio"] = round(
                prompt_profile["max_retry_prompt_chars"] / max(1, prompt_profile.get("initial_prompt_chars", 1)),
                3,
            )

        target_test_file = parsed["target_test_file"]
        chosen_after_parse, choice_reason_after_parse = _choose_viable_test_file(
            context,
            target_test_file,
        )
        if chosen_after_parse and chosen_after_parse != target_test_file:
            logger.warning(
                "LLM chose unusable test file %s; overriding to %s (%s)",
                target_test_file,
                chosen_after_parse,
                choice_reason_after_parse,
            )
            target_test_file = chosen_after_parse
            parsed["target_test_file"] = target_test_file
            prompt_profile["target_test_file_choice_reason"] = choice_reason_after_parse

        # ── __init__.py guard: LLM이 __init__.py를 선택한 경우, 시나리오 힌트로 교체 ──
        if target_test_file.endswith("__init__.py") and target_test_file_hint and not target_test_file_hint.endswith("__init__.py"):
            logger.warning(
                "LLM chose __init__.py as test file %s; overriding to %s",
                target_test_file, target_test_file_hint,
            )
            target_test_file = target_test_file_hint
            parsed["target_test_file"] = target_test_file

        # ── skip-guard: LLM이 module-level skip 파일을 선택한 경우, 시나리오 힌트로 교체 ──
        _skip_set = {
            cf["path"]
            for cf in context.get("candidate_test_files", [])
            if cf.get("has_module_skip")
        }
        if target_test_file in _skip_set and target_test_file_hint and target_test_file_hint not in _skip_set:
            logger.warning(
                "LLM chose skip-flagged file %s; overriding to %s",
                target_test_file, target_test_file_hint,
            )
            target_test_file = target_test_file_hint
            parsed["target_test_file"] = target_test_file

        target_test_abspath = Path(repo_path) / target_test_file

        if not target_test_abspath.exists():
            raise FileNotFoundError(f"Target test file does not exist: {target_test_abspath}")

        original_content = read_text(target_test_abspath)

        # ── base_commit 버전의 파일 내용 가져오기 (patch context 불일치 방지) ──
        # 로컬 파일은 최신 커밋 기준이지만 Docker 컨테이너는 base_commit에서 실행됨.
        # patch를 base_commit 버전 파일 기준으로 생성해야 git apply가 성공함.
        base_commit = getattr(instance, "base_commit", None)
        content_for_patch = original_content  # fallback
        if base_commit and repo_path:
            try:
                r = subprocess.run(
                    ["git", "show", f"{base_commit}:{target_test_file}"],
                    capture_output=True, text=True, cwd=repo_path,
                )
                if r.returncode == 0 and r.stdout.strip():
                    content_for_patch = r.stdout
                    logger.debug(
                        "Using base_commit content for patch: %s@%s",
                        target_test_file, base_commit[:8],
                    )
            except Exception as e:
                logger.warning("git show failed for %s@%s: %s", target_test_file, base_commit, e)

        # ── 실제 선택된 파일 기준 import 보완 ──
        # LLM이 hint와 다른 파일을 선택했을 수 있으므로,
        # 실제 파일의 기존 imports를 확인하여 append_block에 필요한 import를 추가 주입한다.
        parsed = _fix_append_block_imports_against_file(parsed, original_content)

        if parsed.get("insert_mode") == "append_block":
            # 단순 append — base_commit 버전 파일에 붙임
            modified_content = content_for_patch.rstrip() + "\n\n" + parsed["append_block"] + "\n"
        else:
            # 구 방식 (하위 호환)
            modified_content = self._build_modified_test_file_content(
                original_content=content_for_patch,
                imports=parsed["imports"],
                test_code=parsed["test_code"],
            )
        test_patch = self._build_unified_patch(
            original_content=content_for_patch,
            modified_content=modified_content,
            relative_path=target_test_file,
        )

        return GeneratedReproductionTest(
            instance_id=instance.instance_id,
            scenario_id=scenario.get("scenario_id", "unknown"),
            model_name=self.client.config.model_name,
            repo_path=repo_path,
            target_test_file=target_test_file,
            target_test_file_abspath=str(target_test_abspath),
            target_source_file=scenario.get("target_location", {}).get("source_file", ""),
            insert_mode=parsed["insert_mode"],
            insertion_hint=parsed["insertion_hint"],
            imports=parsed["imports"],
            test_code=parsed["test_code"],
            original_test_file_content=original_content,
            modified_test_file_content=modified_content,
            test_patch=test_patch,
            raw_response=last_raw_response,
            prompt=prompt,
            repair_attempted=bool(repair_actions_accum or repair_failed_reason),
            repair_actions=sorted(set(repair_actions_accum)),
            repair_failed_reason=repair_failed_reason,
            repair_retry_count=repair_retry_count,
            retry_required_oracle_risks=retry_required_oracle_risks,
            semantic_risk_flags=semantic_risk_flags,
            prompt_profile=prompt_profile,
            token_usage=accumulated_tokens,
        )

    def save(self, result: GeneratedReproductionTest, output_path: str) -> None:
        """
        output_path 예:
        outputs/<instance_id>/generated_test.json

        같이 저장되는 파일:
        - generated_test.json
        - generated_test.patch
        - generated_test_rendered.py
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        result.patch_sha256 = sha256_text(result.test_patch)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

        patch_path = path.with_suffix(".patch")
        with open(patch_path, "w", encoding="utf-8") as f:
            f.write(result.test_patch)

        rendered_path = path.with_name(path.stem + "_rendered.py")
        with open(rendered_path, "w", encoding="utf-8") as f:
            f.write(result.modified_test_file_content)

    @staticmethod
    def _build_fault_location_section(clue: Dict[str, Any]) -> str:
        """clue의 fault_locations를 프롬프트 섹션으로 변환한다."""
        fault_locations = clue.get("fault_locations", [])
        if not fault_locations:
            return ""
        traceback_lines = []
        inferred_lines = []
        for fl in fault_locations[:5]:
            fp = fl.get("file_path", "").replace("\\", "/")
            fn = fl.get("function_name", "")
            ln = fl.get("line_no", "?")
            source = fl.get("source", "traceback")
            confidence = fl.get("confidence", "high" if source == "traceback" else "medium")
            parts = fp.split("/")
            rel_guess = "/".join(parts[-4:]) if len(parts) >= 4 else fp
            line = f"  - {rel_guess}  line {ln}  in {fn}"
            if source == "traceback" and confidence == "high":
                traceback_lines.append(line)
            else:
                inferred_lines.append(line)
        sections = []
        if traceback_lines:
            sections.append(
                "\n[CRITICAL: Fault Locations from Issue Traceback]\n"
                "The issue's stack trace explicitly points to these code locations.\n"
                "Your test MUST directly call the function identified here (or its public wrapper):\n"
                + "\n".join(traceback_lines)
                + "\n"
                "If this function is private (starts with _), call its nearest public caller instead.\n"
            )
        if inferred_lines:
            sections.append(
                "\n[Inferred Fault Location Candidates]\n"
                "These are medium-confidence hints, not mandatory traceback locations:\n"
                + "\n".join(inferred_lines)
                + "\n"
            )
        return "".join(sections)

    def _build_prompt(
        self,
        instance: Any,
        clue: Dict[str, Any],
        context: Dict[str, Any],
        scenario: Dict[str, Any],
        existing_test_imports: str = "",
        target_test_example: str = "",
        runtime_error_hint: Optional[str] = None,
        prompt_profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        if prompt_profile is not None:
            prompt_profile.setdefault("budget_mode", "compact")
            prompt_profile.setdefault("sections_included", [])
            prompt_profile.setdefault("truncated_sections", [])

        _identifiers = scenario.get("identifiers") or clue.get("identifiers", {})
        noisy_functions = {
            "arange", "rand", "random", "seed", "platform", "get_backend",
            "show_versions",
        }
        issue_functions = [
            fn for fn in _identifiers.get("functions", [])
            if fn not in noisy_functions
        ]
        issue_classes = _identifiers.get("classes", [])
        issue_error_keywords = scenario.get("error_keywords") or clue.get("error_keywords", [])
        target_location = scenario.get("target_location", {})
        target_test_file = target_location.get("candidate_test_file") or (
            scenario.get("relevant_test_files", [""])[0] if scenario.get("relevant_test_files") else ""
        )

        project_framework = context.get("project_test_style", {}).get("framework", "unknown")
        runner = context.get("project_test_style", {}).get("runner", "pytest")
        test_example = context.get("test_example_snippet", "")
        source_candidates = [x.get("path", "") for x in context.get("candidate_source_files", [])[:3]]
        test_candidates = [x.get("path", "") for x in context.get("candidate_test_files", [])[:3]]

        # available_imports 정보를 compact budget에 맞춰 변환
        available_imports = context.get("available_imports", {})
        import_map_lines = []
        for module, symbols in _prioritized_available_imports(available_imports, clue, context, scenario):
            if symbols:
                import_map_lines.append(f"  {module}: {', '.join(symbols[:_PROMPT_IMPORT_SYMBOLS])}")
        import_map_text = "\n".join(import_map_lines) if import_map_lines else "  (not available)"
        if import_map_lines:
            _mark_prompt_section(prompt_profile, "available_imports_compact")
        if len(available_imports or {}) > len(import_map_lines):
            if prompt_profile is not None:
                prompt_profile.setdefault("truncated_sections", []).append("available_imports")

        # conftest fixtures 섹션
        conftest_fixtures = context.get("conftest_fixtures", {})
        conftest_section = ""
        if conftest_fixtures:
            lines = ["[Available Pytest Fixtures (from conftest.py)]"]
            for path, names in list(conftest_fixtures.items())[:_PROMPT_CONFTEST_PATHS]:
                lines.append(f"  {path}: {', '.join(names[:_PROMPT_CONFTEST_FIXTURES])}")
            conftest_section = "\n".join(lines) + "\n"
            _mark_prompt_section(prompt_profile, "conftest_fixtures")
            if len(conftest_fixtures) > _PROMPT_CONFTEST_PATHS and prompt_profile is not None:
                prompt_profile.setdefault("truncated_sections", []).append("conftest_fixtures")

        # scenario의 required_fixtures 섹션
        test_env = scenario.get("test_environment", {})
        required_fixtures = test_env.get("required_fixtures", []) if test_env else []
        required_fixtures_section = ""
        if required_fixtures:
            required_fixtures_section = (
                f"[Required Fixtures for This Scenario]\n"
                f"  {', '.join(required_fixtures)}\n"
            )

        test_symbol_catalog = context.get("test_symbol_catalog", {}) or {}
        target_symbol_catalog = test_symbol_catalog.get(target_test_file, {}) if target_test_file else {}
        test_symbol_section = ""
        if target_symbol_catalog:
            def _cat_items(name: str, limit: int = 20) -> str:
                values = target_symbol_catalog.get(name, [])
                if not isinstance(values, list):
                    values = []
                return ", ".join(str(v) for v in values[:limit]) or "(none)"

            test_symbol_section = f"""
[Target Test File Symbol Catalog]
Grounded symbols already present in {target_test_file}. Prefer these over inventing new setup.
imports: {_cat_items("imports", 12)}
imported_symbols: {_cat_items("imported_symbols", 30)}
classes: {_cat_items("classes", 25)}
models: {_cat_items("models", 25)}
helpers: {_cat_items("helpers", 20)}
collection_risk: {_cat_items("collection_risk", 5)}
"""
            _mark_prompt_section(prompt_profile, "test_symbol_catalog")

        # 기존 테스트 파일 import 블록 — 가져온 심볼 이름 추출
        existing_imports_section = ""
        if existing_test_imports:
            # import 라인에서 이름 파싱 (전체 블록에서 파싱하되, 출력은 cap)
            _imported_names: list[str] = []
            for _line in existing_test_imports.splitlines():
                _line = _line.strip()
                if _line.startswith("from ") and " import " in _line:
                    _names_part = _line.split(" import ", 1)[1].split("#")[0]
                    for _n in _names_part.split(","):
                        _n = _n.strip().split(" as ")[-1].strip()
                        if _n and _n.isidentifier():
                            _imported_names.append(_n)
                elif _line.startswith("import "):
                    _n = _line[7:].strip().split(" as ")[-1].split("#")[0].strip()
                    if _n and _n.isidentifier():
                        _imported_names.append(_n)
            _imported_names_str = ", ".join(_imported_names[:_PROMPT_EXISTING_SYMBOLS]) if _imported_names else "(none)"

            # 토큰 예산 보호: import 블록이 길면 잘라냄
            _imports_display = _clip_prompt_text(
                existing_test_imports,
                _PROMPT_EXISTING_IMPORTS_CHARS,
                "existing_test_imports",
                prompt_profile,
            )
            _mark_prompt_section(prompt_profile, "existing_test_imports")

            existing_imports_section = f"""
[Existing Imports in Target Test File]
The following imports already exist in {target_test_file}.
```
{_imports_display}
```
Available symbols (already imported — use them DIRECTLY, do NOT re-import):
{_imported_names_str}

CRITICAL:
- Do NOT define new class or model definitions (e.g., class MyModel(models.Model): ...).
- Do NOT import symbols that are not listed in [Available Imports from Repository] or above.
- Use only the classes/functions that are already imported in this file.
"""

        # === 이슈 원문의 코드 예시 섹션 구축 ===
        # scenario에 merge된 값 우선 사용 (run_single._merge_clue_into_scenarios로 주입됨)
        code_examples = scenario.get("reproduction_code") or clue.get("code_examples", [])
        expected_outputs = scenario.get("expected_outputs") or clue.get("expected_outputs", [])
        actual_outputs = scenario.get("actual_outputs") or clue.get("actual_outputs", [])
        oracle_hints = scenario.get("oracle_hints") or []
        oracle_text = scenario.get("oracle", "")

        oracle_hint_section = ""
        if oracle_hints or oracle_text:
            lines = ["[Synthesized Oracle Hints — use before raw issue text]"]
            if oracle_text:
                lines.append(_clip_prompt_text(oracle_text, _PROMPT_ORACLE_TEXT_CHARS, "oracle_text", prompt_profile))
            for hint in oracle_hints[:_PROMPT_ORACLE_HINTS_MAX]:
                hint_str = str(hint).strip()
                if hint_str:
                    lines.append(f"- {_clip_prompt_text(hint_str, _PROMPT_ORACLE_HINT_CHARS, 'oracle_hints', prompt_profile)}")
            oracle_hint_section = "\n".join(lines) + "\n"
            _mark_prompt_section(prompt_profile, "oracle_hints")

        issue_code_section = ""
        if code_examples:
            code_parts = []
            included_blocks = 0
            for i, block in enumerate(code_examples):
                if included_blocks >= _PROMPT_CODE_EXAMPLES_MAX:
                    break
                if isinstance(block, dict) and block.get("is_system_or_output"):
                    continue
                if not isinstance(block, dict):
                    block = {"code": str(block)}
                ctx = block.get("context_before", "")
                code = block.get("code", "")
                interactive_in = block.get("interactive_input", "")
                interactive_out = block.get("interactive_output", "")

                label = f"Code Block {i + 1}"
                if ctx:
                    label += f" (context: \"{ctx[:100]}\")"

                code_trunc = _clip_prompt_text(code, _PROMPT_CODE_CHARS, "issue_code_examples", prompt_profile)
                code_parts.append(f"### {label}\n```python\n{code_trunc}\n```")
                if interactive_in:
                    code_parts.append(
                        "Interactive input:\n```python\n"
                        f"{_clip_prompt_text(interactive_in, _PROMPT_INTERACTIVE_CHARS, 'interactive_input', prompt_profile)}\n```"
                    )
                if interactive_out:
                    code_parts.append(
                        "Output:\n```\n"
                        f"{_clip_prompt_text(interactive_out, _PROMPT_INTERACTIVE_CHARS, 'interactive_output', prompt_profile)}\n```"
                    )
                included_blocks += 1

            if code_parts:
                issue_code_section += "\n[Issue Reproduction Code from Original Issue]\n"
                issue_code_section += "The following code blocks are extracted from the original GitHub issue.\n"
                issue_code_section += "Use these EXACT patterns to construct your test.\n\n"
                issue_code_section += "\n\n".join(code_parts)
                _mark_prompt_section(prompt_profile, "issue_code_examples")

            if expected_outputs:
                issue_code_section += "\n\n[Expected Correct Output (from issue)]\n"
                issue_code_section += "These outputs represent the CORRECT behavior (what the code should produce after the fix):\n"
                for out in expected_outputs[:_PROMPT_OUTPUTS_MAX]:
                    out_str = out if isinstance(out, str) else str(out)
                    issue_code_section += (
                        "```\n"
                        f"{_clip_prompt_text(out_str, _PROMPT_OUTPUT_CHARS, 'expected_outputs', prompt_profile)}\n"
                        "```\n"
                    )
                _mark_prompt_section(prompt_profile, "expected_outputs")

            if actual_outputs:
                issue_code_section += "\n[Actual Buggy Output (from issue)]\n"
                issue_code_section += "These outputs represent the BUGGY behavior (what the code currently produces):\n"
                for out in actual_outputs[:_PROMPT_OUTPUTS_MAX]:
                    out_str = out if isinstance(out, str) else str(out)
                    issue_code_section += (
                        "```\n"
                        f"{_clip_prompt_text(out_str, _PROMPT_OUTPUT_CHARS, 'actual_outputs', prompt_profile)}\n"
                        "```\n"
                    )
                _mark_prompt_section(prompt_profile, "actual_outputs")

            issue_code_section += "\n"

        if not code_examples and (expected_outputs or actual_outputs):
            if expected_outputs:
                issue_code_section += "\n[Expected Correct Output (from issue)]\n"
                for out in expected_outputs[:_PROMPT_OUTPUTS_MAX]:
                    issue_code_section += (
                        "```\n"
                        f"{_clip_prompt_text(out, _PROMPT_OUTPUT_CHARS, 'expected_outputs', prompt_profile)}\n"
                        "```\n"
                    )
                _mark_prompt_section(prompt_profile, "expected_outputs")
            if actual_outputs:
                issue_code_section += "\n[Actual Buggy Output (from issue)]\n"
                for out in actual_outputs[:_PROMPT_OUTPUTS_MAX]:
                    issue_code_section += (
                        "```\n"
                        f"{_clip_prompt_text(out, _PROMPT_OUTPUT_CHARS, 'actual_outputs', prompt_profile)}\n"
                        "```\n"
                    )
                _mark_prompt_section(prompt_profile, "actual_outputs")

        # === raw_issue_text 섹션 (코드 예시가 없을 때의 fallback이자 보충) ===
        raw_issue_text = clue.get("raw_issue_text", "")
        raw_issue_section = ""
        if raw_issue_text:
            truncated = _clip_prompt_text(raw_issue_text, _PROMPT_RAW_ISSUE_CHARS, "raw_issue_text", prompt_profile)
            raw_issue_section = f"""
[Full Issue Description]
{truncated}
"""
            _mark_prompt_section(prompt_profile, "raw_issue_text")

        # runner별 테스트 구조 힌트 (모든 레포에 대해 적응적으로 생성)
        framework_constraint = ""
        if runner == "django-test":
            framework_constraint = """
[CRITICAL: Test Structure — Django Test Runner]
This project runs tests via Django's test runner (not raw pytest).
- MUST use `from django.test import TestCase` (NOT `from unittest import TestCase`)
- MUST inherit from django.test.TestCase (or SimpleTestCase if no DB needed)
- Test method MUST be inside the class (standalone `def test_xxx():` is NOT discovered by Django runner)
- Class name MUST start with "Test", method name must start with "test_"
- NEVER use relative imports (e.g. `from .models import X` will FAIL — use the full module path)
- NEVER import from the `tests` package (e.g. `from tests.xxx import Y` will FAIL with ModuleNotFoundError).
  Instead, import Django app modules directly (e.g. `from django.xxx import Y`).
- Example structure:
  from django.test import TestCase

  class TestMyFeature(TestCase):
      def test_behavior(self):
          self.assertEqual(expected, actual)

FORBIDDEN (these will cause OperationalError, LookupError, or test not collected):
- Defining new Django models inline: `class MyModel(models.Model): ...` → no migration, no table
- Using models NOT imported in [Existing Imports in Target Test File] or [Target Test File Symbol Catalog]
- Accessing the database without inheriting from django.test.TestCase
- Creating custom app labels or INSTALLED_APPS entries
- Standalone test functions with `self` parameter: `def test_foo(self):` outside a class
"""
        elif runner == "unittest":
            framework_constraint = """
[Test Structure — unittest/pytest-compatible Repository]
This repository has unittest-style tests, but many files are still collected by pytest.
- Mirror the exact style shown in [Example: Actual Test Methods] when available.
- If the target file mostly uses top-level test functions, write one top-level `def test_*():`.
- If the target file uses TestCase classes, add one method inside a compatible TestCase subclass.
- Use `self.assert*()` only inside TestCase methods; use plain `assert` in top-level functions.
- Do NOT force a new unittest.TestCase class when the target file's existing tests are top-level functions.
"""
        elif runner == "sympy-bin-test":
            framework_constraint = """
[CRITICAL: Test Structure — SymPy (pytest)]
This project runs tests via pytest (sympy test files are pytest-compatible).
- Tests are top-level functions starting with "test_"
- Use plain `assert` statements directly (NOT self.assert*)
- pytest fixtures (e.g. tmp_path, monkeypatch) are allowed if needed
- Keep the test simple and self-contained

[CRITICAL: SymPy Import Rules]
- Import ONLY from modules listed in [Available Imports from Repository]
- NEVER import from deep sub-modules like `sympy.sets.sets`, `sympy.core.core`, etc.
  unless they are explicitly listed in [Available Imports]
- ALWAYS prefer top-level imports: `from sympy import symbols, Function, Lambda, Eq, solve, S`
- If unsure whether a symbol exists in a sub-module, use `from sympy import XYZ` (top-level namespace)
- Common safe top-level imports: symbols, Function, Lambda, Eq, solve, S, I, oo, pi,
  Rational, Integer, Float, Matrix, Symbol, Expr, Add, Mul, Pow, Number
"""

        # matplotlib 특수 처리: Agg backend 필수 지시
        if "matplotlib" in instance.repo.lower():
            framework_constraint += """
[CRITICAL: matplotlib Backend]
REQUIRED at the very TOP of append_block (before any other matplotlib/pyplot import):
  import matplotlib
  matplotlib.use('Agg')  # MUST be set before importing matplotlib.pyplot
Do NOT call plt.show() — the test environment has no display.
"""

        # sphinx 특수 처리: sphinx.testing.fixtures 의존성 문제 회피
        if "sphinx" in instance.repo.lower():
            framework_constraint += """
[CRITICAL: sphinx Test Constraints]
- NEVER use @pytest.mark.sphinx decorator — it loads sphinx.testing.fixtures which has a broken
  dependency (docutils.utils.roman removed in Python 3.9+) and will cause ImportError
- NEVER use 'app', 'status', 'warning' as test function parameters (these are sphinx fixtures)
- Write STANDALONE tests: import sphinx modules directly, no fixture injection
- If you need a Sphinx app, create it directly:
    import tempfile, os
    from sphinx.application import Sphinx
    with tempfile.TemporaryDirectory() as tmpdir:
        app = Sphinx(srcdir, confdir, outdir, doctreedir, buildername)
"""

        # pytest-dev 특수 처리: 외부 패키지 import 금지
        if "pytest" in instance.repo.lower():
            framework_constraint += """
[CRITICAL: pytest-dev Test Constraints]
- NEVER import external packages unrelated to pytest (e.g. youtube_dl, roman, requests, etc.)
- Only import from: pytest, _pytest.*, testing.*, and Python standard library
- Use pytester fixture or tmp_path for creating temporary test files
- For skip/PDB issues, create a temporary test file and run pytester; do NOT append a skipped class with undefined names into the repository test file.
- Ensure all f-strings, brackets, and quotes are properly closed (no SyntaxError)
"""

        # django 추가 import 제약 (기존 django-test 블록 이후에 추가)
        if runner == "django-test":
            framework_constraint += """
[CRITICAL: Django Import Rules]
- NEVER import from app names that are NOT listed in [Available Imports from Repository]
  (e.g. `from app import X`, `from myapp import X`, `from myapp1 import X` are FORBIDDEN
   unless that exact module path appears in [Available Imports])
- ONLY import from: django.*, and the exact module paths shown in [Available Imports]
- If a symbol you need is not in [Available Imports], do NOT use it
- Use model classes from [Target Test File Symbol Catalog]. If no suitable model is listed,
  prefer SimpleTestCase and a public API/value-level reproduction that avoids database setup.
- For query tests that touch Author.objects or Meta.ordering, mirror existing tests and add explicit .order_by(...) when warnings would become errors.
"""

        if "requests" in instance.repo.lower():
            framework_constraint += """
[CRITICAL: Requests HTTP/Auth Test Constraints]
- Do not make external network calls.
- For digest auth issues, use existing local helper/mock challenge flow when Authorization depends on WWW-Authenticate challenge data.
- A bare PreparedRequest with HTTPDigestAuth but no challenge is usually not enough to test digest header behavior.
"""

        if any(name in instance.repo.lower() for name in ("sphinx", "matplotlib", "seaborn")):
            framework_constraint += """
[CRITICAL: Public Semantic Oracle Constraints]
- Do not inspect private attributes or long raw rendered strings.
- Prefer public artists, axes, legend/text objects, rendered node properties, or minimal semantic markers.
"""

        # 타겟 파일의 실제 test 메서드 예시 (타겟 파일 우선, fallback은 context snippet)
        display_example = target_test_example or test_example
        if display_example:
            display_example_trunc = _clip_prompt_text(
                display_example,
                _PROMPT_TEST_EXAMPLE_CHARS,
                "target_test_example",
                prompt_profile,
            )
            example_source = f"from {target_test_file}" if target_test_example else "from this repository"
            framework_constraint += f"""
[Example: Actual Test Methods {example_source} — Mirror This Style Exactly]
These are REAL test methods from the exact file where your test will be appended.
Follow the same class hierarchy, decorator usage, and assertion style:
```python
{display_example_trunc}
```
"""
            _mark_prompt_section(prompt_profile, "target_test_example")

        runtime_error_section = ""
        if runtime_error_hint:
            hint_trunc = runtime_error_hint[:500] + "…" if len(runtime_error_hint) > 500 else runtime_error_hint
            self_fix = (
                '"missing 1 required positional argument: self" → for Django/TestCase files, put the method inside a TestCase class; '
                'for pytest-style files, remove the self parameter from the top-level test function'
            )
            runtime_error_section = f"""
[Previous Execution Error — MUST FIX]
The previous attempt ran in the test environment but produced this runtime error:
  {hint_trunc}
You MUST fix this error in your new attempt. Common fixes:
- {self_fix}
- "RuntimeError: Model class ... INSTALLED_APPS" → only use model classes already imported from the existing test file; do NOT define new models
- "no such table" → use SimpleTestCase instead of TestCase, or mock DB calls
- "AttributeError" / "ImportError" → verify the attribute/module actually exists in this repo version
"""

        prompt = f"""
You are generating a Python issue-reproducing test for a real repository.

Task:
Generate one focused reproduction test for the issue below.
The test must be designed to fail on the pre-patch code because of the issue behavior described.
The test must PASS after the bug is fixed (i.e., the assertion reflects correct/expected behavior).
{runtime_error_section}{framework_constraint}
Repository: {instance.repo}
Instance ID: {instance.instance_id}
Base Commit: {instance.base_commit}

[Issue Clue]
Observed behavior:
{json.dumps(clue.get("observed_behavior", []), ensure_ascii=False, indent=2)}

Expected behavior:
{json.dumps(clue.get("expected_behavior", []), ensure_ascii=False, indent=2)}

Reproduction conditions:
{json.dumps(clue.get("repro_conditions", []), ensure_ascii=False, indent=2)}

Related identifiers:
- functions: {issue_functions}
- classes: {issue_classes}
{f"- error/exception keywords: {issue_error_keywords}" + chr(10) if issue_error_keywords else ""}{self._build_fault_location_section(clue)}{oracle_hint_section}{raw_issue_section}
{issue_code_section}
[Code Context]
Framework: {project_framework} (runner: {runner})
Candidate source files: {source_candidates}
Candidate test files: {test_candidates}
{conftest_section}{required_fixtures_section}
{test_symbol_section}
[Available Imports from Repository]
CRITICAL: You MUST explicitly import every symbol you use. Do NOT assume anything is pre-imported.
If you use a symbol listed below, you MUST include the corresponding import in your append_block.
Only use imports from the following verified module paths:
{import_map_text}
{existing_imports_section}
[Validated Scenario]
{json.dumps(_truncate_scenario_for_prompt(scenario), ensure_ascii=False, separators=(',', ':'))}

[Oracle Contract — follow this before writing assertions]
oracle_type: {(scenario.get("oracle_contract") or {}).get("oracle_type") or scenario.get("oracle_type", "")}
oracle_source: {(scenario.get("oracle_contract") or {}).get("oracle_source") or scenario.get("oracle_source", "")}
rule: {(scenario.get("oracle_contract") or {}).get("rule", "")}

[CRITICAL: Bug Reproduction Contract]
The test must FAIL on buggy pre-patch code and PASS on fixed post-patch code.
Write assertions from the fixed behavior perspective:
- If [Expected Correct Output] exists, assert that exact value/output. Use np.testing for arrays and pytest.approx for floats.
- If only buggy output is known, assert the function return value or public state is not that buggy value.
- If the fix should remove an exception, assert the success path; do not use pytest.raises for that case.
- If the fix should introduce/correct an exception, pytest.raises must wrap the actual triggering call.
- Last resort only: type/non-None/length structural assertions.

Forbidden oracle patterns:
- bug symptom assertions: BUG_STRING in str(exc), exact exception message, raw rendered HTML/LaTeX/Sphinx strings
- negative assertion on a local constant such as expected_matrix/baseline_value/correct_output
- guessed exact expected arrays/values not stated by the issue
- numpy direct equality, @image_comparison, external network calls, private attribute reads, Django inline models
- warning count/type alone; assert the returned value or public state instead

[Generation Constraints]
1. Do not invent issue-irrelevant APIs or identifiers.
2. Prefer using the validated target function and target source file.
3. Prefer inserting into this test file if suitable: {target_test_file}
4. Prefer ONE focused reproduction test function or method. If helpers are needed, helper names must not start with test_.
5. Use appropriate assertions: plain `assert` for pytest/sympy, `self.assert*()` for unittest/Django.
6. The test should reproduce the issue described by the scenario, not a generic failure.
7. NEVER make real network calls. Use PreparedRequest, mock responses, or existing local HTTP helper classes.
8. CRITICAL: Copy the EXACT API call pattern from [Issue Reproduction Code]. If the issue shows
   code like `func(a, b)`, use that exact signature. Do NOT invent call patterns.
   If [Issue Reproduction Code] shows a class/object, instantiate it the same way.
   The assertion target MUST be the return value or state change of the function under test —
   never a local constant you defined (e.g., expected_matrix, baseline_value, correct_output).
9. Return JSON only. No explanation.
10. CRITICAL: Only import symbols that actually exist in [Available Imports] or [Existing Imports].
11. NEVER define Django model classes inside tests. Use existing models/imports only.
12. Prefer reusing existing imports from the target test file over adding new ones.
13. CRITICAL: Use the EXACT patterns from [Issue Reproduction Code], including operators and class instantiation.
14. For file-based tools, write content to a tempfile and pass the path.
15. NEVER access private attributes (names starting with _) of library objects. Test through public API.
16. For requests/http issues, NEVER call external URLs. Build and inspect PreparedRequest objects or use existing local test helpers.

[Required Output JSON Schema]
{{
  "target_test_file": "string (relative path to test file)",
  "append_block": "complete Python code to append at the END of the target file"
}}

The append_block will be placed verbatim at the end of the target file.
- Include any NEW imports at the top of append_block (only imports NOT already in [Existing Imports] above)
- Then include the complete focused reproduction test function or class
- Do NOT duplicate imports already shown in [Existing Imports]
- The function/method name must start with: test_
- Prefer one test_* function/method total; if multiple are returned, the pipeline may keep only the strongest reproduction candidate.
- The code must be valid Python
- The code must directly exercise: {target_location.get("target_function", "")}
""".strip()

        if prompt_profile is not None:
            prompt_profile["prompt_chars"] = len(prompt)
            prompt_profile["sections_included"] = _dedup_text_items(
                prompt_profile.get("sections_included", [])
            )
            prompt_profile["truncated_sections"] = _dedup_text_items(
                prompt_profile.get("truncated_sections", [])
            )

        return prompt

    def _parse_model_output(
        self,
        raw_response: str,
        scenario: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        text = raw_response.strip() if isinstance(raw_response, str) else str(raw_response or "")

        # ── 1단계: 코드 펜스 안의 JSON 추출 ──
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()

        # ── 2단계: JSON 파싱 시도 ──
        data = None
        json_error = None
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            json_error = str(e)
            data = self._extract_outermost_json(text)

        # ── 3단계: JSON 파싱 실패 → 절단된 JSON 복구 시도 ──
        if data is None:
            data = self._try_repair_truncated_json(raw_response)

        # ── 4단계: 여전히 실패 → Python 코드 직접 추출 ──
        if data is None:
            fallback = self._extract_python_code_fallback(raw_response, scenario)
            if fallback is not None:
                logger.warning("JSON parsing failed (%s); recovered test code via Python fallback.", json_error)
                return fallback
            raise ValueError(f"Model output is not valid JSON and contains no extractable test code. JSON error: {json_error}")

        # ── 이하: data dict 처리 ──
        if not isinstance(data, dict):
            raise ValueError(f"Model output JSON must be an object, got {type(data).__name__}")

        target_test_file = self._coerce_model_string(data.get("target_test_file", ""))
        if not target_test_file:
            target = scenario.get("target_location", {})
            if not isinstance(target, dict):
                target = {}
            target_test_file = self._coerce_model_string(target.get("candidate_test_file")) or (
                scenario.get("relevant_test_files", [""])[0] if scenario.get("relevant_test_files") else ""
            )

        # 새 방식: append_block
        if "append_block" in data:
            append_block = self._coerce_model_code(data["append_block"])
            if not append_block:
                raise ValueError("append_block is empty.")
            # def test_ 없으면 Python fallback으로 재시도
            if "def test_" not in append_block:
                fallback = self._extract_python_code_fallback(raw_response, scenario)
                if fallback is not None:
                    logger.warning("append_block has no test function; recovered via Python fallback.")
                    return fallback
                raise ValueError("append_block has no valid test function.")
            # 기존 테스트와 이름 충돌 방지: test 함수명에 _repro 접미사 보장
            append_block = _ensure_repro_suffix(append_block)
            target_function = scenario.get("target_location", {}).get("target_function", "")
            if target_function and not self._check_target_function_presence(target_function, append_block):
                logger.warning("Generated test may not call target function '%s' — continuing anyway.", target_function)
            return {
                "target_test_file": target_test_file,
                "insert_mode": "append_block",
                "insertion_hint": "end_of_file",
                "imports": [],
                "test_code": append_block,  # 하위 호환용
                "append_block": append_block,
            }

        # 구 방식: imports + test_code → append_block으로 통합 처리
        imports = data.get("imports", [])
        test_code = self._coerce_model_code(data.get("test_code", "")).rstrip()

        if not isinstance(imports, list):
            imports = []
        imports = [x.strip() for x in imports if isinstance(x, str) and x.strip()]

        if not test_code or "def test_" not in test_code:
            raise ValueError("Generated test_code has no valid test function.")

        target_function = scenario.get("target_location", {}).get("target_function", "")
        if target_function and not self._check_target_function_presence(target_function, test_code):
            logger.warning("Generated test may not call target function '%s' — continuing anyway.", target_function)

        # imports + test_code를 하나의 append_block으로 합침
        parts = []
        if imports:
            parts.extend(imports)
            parts.append("")
        parts.append(test_code)
        append_block = "\n".join(parts)

        return {
            "target_test_file": target_test_file,
            "insert_mode": "append_block",
            "insertion_hint": "end_of_file",
            "imports": imports,
            "test_code": test_code,
            "append_block": append_block,
        }

    @staticmethod
    def _coerce_model_string(value: Any) -> str:
        """LLM이 string 필드에 dict/list를 넣어도 가능한 문자열만 뽑는다."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            for key in ("path", "file", "target_test_file", "value", "text", "code"):
                if key in value:
                    coerced = ReproductionTestGenerator._coerce_model_string(value[key])
                    if coerced:
                        return coerced
            return ""
        if isinstance(value, list):
            parts = [
                ReproductionTestGenerator._coerce_model_string(item)
                for item in value
            ]
            return "\n".join(part for part in parts if part).strip()
        return str(value).strip()

    @staticmethod
    def _coerce_model_code(value: Any) -> str:
        """append_block/test_code 필드를 Python 코드 문자열로 정규화한다."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            for key in ("code", "append_block", "test_code", "content", "text", "value"):
                if key in value:
                    coerced = ReproductionTestGenerator._coerce_model_code(value[key])
                    if coerced:
                        return coerced
            return ""
        if isinstance(value, list):
            parts = [
                ReproductionTestGenerator._coerce_model_code(item)
                for item in value
            ]
            return "\n".join(part for part in parts if part).strip()
        return str(value).strip()

    @staticmethod
    def _extract_outermost_json(text: str) -> Optional[dict]:
        """텍스트에서 가장 바깥쪽 { ... } 블록을 brace-depth counting으로 추출 후 JSON 파싱."""
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        return None
        return None

    @staticmethod
    def _try_repair_truncated_json(text: str) -> Optional[dict]:
        """max_tokens 초과로 JSON이 중간에 잘린 경우 복구 시도.

        target_test_file과 append_block의 시작을 찾아, 잘린 Python 코드라도 추출한다.
        """
        file_match = re.search(r'"target_test_file"\s*:\s*"([^"]+)"', text)
        if not file_match:
            return None
        target_file = file_match.group(1)

        block_start = re.search(r'"append_block"\s*:\s*"', text)
        if not block_start:
            return None

        raw = text[block_start.end():]
        decoded: List[str] = []
        i = 0
        while i < len(raw):
            ch = raw[i]
            if ch == "\\" and i + 1 < len(raw):
                nc = raw[i + 1]
                mapping = {"n": "\n", "t": "\t", '"': '"', "\\": "\\", "'": "'", "r": "\r"}
                decoded.append(mapping.get(nc, nc))
                i += 2
            elif ch == '"':
                break  # JSON 문자열 종료
            else:
                decoded.append(ch)
                i += 1

        code = "".join(decoded).strip()
        if not code or "def test_" not in code:
            return None

        logger.warning("Repaired truncated JSON: recovered %d chars of append_block.", len(code))
        return {
            "target_test_file": target_file,
            "insert_mode": "append_block",
            "insertion_hint": "end_of_file",
            "imports": [],
            "test_code": code,
            "append_block": code,
        }

    @staticmethod
    def _extract_python_code_fallback(text: str, scenario: Dict[str, Any]) -> Optional[dict]:
        """LLM이 JSON 대신 Python 코드를 직접 출력했거나 코드 펜스만 있는 경우 처리.

        우선순위:
        1. ```python ... ``` 코드 펜스 안의 test 코드
        2. 응답 전체에서 def test_ / class Test 로 시작하는 블록
        """
        target_location = scenario.get("target_location", {})
        target_test_file = target_location.get("candidate_test_file") or (
            scenario.get("relevant_test_files", [""])[0] if scenario.get("relevant_test_files") else ""
        )

        def _make_result(code: str) -> dict:
            return {
                "target_test_file": target_test_file,
                "insert_mode": "append_block",
                "insertion_hint": "end_of_file",
                "imports": [],
                "test_code": code,
                "append_block": code,
            }

        # 1) Python 코드 펜스
        for m in re.finditer(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL):
            code = m.group(1).strip()
            if "def test_" in code:
                return _make_result(code)

        # 2) 텍스트 전체에서 def test_ / class Test 블록 추출
        m = re.search(r"^(?:def test_|class Test)", text, re.MULTILINE)
        if m:
            code = text[m.start():].strip()
            # 이후 불필요한 설명 텍스트 제거: 연속된 non-indented non-def/class 줄이 나오면 자름
            lines = code.splitlines()
            kept: List[str] = []
            for line in lines:
                if kept and line and not line[0].isspace() and not line.startswith(("def ", "class ", "@", "#")):
                    break
                kept.append(line)
            code = "\n".join(kept).strip()
            if "def test_" in code:
                return _make_result(code)

        return None

    @staticmethod
    def _check_target_function_presence(target_function: str, test_code: str) -> bool:
        """Check if the target function appears in test code (flexible).

        Dunder methods are invoked indirectly (e.g. obj() → __call__), so we
        only skip those.  Everything else is checked via string containment or
        a call-pattern regex.
        """
        # dunder methods: used indirectly (e.g. obj() → __call__)
        if target_function.startswith("__") and target_function.endswith("__"):
            return True

        # very short names (<=2 chars) cause too many false positives
        if len(target_function) <= 2:
            return True

        # direct string containment (covers attribute access, keyword args, etc.)
        if target_function in test_code:
            return True

        # for private methods, also check the bare name without leading underscores
        if target_function.startswith("_"):
            bare = target_function.lstrip("_")
            if bare and bare in test_code:
                return True

        # function call pattern: .func_name( or func_name(
        call_pattern = re.compile(
            r'(?:^|[.\s(,=])' + re.escape(target_function) + r'\s*\(',
            re.MULTILINE,
        )
        if call_pattern.search(test_code):
            return True

        return False

    def _validate_generated_code(
        self,
        parsed: Dict[str, Any],
        repo_path: str,
        context: Dict[str, Any],
        clue: Optional[Dict[str, Any]] = None,
        scenario: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        생성된 코드에 대해 정적 검증을 수행한다.
        1) 구문 검증 (ast.parse)
        2) import 경로가 repo 내에 존재하는지 확인
        3) test_code에서 사용하지 않는 import 제거
        4) 이슈 코드 예시의 핵심 식별자가 테스트에 포함되는지 soft 검증
        잘못된 import를 교정할 수 있으면 fixed_imports를 반환한다.
        """
        errors: List[str] = []
        imports = list(parsed.get("imports", []))
        test_code = parsed.get("test_code", "")
        target_test_file = parsed.get("target_test_file", "")

        repo = Path(repo_path)
        available_imports = context.get("available_imports", {})

        # append_block 방식: 단순 append 후 구문 검증
        if parsed.get("insert_mode") == "append_block":
            append_block = parsed.get("append_block", "")
            try:
                ast.parse(append_block)
            except SyntaxError as e:
                errors.append(f"append_block SyntaxError: {e}")
                return ValidationResult(is_valid=False, errors=errors)
            test_count = _count_generated_tests(append_block)
            if test_count == 0:
                errors.append(
                    "CRITICAL: generated append_block must define at least one test function/method."
                )
                return ValidationResult(is_valid=False, errors=errors)
            if test_count > 1:
                errors.append(
                    f"CRITICAL: generated append_block defines {test_count} test functions/methods; "
                    "keep exactly one focused reproduction test."
                )
                return ValidationResult(is_valid=False, errors=errors)
            test_file_abs = repo / target_test_file if target_test_file else None
            original_content = ""
            if test_file_abs and test_file_abs.exists():
                original_content = read_text(test_file_abs)
                trial_content = original_content.rstrip() + "\n\n" + append_block + "\n"
                try:
                    ast.parse(trial_content)
                except SyntaxError as e:
                    errors.append(f"appended file SyntaxError: {e}")
                    return ValidationResult(is_valid=False, errors=errors)
            # 관용 alias가 import 없이 쓰이면 invalid 처리
            undefined = _detect_missing_common_aliases(append_block, original_content)
            if undefined:
                errors.append(
                    f"Missing imports for common aliases: {undefined}. "
                    "Add explicit import statements at the top of append_block."
                )
                return ValidationResult(is_valid=False, errors=errors)
            oracle_risks = _detect_blocking_oracle_risks(append_block, clue=clue)
            if oracle_risks:
                errors.extend(oracle_risks)
                return ValidationResult(is_valid=False, errors=errors)
            semantic_risks = _detect_semantic_risk_flags(
                append_block,
                clue,
                context,
                scenario,
                original_content=original_content,
            )
            if semantic_risks:
                errors.extend(f"CRITICAL: semantic risk: {risk}" for risk in semantic_risks)
                return ValidationResult(is_valid=False, errors=errors)
            return ValidationResult(is_valid=True, errors=[], fixed_imports=None)

        # 구 방식: 1) 구문 검증 — test_code 자체
        try:
            ast.parse(test_code)
        except SyntaxError as e:
            errors.append(f"test_code SyntaxError: {e}")
            return ValidationResult(is_valid=False, errors=errors)
        test_count = _count_generated_tests(test_code)
        if test_count == 0:
            errors.append(
                "CRITICAL: generated test_code must define at least one test function/method."
            )
            return ValidationResult(is_valid=False, errors=errors)
        if test_count > 1:
            errors.append(
                f"CRITICAL: generated test_code defines {test_count} test functions/methods; "
                "keep exactly one focused reproduction test."
            )
            return ValidationResult(is_valid=False, errors=errors)

        # 2) 전체 파일 구문 검증 (imports + test_code)
        test_file_abs = repo / target_test_file if target_test_file else None
        if test_file_abs and test_file_abs.exists():
            original_content = read_text(test_file_abs)
            trial_content = self._build_modified_test_file_content(
                original_content=original_content,
                imports=imports,
                test_code=test_code,
            )
            try:
                ast.parse(trial_content)
            except SyntaxError as e:
                errors.append(f"modified file SyntaxError: {e}")
                return ValidationResult(is_valid=False, errors=errors)

        # 3) import 검증 — 각 import 문의 심볼이 repo에 존재하는지
        validated_imports: List[str] = []
        import_errors: List[str] = []

        # 기존 테스트 파일의 import를 수집 (이미 있는 건 검증 skip)
        existing_imports: set = set()
        if test_file_abs and test_file_abs.exists():
            original_content = read_text(test_file_abs)
            for line in original_content.splitlines():
                if (line.startswith("import ") or line.startswith("from ")) and not line.startswith((" ", "\t")):
                    existing_imports.add(line.strip())

        # unittest 자동 주입: test_code에서 unittest.를 사용하는데 import가 없으면 추가
        if "unittest." in test_code or "(unittest.TestCase)" in test_code:
            has_unittest = (
                "import unittest" in existing_imports
                or any("import unittest" in imp for imp in imports)
            )
            if not has_unittest:
                imports = ["import unittest"] + imports

        for imp in imports:
            # 기존 파일에 이미 있는 import는 중복이므로 제거
            if imp in existing_imports:
                continue

            check = self._check_import_validity(imp, repo, available_imports)
            if check is True:
                validated_imports.append(imp)
            elif isinstance(check, str):
                # 교정된 import
                validated_imports.append(check)
                import_errors.append(f"corrected: '{imp}' -> '{check}'")
            else:
                # 사용 여부 확인: test_code에서 실제 사용하는 심볼인지
                symbols = self._extract_imported_symbols(imp)
                used = any(sym in test_code for sym in symbols)
                if used:
                    import_errors.append(f"import path unverifiable (keeping, in use): {imp}")
                    validated_imports.append(imp)
                else:
                    import_errors.append(f"removed unused/unverifiable import: {imp}")

        if import_errors:
            errors.extend(import_errors)

        # 4) TestCase 상속 검증.  Only Django's runner strictly requires
        # class-based test methods here; several pytest-collected repositories
        # are classified as "unittest" because they contain TestCase classes
        # while still accepting top-level test functions.
        runner = context.get("project_test_style", {}).get("runner", "pytest")
        if runner == "django-test":
            try:
                tree = ast.parse(test_code)
            except SyntaxError:
                pass  # 이미 위에서 SyntaxError로 처리됨
            else:
                has_testcase_class = False
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        for base in node.bases:
                            if isinstance(base, ast.Name):
                                base_name = base.id
                            elif isinstance(base, ast.Attribute):
                                base_name = base.attr
                            else:
                                base_name = ""
                            if "TestCase" in base_name or "SimpleTestCase" in base_name:
                                has_testcase_class = True
                                break

                standalone_test_funcs = [
                    node.name
                    for node in ast.iter_child_nodes(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name.startswith("test")
                ]

                if not has_testcase_class:
                    func_hint = (
                        f" (found standalone functions: {', '.join(standalone_test_funcs[:2])})"
                        if standalone_test_funcs else ""
                    )
                    errors.append(
                        f"CRITICAL: {runner} runner requires a class inheriting from "
                        f"django.test.TestCase or SimpleTestCase.{func_hint} "
                        "Wrap all test methods inside a TestCase subclass."
                    )

        # 5) 이슈 코드 예시 alignment soft 검증
        if clue:
            self._soft_validate_issue_alignment(test_code, clue, errors)

        errors.extend(_detect_blocking_oracle_risks(test_code, clue=clue))
        errors.extend(
            f"CRITICAL: semantic risk: {risk}"
            for risk in _detect_semantic_risk_flags(
                test_code,
                clue,
                context,
                scenario,
                original_content=original_content if test_file_abs and test_file_abs.exists() else "",
            )
        )

        # critical error가 있으면 실패 (SyntaxError, TestCase 미상속 등)
        has_critical = any("SyntaxError" in e or "CRITICAL:" in e for e in errors)

        return ValidationResult(
            is_valid=not has_critical,
            errors=errors,
            fixed_imports=validated_imports,
        )

    def _soft_validate_issue_alignment(
        self,
        test_code: str,
        clue: Dict[str, Any],
        errors: List[str],
    ) -> None:
        """이슈 원문의 핵심 식별자가 생성된 테스트에 포함되는지 soft 검증 (경고만)."""
        code_examples = clue.get("code_examples", [])
        if not code_examples:
            return

        # 코드 예시에서 class/function 호출 패턴 추출
        issue_identifiers: set = set()
        for block in code_examples:
            code = block.get("code", "") + " " + block.get("interactive_input", "")
            # ClassName( 패턴
            issue_identifiers.update(re.findall(r'\b([A-Z][A-Za-z0-9_]+)\s*\(', code))
            # function_call( 패턴
            issue_identifiers.update(re.findall(r'\b([a-z_][a-z0-9_]+)\s*\(', code))

        # 너무 일반적인 식별자 제외
        generic = {"print", "range", "len", "type", "str", "int", "float", "list",
                   "dict", "set", "tuple", "isinstance", "assert", "True", "False",
                   "None", "array", "import", "from", "def", "class", "return"}
        issue_identifiers -= generic

        if not issue_identifiers:
            return

        found = {ident for ident in issue_identifiers if ident in test_code}
        missing = issue_identifiers - found

        if found:
            hit_ratio = len(found) / len(issue_identifiers)
            if hit_ratio < 0.3 and len(missing) > 2:
                logger.warning(
                    "[soft-validation] issue code identifier alignment low: %.0f%% (%d/%d). "
                    "missing: %s",
                    hit_ratio * 100, len(found), len(issue_identifiers),
                    ", ".join(sorted(missing)[:10]),
                )
                errors.append(
                    f"[warning] issue code identifier alignment low: "
                    f"{len(found)}/{len(issue_identifiers)} matched. "
                    f"missing: {', '.join(sorted(missing)[:5])}"
                )
        else:
            logger.warning(
                "[soft-validation] no issue code identifiers found in test: %s",
                ", ".join(sorted(issue_identifiers)[:10]),
            )
            errors.append(
                f"[warning] no issue code identifiers found in test: "
                f"{', '.join(sorted(issue_identifiers)[:5])}"
            )

    def _check_import_validity(
        self,
        import_line: str,
        repo: Path,
        available_imports: Dict[str, List[str]],
    ) -> Any:
        """
        import 문이 repo 내에서 유효한지 확인한다.
        Returns:
            True: 유효
            str: 교정된 import 문
            False: 확인 불가
        """
        # 표준 라이브러리 / 서드파티는 검증 skip → True
        stdlib_prefixes = {
            "os", "sys", "re", "json", "math", "collections", "itertools",
            "functools", "pathlib", "typing", "abc", "copy", "io", "datetime",
            "logging", "unittest", "dataclasses", "contextlib", "textwrap",
            "warnings", "traceback", "inspect", "importlib", "operator",
        }
        thirdparty_prefixes = {
            "pytest", "numpy", "np", "scipy", "matplotlib", "pandas",
            "requests", "yaml", "toml", "setuptools", "pkg_resources",
            "django", "flask", "sqlalchemy", "celery", "redis",
            "astropy", "sympy", "sphinx", "sklearn", "cv2", "PIL",
        }

        stripped = import_line.strip()

        # relative import는 독립 실행 환경에서 항상 실패 → 즉시 거부
        if stripped.startswith("from ."):
            return False

        # "import X" or "from X import Y" 에서 최상위 모듈 추출
        if stripped.startswith("from "):
            match = re.match(r"from\s+([\w.]+)\s+import\s+(.*)", stripped)
            if not match:
                return True  # 파싱 불가 → skip
            module_path = match.group(1)
            import_names = [n.strip().split(" as ")[0].strip() for n in match.group(2).split(",")]
        elif stripped.startswith("import "):
            match = re.match(r"import\s+([\w.]+)", stripped)
            if not match:
                return True
            module_path = match.group(1)
            import_names = []
        else:
            return True

        top_module = module_path.split(".")[0]

        if top_module in stdlib_prefixes or top_module in thirdparty_prefixes:
            return True

        # available_imports에서 확인
        if module_path in available_imports:
            available_symbols = set(available_imports[module_path])
            if not import_names:
                return True
            missing = [n for n in import_names if n not in available_symbols]
            if not missing:
                return True
            # 누락된 심볼이 있으면 다른 모듈에서 찾기
            for alt_module, alt_symbols in available_imports.items():
                if all(n in alt_symbols for n in import_names):
                    corrected = f"from {alt_module} import {', '.join(import_names)}"
                    return corrected
            # 사용된 심볼만 유효한 것으로 필터
            valid_names = [n for n in import_names if n in available_symbols]
            if valid_names and len(valid_names) < len(import_names):
                corrected = f"from {module_path} import {', '.join(valid_names)}"
                return corrected
            return False

        # repo 파일 시스템에서 모듈 존재 여부
        module_file = repo / module_path.replace(".", "/")
        if (module_file.with_suffix(".py")).exists() or (module_file / "__init__.py").exists():
            # 모듈 파일은 존재 — from X import Y 인 경우 심볼도 검증
            if import_names:
                py_file = module_file.with_suffix(".py")
                init_file = module_file / "__init__.py"
                target_file = py_file if py_file.exists() else init_file
                if target_file.exists():
                    try:
                        src = target_file.read_text(encoding="utf-8", errors="replace")
                        defined = set(re.findall(r"(?:def|class)\s+(\w+)", src))
                        # __all__ exports
                        all_match = re.search(r"__all__\s*=\s*\[([^\]]+)\]", src)
                        if all_match:
                            defined.update(
                                n.strip().strip("'\"")
                                for n in all_match.group(1).split(",")
                            )
                        missing = [n for n in import_names if n not in defined]
                        if missing:
                            valid = [n for n in import_names if n in defined]
                            if valid:
                                corrected = f"from {module_path} import {', '.join(valid)}"
                                return corrected
                    except Exception:
                        pass
            return True

        # 부모 모듈에서 찾기
        if "." in module_path:
            parent = module_path.rsplit(".", 1)[0]
            if parent in available_imports:
                parent_symbols = set(available_imports[parent])
                if import_names and all(n in parent_symbols for n in import_names):
                    corrected = f"from {parent} import {', '.join(import_names)}"
                    return corrected

        return False

    def _extract_imported_symbols(self, import_line: str) -> List[str]:
        """import 문에서 가져오는 심볼 이름 추출"""
        stripped = import_line.strip()
        if stripped.startswith("from "):
            match = re.match(r"from\s+[\w.]+\s+import\s+(.*)", stripped)
            if match:
                parts = match.group(1).split(",")
                symbols = []
                for p in parts:
                    p = p.strip()
                    if " as " in p:
                        symbols.append(p.split(" as ")[-1].strip())
                    else:
                        symbols.append(p.strip())
                return symbols
        elif stripped.startswith("import "):
            match = re.match(r"import\s+([\w.]+)(?:\s+as\s+(\w+))?", stripped)
            if match:
                alias = match.group(2) or match.group(1).split(".")[-1]
                return [alias]
        return []

    def _extract_test_examples_from_file(
        self,
        file_path: str,
        n: int = 2,
        max_lines_per: int = 30,
    ) -> str:
        """타겟 테스트 파일에서 완성된 test_ 메서드 n개를 추출 (decorator 포함).

        LLM이 타겟 파일의 실제 클래스 구조, decorator 패턴, assert 스타일을 직접 볼 수 있도록
        한다. 이를 통해 import 삽입 위치 오류(decorator 앞 삽입 등)를 방지한다.
        """
        path = Path(file_path)
        if not path.exists():
            return ""
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(src)
        except Exception:
            return ""

        lines = src.splitlines()
        snippets: List[str] = []

        # TestClass 내부 test_ 메서드 우선 (ast.walk는 순서 보장 안 됨 → 직접 순회)
        # "Test"로 시작하거나 "Tests"로 끝나는 클래스 포함
        for node in ast.parse(src).body:  # top-level 문장만
            if isinstance(node, ast.ClassDef) and (
                node.name.startswith("Test") or node.name.endswith("Tests") or node.name.endswith("TestCase")
            ):
                class_header = lines[node.lineno - 1]  # e.g. "class TestFoo(TestCase):"
                for item in node.body:
                    if (
                        isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and item.name.startswith("test_")
                    ):
                        # decorator부터 포함
                        start = (
                            item.decorator_list[0].lineno - 1
                            if item.decorator_list
                            else item.lineno - 1
                        )
                        end = min(start + max_lines_per, len(lines))
                        method_lines = "\n".join(lines[start:end])
                        # 클래스 컨텍스트(헤더 한 줄)도 포함
                        snippet = f"{class_header}\n    ...\n{method_lines}"
                        snippets.append(snippet)
                        if len(snippets) >= n:
                            break
                if snippets:
                    break

        # TestClass 없으면 standalone test_ 함수
        if not snippets:
            for node in ast.walk(tree):
                if (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name.startswith("test_")
                ):
                    start = (
                        node.decorator_list[0].lineno - 1
                        if node.decorator_list
                        else node.lineno - 1
                    )
                    end = min(start + max_lines_per, len(lines))
                    snippets.append("\n".join(lines[start:end]))
                    if len(snippets) >= n:
                        break

        return "\n\n".join(snippets)

    def _extract_import_block(self, file_path: Path) -> str:
        """파일에서 top-level import 블록만 추출"""
        try:
            content = read_text(file_path)
        except Exception:
            return ""

        lines = content.splitlines()
        import_lines: List[str] = []
        seen_import = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")) and not line.startswith((" ", "\t")):
                import_lines.append(line)
                seen_import = True
            elif stripped == "" or stripped.startswith("#"):
                if seen_import:
                    import_lines.append(line)
            else:
                if seen_import:
                    break

        return "\n".join(import_lines).strip()

    def _build_fix_prompt(
        self,
        original_prompt: str,
        previous_response: str,
        error_message: str,
        attempt: int,
    ) -> str:
        """이전 시도의 에러를 포함한 compact 수정 요청 프롬프트.

        Retry에서 original prompt 전체를 반복하면 토큰이 기하급수적으로 증가한다.
        여기서는 task summary, oracle contract, 핵심 에러, 이전 응답 일부만 전달한다.
        """
        # JSON 파싱 실패인지 감지 → 더 구체적인 포맷 지침 추가
        is_json_fail = (
            "not valid JSON" in error_message
            or "JSONDecodeError" in error_message
            or "Model output parsing" in error_message
        )
        if is_json_fail:
            format_hint = """
[CRITICAL FORMAT REQUIREMENT]
Your previous response could not be parsed as JSON. You MUST return ONLY a valid JSON object.
- Do NOT write any explanation before or after the JSON.
- Do NOT use markdown outside the JSON value.
- The "append_block" value must be a JSON string: escape newlines as \\n and quotes as \\".
- Example of correct response format:
  {"target_test_file": "path/to/test_file.py", "append_block": "def test_foo():\\n    assert bar() == 1\\n"}
- If your test code is long, keep it concise to avoid truncation.
"""
        else:
            format_hint = ""

        oracle_retry_hint = ""
        if "retry required" in error_message or "oracle" in error_message.lower():
            oracle_retry_hint = self._oracle_rewrite_hint(error_message)

        semantic_retry_hint = ""
        if "semantic risk" in error_message:
            semantic_retry_hint = """
[SEMANTIC REWRITE REQUIREMENT]
Your previous test used APIs or setup that are not grounded in the issue target.
- Do not introduce unrelated high-level APIs just to manufacture data.
- Directly exercise the target/source API from [Validated Scenario] and [Issue Reproduction Code].
- For pytest skip/PDB issues, create a temporary test file with pytester/tmp_path instead of appending a skipped class with undefined names.
- For Django query issues, avoid ordering warnings by mirroring existing query tests and adding order_by(...) when needed.
- For Django model/query tests, use only models/helpers already imported by the target test file or explicitly available in repository imports.
- Do not use placeholder symbols such as xxx, MockModel, Dummy*, FooModel, or BarModel.
- If target_function_not_called appears, rewrite the stimulus to directly call the target/source API from the scenario.
- If target_function_public_api_rewrite appears, call the target function from [Validated Scenario],
  or call the public wrapper shown in the issue that reaches the same behavior. Do not switch to an unrelated API.
- If private_attribute_public_api_rewrite appears, replace any `._private` assertion with a public API assertion:
  use return values, public object state, Matplotlib axis/legend accessors, Seaborn artist/axis properties,
  or a small issue-visible semantic invariant.
- For Requests digest auth, use the repository's digest auth helper/mock challenge flow instead of a bare PreparedRequest with no challenge.
- For Sphinx tests, do not import sphinx.testing.fixtures directly; mirror existing local test helpers or use a minimal parser/app pattern.
"""

        task_summary = _extract_retry_task_summary(original_prompt)
        compact_errors = _compact_validation_errors(error_message)
        compact_response = _clip_prompt_text(
            previous_response,
            _RETRY_PREVIOUS_RESPONSE_CHARS,
        )

        return f"""You are fixing a previously generated reproduction test. Return corrected JSON only.

[Compact Task Summary]
{task_summary}

[Oracle Contract]
- The assertion must define fixed/post-patch behavior, not the bug symptom.
- Use issue-stated expected output when available.
- If only buggy output is known, compare the function return value/state against the buggy value.
- Do not invent exact expected arrays/values without issue evidence.
- Do not use private attributes, external network calls, raw rendered exact strings, or warning count/type alone.

[Previous Attempt #{attempt} - FAILED]
Key validation errors:
{compact_errors}

Previous response snippet:
```
{compact_response}
```
{format_hint}
{oracle_retry_hint}
{semantic_retry_hint}
[Fix Instructions]
1. Fix all errors listed above.
2. Only import symbols that actually exist in the repository.
3. Ensure the test code is syntactically valid Python.
4. Keep one focused test_* function/method and use the same target/source API as the issue.
5. Return corrected JSON only — no explanation, no markdown outside the JSON.
""".strip()

    @staticmethod
    def _oracle_rewrite_hint(error_message: str) -> str:
        hints = [
            "[ORACLE REWRITE REQUIREMENT]",
            "Your previous test used an oracle that is likely to fail on both buggy and fixed code.",
        ]
        if "no explicit oracle" in error_message.lower():
            hints.append("- Add exactly one explicit assertion. Priority: issue-stated expected output; else public return/state differs from known buggy output; else a public semantic invariant from the scenario.")
            hints.append("- Do not leave only setup/execution code. The final line should assert a fixed-behavior value, state, exception type, or no-exception success path.")
        if "guessed_expected_array" in error_message or "guessed_expected_value" in error_message:
            hints.append("- Do NOT invent exact expected arrays/values. Use issue-stated expected output only, or assert shape/type/finite/range/semantic invariant.")
        if "raises_only_no_body_assertion" in error_message or "fix_disappearing_exception_oracle" in error_message:
            hints.append("- Use pytest.raises/assertRaises only when the issue explicitly says the fixed behavior should raise. Otherwise assert the success path and post-call state/value.")
        if "private_attribute_oracle" in error_message:
            hints.append("- Do NOT read private attributes such as _legend_data, _gridOnMajor, or _legend_labels. Use public axis/artist/legend APIs.")
        if "raw_rendered_output_exact_match" in error_message:
            hints.append("- Do NOT assert long raw HTML/LaTeX/Sphinx strings. Use a minimal public semantic marker or whitespace invariant.")
        if "warning_presence_oracle" in error_message or "warning_catch_only" in error_message:
            hints.append("- Do NOT assert warning count/type alone. Assert the returned value or public state after the call.")
        if len(hints) == 2:
            hints.append("- Do NOT use @image_comparison, private attributes, raw rendered strings, warning-count assertions, or guessed expected arrays/values.")
            hints.append("- The assertion must describe the fixed behavior using issue expected output, public API state, or a small semantic invariant.")
        return "\n".join(hints) + "\n"

    def _build_modified_test_file_content(
        self,
        original_content: str,
        imports: List[str],
        test_code: str,
    ) -> str:
        lines = original_content.splitlines()

        # 기존 top-level import만 수집
        existing_top_imports = set()
        for line in lines:
            stripped = line.strip()
            if (line.startswith("import ") or line.startswith("from ")) and not line.startswith((" ", "\t")):
                existing_top_imports.add(stripped)

        new_imports = [imp.strip() for imp in imports if imp.strip() and imp.strip() not in existing_top_imports]

        # top-level import block 끝 찾기
        insert_idx = 0
        seen_top_import = False
        past_decorator = False  # @decorator를 이미 지났으면 import 삽입 금지
        in_paren = 0  # 괄호 depth (multi-line import 추적)

        for i, line in enumerate(lines):
            stripped = line.strip()

            # 괄호 depth 추적
            in_paren += line.count("(") - line.count(")")
            if in_paren < 0:
                in_paren = 0

            # @decorator를 지난 이후에는 빈 줄 포함 모든 줄 무시 → import block 종료
            if past_decorator:
                break

            # 빈 줄은 import block 안에서는 허용
            if not seen_top_import and stripped == "":
                continue
            # import block을 이미 봤고 빈 줄 → 아직 계속 허용 (import 사이 빈 줄)
            if seen_top_import and stripped == "" and in_paren == 0:
                continue

            # top-level import — 괄호 안에 있으면 무시
            if (line.startswith("import ") or line.startswith("from ")) and not line.startswith((" ", "\t")):
                seen_top_import = True
                # 괄호가 이 줄에서 닫히면 다음 줄, 아니면 괄호 닫힐 때까지 대기
                if in_paren == 0:
                    insert_idx = i + 1
                continue

            # multi-line import 괄호 닫힘
            if in_paren == 0 and seen_top_import and stripped.endswith(")"):
                insert_idx = i + 1
                continue

            # 괄호 안에 있으면 import block 계속
            if in_paren > 0:
                continue

            # 첫 번째 top-level 비-import 문장을 만나면 종료
            if not line.startswith((" ", "\t")) and stripped != "":
                if stripped.startswith("@"):
                    # decorator → import block 종료, 이후 줄은 무시
                    past_decorator = True
                    continue
                if seen_top_import:
                    break
                else:
                    insert_idx = i
                    break
        else:
            insert_idx = len(lines)

        updated_lines = lines[:]

        if new_imports:
            block = []
            if insert_idx > 0 and updated_lines[insert_idx - 1].strip() != "":
                block.append("")
            block.extend(new_imports)
            block.append("")
            updated_lines[insert_idx:insert_idx] = block

        rendered_test_code = test_code.rstrip()
        updated_content = "\n".join(updated_lines).rstrip() + "\n\n" + rendered_test_code + "\n"
        return updated_content

    def _build_unified_patch(
        self,
        original_content: str,
        modified_content: str,
        relative_path: str,
    ) -> str:
        old_lines = original_content.splitlines(keepends=True)
        new_lines = modified_content.splitlines(keepends=True)

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{relative_path}",
            tofile=f"b/{relative_path}",
        )
        return "".join(diff)
