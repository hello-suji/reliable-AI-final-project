from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class IssueClueFile:
    instance_id: str
    observed_behavior: List[str]
    expected_behavior: List[str]
    repro_conditions: List[str]
    environment: List[str]
    identifiers: Dict[str, List[str]]
    raw_issue_text: str
    code_examples: List[Dict[str, str]] = field(default_factory=list)
    expected_outputs: List[str] = field(default_factory=list)
    actual_outputs: List[str] = field(default_factory=list)
    error_keywords: List[str] = field(default_factory=list)
    # 이슈 traceback에서 파싱된 고신뢰도 fault location 후보
    # 각 항목: {"file_path": "...", "line_no": N, "function_name": "..."}
    fault_locations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


class IssueClueExtractor:
    def __init__(self) -> None:
        self.class_pattern = re.compile(r"\b[A-Z][A-Za-z0-9_]+\b")
        self.file_pattern = re.compile(r"\b[\w\-/\\]+\.(py|txt|json|yaml|yml|ini|cfg)\b")
        self.exception_pattern = re.compile(r"\b[A-Z][A-Za-z0-9_]*(Error|Exception)\b")
        self.func_pattern = re.compile(r"\b[a-z_][a-zA-Z0-9_]*\(")
        self.class_stopwords = {
            # Original
            "Consider", "If", "It", "The", "This", "That", "Suddenly",
            "True", "False", "Modeling", "Expected", "Actual",
            # Markdown headers / section titles
            "Description", "Example", "Reproduction", "Note", "Warning",
            "See", "Also", "Summary", "Details", "Steps", "Problem",
            "Solution", "Result", "Output", "Input", "Background",
            "Context", "Workaround", "Suggestion", "Resolution",
            # English common words that match CamelCase pattern
            "For", "So", "Use", "Using", "Fix", "Fixes", "Fixed",
            "Thanks", "After", "Before", "When", "Where", "How",
            "What", "About", "Between", "Into", "With", "Without",
            "From", "Based", "During", "Through", "However", "Since",
            "Because", "Although", "While", "Each", "Every", "Some",
            "Any", "Other", "Another", "Both", "Here", "There",
            # Python literals / builtins
            "None", "True", "False", "NotImplemented",
            # Common framework names (not identifier targets)
            "Django", "Astropy", "Flask", "React", "Rails",
            "Python", "Pytest", "Numpy", "Scipy", "Matplotlib",
            "GitHub", "Windows", "Linux", "MacOS",
            # IPython/Jupyter notebook style
            "In", "Out",
            # Generic prose words that match CamelCase
            "File", "New", "Old", "Line", "Code", "Test", "Class",
            "Module", "Calling", "Documents", "Looking", "Users", "Correct",
            "Traceback", "Error", "Exception",
        }
        self.function_stopwords = {
            "array", "print", "len",
            # Reproduction/environment helpers that often appear in issue snippets
            # but are poor root-cause candidates.
            "arange", "rand", "random", "seed", "platform", "get_backend",
            "show_versions",
            # Python builtins
            "range", "type", "str", "int", "list", "dict", "set",
            "format", "open", "input", "super", "isinstance",
            "getattr", "setattr", "hasattr", "delattr",
            "repr", "hash", "iter", "next", "sorted", "reversed",
            "enumerate", "zip", "map", "filter", "any", "all",
            "min", "max", "sum", "abs", "round", "bool", "float",
            "tuple", "bytes", "bytearray", "object", "property",
            "staticmethod", "classmethod", "vars", "dir", "id",
        }

    def extract(self, instance_id: str, issue_text: str) -> IssueClueFile:
        text = issue_text.strip()
        signal_text = self._strip_template_noise(text)

        observed = self._extract_observed_behavior(text)
        expected = self._extract_expected_behavior(text)
        repro = self._extract_repro_conditions(text)
        env = self._extract_environment(text)
        identifiers = self._extract_identifiers(signal_text)
        code_examples = self._extract_code_blocks(text)
        expected_outputs, actual_outputs = self._extract_output_examples(text, code_examples)
        error_keywords = self._extract_error_keywords(signal_text, identifiers, code_examples)

        fault_locations = self._extract_fault_locations(text, code_examples)

        return IssueClueFile(
            instance_id=instance_id,
            observed_behavior=observed,
            expected_behavior=expected,
            repro_conditions=repro,
            environment=env,
            identifiers=identifiers,
            raw_issue_text=text,
            code_examples=code_examples,
            expected_outputs=expected_outputs,
            actual_outputs=actual_outputs,
            error_keywords=error_keywords,
            fault_locations=fault_locations,
        )

    @staticmethod
    def _strip_template_noise(text: str) -> str:
        """Remove issue-template comments and bulky environment dumps for identifier extraction."""
        text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
        text = re.sub(r"<details>.*?</details>", " ", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(
            r"(?is)installed versions\s*-+\s*.*?(?:\n\s*\n|$)",
            " ",
            text,
        )
        return text

    def save(self, clue: IssueClueFile, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(clue.to_dict(), f, ensure_ascii=False, indent=2)

    def _split_lines(self, text: str) -> List[str]:
        lines = [line.strip() for line in text.splitlines()]
        return [line for line in lines if line]

    def _extract_expected_behavior(self, text: str) -> List[str]:
        lines = self._split_lines(text)
        results = []

        keywords = [
            "expected",
            "should",
            "must",
            "ought to",
            "return",
            "be able to",
        ]

        for line in lines:
            lower = line.lower()
            if any(k in lower for k in keywords):
                results.append(line)

        return self._dedup(results)

    def _extract_observed_behavior(self, text: str) -> List[str]:
        lines = self._split_lines(text)
        results = []

        keywords = [
            "actual",
            "instead",
            "error",
            "wrong",
            "fails",
            "failure",
            "returns",
            "raised",
            "traceback",
            "does not",
            "cannot",
        ]

        for line in lines:
            lower = line.lower()
            if any(k in lower for k in keywords):
                results.append(line)

        return self._dedup(results)

    def _extract_repro_conditions(self, text: str) -> List[str]:
        lines = self._split_lines(text)
        results = []

        keywords = [
            "consider the following",
            "reproduce",
            "steps",
            "example",
            "when",
            "if",
            "using",
        ]

        for line in lines:
            lower = line.lower()
            if any(k in lower for k in keywords):
                results.append(line)

        return self._dedup(results)

    def _extract_environment(self, text: str) -> List[str]:
        lines = self._split_lines(text)
        results = []

        keywords = [
            "version",
            "python ",
            "python:",
            "ubuntu",
            "linux",
            "windows",
            "macos",
            "os:",
        ]

        for line in lines:
            lower = line.lower()

            # 코드블록/코드라인 제외
            if line.startswith("```"):
                continue
            if line.startswith("from ") or line.startswith("import "):
                continue
            if "=" in line and "(" in line:
                continue

            if any(k in lower for k in keywords):
                results.append(line)

        return self._dedup(results)

    # 식별자 추출 상한 (LLM 프롬프트 노이즈 방지)
    _MAX_FUNCTIONS = 10
    _MAX_CLASSES = 10

    _SYSTEM_DETAIL_KEYWORDS = {
        "system details", "matplotlib version", "python version",
        "operating system", "jupyter version", "numpy", "scipy",
        "pyerfa", "platform", "windows", "linux", "macos",
    }

    def _is_system_or_output_block(
        self,
        code: str,
        language: str = "",
        context_before: str = "",
    ) -> bool:
        """Return True for environment dumps or plain outputs, not repro code."""
        language = (language or "").lower()
        context = (context_before or "").lower()
        stripped = code.strip()
        lower = stripped.lower()
        if not stripped:
            return True
        if any(k in context for k in self._SYSTEM_DETAIL_KEYWORDS):
            return True
        if any(k in context for k in ("how to reproduce", "reproduce", "file ", "code example", "minimal example")):
            return False
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if not lines:
            return True
        code_like = sum(
            1
            for line in lines
            if (
                line.startswith(("from ", "import ", "def ", "class ", "with ", "for ", "if ", "assert ", ">>> "))
                or "=" in line
                or re.search(r"\w+\s*\(", line)
            )
        )
        version_like = sum(
            1
            for line in lines
            if re.search(r"\b(version|python|numpy|scipy|matplotlib|windows|linux|macos|jupyter|pyerfa)\b", line.lower())
        )
        array_output_like = bool(re.fullmatch(r"\[?[-+0-9eE.,\s]+\]?", stripped))
        return (
            code_like == 0
            or (code_like <= 1 and version_like >= max(1, len(lines) // 2))
            or array_output_like
        )

    def _extract_identifiers(self, text: str) -> Dict[str, List[str]]:
        """이슈 텍스트에서 Python 식별자를 추출한다.

        전략:
          1) 코드 영역(fence/inline) 우선 추출 — 신뢰도 높음
          2) 코드 영역에서 충분히 얻지 못한 경우에만 산문 보완
             (단, 산문 클래스는 더 엄격한 필터 적용)
          3) 함수/클래스 각각 상한(_MAX_FUNCTIONS/_MAX_CLASSES) 적용
        """
        functions_code: set = set()
        functions_prose: set = set()
        classes_code: set = set()
        classes_prose: set = set()
        files: set = set()
        exceptions: set = set()

        # ── 코드 영역 위치 계산 ──
        code_spans: list = []
        for m in re.finditer(r"```\s*(\w*)\s*\n(.*?)```", text, re.DOTALL):
            preceding = text[max(0, m.start() - 200):m.start()].strip()
            preceding_lines = [l.strip() for l in preceding.splitlines() if l.strip()]
            context_before = preceding_lines[-1] if preceding_lines else ""
            if self._is_system_or_output_block(m.group(2), m.group(1), context_before):
                continue
            code_spans.append((m.start(2), m.end(2)))
        for m in re.finditer(r"`([^`\n]+)`", text):
            code_spans.append((m.start(1), m.end(1)))

        def _is_in_code(pos: int) -> bool:
            return any(s <= pos < e for s, e in code_spans)

        # ── 함수 패턴: 코드/산문 분리 추출 ──
        for match in self.func_pattern.finditer(text):
            fn = match.group(0)[:-1]
            if fn in self.function_stopwords:
                continue
            if _is_in_code(match.start()):
                functions_code.add(fn)
            else:
                functions_prose.add(fn)

        # ── 클래스 패턴: 코드/산문 분리 추출 ──
        for match in self.class_pattern.finditer(text):
            name = match.group(0)
            if name.endswith(("Error", "Exception")):
                continue
            if name in self.class_stopwords:
                continue
            if name.isupper():
                continue
            has_underscore = "_" in name
            # 내부 대문자 개수: Python 복합 클래스명(FileSystemStorage)은 1개 이상
            # 단순 영단어(Browser, Additional, Enabling)는 0개
            interior_upper = sum(1 for c in name[1:] if c.isupper())
            is_compound_pascal = interior_upper >= 1 or has_underscore

            if _is_in_code(match.start()):
                # 코드 영역: 소문자 포함이면 허용 (주석/docstring 내 단순 단어 제외)
                has_lower = any(c.islower() for c in name[1:])
                if has_lower or has_underscore:
                    classes_code.add(name)
            else:
                # 산문: 복합 PascalCase(FileSystemStorage)만 허용, 단순 영단어 제외
                if is_compound_pascal:
                    classes_prose.add(name)

        # ── 파일/예외 ──
        files.update(self._extract_file_references(text))
        for m in self.exception_pattern.finditer(text):
            exceptions.add(m.group(0))

        # ── 최종 병합: 코드 우선, 필요 시 산문 보완, 상한 적용 ──
        final_functions = list(functions_code)
        if len(final_functions) < self._MAX_FUNCTIONS:
            for fn in sorted(functions_prose):
                if fn not in functions_code:
                    final_functions.append(fn)
                if len(final_functions) >= self._MAX_FUNCTIONS:
                    break

        final_classes = list(classes_code)
        if len(final_classes) < self._MAX_CLASSES:
            for cls in sorted(classes_prose):
                if cls not in classes_code:
                    final_classes.append(cls)
                if len(final_classes) >= self._MAX_CLASSES:
                    break

        return {
            "functions": sorted(final_functions[:self._MAX_FUNCTIONS]),
            "classes": sorted(final_classes[:self._MAX_CLASSES]),
            "files": sorted(files),
            "exceptions": sorted(exceptions),
        }

    def _extract_file_references(self, text: str) -> List[str]:
        """Extract repo-relative file hints from prose, line refs, and GitHub URLs."""
        files: set[str] = set()

        def add_path(value: str) -> None:
            path = (value or "").strip().strip("`'\"()[]{}.,;:")
            path = path.replace("\\", "/")
            path = re.sub(r"^(?:\./|a/|b/)+", "", path)
            if "/blob/" in path or path.startswith(("http", "www.", "github.com/", "com/")):
                return
            if path and "." in Path(path).name:
                files.add(path)

        for m in self.file_pattern.finditer(text):
            add_path(m.group(0))

        # path/to/file.py:859 and Windows variants.
        for m in re.finditer(r"(?<!\w)([\w./\\-]+\.(?:py|txt|json|yaml|yml|ini|cfg)):(\d+)", text):
            add_path(m.group(1))

        # GitHub blob URL:
        # https://github.com/org/repo/blob/<sha-or-branch>/path/to/file.py#L10
        blob_re = re.compile(
            r"https?://github\.com/[^/\s]+/[^/\s]+/blob/[^/\s]+/"
            r"([^\s#?]+?\.(?:py|txt|json|yaml|yml|ini|cfg))(?:[#?][^\s]*)?",
            re.IGNORECASE,
        )
        for m in blob_re.finditer(text):
            add_path(m.group(1))

        return sorted(files)

    def _extract_code_blocks(self, text: str) -> List[Dict[str, str]]:
        """
        이슈 텍스트에서 코드 블록을 추출한다.
        마크다운 ``` 코드 펜스와 >>> 인터랙티브 예시를 모두 처리한다.
        각 블록에 대해 앞쪽 문맥(context_before)도 함께 저장한다.
        """
        blocks: List[Dict[str, str]] = []

        # 1) 마크다운 코드 펜스 추출: ```python ... ``` 또는 ``` ... ```
        fence_pattern = re.compile(
            r"```\s*(\w*)\s*\n(.*?)```",
            re.DOTALL,
        )
        for match in fence_pattern.finditer(text):
            language = match.group(1) or "python"
            code = match.group(2).strip()
            if not code:
                continue

            # 코드 블록 바로 앞의 텍스트(최대 200자)를 context로 저장
            start = match.start()
            preceding = text[max(0, start - 200):start].strip()
            # 마지막 문장만 추출
            preceding_lines = [l.strip() for l in preceding.splitlines() if l.strip()]
            context_before = preceding_lines[-1] if preceding_lines else ""

            blocks.append({
                "language": language,
                "code": code,
                "context_before": context_before,
                "is_system_or_output": self._is_system_or_output_block(
                    code, language, context_before
                ),
            })

        # 2) 코드 펜스 내의 >>> 인터랙티브 블록에서 코드+출력 분리
        #    (이미 위에서 추출한 블록 안에서 >>> 패턴을 해석)
        for block in blocks:
            code = block["code"]
            if ">>>" not in code:
                continue

            input_lines = []
            output_lines = []
            for line in code.splitlines():
                stripped = line.strip()
                if stripped.startswith(">>> "):
                    input_lines.append(stripped[4:])
                elif stripped.startswith("..."):
                    input_lines.append(stripped[4:] if len(stripped) > 4 else "")
                else:
                    output_lines.append(stripped)

            if input_lines:
                block["interactive_input"] = "\n".join(input_lines)
            if output_lines:
                block["interactive_output"] = "\n".join(output_lines)

        return blocks

    def _extract_output_examples(
        self, text: str, code_blocks: List[Dict[str, str]]
    ) -> tuple:
        """
        코드 블록의 context_before를 분석하여 기대 출력과 실제(버그) 출력을 분류한다.
        Returns: (expected_outputs, actual_outputs)
        """
        # "expect" 포함 변형도 매칭 (might expect, as you expect, etc.)
        expected_keywords = {
            "expected", "as expected", "correct", "should", "works", "expect",
            "ideally", "want", "desired", "suppose",
        }
        actual_keywords = {
            "however", "suddenly", "bug", "wrong", "instead", "no longer",
            "does not", "doesn't", "broken", "fail", "issue", "problem",
            "currently", "actually", "incorrectly", "unexpected",
        }

        expected_outputs: List[str] = []
        actual_outputs: List[str] = []

        blocks_with_output = []
        for block in code_blocks:
            output = block.get("interactive_output", "").strip()

            if not output:
                # interactive 출력이 없어도 traceback 블록이면 actual_output으로 추가
                code = block.get("code", "")
                if re.search(
                    r"Traceback \(most recent call last\)|^\w+Error:",
                    code,
                    re.MULTILINE,
                ):
                    lines = code.strip().splitlines()
                    last_err = next(
                        (
                            l.strip()
                            for l in reversed(lines)
                            if re.match(r"\w+Error:", l.strip())
                        ),
                        None,
                    )
                    if last_err:
                        actual_outputs.append(last_err)
                continue

            blocks_with_output.append((block, output))

        for i, (block, output) in enumerate(blocks_with_output):
            context = block.get("context_before", "").lower()

            is_expected = any(kw in context for kw in expected_keywords)
            is_actual = any(kw in context for kw in actual_keywords)

            if is_actual:
                actual_outputs.append(output)
            elif is_expected:
                expected_outputs.append(output)
            else:
                # 순서 기반 fallback: 마지막 블록 → actual(버그 시연), 첫 블록 → expected(정상 동작)
                if i == 0 and len(blocks_with_output) > 1:
                    expected_outputs.append(output)
                else:
                    actual_outputs.append(output)

        # Inline comment patterns in minimal examples, e.g.
        # "# correct ... => array([...])" / "# incorrect ... => array([...])".
        for block in code_blocks:
            comment_label = ""
            for line in block.get("code", "").splitlines():
                stripped = line.strip()
                if not stripped.startswith("#"):
                    continue
                comment = stripped[1:].strip()
                comment_lower = comment.lower()
                if "=>" not in comment:
                    if any(k in comment_lower for k in ("incorrect", "actual", "bug", "wrong")):
                        comment_label = "actual"
                    elif any(k in comment_lower for k in ("correct", "expected", "ideally", "desired")):
                        comment_label = "expected"
                    continue
                label, value = comment.split("=>", 1)
                label_lower = label.lower()
                value = value.strip()
                if not value:
                    continue
                if comment_label == "expected" or any(k in label_lower for k in ("correct", "expected", "ideally", "desired")):
                    expected_outputs.append(value)
                if comment_label == "actual" or any(k in label_lower for k in ("incorrect", "actual", "bug", "wrong")):
                    actual_outputs.append(value)

        return self._dedup(expected_outputs), self._dedup(actual_outputs)

    def _extract_error_keywords(
        self,
        text: str,
        identifiers: Dict[str, List[str]],
        code_blocks: List[Dict[str, str]],
    ) -> List[str]:
        """이슈에서 에러/예외 관련 핵심 키워드를 추출한다.

        1) identifiers.exceptions에서 이미 추출된 예외 타입 포함
        2) code_blocks의 interactive_output에서 "ExcType: message" 패턴 추출
        3) 이슈 텍스트 전체에서 예외 타입명 보완
        """
        keywords: List[str] = []

        # 1) identifiers에서 이미 추출된 예외 타입 그대로 포함
        keywords.extend(identifiers.get("exceptions", []))

        # 2) interactive_output에서 "ExcType: message" 한 줄짜리 에러 메시지 추출
        for block in code_blocks:
            output = block.get("interactive_output", "")
            for m in re.finditer(
                r"\b([A-Z]\w*(?:Error|Exception)): ([^\n]{5,100})",
                output,
            ):
                entry = m.group(0)[:120]
                if entry not in keywords:
                    keywords.append(entry)

        # 3) 이슈 텍스트 전체에서 예외 타입명 보완 (identifiers에 없는 것만)
        existing_exc = set(identifiers.get("exceptions", []))
        for m in re.finditer(r"\b([A-Z]\w*(?:Error|Exception))\b", text):
            exc = m.group(0)
            if exc not in existing_exc and exc not in keywords:
                keywords.append(exc)
                existing_exc.add(exc)

        return self._dedup(keywords)[:10]

    # traceback frame 패턴: File "path", line N, in func_name
    _TRACEBACK_FRAME = re.compile(
        r'File\s+"([^"]+\.py)",\s+line\s+(\d+),\s+in\s+(\S+)'
    )
    # site-packages / stdlib 등 제외할 경로 패턴
    _SKIP_PATH_PATTERNS = re.compile(
        r"site-packages|dist-packages|lib/python\d+\.\d+/(?!site)|"
        r"\.tox/|/tmp/|\\tmp\\|<.*>"
    )

    def _extract_fault_locations(
        self,
        text: str,
        code_examples: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """이슈 텍스트의 traceback에서 fault location 후보를 추출한다.

        `File "path/to/file.py", line N, in func_name` 패턴을 파싱하여
        repo 내부 파일에 해당하는 항목만 반환한다. (site-packages / stdlib 제외)

        Returns:
            [{"file_path": "...", "line_no": N, "function_name": "..."}, ...]
            파일 경로는 원문 절대 경로 그대로 — CodeContextExtractor가 suffix 매칭으로 처리.
        """
        seen: set[str] = set()
        results: List[Dict[str, Any]] = []

        # Only actual traceback text is high-confidence. Reproduction snippets
        # and environment dumps must not be promoted to CRITICAL fault locations.
        search_texts = [text]
        search_texts.extend(
            b.get("code", "")
            for b in code_examples
            if not b.get("is_system_or_output")
            and "Traceback (most recent call last)" in b.get("code", "")
        )

        for src in search_texts:
            if "Traceback (most recent call last)" not in src:
                continue
            for m in self._TRACEBACK_FRAME.finditer(src):
                file_path = m.group(1)
                line_no = int(m.group(2))
                func_name = m.group(3)

                # stdlib / venv / tox 경로 제외
                if self._SKIP_PATH_PATTERNS.search(file_path):
                    continue

                key = f"{file_path}:{func_name}"
                if key in seen:
                    continue
                seen.add(key)

                results.append({
                    "file_path": file_path,
                    "line_no": line_no,
                    "function_name": func_name,
                    "source": "traceback",
                    "confidence": "high",
                })

        # 상한 10건
        return results[:10]

    def _dedup(self, items: List[str]) -> List[str]:
        seen = set()
        results = []

        for item in items:
            normalized = item.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                results.append(normalized)

        return results
