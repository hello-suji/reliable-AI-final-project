from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.models.client import LLMClient, ModelConfig
from src.models.config import load_model_config
from src.scenario.scenario_hydrator import hydrate_scenario_dict

logger = logging.getLogger(__name__)


@dataclass
class TestScenario:
    scenario_id: str
    target_location: Dict[str, Any]
    setup_steps: List[str]
    execution_stimulus: List[str]
    expected_failure: str
    relevant_source_files: List[str]
    relevant_test_files: List[str]
    test_environment: Dict[str, Any] = None  # {"required_fixtures": [...], "runner": "pytest"}
    # clue에서 merge되는 필드 (run_single.py에서 validation 후 채워짐)
    reproduction_code: List[Dict[str, str]] = None   # clue.code_examples
    expected_outputs: List[str] = None               # clue.expected_outputs
    actual_outputs: List[str] = None                 # clue.actual_outputs
    error_keywords: List[str] = None                 # clue.error_keywords
    identifiers: Dict[str, List[str]] = None         # clue.identifiers
    oracle_hints: List[str] = None                   # synthesized oracle guidance
    oracle: str = ""                                 # compact oracle guidance string
    oracle_contract: Dict[str, str] = None           # {"oracle_type": ..., "oracle_source": ..., "rule": ...}
    oracle_type: str = ""                            # compact top-level copy for prompts
    oracle_source: str = ""                          # compact top-level copy for prompts

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d.get("reproduction_code") is None:
            d["reproduction_code"] = []
        if d.get("expected_outputs") is None:
            d["expected_outputs"] = []
        if d.get("actual_outputs") is None:
            d["actual_outputs"] = []
        if d.get("error_keywords") is None:
            d["error_keywords"] = []
        if d.get("identifiers") is None:
            d["identifiers"] = {}
        if d.get("test_environment") is None:
            d["test_environment"] = {}
        if d.get("oracle_hints") is None:
            d["oracle_hints"] = []
        if d.get("oracle") is None:
            d["oracle"] = ""
        if d.get("oracle_contract") is None:
            d["oracle_contract"] = {}
        if d.get("oracle_type") is None:
            d["oracle_type"] = ""
        if d.get("oracle_source") is None:
            d["oracle_source"] = ""
        return d


class ScenarioGenerator:

    SYSTEM_PROMPT = (
        "You are a software test scenario planner. "
        "Analyze the issue and generate structured test scenarios. "
        "Return JSON only."
    )

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        model_key: str = "qwen",
    ) -> None:
        self.client = client or LLMClient(load_model_config(model_key))

    def extract(
        self,
        instance: Any,
        clue: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[TestScenario]:
        prompt = self._build_prompt(instance, clue, context)
        raw_response = self.client.generate(prompt, system_prompt=self.SYSTEM_PROMPT)
        scenarios = self._parse_response(raw_response, clue, context)
        scenarios = self._enrich_with_probe(scenarios, instance, context)
        scenarios = [
            self._dict_to_hydrated_scenario(
                s.to_dict(),
                clue=clue,
                context=context,
                repo=getattr(instance, "repo", ""),
            )
            for s in scenarios
        ]
        return scenarios[:3]

    # ── Probe execution ──

    def _enrich_with_probe(
        self,
        scenarios: List[TestScenario],
        instance: Any,
        context: Dict[str, Any],
    ) -> List[TestScenario]:
        """Top 시나리오에 대해 probe test를 실행하여 actual_outputs를 채운다."""
        if not scenarios:
            return scenarios
        top = scenarios[0]
        if top.actual_outputs:
            return scenarios  # 이미 있으면 skip

        repro = top.reproduction_code or []
        if not repro:
            return scenarios
        code = repro[0].get("code", "") if isinstance(repro[0], dict) else str(repro[0])
        if not code.strip():
            return scenarios

        probe_code = self._transform_to_probe(code)
        if not probe_code:
            return scenarios

        target_test_file = top.target_location.get("candidate_test_file", "")
        if not target_test_file:
            return scenarios

        # repo_path 결정: data/repos/{repo_owner}/{repo_name}
        # instance.instance_id 형식: "owner__repo-number" or instance.raw["repo"] = "owner/repo"
        try:
            repo_slug = getattr(instance, "repo", "") or ""
            if repo_slug:
                repo_path = Path("data/repos") / repo_slug.replace("/", "__")
                if not repo_path.exists():
                    repo_path = Path("data/repos") / repo_slug
            else:
                repo_path = Path("data/repos") / instance.instance_id.rsplit("-", 1)[0]
        except Exception:
            return scenarios

        if not repo_path.exists():
            logger.debug("probe: repo_path not found: %s", repo_path)
            return scenarios

        try:
            probe_result = self._run_probe(probe_code, instance, target_test_file, repo_path)
        except Exception as e:
            logger.debug("probe run failed: %s", e)
            return scenarios

        raw_output = probe_result.get("raw_output", "") if isinstance(probe_result, dict) else (probe_result or "")
        probe_cov = probe_result.get("coverage_data", {}) if isinstance(probe_result, dict) else {}

        if not raw_output:
            return scenarios

        # actual_outputs 채우기
        actual = self._parse_probe_output(raw_output)
        if actual:
            top.actual_outputs = [actual]
            logger.info("probe enriched actual_outputs for %s: %s", top.scenario_id, actual[:120])

        # Fault location 검증: target_source_file이 probe에서 실행됐는지 확인
        target_src = top.target_location.get("source_file", "")
        if target_src and probe_cov:
            src_covered = self._check_source_covered(probe_cov, target_src)
            if not src_covered:
                logger.warning(
                    "probe: source file '%s' NOT covered — fault localization may be wrong. "
                    "Trying alternative scenario.",
                    target_src,
                )
                # 다른 시나리오 중 더 나은 것으로 교체 시도
                for i, alt in enumerate(scenarios[1:], 1):
                    alt_src = alt.target_location.get("source_file", "")
                    if alt_src and alt_src != target_src:
                        # 대안 시나리오가 다른 파일을 타겟한다면 순서 교체
                        scenarios[0], scenarios[i] = scenarios[i], scenarios[0]
                        logger.info("probe: switched to scenario %s (source: %s)", alt.scenario_id, alt_src)
                        break

        return scenarios

    def _check_source_covered(self, coverage_data: Dict[str, Any], source_file: str) -> bool:
        """probe coverage에서 target source file이 실행됐는지 확인."""
        for fname, info in coverage_data.items():
            if not isinstance(info, dict):
                continue
            if fname.endswith(source_file) or source_file.endswith(fname):
                return info.get("cover", 0) > 5  # 5% 이상이면 실행됐다고 판단
        return False

    def _transform_to_probe(self, code: str) -> str:
        """reproduction_code에서 assertion을 제거하고 결과 캡처 코드로 변환."""
        lines = code.strip().splitlines()
        if not lines:
            return ""

        new_lines: List[str] = []
        func_indent = "    "

        for line in lines:
            # 함수명 변경
            m_def = re.match(r'(\s*def\s+)(test_\w+)(.*)', line)
            if m_def:
                line = m_def.group(1) + "test_probe_capture" + m_def.group(3)
                new_lines.append(line)
                continue

            stripped = line.strip()

            # assert 문 및 assert_*() 헬퍼 호출 제거 (probe가 결과를 캡처하려면 예외가 나면 안 됨)
            if (stripped.startswith("assert ")
                    or re.match(r'assert_\w+\s*\(', stripped)
                    or re.match(r'self\.assert\w*\s*\(', stripped)
                    or re.match(r'np\.testing\.assert\w*\s*\(', stripped)):
                indent = len(line) - len(line.lstrip())
                new_lines.append(" " * indent + "# [probe: assertion removed]")
                continue

            # pytest.raises 블록 제거
            if stripped.startswith("with pytest.raises") or stripped.startswith("with self.assertRaises"):
                indent = len(line) - len(line.lstrip())
                new_lines.append(" " * indent + "# [probe: pytest.raises removed]")
                continue

            # result = func(...) 패턴: __probe_result에 캡처
            m_assign = re.match(r'^(\s*)(\w+)\s*=\s*(.+\(.*)', stripped)
            if (
                m_assign
                and not stripped.startswith(("import ", "from ", "def ", "class "))
            ):
                var_name = m_assign.group(2)
                indent = len(line) - len(line.lstrip())
                new_lines.append(line)
                new_lines.append(" " * indent + f"__probe_result = {var_name}")
                continue

            new_lines.append(line)

        # 함수 마지막에 출력 캡처 블록 추가
        new_lines += [
            f"{func_indent}# --- probe output capture ---",
            f"{func_indent}import sys as __sys",
            f"{func_indent}if '__probe_result' in dir():",
            f"{func_indent}    print(f'__PROBE_RESULT__:{{repr(__probe_result)}}', "
            f"file=__sys.stderr)",
        ]
        return "\n".join(new_lines)

    def _run_probe(
        self,
        probe_code: str,
        instance: Any,
        target_test_file: str,
        repo_path: Path,
    ) -> Dict[str, Any]:
        """probe test를 Docker에서 실행하고 raw_output + coverage_data를 반환."""
        import json as _json
        import os
        import tempfile

        from src.executor.alignment_runner import AlignmentRunner
        from src.generator.repro_test_generator import ReproductionTestGenerator

        runner = AlignmentRunner()

        abs_path = repo_path / target_test_file
        original = ""
        if abs_path.exists():
            original = abs_path.read_text(encoding="utf-8", errors="ignore")

        modified = original.rstrip() + "\n\n" + probe_code + "\n"

        # ReproductionTestGenerator의 _build_unified_patch 재사용
        rtg = ReproductionTestGenerator.__new__(ReproductionTestGenerator)
        test_patch = rtg._build_unified_patch(original, modified, target_test_file)

        probe_dict = {
            "instance_id": instance.instance_id,
            "scenario_id": "probe",
            "model_name": "probe",
            "target_test_file": target_test_file,
            "test_patch": test_patch,
            "modified_test_file_content": modified,
            "test_code": probe_code,
            "insert_mode": "append_block",
            "append_block": probe_code,
            "imports": [],
            "original_test_file_content": original,
            "insertion_hint": "",
            "raw_response": "",
            "prompt": "",
            "repo_path": str(repo_path),
            "target_test_file_abspath": str(abs_path),
            "target_source_file": "",
            "token_usage": {},
        }

        tmp_json = tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False, encoding="utf-8"
        )
        _json.dump(probe_dict, tmp_json)
        tmp_json.close()
        probe_json_path = tmp_json.name
        probe_patch_path = probe_json_path.replace(".json", ".patch")

        try:
            with open(probe_patch_path, "w", encoding="utf-8") as f:
                f.write(test_patch)

            result = runner.run(
                instance=instance,
                generated_test_json_path=probe_json_path,
                run_id=f"probe-{instance.instance_id}",
            )
            return {
                "raw_output": result.raw_output or "",
                "coverage_data": result.coverage_data or {},
            }
        finally:
            if os.path.exists(probe_json_path):
                os.unlink(probe_json_path)
            if os.path.exists(probe_patch_path):
                os.unlink(probe_patch_path)

    def _parse_probe_output(self, raw_output: str) -> str:
        """raw Docker output에서 __PROBE_RESULT__: 값 추출."""
        m = re.search(r"__PROBE_RESULT__:(.+)", raw_output)
        if m:
            return m.group(1).strip()[:300]
        return ""

    def save(self, scenarios: List[TestScenario], output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([s.to_dict() for s in scenarios], f, ensure_ascii=False, indent=2)

    # ── LLM 기반 fault location 추론 (traceback 없는 경우) ──

    def infer_fault_locations(
        self,
        clue: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """traceback이 없는 이슈에서 소스 코드 스니펫을 분석하여 버그 위치를 추론한다.

        Returns:
            fault_locations 형식의 리스트:
            [{"file_path": "...", "function_name": "...", "line_no": 0, "inferred": True}, ...]
        """
        observed = clue.get("observed_behavior", [])
        expected = clue.get("expected_behavior", [])
        noisy_functions = {
            "arange", "rand", "random", "seed", "platform", "get_backend",
            "show_versions",
        }
        functions = [
            fn for fn in clue.get("identifiers", {}).get("functions", [])
            if fn not in noisy_functions
        ]
        classes = clue.get("identifiers", {}).get("classes", [])
        error_keywords = clue.get("error_keywords", [])

        # 소스 코드 스니펫 수집 (최대 5개 파일)
        snippet_parts = []
        for sf in context.get("candidate_source_files", [])[:5]:
            snippets = sf.get("code_snippets") or {}
            if not snippets:
                continue
            parts = [f"File: {sf['path']}"]
            for ident, snippet in list(snippets.items())[:3]:
                parts.append(f"```python\n{snippet[:800]}\n```")
            snippet_parts.append("\n".join(parts))

        if not snippet_parts:
            return []

        snippet_section = "\n\n".join(snippet_parts)

        prompt = f"""You are analyzing a GitHub issue to identify which function is most likely buggy.

Issue observed behavior: {json.dumps(observed[:3], ensure_ascii=False)}
Issue expected behavior: {json.dumps(expected[:3], ensure_ascii=False)}
Related identifiers — functions: {functions[:8]}, classes: {classes[:8]}
{f"Error keywords: {error_keywords[:5]}" if error_keywords else ""}

Source code snippets from the repository:
{snippet_section}

Based on the issue description and code above, identify the most likely buggy function(s).

Return JSON only:
{{"fault_locations": [{{"file_path": "relative/path/to/file.py", "function_name": "function_name", "line_no": 0}}]}}

Rules:
- Return at most 3 candidates, most likely first.
- Use the exact file paths shown above.
- If you are uncertain, return {{"fault_locations": []}}
- Do NOT include test files.
"""
        try:
            raw = self.client.generate(
                prompt,
                system_prompt="You are a bug localization assistant. Return JSON only.",
            )
        except Exception as e:
            logger.warning("infer_fault_locations LLM call failed: %s", e)
            return []

        # JSON 파싱
        text = raw.strip()
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            obj_match = re.search(r"(\{.*\})", text, re.DOTALL)
            if not obj_match:
                logger.warning("infer_fault_locations: failed to parse LLM response")
                return []
            try:
                data = json.loads(obj_match.group(1))
            except json.JSONDecodeError:
                logger.warning("infer_fault_locations: JSON parse error in extracted object")
                return []

        locations = data.get("fault_locations", [])
        if not isinstance(locations, list):
            return []

        result = []
        for loc in locations[:3]:
            if not isinstance(loc, dict):
                continue
            fp = loc.get("file_path", "")
            fn = loc.get("function_name", "")
            if fp and fn:
                result.append({
                    "file_path": fp,
                    "function_name": fn,
                    "line_no": loc.get("line_no", 0),
                    "inferred": True,
                    "source": "inferred_llm",
                    "confidence": "medium",
                })

        if result:
            logger.info("infer_fault_locations: found %d location(s): %s", len(result), result)
        return result

    # ── LLM 기반 시나리오 생성 ──

    def _build_prompt(
        self,
        instance: Any,
        clue: Dict[str, Any],
        context: Dict[str, Any],
    ) -> str:
        noisy_functions = {
            "arange", "rand", "random", "seed", "platform", "get_backend",
            "show_versions",
        }
        functions = [
            fn for fn in clue.get("identifiers", {}).get("functions", [])
            if fn not in noisy_functions
        ]
        classes = clue.get("identifiers", {}).get("classes", [])
        error_keywords = clue.get("error_keywords", [])
        observed = clue.get("observed_behavior", [])
        expected = clue.get("expected_behavior", [])
        repro = clue.get("repro_conditions", [])
        raw_issue_text = clue.get("raw_issue_text", "")
        code_examples = clue.get("code_examples", [])
        expected_outputs = clue.get("expected_outputs", [])
        actual_outputs = clue.get("actual_outputs", [])
        fault_locations = clue.get("fault_locations", [])

        source_files = [x["path"] for x in context.get("candidate_source_files", [])]
        test_files = [x["path"] for x in context.get("candidate_test_files", [])]
        framework = context.get("project_test_style", {}).get("framework", "unknown")
        runner = context.get("project_test_style", {}).get("runner", "pytest")
        conftest_fixtures = context.get("conftest_fixtures", {})

        # 소스 파일별 실제 공개 함수 목록 (AST 추출, 환각 방지용)
        func_list_lines = []
        for sf in context.get("candidate_source_files", [])[:3]:
            funcs = sf.get("top_level_functions") or []
            if funcs:
                func_list_lines.append(f"  {sf['path']}: {funcs[:20]}")
        func_list_section = (
            "\n[Available Functions per Source File — from AST]\n"
            + "\n".join(func_list_lines)
            + "\n"
        ) if func_list_lines else ""

        # 소스 코드 스니펫 섹션 (matched identifiers의 실제 함수/클래스 정의)
        source_snippet_parts = []
        for sf in context.get("candidate_source_files", [])[:2]:
            snippets = sf.get("code_snippets") or {}
            if not snippets:
                continue
            parts = [f"#### {sf['path']}"]
            for ident, snippet in list(snippets.items())[:2]:
                parts.append(f"```python\n{snippet[:600]}\n```")
            source_snippet_parts.append("\n".join(parts))
        source_snippet_section = "\n\n".join(source_snippet_parts) if source_snippet_parts else "(none available)"

        # 코드 예시 섹션
        code_section = ""
        if code_examples:
            parts = []
            for i, block in enumerate(code_examples):
                if block.get("is_system_or_output"):
                    continue
                code = block.get("code", "")
                ctx = block.get("context_before", "")
                label = f"Code Block {i + 1}"
                if ctx:
                    label += f' (context: "{ctx[:100]}")'
                parts.append(f"### {label}\n```python\n{code}\n```")
            code_section = "\n\n".join(parts)

        expected_output_section = ""
        if expected_outputs:
            expected_output_section = "\n[Expected Correct Output]\n" + "\n".join(
                f"```\n{out[:500]}\n```" for out in expected_outputs[:3]
            )

        actual_output_section = ""
        if actual_outputs:
            actual_output_section = "\n[Actual Buggy Output]\n" + "\n".join(
                f"```\n{out[:500]}\n```" for out in actual_outputs[:3]
            )

        # 이슈 원문 (truncated)
        raw_issue_section = ""
        if raw_issue_text:
            truncated = raw_issue_text[:800]
            if len(raw_issue_text) > 800:
                truncated += "\n... (truncated)"
            raw_issue_section = f"\n[Issue Description]\n{truncated}\n"

        # fault locations 섹션: real traceback and inferred candidates are separate
        fault_location_section = ""
        if fault_locations:
            traceback_lines = []
            inferred_lines = []
            for fl in fault_locations[:5]:
                fp = fl.get("file_path", "")
                fn = fl.get("function_name", "")
                ln = fl.get("line_no", "?")
                source = fl.get("source", "traceback")
                confidence = fl.get("confidence", "high" if source == "traceback" else "medium")
                # 절대 경로에서 repo 내 상대 경로 추정 (마지막 의미 있는 부분)
                # e.g. /home/user/.../astropy/coordinates/sky_coordinate.py → astropy/coordinates/sky_coordinate.py
                parts = fp.replace("\\", "/").split("/")
                # site-packages 이후 경로는 제외했으므로 그냥 마지막 3~4 segments 사용
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
                    "These locations were explicitly identified in the issue's stack trace.\n"
                    "They are HIGH-CONFIDENCE indicators of where the bug lives.\n"
                    + "\n".join(traceback_lines)
                    + "\n"
                    "→ S1 MUST use one of these as target_location (source_file + target_function).\n"
                    "→ If the function listed here is a private helper (starts with _), use the\n"
                    "  closest public caller function visible in [Source Code Snippets] instead.\n"
                )
            if inferred_lines:
                sections.append(
                    "\n[Inferred Fault Location Candidates]\n"
                    "These are MEDIUM-CONFIDENCE guesses from issue text and source snippets.\n"
                    "Use them only if they agree with the issue behavior and available functions.\n"
                    + "\n".join(inferred_lines)
                    + "\n"
                )
            fault_location_section = "".join(sections)

        prompt = f"""Generate 1-3 bug reproduction test scenarios for the issue below. Return JSON array only.

Repository: {instance.repo}

[Issue Clue]
Observed: {json.dumps(observed[:3], ensure_ascii=False)}
Expected: {json.dumps(expected[:3], ensure_ascii=False)}
Repro: {json.dumps(repro[:3], ensure_ascii=False)}
Functions: {functions[:8]}  Classes: {classes[:8]}
{f"Errors: {error_keywords[:5]}" if error_keywords else ""}{raw_issue_section}
{fault_location_section}
{f"[Issue Code Examples]{chr(10)}{code_section}" if code_section else ""}
{expected_output_section}
{actual_output_section}

[Code Context]
Test framework: {framework}
Test runner: {runner}
Candidate source files: {source_files[:5]}
Candidate test files: {test_files[:5]}
{self._build_conftest_section(conftest_fixtures)}
[Source Code Snippets - Use these to write concrete execution_stimulus and setup_steps]
{source_snippet_section}

[Output Format]
Return a JSON array of 1-3 scenario objects. Each object must have exactly these fields:
{{
  "scenario_id": "S1",
  "target_location": {{
    "source_file": "path/to/source.py",
    "target_function": "function_name",
    "related_classes": ["ClassName1", "ClassName2"],
    "candidate_test_file": "path/to/test_file.py",
    "confidence": "high|medium|low"
  }},
  "setup_steps": ["step 1 (imports/preconditions)", "step 2 (object/fixture setup)"],
  "execution_stimulus": ["action 1", "action 2"],
  "expected_failure": "Description of how the test should fail on buggy code",
  "test_environment": {{
    "required_fixtures": ["fixture_name_if_needed"],
    "runner": "{runner}"
  }},
  "reproduction_code": [{{"language": "python", "code": "def test_<name>():\n    # imports\n    # setup from setup_steps\n    # execution from execution_stimulus\n    assert <expected_failure condition>"}}],
  "identifiers": {{"functions": ["target_func"], "classes": ["RelatedClass"]}},
  "expected_outputs": [],
  "actual_outputs": [],
  "error_keywords": []
}}

[Target Function Selection Guide]
- PREFER: domain-specific functions explicitly named in the issue (e.g., `separability_matrix`, `register_cmap`, `get_cmap`)
- PREFER: public API functions that a user would call directly to trigger the bug
- AVOID: dunder methods (`__str__`, `__hash__`, `__init__`, `__setitem__`, etc.) unless the issue is explicitly about that dunder's behavior AND it appears in the identified functions list
- If dunder is the only option, make sure execution_stimulus calls it indirectly (e.g., `str(obj)` not `obj.__str__()`) so it actually appears in the test
- If functions list is empty: infer the most specific public function from the issue description and source file name

{func_list_section}[Constraints]
1. target_function selection priority:
   a. IF [Fault Locations from Issue Traceback] exists → use that function/file for S1 (HIGHEST PRIORITY)
   b. ELSE IF [Inferred Fault Location Candidates] exists → treat them as hints, not mandatory truth
   c. ELSE use the issue's identified functions: {functions[:10]}
   d. If both are empty, pick from [Available Functions per Source File] above
   CRITICAL: target_function MUST appear in [Available Functions per Source File] for the chosen
   source_file. NEVER invent a function name not listed there or visible in [Source Code Snippets].
2. source_file MUST be one of the candidate source files: {source_files[:5]}
3. candidate_test_file SHOULD be one of the candidate test files: {test_files[:5]}
4. execution_stimulus must describe concrete actions, not abstract descriptions
5. expected_failure must describe what goes wrong on the buggy version
6. setup_steps should include both preconditions (environment/imports) and concrete setup actions
7. At least one scenario (preferably S3) should include a candidate_test_file
8. CRITICAL — actual_outputs / expected_outputs: extract from [Issue Description] text.
   actual_outputs — the BUGGY value the code currently produces:
     Search [Issue Description] for: "currently returns X" / "outputs Y" / "produces Z" /
     "Actual: Z" / numbers or arrays described as WRONG / error messages shown.
     Extract VERBATIM. Even a single token like ["1"] or ["None"] is valuable.
     Example: issue says "distance is calculated as 1" → actual_outputs: ["1"]
     Leave [] ONLY if the issue gives absolutely NO observable output or value.
   expected_outputs — the CORRECT value after the fix:
     ONLY fill if issue EXPLICITLY states: "should return X" / "correct output: Y" / "Expected: Z".
     DO NOT guess or calculate. If not stated, leave [].
9. reproduction_code MUST be a runnable Python test function (def test_<name>():) that:
   - Imports necessary modules
   - Executes the setup_steps and execution_stimulus as code
   - Contains an assert statement that will FAIL on the buggy code
   - Is self-contained (no fixtures, no class required unless runner=django-test)
10. Do not use placeholder symbols such as xxx, foo, bar, Dummy*, MockModel, FooModel, or BarModel.
11. Do not define Django model classes in scenarios. Reuse candidate test file models/helpers only.
12. Return JSON array only. No explanation.
""".strip()

        return prompt

    def _parse_response(
        self,
        raw_response: str,
        clue: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[TestScenario]:
        text = raw_response.strip()

        # 코드 펜스 안의 JSON 추출
        fence_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # 배열 패턴 추출 시도
            arr_match = re.search(r"(\[.*\])", text, re.DOTALL)
            obj_match = re.search(r"(\{.*\})", text, re.DOTALL)
            if arr_match:
                try:
                    data = json.loads(arr_match.group(1))
                except json.JSONDecodeError:
                    # 배열 파싱도 실패 → 단일 객체 시도
                    if obj_match:
                        try:
                            data = [json.loads(obj_match.group(1))]
                        except json.JSONDecodeError:
                            logger.error("LLM scenario response parsing failed, using fallback scenarios")
                            return self._build_fallback_scenarios(clue, context)
                    else:
                        logger.error("LLM scenario response parsing failed, using fallback scenarios")
                        return self._build_fallback_scenarios(clue, context)
            elif obj_match:
                # 단일 객체 → 배열로 래핑
                try:
                    data = [json.loads(obj_match.group(1))]
                except json.JSONDecodeError:
                    logger.error("LLM scenario response parsing failed, using fallback scenarios")
                    return self._build_fallback_scenarios(clue, context)
            else:
                logger.error("LLM scenario response parsing failed, using fallback scenarios")
                return self._build_fallback_scenarios(clue, context)

        if isinstance(data, dict):
            data = [data]

        if not isinstance(data, list) or not data:
            logger.warning("LLM scenario response is empty, using fallback scenarios")
            return self._build_fallback_scenarios(clue, context)

        source_files = [x["path"] for x in context.get("candidate_source_files", [])]
        test_files = [x["path"] for x in context.get("candidate_test_files", [])]
        functions = clue.get("identifiers", {}).get("functions", [])
        classes = clue.get("identifiers", {}).get("classes", [])
        code_examples = clue.get("code_examples", [])
        expected_outputs = clue.get("expected_outputs", [])
        actual_outputs = clue.get("actual_outputs", [])
        error_keywords = clue.get("error_keywords", [])
        clue_identifiers = clue.get("identifiers", {})
        runner = context.get("project_test_style", {}).get("runner", "pytest")

        # source_file → matched_identifiers 맵 (target_function 실존 검증용)
        candidate_ident_map: Dict[str, List[str]] = {
            sf["path"]: sf.get("matched_identifiers", [])
            for sf in context.get("candidate_source_files", [])
        }

        items: List[Dict[str, Any]] = []
        for item in data:
            if isinstance(item, dict):
                items.append(item)
            elif isinstance(item, list):
                nested = [x for x in item if isinstance(x, dict)]
                if nested:
                    logger.warning("LLM scenario response contained nested list; flattening it")
                    items.extend(nested)
                else:
                    logger.warning("skipping non-dict scenario list item: %s", type(item).__name__)
            else:
                logger.warning("skipping non-dict scenario item: %s", type(item).__name__)

        if not items:
            logger.warning("LLM scenario response has no dict items, using fallback scenarios")
            return self._build_fallback_scenarios(clue, context)

        scenarios: List[TestScenario] = []
        for i, item in enumerate(items[:3]):
            # target_function 실존 검증 + 자동 교체
            tl = item.get("target_location", {})
            if not isinstance(tl, dict):
                tl = {}
            src = tl.get("source_file", "")
            tfunc = tl.get("target_function", "")
            actual_idents = candidate_ident_map.get(src, [])
            if tfunc and actual_idents and tfunc not in actual_idents:
                # clue functions 중 해당 파일에 실제로 있는 것으로 교체
                valid_alts = [f for f in functions if f in actual_idents]
                if valid_alts:
                    logger.warning(
                        "target_function '%s' not found in %s — replacing with '%s'",
                        tfunc, src, valid_alts[0],
                    )
                    tl["target_function"] = valid_alts[0]
                    item["target_location"] = tl

            try:
                scenario = self._dict_to_scenario(
                    item,
                    fallback_id=f"S{i + 1}",
                    source_files=source_files,
                    test_files=test_files,
                    functions=functions,
                    classes=classes,
                    code_examples=code_examples,
                    expected_outputs=expected_outputs,
                    actual_outputs=actual_outputs,
                    error_keywords=error_keywords,
                    clue_identifiers=clue_identifiers,
                    runner=runner,
                )
                scenarios.append(scenario)
            except Exception as e:
                logger.warning("scenario parsing failed (index %d): %s", i, e)

        if not scenarios:
            logger.warning("all LLM scenario parsing failed, using fallback scenarios")
            return self._build_fallback_scenarios(clue, context)

        return scenarios

    def _dict_to_scenario(
        self,
        item: Dict[str, Any],
        fallback_id: str,
        source_files: List[str],
        test_files: List[str],
        functions: List[str],
        classes: List[str],
        code_examples: List[Dict[str, str]],
        expected_outputs: List[str],
        actual_outputs: List[str],
        error_keywords: Optional[List[str]] = None,
        clue_identifiers: Optional[Dict[str, Any]] = None,
        runner: str = "pytest",
    ) -> TestScenario:
        target = item.get("target_location", {})
        if not isinstance(target, dict):
            target = {}

        # source_file / target_function 보정
        src_file = target.get("source_file", "")
        if not src_file and source_files:
            src_file = source_files[0]

        tgt_func = target.get("target_function", "")
        if not tgt_func and functions:
            tgt_func = functions[0]

        related = self._ensure_list(target.get("related_classes", classes[:3]))
        candidate_test = target.get("candidate_test_file", "")
        if not candidate_test and test_files:
            candidate_test = test_files[0]

        confidence = target.get("confidence", "medium")
        test_environment = item.get("test_environment", {})
        if not isinstance(test_environment, dict):
            test_environment = {}
        reproduction_code = self._ensure_reproduction_code(
            item.get("reproduction_code") or code_examples
        )
        identifiers = item.get("identifiers") or clue_identifiers or {}
        if not isinstance(identifiers, dict):
            identifiers = clue_identifiers or {}

        # preconditions가 있으면 setup_steps 앞에 병합 (하위호환)
        preconditions = self._ensure_list(item.get("preconditions", []))
        setup_steps = self._ensure_list(item.get("setup_steps", []))
        merged_setup = preconditions + [s for s in setup_steps if s not in preconditions]

        return TestScenario(
            scenario_id=item.get("scenario_id", fallback_id),
            target_location={
                "source_file": src_file,
                "target_function": tgt_func,
                "related_classes": related[:5],
                "candidate_test_file": candidate_test,
                "confidence": confidence,
            },
            setup_steps=merged_setup,
            execution_stimulus=self._ensure_list(item.get("execution_stimulus", [])),
            expected_failure=self._ensure_str(
                item.get("expected_failure"), "The test should fail due to the reported bug."
            ),
            relevant_source_files=source_files[:3],
            relevant_test_files=test_files[:3],
            test_environment={
                "required_fixtures": self._ensure_list(
                    test_environment.get("required_fixtures", [])
                ),
                "runner": test_environment.get("runner", runner),
            },
            # clue 파생 필드: LLM이 생성했으면 그대로, 없으면 clue 데이터로 fallback
            reproduction_code=reproduction_code,
            expected_outputs=item.get("expected_outputs") or expected_outputs,
            actual_outputs=item.get("actual_outputs") or actual_outputs,
            error_keywords=item.get("error_keywords") or error_keywords or [],
            identifiers=identifiers,
            oracle_hints=self._ensure_list(item.get("oracle_hints", [])),
            oracle=self._ensure_str(item.get("oracle"), ""),
            oracle_contract=item.get("oracle_contract") if isinstance(item.get("oracle_contract"), dict) else {},
            oracle_type=self._ensure_str(item.get("oracle_type"), ""),
            oracle_source=self._ensure_str(item.get("oracle_source"), ""),
        )

    def _dict_to_hydrated_scenario(
        self,
        item: Dict[str, Any],
        clue: Dict[str, Any],
        context: Dict[str, Any],
        repo: str = "",
    ) -> TestScenario:
        hydrated = hydrate_scenario_dict(item, clue, repo=repo, context=context)
        target = hydrated.get("target_location", {}) if isinstance(hydrated.get("target_location"), dict) else {}
        test_env = hydrated.get("test_environment", {}) if isinstance(hydrated.get("test_environment"), dict) else {}
        return TestScenario(
            scenario_id=hydrated.get("scenario_id", "S1"),
            target_location=target,
            setup_steps=self._ensure_list(hydrated.get("setup_steps", [])),
            execution_stimulus=self._ensure_list(hydrated.get("execution_stimulus", [])),
            expected_failure=self._ensure_str(hydrated.get("expected_failure"), ""),
            relevant_source_files=self._ensure_list(hydrated.get("relevant_source_files", [])),
            relevant_test_files=self._ensure_list(hydrated.get("relevant_test_files", [])),
            test_environment=test_env,
            reproduction_code=self._ensure_reproduction_code(hydrated.get("reproduction_code", [])),
            expected_outputs=self._ensure_list(hydrated.get("expected_outputs", [])),
            actual_outputs=self._ensure_list(hydrated.get("actual_outputs", [])),
            error_keywords=self._ensure_list(hydrated.get("error_keywords", [])),
            identifiers=hydrated.get("identifiers", {}) if isinstance(hydrated.get("identifiers"), dict) else {},
            oracle_hints=self._ensure_list(hydrated.get("oracle_hints", [])),
            oracle=self._ensure_str(hydrated.get("oracle"), ""),
            oracle_contract=hydrated.get("oracle_contract", {}) if isinstance(hydrated.get("oracle_contract"), dict) else {},
            oracle_type=self._ensure_str(hydrated.get("oracle_type"), ""),
            oracle_source=self._ensure_str(hydrated.get("oracle_source"), ""),
        )

    def _build_fallback_scenarios(
        self,
        clue: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[TestScenario]:
        """LLM 호출 실패 시 clue/context 기반 최소 시나리오 생성."""
        functions = clue.get("identifiers", {}).get("functions", [])
        classes = clue.get("identifiers", {}).get("classes", [])
        observed = clue.get("observed_behavior", [])
        expected = clue.get("expected_behavior", [])

        source_files = [x["path"] for x in context.get("candidate_source_files", [])]
        test_files = [x["path"] for x in context.get("candidate_test_files", [])]
        framework = context.get("project_test_style", {}).get("framework", "unknown")
        runner = context.get("project_test_style", {}).get("runner", "pytest")

        primary_source = source_files[0] if source_files else ""
        primary_test = test_files[0] if test_files else ""
        primary_func = ""
        for fl in clue.get("fault_locations", []) or []:
            if fl.get("function_name"):
                primary_func = fl["function_name"]
                break
        if not primary_func:
            primary_func = functions[0] if functions else ""

        expected_outputs = clue.get("expected_outputs", [])
        actual_outputs = clue.get("actual_outputs", [])
        if expected_outputs:
            oracle_type = "positive_value"
            oracle_source = "issue_expected"
            oracle_rule = "Assert the fixed behavior stated in expected_outputs."
        elif actual_outputs:
            oracle_type = "semantic_invariant"
            oracle_source = "actual_buggy_output"
            oracle_rule = "Call the target API and assert a public invariant that excludes the buggy output."
        else:
            oracle_type = "semantic_invariant"
            oracle_source = "inferred_semantic"
            oracle_rule = "Call the target API and assert public state/value behavior, not local constants."

        failure_text = observed[0] if observed else "The assertion should fail due to the bug described in the issue."

        framework_step = [f"Import the relevant module (framework: {framework})."] if framework != "unknown" else ["Import the relevant module."]

        scenarios = [
            TestScenario(
                scenario_id="S1",
                target_location={
                    "source_file": primary_source,
                    "target_function": primary_func,
                    "related_classes": classes[:3],
                    "candidate_test_file": primary_test,
                    "confidence": "medium",
                },
                setup_steps=framework_step + [
                    f"Set up an environment where {primary_func} can be called." if primary_func else "Reproduce the input conditions described in the issue.",
                ],
                execution_stimulus=[
                    f"Call {primary_func} with the reproduction conditions from the issue." if primary_func else "Execute the reproduction code from the issue.",
                    "Compare the result with the expected value.",
                ],
                expected_failure=failure_text,
                relevant_source_files=source_files[:3],
                relevant_test_files=test_files[:3],
                test_environment={"required_fixtures": [], "runner": runner},
                reproduction_code=clue.get("code_examples", []),
                expected_outputs=expected_outputs,
                actual_outputs=actual_outputs,
                error_keywords=clue.get("error_keywords", []),
                identifiers=clue.get("identifiers", {}),
                oracle_hints=[oracle_rule],
                oracle=oracle_rule,
                oracle_contract={
                    "oracle_type": oracle_type,
                    "oracle_source": oracle_source,
                    "rule": oracle_rule,
                },
                oracle_type=oracle_type,
                oracle_source=oracle_source,
            ),
        ]

        return scenarios

    @staticmethod
    def _build_conftest_section(conftest_fixtures: Dict[str, List[str]]) -> str:
        if not conftest_fixtures:
            return ""
        lines = ["[Available Pytest Fixtures (from conftest.py)]"]
        for path, names in conftest_fixtures.items():
            lines.append(f"  {path}: {', '.join(names)}")
        lines.append("")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _ensure_list(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(x) for x in value if x]
        if isinstance(value, str):
            return [value]
        return []

    @staticmethod
    def _ensure_str(value: Any, default: str = "") -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return " ".join(str(x) for x in value)
        return str(value) if value else default

    @staticmethod
    def _ensure_reproduction_code(value: Any) -> List[Dict[str, str]]:
        if isinstance(value, list):
            result: List[Dict[str, str]] = []
            for item in value:
                if isinstance(item, dict):
                    code = item.get("code", "")
                    result.append({
                        "language": str(item.get("language", "python")),
                        "code": str(code) if code else "",
                    })
                elif isinstance(item, str):
                    result.append({"language": "python", "code": item})
            return result
        if isinstance(value, dict):
            return [{
                "language": str(value.get("language", "python")),
                "code": str(value.get("code", "")),
            }]
        if isinstance(value, str):
            return [{"language": "python", "code": value}]
        return []
