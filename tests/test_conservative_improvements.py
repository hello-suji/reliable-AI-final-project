from pathlib import Path

from src.context_builder.code_context import (
    CandidateFile,
    CodeContextExtractor,
    IndexedFile,
)
from src.generator.repro_test_generator import (
    _choose_viable_test_file,
    _fill_empty_exception_blocks,
)
from src.issue_parser.issue_clues import IssueClueExtractor
from src.pipeline.run_batch import _entry_from_artifacts


def test_issue_clue_extracts_file_traceback_and_code_identifiers():
    issue = '''
Something fails in `pkg/module.py:12`.

```python
result = target_func(1)
```

Traceback (most recent call last):
  File "pkg/module.py", line 12, in target_func
ValueError: bad value
'''
    clue = IssueClueExtractor().extract("x", issue).to_dict()

    assert "target_func" in clue["identifiers"]["functions"]
    assert "pkg/module.py" in clue["identifiers"]["files"]
    assert clue["fault_locations"][0]["file_path"] == "pkg/module.py"
    assert clue["fault_locations"][0]["function_name"] == "target_func"


def test_context_graph_expansion_adds_called_neighbor():
    extractor = CodeContextExtractor(top_k_source=3)
    indexed = [
        IndexedFile(
            path="pkg/a.py",
            is_test_file=False,
            classes=[],
            functions=["entry"],
            methods=[],
            test_functions=[],
            imports=["pkg.b"],
            call_names=["helper"],
        ),
        IndexedFile(
            path="pkg/b.py",
            is_test_file=False,
            classes=[],
            functions=["helper"],
            methods=[],
            test_functions=[],
            imports=[],
            call_names=[],
        ),
    ]
    first_pass = [
        CandidateFile(
            path="pkg/a.py",
            score=10,
            matched_identifiers=["entry"],
            reasons=["function_match:entry"],
            localization_signals=["issue_identifier"],
        )
    ]

    expanded = extractor._expand_source_candidates_with_graph(first_pass, indexed)
    by_path = {item.path: item for item in expanded}

    assert "pkg/b.py" in by_path
    assert "graph_neighbor" in by_path["pkg/b.py"].localization_signals


def test_context_marks_sphinx_collection_risk():
    source = """
import pytest
pytestmark = pytest.mark.sphinx('html')

def test_example():
    pass
"""
    item = CodeContextExtractor()._parse_python_file("tests/test_app.py", source, True)

    assert "pytest_mark_sphinx" in item.collection_risk


def test_generator_repairs_empty_raises_block_and_picks_viable_file(tmp_path):
    repaired = _fill_empty_exception_blocks(
        "def test_x():\n"
        "    try:\n"
        "    except ValueError:\n"
        "    assert True\n"
    )
    assert "except ValueError:\n        pass" in repaired

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "tests").mkdir()
    (repo / "tests" / "test_good.py").write_text("def test_existing():\n    pass\n")
    context = {
        "repo_path": str(repo),
        "candidate_test_files": [
            {"path": "tests/test_missing.py", "has_module_skip": False},
            {"path": "tests/test_good.py", "has_module_skip": False, "collection_risk": ""},
        ],
    }

    chosen, reason = _choose_viable_test_file(context, "tests/test_missing.py")

    assert chosen == "tests/test_good.py"
    assert reason == "preferred_test_file_missing_or_unusable"


def test_batch_summary_does_not_promote_relaxed_alignment(tmp_path):
    output_dir = tmp_path / "instance"
    output_dir.mkdir()
    (output_dir / "alignment_result.json").write_text(
        """
{
  "failure_type": "NO_COVERAGE",
  "iteration": 5,
  "test_results": {"test_repro": "FAILED"},
  "score_breakdown": {
    "bug_fail_score": 1.0,
    "coverage_score": 0.5,
    "issue_alignment_score": 1.0
  }
}
"""
    )

    entry = _entry_from_artifacts("repo__repo-1", output_dir)

    assert entry["failure_type"] == "NO_COVERAGE"
    assert entry["strict_gate_pass"] is False
    assert "relaxed_alignment_detail" not in entry
