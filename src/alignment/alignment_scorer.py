"""
AlignmentScorer — patch-free 규칙기반 정합성 평가.

alignment_runner의 before-patch 결과를 받아:
  1) s_b: 버그 코드에서 FAIL 여부 (Bug Reproduction Score, 0~1)
  2) s_a: 이슈-테스트 규칙기반 정합성 (Issue Alignment Score, 0~1)
  3) s_c: 의심 위치 커버리지 (Coverage Score, 0~1)
를 독립 계산하고, 세 점수를 순차 게이트로 검사하여 ALIGNED 판정한다.

논문 §3.3 게이트 방식:
  Gate 1 (s_b) → Gate 2 (s_c) → Gate 3 (s_a) 순서로 각 임계값 검사.
  모두 통과 시 ALIGNED; 실패 시 해당 유형 반환.

판정 기준:
  최종 label은 s_b, s_c, s_a의 순차 게이트 통과 여부로만 결정한다.
  평균 alignment score는 배치 요약 지표로 사용하지 않는다.
"""
from __future__ import annotations

import copy
import logging
import re
import ast
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

ALIGNMENT_SCORE_SCHEMA_VERSION = "gate-v4-0to1"

# ---------------------------------------------------------------------------
# 점수 구성 — 각 컴포넌트 독립 계산 후 순차 게이트 검사 (논문 §3.3)
# label은 s_b, s_c, s_a 게이트로 판정한다. 평균 score는 산출하지 않는다.
# ---------------------------------------------------------------------------

# bug_fail_score (s_b): 0 ~ 1  (before-patch 실패 관측 게이트)
_BUG_FAIL_MAX           = 1.0
_BUG_FAIL_PARTIAL       = 1.0   # ERROR+PASSED 혼재 (부분 재현) 점수
_ALIGNED_BUG_FAIL_MIN   = 0.70  # ALIGNED 판정을 위한 τb

# 피처 가중치.
#
# s_b = clip(Σ_t w_t f_t, 0, 1). ERROR/NOT_VALID 계열 신호는 버그 재현
# 점수에서 감점하지 않고 failure_type과 feedback에서 별도로 다룬다.
_BUG_FAIL_FEATURE_WEIGHTS = {
    "f_fail":     0.50,  # before-patch에서 생성 테스트가 FAILED로 관측된 정도
    "f_assert":   0.25,  # assertion failure 또는 expected/actual mismatch 신호
    "f_symptom":  0.25,  # 실패 출력/테스트가 이슈 증거 토큰과 겹치는 정도
}

# issue_alignment_score (s_a): 0 ~ 1  (이슈-테스트 정합성, 4개 서브항목)
# =====================================================================
# 설계 근거:
# • 4개 sub-component를 균등 배분하는 이유: 각각이 독립적으로 중요
#   1) 식별자 매칭: 이슈에서 언급한 함수/클래스가 테스트에 등장
#   2) 코드 패턴 매칭: 이슈 코드 예시의 패턴이 테스트에 반영
#   3) 기대값/에러 매칭: 이슈의 기대값, 에러 키워드가 테스트에서 검증
#   4) 타겟 위치 일치: 시나리오의 target_function, source_file이 테스트에서 사용
# • Token hit threshold 0.1: 코드/값에서 10% 이상 토큰이 매칭되면 성공 판정
#   - 너무 높으면(e.g., 100%) 엄격하여 거짓 음성 증가
#   - 너무 낮으면 포용적이어서 거짓 양성 증가
#   - 0.1은 부분 매칭을 허용하기 위한 완화값 (검증 필요: 향후 데이터 기반 분석 권장)
# 
# ⚠️ 미해결 이슈:
# • 4개 sub-component의 중요도가 실제로 동등한지 검증 필요
# • 0.1 threshold의 근거 데이터 없음 (향후 별도 분석)
# • 식별자 ratio 계산: N개 중 M개 매칭 시 M/N점수 → 낮은 식별자 개수일 때 미안정
_ISSUE_ALIGN_MAX        = 1.0
_ISSUE_ALIGN_SUB        = _ISSUE_ALIGN_MAX / 4   # 서브항목당 최대 0.25
_ISSUE_ALIGN_TOKEN_HIT  = 0.1   # 토큰 히트율 임계값 (0.2→0.1: 부분 매칭 허용, 식별자 소수일 때 불리함 보정)
_ISSUE_ALIGN_STRONG_GATE = 0.75  # token overlap only인 경우 ALIGNED를 막기 위한 강화 게이트

# coverage_score (s_c): 0 ~ 1  (의심 위치 커버리지)
# =====================================================================
# 설계 근거:
# • 목표: 테스트가 의심 코드 라인 자체를 실행했는지 검증
# • 의심 라인 L_s:
#   1) clue.fault_locations의 line_no 주변 window
#   2) 없으면 scenario.target_location의 target_function AST statement lines
#   3) 둘 다 없으면 target source file coverage ratio fallback
# • s_c = |covered(L_s)| / |L_s| ∈ [0, 1]
#   - covered 여부는 coverage missing_lines에 포함되지 않는 statement line으로 판단
#   - gold patch/final-eval patch line coverage는 alignment에서 사용하지 않음
_COVERAGE_MAX           = 1.0
_COVERAGE_BASE          = 0.0   # base bonus 제거
_COVERAGE_BONUS_MAX     = 1.0   # 커버리지 비율(0~100%)을 그대로 0~1로 환산
_COVERAGE_FALLBACK      = 1.0  # source_file 미지정 시, non-test 파일 커버 비율로 환산
_SUSPICIOUS_LINE_WINDOW = 3

# WEAK_ALIGNMENT → switch_scenario 임계값
_SWITCH_SCENARIO_THRESHOLD = 0.3  # 가장 약한 게이트 점수가 이 미만이면 시나리오 전환

# Requirements-Based Scoring (V2) gate 임계값
# 각 컴포넌트의 최솟값을 만족해야 ALIGNED 판정 가능
_COVERAGE_MIN_GATE    = 0.60  # coverage < 이 값이면 NO_COVERAGE로 차단
_ISSUE_ALIGN_MIN_GATE = 0.65  # issue_align < 이 값이면 WEAK_ALIGNMENT로 차단
_ALIGNED_REPORT_BUG_FAIL_MIN = _ALIGNED_BUG_FAIL_MIN
_ALIGNED_REPORT_COVERAGE_MIN = _COVERAGE_MIN_GATE
_ALIGNED_REPORT_ISSUE_ALIGN_MIN = _ISSUE_ALIGN_MIN_GATE

# 피드백 임계값
_FEEDBACK_BUG_FAIL_WEAK    = _ALIGNED_BUG_FAIL_MIN
_FEEDBACK_ISSUE_ALIGN_WEAK = _ISSUE_ALIGN_MIN_GATE

# ---------------------------------------------------------------------------
# 피드백 문자열 및 반복 횟수 설정
# ---------------------------------------------------------------------------
# 텍스트 길이 제한 (피드백 메시지 가독성 및 토큰 절약)
_FEEDBACK_SHORT_STR_LEN     = 200   # 에러 메시지, assertion 등의 기본 길이
_FEEDBACK_MID_STR_LEN       = 300   # 기대값/실제값 표시 길이
_FEEDBACK_LONG_STR_LEN      = 500   # 상세 설명용 최대 길이

# 반복 항목 표시 개수
_FEEDBACK_ERROR_MSGS_MAX    = 3     # error_messages에서 보여줄 최대 개수
_FEEDBACK_ASSERTION_MAX     = 4     # NOT_FAILED의 assertion 라인 최대 표시
_FEEDBACK_CODE_EXAMPLES_MAX = 2     # 이슈 코드 예시 최대 표시
_FEEDBACK_OUTPUTS_MAX       = 2     # 기대값/실제값 최대 표시
_FEEDBACK_MISSING_IDS_MAX   = 10    # 누락된 식별자 최대 표시
_FEEDBACK_TRACEBACK_LINES   = 10    # fallback traceback에서 표시할 라인 수

# 피드백 메시지 이전 실행 타겟
_FEEDBACK_PREV_ITERATION_KEEP = 1    # 유지할 이전 iteration 개수 (현재 + 이전 1개)
_FEEDBACK_PROMPT_VISIBLE_MAX = 2      # oracle/stimulus/precondition별 prompt-visible 추가 상한


# ---------------------------------------------------------------------------
# 1. 실패 유형 분류
# ---------------------------------------------------------------------------

class FailureType(str, Enum):
    ALIGNED = "ALIGNED"               # 세 게이트(s_b, s_c, s_a) 모두 통과
    NOT_FAILED = "NOT_FAILED"         # 버그 코드에서 테스트가 FAIL하지 않음
    ERROR = "ERROR"                    # 실행 에러 (import/syntax/환경 문제)
    NOT_VALID = "NOT_VALID"            # 생성 테스트가 유효하지 않아 실행기가 찾거나 실행하지 못함
    NO_COVERAGE = "NO_COVERAGE"        # 의심 위치를 커버하지 못함
    WEAK_ALIGNMENT = "WEAK_ALIGNMENT"  # score < threshold (정합성 부족)


@dataclass
class OracleQuality:
    score: float
    risk_flags: List[str]
    feedback: List[str]


def evaluate_oracle_quality(
    generated_test: Dict[str, Any],
    clue: Optional[Dict[str, Any]] = None,
) -> OracleQuality:
    """Static oracle-risk gate used before accepting ALIGNED.

    Patch-free alignment can confirm that a generated test fails on the buggy
    version, but final resolve also requires the same test to pass after the
    patch. These patterns are frequent causes of fail→fail final outcomes.
    """
    code = generated_test.get("test_code") or generated_test.get("append_block") or ""
    lower = code.lower()
    risk_flags: List[str] = []
    feedback: List[str] = []
    expected_outputs = (clue or {}).get("expected_outputs", [])
    actual_outputs = (clue or {}).get("actual_outputs", [])

    assertion_lines = [
        line.strip()
        for line in code.splitlines()
        if re.search(
            r"\bassert\b|self\.assert|pytest\.raises|with\s+.*raises"
            r"|assert_allclose|assert_array|assert_equal|assert_raises",
            line,
        )
    ]

    def add_flag(flag: str, message: str) -> None:
        if flag not in risk_flags:
            risk_flags.append(flag)
            feedback.append(message)

    test_def_count = len(
        re.findall(r"^\s*(?:async\s+)?def\s+test_\w+\s*\(", code, flags=re.MULTILINE)
    )
    if test_def_count > 1:
        add_flag(
            "multiple_generated_tests",
            "하나의 generated patch에 test_*가 여러 개 있으면 일부는 flip되고 일부는 after에서 실패할 수 있다. "
            "가능하면 하나의 focused reproduction test만 생성하고, 보조 함수는 test_ prefix를 쓰지 마라.",
        )

    def _has_issue_expected_signal() -> bool:
        norm_code = re.sub(r"\s+", "", lower)
        for out in expected_outputs[:3]:
            norm_expected = re.sub(r"\s+", "", str(out).lower())
            if norm_expected and norm_expected[:80] in norm_code:
                return True
        return False

    trivial_assertions = [
        line for line in assertion_lines
        if re.search(
            r"^(?:self\.)?assertTrue\s*\(\s*(?:True|1)\s*(?:,\s*[^)]*)?\)\s*(?:#.*)?$"
            r"|^assert\s+(?:True|1)\s*(?:#.*)?$",
            line,
            re.IGNORECASE,
        )
    ]
    if trivial_assertions:
        add_flag(
            "trivial_oracle",
            "`assertTrue(True)`/`assert 1` 같은 무의미한 oracle은 제거하고 "
            "수정 후 통과해야 하는 실제 return value 또는 state change를 검증하라.",
        )

    # @image_comparison 데코레이터: baseline 이미지 없으면 항상 fail
    if re.search(r"@image_comparison", code):
        add_flag(
            "image_comparison_decorator",
            "@image_comparison 데코레이터는 baseline 이미지가 없으면 항상 실패한다. "
            "직접 값/속성을 검증하는 assertion으로 교체하라.",
        )

    # warning 기반 테스트: 패치 후 warning이 사라지면 assertion도 실패
    if re.search(r"catch_warnings|assertWarns|warnings\.warn", code):
        has_non_warning_assertion = bool(re.search(
            r"\bassert\b(?!.*warning)|assertEqual(?!.*warning)|assertIn(?!.*warning)", code, re.IGNORECASE
        ))
        if not has_non_warning_assertion:
            add_flag(
                "warning_catch_only",
                "warning 기반 테스트는 패치 후 경고가 사라지면 assertion이 실패한다. "
                "경고 대신 실제 동작(return value, side effect)을 검증하라.",
            )

    if re.search(
        r"len\s*\(\s*\w+\s*\)\s*==\s*1.*(?:warning|warn)|"
        r"issubclass\s*\([^)]*warning|"
        r"\.category\s*,\s*(?:RuntimeWarning|Warning)",
        code,
        re.IGNORECASE | re.DOTALL,
    ):
        add_flag(
            "warning_presence_oracle",
            "warning 개수/타입만 검증하면 패치 후 경고가 사라지는 경우 after에서도 실패한다. "
            "warning이 아니라 수정 후 값/상태 또는 no-warning 성공 경로를 검증하라.",
        )

    # 예외 메시지 exact match: cm.exception, exc.value 등 다양한 패턴
    if re.search(
        r"str\s*\(\s*\w+[\.\w]*exception\s*\)\s*==|"
        r"\w+[\.\w]*args\[\d+\]\s*==|"
        r"str\s*\(\s*\w+\s*\)\s*==\s*['\"]",
        code,
    ):
        add_flag(
            "exception_message_match",
            "예외 메시지 exact match는 버전별로 흔들린다. 예외 타입 또는 의미 조건만 검증하라.",
        )

    if re.search(
        r"assert\s+str\s*\(\s*[\w.]+\s*\)\s*!=\s*['\"]|"
        r"\w+(?:\.value)?\.args\[\d+\]\s*!=\s*['\"]|"
        r"assert\s+['\"].+['\"]\s+not\s+in\s+str\s*\(|"
        r"self\.assert(?:NotIn|NotRegex)\s*\([^,\n]+,\s*str\s*\(|"
        r"self\.assertNotEqual\s*\(\s*str\s*\(|"
        r"self\.assertNotIn\s*\([^,\n]+,\s*[\w.]+\.args\[\d+\]",
        code,
        re.IGNORECASE,
    ):
        add_flag(
            "exception_message_negative_oracle",
            "예외 메시지 부재/변경을 oracle로 쓰면 fix 후 예외가 사라질 때도 실패한다. "
            "예외가 없어지는 성공 경로 또는 올바른 exception type만 검증하라.",
        )

    if not assertion_lines and "image_comparison_decorator" not in risk_flags:
        add_flag("no_explicit_oracle", "테스트에 명시적인 assertion/raises oracle이 없다.")

    # pytest.raises/assertRaises가 있지만 body에 assertion이 없음
    # → 패치 전/후 모두 예외가 발생하면 flip이 안 일어남
    raises_only = (
        assertion_lines
        and all(
            re.search(r"pytest\.raises|assertRaises|assert_raises|with\s+.*raises", line)
            for line in assertion_lines
        )
        and not re.search(r"^\s*(assert\s+(?!.*raises)|self\.assertEqual|self\.assertIn)", code, re.MULTILINE)
    )
    if raises_only:
        add_flag(
            "raises_only_no_body_assertion",
            "pytest.raises / assertRaises만 있고 body에 assertion이 없다. "
            "예외 타입 체크만으로는 패치 전/후를 구분하지 못한다. "
            "예외 없이 성공하는 동작 또는 result를 직접 검증하는 assertion을 추가하라.",
        )
    issue_text = " ".join(
        str(x)
        for x in (
            (clue or {}).get("observed_behavior", [])
            + (clue or {}).get("expected_behavior", [])
            + (clue or {}).get("repro_conditions", [])
            + [(clue or {}).get("raw_issue_text", "")]
        )
    ).lower()
    issue_says_success_path = bool(re.search(
        r"should\s+not\s+(?:raise|error|fail|crash|warn)|"
        r"must\s+not\s+(?:raise|error|fail|crash|warn)|"
        r"does\s+not\s+(?:raise|error|fail|crash|warn)|"
        r"doesn't\s+(?:raise|error|fail|crash|warn)|"
        r"without\s+(?:raising|error|failing|crashing|warning)|"
        r"no\s+(?:exception|error|warning)|"
        r"no\s+longer\s+(?:raises|errors|fails|crashes|warns)",
        issue_text,
    ))
    issue_says_exception_expected = bool(re.search(
        r"should\s+raise|must\s+raise|expected\s+(?:error|exception)|"
        r"should\s+(?:error|fail)\b|raises?\s+(?:a\s+)?(?:typeerror|valueerror|attributeerror|runtimeerror)",
        issue_text,
    ))
    has_raises_oracle = bool(re.search(r"pytest\.raises|assertRaises|assert_raises|with\s+.*raises", code))
    if has_raises_oracle and (
        issue_says_success_path
        or (not issue_says_exception_expected and re.search(r"post[- ]fix|should\s+accept|fit\s+success|succeed", code, re.IGNORECASE))
    ):
        add_flag(
            "fix_disappearing_exception_oracle",
            "이슈가 예외/에러가 없어져야 하는 성공 경로를 말하는데 raises oracle을 사용하고 있다. "
            "try/call success path와 post-call value/state assertion으로 재작성하라.",
        )

    structural_patterns = (
        r"\bis\s+not\s+none\b",
        r"\bisinstance\s*\(",
        r"\blen\s*\([^)]+\)\s*>\s*0\b",
        r"\.assertisnotnone\s*\(",
        r"\.asserttrue\s*\(\s*len\s*\(",
    )
    if assertion_lines and all(
        any(re.search(p, line, re.IGNORECASE) for p in structural_patterns)
        for line in assertion_lines
    ):
        add_flag("structural_oracle_only", "구조적 assertion만 사용하지 말고 이슈의 올바른 값/동작을 직접 검증하라.")

    structural_hits = [
        line for line in assertion_lines
        if any(re.search(p, line, re.IGNORECASE) for p in structural_patterns)
        or re.search(r"\.assertisinstance\s*\(|\.assertisnotnone\s*\(", line, re.IGNORECASE)
    ]
    if structural_hits and len(structural_hits) >= max(1, len(assertion_lines) - 1):
        add_flag(
            "weak_structural_oracle",
            "`assertIsInstance`/`assertIsNotNone`/length 같은 구조적 oracle은 post-fix 동작을 충분히 특정하지 못한다. "
            "issue-specific value 또는 state change를 검증하라.",
        )

    negative_assertions = [
        line for line in assertion_lines
        if (
            "!=" in line
            or ".assertnot" in line.lower()
            or re.search(r"\bnot\s+np\.array_equal\b", line.lower())
        )
    ]
    if assertion_lines and len(negative_assertions) == len(assertion_lines):
        add_flag("negative_oracle_only", "`!= buggy value`만 검증하면 fix 후에도 실패할 수 있다. 가능한 positive oracle을 사용하라.")

    if re.search(
        r"(?:expected|baseline|correct|desired|known)_[A-Za-z0-9_]*\s*!=|"
        r"assert\s+repr\s*\(\s*(?:expected|baseline|correct|desired|known)_[A-Za-z0-9_]*\s*\)\s*!=",
        code,
        re.IGNORECASE,
    ):
        add_flag(
            "constant_negative_oracle",
            "테스트 내부 local constant에 대한 negative assertion은 함수 결과를 검증하지 않는다. "
            "반드시 function_under_test의 return value 또는 state change를 assertion 대상으로 삼아라.",
        )

    if re.search(
        r"(?:expected|baseline|correct|desired)_(?:matrix|array|result|values?)\s*=.*\n"
        r"(?s:.*?)(?:assert_array_equal|assert_allclose|assert_equal)\s*\([^,\n]+,\s*"
        r"(?:expected|baseline|correct|desired)_(?:matrix|array|result|values?)",
        code,
        re.IGNORECASE,
    ) and not _has_issue_expected_signal():
        add_flag(
            "guessed_expected_array",
            "테스트 내부에서 만든 expected_matrix/array를 exact oracle로 쓰고 있다. "
            "issue expected_outputs에 근거한 값이 아니면 property invariant나 실제 post-fix 의미 조건으로 바꿔라.",
        )
    if re.search(
        r"(?:expected|baseline|correct|desired)_(?:value|output|result)\s*=.*\n"
        r"(?s:.*?)(?:assert\s+[^=\n]+==\s*|self\.assertEqual\s*\([^,\n]+,\s*)"
        r"(?:expected|baseline|correct|desired)_(?:value|output|result)",
        code,
        re.IGNORECASE,
    ) and not _has_issue_expected_signal():
        add_flag(
            "guessed_expected_value",
            "테스트 내부에서 만든 expected_value/output/result를 exact oracle로 쓰고 있다. "
            "issue expected_outputs 근거가 없으면 semantic invariant로 바꿔라.",
        )

    if re.search(r"float\s*\(\s*['\"]nan['\"]\s*\)|\bnp\.nan\b", lower):
        add_flag("nan_comparison", "NaN은 직접 비교하지 말고 `np.isnan(...)` 또는 warning 검증을 사용하라.")

    if re.search(r"^\s*assert\s+.+==\s*np\.array\s*\(", code, re.MULTILINE):
        add_flag("numpy_direct_equality", "numpy/object array는 직접 `assert a == np.array(...)` 대신 `np.testing.assert_array_equal` 또는 객체 identity를 검증하라.")

    if re.search(r"pytest\.raises\([^)]*match\s*=", code) or re.search(
        r"assert\s+['\"].+['\"]\s+in\s+str\s*\(", code
    ):
        add_flag("exception_message_match", "예외 메시지 exact match는 버전별로 흔들린다. 예외 타입 또는 의미 조건만 검증하라.")

    if re.search(r"requests\.(get|post|put|delete|request)\s*\(\s*['\"]https?://", code):
        add_flag("external_network_call", "외부 네트워크 호출은 금지된다. PreparedRequest, mock, 기존 HTTP helper로 header/request 객체를 검증하라.")

    if re.search(r"class\s+\w+\s*\([^)]*models\.Model[^)]*\)", code):
        add_flag("django_inline_model", "Django model class를 테스트 안에 새로 정의하지 말고 기존 테스트 model/import를 사용하라.")

    if re.search(r"get_[xy]lim\(\)\s*\[\s*[01]\s*\]\s*==", code):
        add_flag("raw_axis_limit_equality", "Matplotlib 축 반전은 raw limit equality보다 `ax.yaxis_inverted()`/`xaxis_inverted()` 같은 의미 기반 oracle을 사용하라.")

    raw_string_asserts = [
        line for line in assertion_lines
        if re.search(r"(?:self\.)?assert(?:In|NotIn)\s*\(\s*['\"]", line)
        and (
            len(line) > 140
            or re.search(r"\\PYG|\\sphinx|<[^>]+>|html_content|tex_content|latex", line, re.IGNORECASE)
        )
    ]
    if raw_string_asserts:
        add_flag(
            "raw_rendered_output_exact_match",
            "Sphinx/HTML/LaTeX raw rendered string exact match는 너무 brittle하다. "
            "최소 semantic marker나 구조적 invariant로 재작성하라.",
        )
    if re.search(r"\._[A-Za-z]\w*", code):
        add_flag(
            "private_attribute_oracle",
            "private attribute를 oracle로 읽으면 내부 구현에 묶여 post-fix에서도 실패하기 쉽다. "
            "public API의 return value/state로 검증하라.",
        )

    has_positive_signal = bool(expected_outputs) and any(
        str(out).strip() and str(out).strip()[:40].lower() in lower
        for out in expected_outputs[:2]
    )
    has_only_buggy_signal = bool(actual_outputs) and not has_positive_signal and any(
        str(out).strip() and str(out).strip()[:40].lower() in lower
        for out in actual_outputs[:2]
    )
    if has_only_buggy_signal and "negative_oracle_only" not in risk_flags:
        add_flag("buggy_output_as_oracle", "버그 출력만 oracle로 쓰지 말고 수정 후 올바른 기대 출력/동작을 검증하라.")

    penalty = min(0.15 * len(risk_flags), 0.85)
    return OracleQuality(score=round(1.0 - penalty, 4), risk_flags=risk_flags, feedback=feedback)


# ---------------------------------------------------------------------------
# 2. 세 게이트 점수 산출
# ---------------------------------------------------------------------------

def extract_failure_features(
    raw_output: str,
    test_results: Dict[str, str],
) -> Dict[str, int]:
    """raw_output traceback에서 bug-fail 관련 이진 피처를 추출한다.

    Returns:
        각 피처명 → 0 또는 1 (존재 여부). 가중치는
        _BUG_FAIL_FEATURE_WEIGHTS에 정의한다.
    """
    statuses = set(test_results.values()) if test_results else set()
    return {
        "f_assert_diff": int(bool(re.search(
            r"AssertionError"                        # assertIs/assertEqual/assertTrue/assertRaises 등 모든 assertion 실패
            r"|\s!=\s",                              # 직접 비교
            raw_output,
        ))),
        "f_semantic_err": int(bool(re.search(
            r"\b(TypeError|ValueError|AttributeError|RuntimeError|KeyError|IndexError)\b",
            raw_output,
        ))),
        "f_import_err": int(bool(re.search(
            r"\b(NameError|ImportError|ModuleNotFoundError): ",  # 'except ImportError:' 오탐 방지
            raw_output,
        ))),
        "f_db_err": int(bool(re.search(
            r"\b(OperationalError|DatabaseError|ProgrammingError)\b|no such table",
            raw_output,
        ))),
        "f_setup_assert": int(bool(re.search(
            r"(setUp\b|setUpClass|setUpTestData"
            r"|Database queries to .+ not allowed)",
            raw_output,
        ))),
        "f_has_passed": int("PASSED" in statuses),
        "f_test_failed_summary": int(bool(re.search(
            r"\d+\s+failed",  # "1 failed", "2 failed", etc. — pytest가 명시적으로 보고하는 실패
            raw_output,
        ))),
    }


def compute_bug_fail_score(
    test_results: Dict[str, str],
    has_error: bool,
    raw_output: str = "",
    clue: Optional[Dict[str, Any]] = None,
    scenario: Optional[Dict[str, Any]] = None,
    generated_test: Optional[Dict[str, Any]] = None,
) -> float:
    """before-patch 실패 신호를 0~1로 스케일링한 bug reproduction score.

    s_b = clip(Σ_t w_t f_t, 0, 1). ERROR/NOT_VALID 계열 신호는 감점하지
    않고 failure_type과 feedback에서 별도로 다룬다.
    """
    if not test_results:
        return 0.0
    features = compute_bug_fail_features(
        test_results=test_results,
        raw_output=raw_output,
        clue=clue or {},
        scenario=scenario or {},
        generated_test=generated_test or {},
    )
    score = sum(
        _BUG_FAIL_FEATURE_WEIGHTS[name] * features.get(name, 0.0)
        for name in _BUG_FAIL_FEATURE_WEIGHTS
    )
    return round(_clamp01(score), 4)


def _token_set(text: str) -> Set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z_]\w+", text or "")
        if len(token) > 1
    }


def _issue_symptom_tokens(
    clue: Dict[str, Any],
    scenario: Dict[str, Any],
) -> Set[str]:
    tokens: Set[str] = set()
    clue_ids = clue.get("identifiers", {}) if isinstance(clue, dict) else {}
    if isinstance(clue_ids, dict):
        for key in ("functions", "classes", "exceptions"):
            for value in clue_ids.get(key, []) or []:
                tokens.update(_token_set(str(value)))
    for key in ("expected_outputs", "actual_outputs", "error_keywords"):
        for value in clue.get(key, []) or []:
            tokens.update(_token_set(str(value)))
    target = scenario.get("target_location", {}) if isinstance(scenario, dict) else {}
    if isinstance(target, dict):
        for key in ("source_file", "target_function"):
            tokens.update(_token_set(str(target.get(key, ""))))
        for value in target.get("related_classes", []) or []:
            tokens.update(_token_set(str(value)))
    return tokens


def compute_bug_fail_features(
    test_results: Dict[str, str],
    raw_output: str = "",
    clue: Optional[Dict[str, Any]] = None,
    scenario: Optional[Dict[str, Any]] = None,
    generated_test: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Return f_fail, f_assert, f_symptom for BugScore(r)."""
    statuses = list(test_results.values()) if test_results else []
    executable = [status for status in statuses if status in {"PASSED", "FAILED"}]
    failed = sum(1 for status in executable if status == "FAILED")
    f_fail = failed / len(executable) if executable else 0.0

    f_assert = 1.0 if re.search(
        r"AssertionError|\bassert\b|expected|actual|\s!=\s",
        raw_output or "",
        re.IGNORECASE,
    ) else 0.0

    evidence = _issue_symptom_tokens(clue or {}, scenario or {})
    observed = _token_set(raw_output or "")
    if generated_test:
        observed.update(_token_set(str(generated_test.get("test_code", ""))))
        observed.update(_token_set(str(generated_test.get("test_patch", ""))))
    f_symptom = len(evidence & observed) / len(evidence) if evidence else 0.0

    return {
        "f_fail": round(_clamp01(f_fail), 4),
        "f_assert": round(_clamp01(f_assert), 4),
        "f_symptom": round(_clamp01(f_symptom), 4),
    }


def compute_issue_alignment_score(
    clue: Dict[str, Any],
    scenario: Dict[str, Any],
    generated_test: Dict[str, Any],
) -> float:
    """이슈-테스트 규칙기반 정합성. 최대 _ISSUE_ALIGN_MAX(1.0)점.

    하위 항목 4개 (각 _ISSUE_ALIGN_SUB = _ISSUE_ALIGN_MAX / 4):
      - 식별자 매칭: 이슈의 핵심 식별자가 테스트 코드에 등장
      - 코드 패턴 매칭: 이슈 코드 예시의 패턴이 테스트에 반영
      - 기대 에러/값 매칭: 이슈에서 언급한 에러나 기대값이 테스트에 등장
      - 타겟 위치 일치: 시나리오의 target_location이 테스트에 사용됨

    정규화: 이슈 데이터에 따라 평가 불가한 항목(코드 예시 없음 등)은
    분모에서 제외하고 적용 가능한 항목 기준으로 0~_ISSUE_ALIGN_MAX 스케일로 환산.
    → 데이터 특성(이슈에 코드 예시/에러 키워드 유무)에 관계없이 공정한 평가.
    """
    test_code = generated_test.get("test_code", "").lower()
    if not test_code:
        return 0.0

    applicable = 0   # 적용 가능한 서브항목 수
    raw_score = 0.0  # applicable 기준 누적 점수 (각 항목 최대 1.0)

    # --- 식별자 매칭 ---
    noisy_functions = {
        "arange", "rand", "random", "seed", "platform", "get_backend",
        "show_versions",
    }
    identifiers = set()
    clue_ids = clue.get("identifiers", {})
    for fn in clue_ids.get("functions", []):
        if isinstance(fn, str) and fn not in noisy_functions:
            identifiers.add(fn.lower())
    for cls in clue_ids.get("classes", []):
        if isinstance(cls, str):
            identifiers.add(cls.lower())
    for exc in clue_ids.get("exceptions", []):
        if isinstance(exc, str):
            identifiers.add(exc.lower())
    if identifiers:
        applicable += 1
        matched = sum(1 for ident in identifiers if ident in test_code)
        if identifiers:
            ratio = matched / len(identifiers)
            # partial credit is intentionally small: identifier overlap alone
            # should not make a generic failing test look aligned.
            raw_score += max(ratio, 0.15) if matched > 0 else 0.0

    # --- 코드 패턴 매칭 (이슈에 코드 예시가 있을 때만 평가) ---
    code_examples = clue.get("code_examples", [])
    if code_examples:
        applicable += 1
        pattern_hits = 0
        for block in code_examples:
            code = block.get("code", "") or block.get("interactive_input", "")
            if not code:
                continue
            tokens = set(re.findall(r"[A-Za-z_]\w+", code))
            hits = sum(1 for t in tokens if t.lower() in test_code)
            if tokens and hits / len(tokens) >= _ISSUE_ALIGN_TOKEN_HIT:
                pattern_hits += 1
        raw_score += min(pattern_hits / max(len(code_examples), 1), 1.0)

    # --- 기대 에러/값 매칭 (이슈에 출력/에러 키워드가 있을 때만 평가) ---
    expected_outputs = clue.get("expected_outputs", [])
    actual_outputs = clue.get("actual_outputs", [])
    error_keywords = clue.get("error_keywords", [])
    value_hits = 0
    total_values = 0
    for vals in [expected_outputs, actual_outputs, error_keywords]:
        for v in vals:
            total_values += 1
            v_lower = str(v).lower()
            tokens = set(re.findall(r"[A-Za-z_]\w+", v_lower))
            if tokens:
                hits = sum(1 for t in tokens if t in test_code)
                if hits / len(tokens) >= _ISSUE_ALIGN_TOKEN_HIT:
                    value_hits += 1
            elif v_lower in test_code:
                value_hits += 1
    if total_values > 0:
        applicable += 1
        raw_score += min(value_hits / total_values, 1.0)

    # --- 타겟 위치 일치 ---
    target = scenario.get("target_location", {})
    target_func = target.get("target_function", "")
    source_file = target.get("source_file", "")
    if not isinstance(target_func, str):
        target_func = ""
    if not isinstance(source_file, str):
        source_file = ""
    target_hits = 0
    target_total = 0
    if target_func:
        target_total += 1
        if target_func.lower() in test_code:
            target_hits += 1
    if source_file:
        target_total += 1
        module_path = source_file.replace("/", ".").replace(".py", "")
        if (source_file.lower() in test_code
                or module_path.lower() in test_code
                or source_file.split("/")[-1].replace(".py", "").lower() in test_code):
            target_hits += 1
    if target_total > 0:
        applicable += 1
        raw_score += target_hits / target_total

    if applicable == 0:
        return 0.0

    # 적용 가능한 항목 기준으로 정규화 → [0, _ISSUE_ALIGN_MAX] 범위로 스케일
    return round((raw_score / applicable) * _ISSUE_ALIGN_MAX, 4)


def has_strong_issue_evidence(
    clue: Dict[str, Any],
    generated_test: Dict[str, Any],
) -> bool:
    """Check for non-trivial issue evidence beyond broad identifier overlap."""
    test_code = generated_test.get("test_code", "").lower()
    if not test_code:
        return False

    for block in clue.get("code_examples", []):
        if block.get("is_system_or_output"):
            continue
        code = block.get("code", "") or block.get("interactive_input", "")
        tokens = {
            t.lower() for t in re.findall(r"[A-Za-z_]\w+", code)
            if t not in {"from", "import", "def", "class", "assert", "with", "print"}
        }
        if tokens:
            hits = sum(1 for t in tokens if t in test_code)
            if hits / len(tokens) >= max(_ISSUE_ALIGN_TOKEN_HIT, 0.2):
                return True

    for vals in (
        clue.get("expected_outputs", []),
        clue.get("actual_outputs", []),
        clue.get("error_keywords", []),
    ):
        for value in vals:
            value_lower = str(value).lower()
            tokens = set(re.findall(r"[A-Za-z_]\w+", value_lower))
            if tokens:
                hits = sum(1 for t in tokens if t in test_code)
                if hits / len(tokens) >= max(_ISSUE_ALIGN_TOKEN_HIT, 0.2):
                    return True
            elif value_lower and value_lower in test_code:
                return True

    return False


def is_target_location_verified(
    scenario: Dict[str, Any],
    context: Optional[Dict[str, Any]],
    execution_result: Optional[Dict[str, Any]],
) -> bool:
    """Require a non-empty target function that exists and, when known, executes."""
    target = scenario.get("target_location") or {}
    if not isinstance(target, dict):
        return False
    source_file = target.get("source_file", "")
    target_func = target.get("target_function", "")
    if not isinstance(source_file, str) or not isinstance(target_func, str):
        return False
    if not source_file or not target_func:
        return False

    bare = target_func.split(".")[-1]
    if context:
        source_entries = context.get("candidate_source_files", [])
        for sf in source_entries:
            if sf.get("path") != source_file:
                continue
            top_funcs = sf.get("top_level_functions") or []
            if top_funcs:
                if bare.startswith("__") and bare.endswith("__"):
                    break
                if not any(tf == target_func or tf.split(".")[-1] == bare for tf in top_funcs):
                    return False
            break

    contributing = (execution_result or {}).get("contributing_functions", {})
    if isinstance(contributing, dict) and contributing:
        funcs_in_target_file = []
        for fname, funcs in contributing.items():
            if fname.endswith(source_file) or source_file.endswith(fname):
                funcs_in_target_file.extend(funcs if isinstance(funcs, list) else [])
        if funcs_in_target_file and not any(
            target_func == fn or fn.endswith(f".{bare}") or f".{bare}." in fn
            for fn in funcs_in_target_file
        ):
            return False

    return True


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _path_tail_matches(candidate: str, expected: str) -> bool:
    if not candidate or not expected:
        return False
    candidate_norm = candidate.replace("\\", "/")
    expected_norm = expected.replace("\\", "/")
    return (
        candidate_norm.endswith(expected_norm)
        or expected_norm.endswith(candidate_norm)
        or candidate_norm.split("/")[-1] == expected_norm.split("/")[-1]
    )


def _match_coverage_file(coverage_data: Dict[str, Dict], source_file: str) -> Optional[str]:
    for fname in coverage_data:
        if _path_tail_matches(fname, source_file):
            return fname
    return None


def _parse_line_no(value: Any) -> Optional[int]:
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str):
        match = re.search(r"\d+", value)
        if match:
            line_no = int(match.group(0))
            return line_no if line_no > 0 else None
    return None


def _repo_source_path(source_file: str, context: Optional[Dict[str, Any]]) -> Optional[Path]:
    if not source_file or not context:
        return None
    repo_path = context.get("repo_path")
    if not repo_path:
        return None
    path = Path(repo_path) / source_file
    return path if path.exists() else None


def _filter_source_code_lines(source_path: Optional[Path], lines: Set[int]) -> Set[int]:
    if not source_path or not source_path.exists() or not lines:
        return {line for line in lines if line > 0}
    try:
        source_lines = source_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return {line for line in lines if line > 0}

    filtered: Set[int] = set()
    for line_no in lines:
        if line_no <= 0 or line_no > len(source_lines):
            continue
        stripped = source_lines[line_no - 1].strip()
        if stripped and not stripped.startswith("#"):
            filtered.add(line_no)
    return filtered


def _fault_location_suspicious_lines(
    clue: Optional[Dict[str, Any]],
    source_file: str,
    source_path: Optional[Path],
) -> Set[int]:
    if not clue:
        return set()
    lines: Set[int] = set()
    for fault in clue.get("fault_locations", []) or []:
        if not isinstance(fault, dict):
            continue
        fault_file = str(
            fault.get("file_path")
            or fault.get("source_file")
            or fault.get("file")
            or ""
        )
        if fault_file and not _path_tail_matches(fault_file, source_file):
            continue
        line_no = _parse_line_no(
            fault.get("line_no")
            or fault.get("line")
            or fault.get("lineno")
        )
        if not line_no:
            continue
        lines.update(range(line_no - _SUSPICIOUS_LINE_WINDOW, line_no + _SUSPICIOUS_LINE_WINDOW + 1))
    return _filter_source_code_lines(source_path, lines)


def _statement_lines_for_target(
    source_path: Optional[Path],
    target_func: str,
    related_classes: Optional[List[Any]] = None,
) -> Set[int]:
    if not source_path or not source_path.exists() or not target_func:
        return set()
    try:
        source_text = source_path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source_text)
    except (OSError, SyntaxError):
        return set()

    bare_target = target_func.split(".")[-1]
    class_names = {
        str(name).split(".")[-1]
        for name in (related_classes or [])
        if str(name).strip()
    }
    target_nodes: List[ast.AST] = []
    if class_names:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in class_names:
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == bare_target:
                        target_nodes.append(child)
                if node.name == bare_target:
                    target_nodes.append(node)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == target_func or node.name == bare_target:
                target_nodes.append(node)

    if not target_nodes:
        return set()

    def node_span(node: ast.AST) -> int:
        start = getattr(node, "lineno", 0) or 0
        end = getattr(node, "end_lineno", start) or start
        return max(end - start, 0)

    # Prefer the smallest matching node so a method named `clean` inside a class
    # beats a broad class wrapper with the same public name.
    selected = min(target_nodes, key=node_span)
    lines: Set[int] = set()
    for node in ast.walk(selected):
        if isinstance(node, ast.stmt):
            line_no = getattr(node, "lineno", None)
            if isinstance(line_no, int) and line_no > 0:
                lines.add(line_no)
    return _filter_source_code_lines(source_path, lines)


def _coverage_file_ratio(info: Dict[str, Any]) -> float:
    try:
        return _clamp01(float(info.get("cover", 0.0)) / 100.0)
    except (TypeError, ValueError):
        return 0.0


def _suspicious_line_coverage_ratio(
    info: Dict[str, Any],
    suspicious_lines: Set[int],
) -> float:
    if not suspicious_lines or _coverage_file_ratio(info) <= 0:
        return 0.0
    missing_lines = {
        line
        for line in (_parse_line_no(value) for value in (info.get("missing_lines") or []))
        if line is not None
    }
    covered = [line for line in suspicious_lines if line not in missing_lines]
    return _clamp01(len(covered) / len(suspicious_lines))


def compute_coverage_score(
    coverage_data: Dict[str, Dict],
    scenario: Dict[str, Any],
    execution_result: Optional[Dict[str, Any]] = None,
    clue: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> float:
    """의심 코드 라인 커버리지. 최대 1.0점.

    gold patch 라인은 쓰지 않는다. clue/scenario/context에서 얻은 의심 위치
    라인 집합 L_s 중 before-patch 실행에서 실제 커버된 비율만 계산한다.
    """
    if not coverage_data:
        return 0.0

    target = scenario.get("target_location", {})
    source_file = target.get("source_file", "")
    target_func = target.get("target_function", "")
    if not isinstance(source_file, str):
        source_file = ""
    if not isinstance(target_func, str):
        target_func = ""

    if not source_file:
        # source_file이 없으면 전체 커버리지 기반으로 추정
        # 소스 파일(test가 아닌) 중 커버된 게 있으면 부분 점수
        source_covered = 0
        source_total = 0
        for fname, info in coverage_data.items():
            if "/test" in fname or "test_" in fname:
                continue
            source_total += 1
            if info.get("cover", 0) > 0:
                source_covered += 1
        if source_total > 0:
            return _COVERAGE_FALLBACK * min(source_covered / source_total, 1.0)
        return 0.0

    # source_file이 있는 경우 — 정확한 매칭
    matched_file = _match_coverage_file(coverage_data, source_file)
    if not matched_file:
        # 의심 파일이 커버리지 리포트에 없음 → 테스트가 해당 파일을 전혀 실행하지 않음
        return 0.0

    info = coverage_data.get(matched_file)
    if not isinstance(info, dict):
        return 0.0
    file_ratio = _coverage_file_ratio(info)
    if file_ratio == 0:
        return 0.0

    source_path = _repo_source_path(source_file, context)
    suspicious_lines = _fault_location_suspicious_lines(clue, source_file, source_path)
    if not suspicious_lines:
        suspicious_lines = _statement_lines_for_target(
            source_path,
            target_func,
            target.get("related_classes") if isinstance(target, dict) else None,
        )

    if suspicious_lines:
        return round(_suspicious_line_coverage_ratio(info, suspicious_lines), 4)

    # 의심 라인을 만들 수 없는 오래된/불완전 산출물은 target source file의
    # coverage ratio로만 fallback한다. 이 경우도 patch line coverage는 쓰지 않는다.
    return round(min(file_ratio, _COVERAGE_MAX), 4)


def classify_and_score_v2(
    test_results: Dict[str, str],
    has_error: bool,
    bug_fail: float,
    coverage: float,
    issue_align: float,
    failure_features: Optional[Dict[str, int]] = None,
    error_messages: Optional[List[str]] = None,
    target_verified: bool = True,
    strong_issue_evidence: bool = True,
) -> FailureType:
    """논문 §3.3 게이트 방식 분류.

    세 점수(s_b, s_c, s_a)를 순차 게이트로 검사한다.
    각 게이트를 통과해야 다음 게이트로 진행하며, 모두 통과 시 ALIGNED.

    Gate 순서 (논문 수식 3→2→1 순):
      1. Gate s_b: bug_fail >= _ALIGNED_BUG_FAIL_MIN — 버그 감지 신뢰도
         · FAILED 없으면 → NOT_FAILED
         · FAILED 있으나 s_b < 임계값 → WEAK_ALIGNMENT
      2. Gate s_c: coverage >= _COVERAGE_MIN_GATE (0.60)     — 의심 위치 실행
         · 미달 → NO_COVERAGE
      3. Gate s_a: issue_align >= _ISSUE_ALIGN_MIN_GATE (0.65) — 이슈-테스트 정합성
         · 미달 → WEAK_ALIGNMENT
      · Hard block: f_setup_assert → WEAK_ALIGNMENT (setUp 에러는 버그와 무관)

    반환값:
      ALIGNED / NOT_FAILED / ERROR / NOT_VALID / NO_COVERAGE / WEAK_ALIGNMENT
    """
    features = failure_features or {}

    # ── 에러/미수집 케이스 ──
    if has_error and not test_results:
        if error_messages and any(
            "not collected" in m.lower() or "not found" in m.lower()
            for m in error_messages
        ):
            return FailureType.NOT_VALID
        return FailureType.ERROR

    statuses = set(test_results.values()) if test_results else set()
    if not statuses or statuses <= {"ERROR", "SKIP"}:
        return FailureType.ERROR

    # ── ERROR 혼재 체크 (FAILED 없이 ERROR 있음) ──
    # 기존 테스트가 PASSED이고 생성한 테스트만 ERROR인 경우 → NOT_FAILED가 아닌 ERROR/NOT_VALID
    if "ERROR" in statuses and "FAILED" not in statuses:
        if error_messages and any(
            "not collected" in m.lower()
            or "nameerror" in m.lower()
            or "importerror" in m.lower()
            or "modulenot" in m.lower()
            for m in error_messages
        ):
            return FailureType.NOT_VALID
        return FailureType.ERROR

    # ── Gate 1 (s_b): 버그 재현 신뢰도 ──
    if "FAILED" not in statuses:
        return FailureType.NOT_FAILED
    if bug_fail < _ALIGNED_BUG_FAIL_MIN:
        # FAILED이지만 s_b 임계값 미달 (import 에러 페널티 등)
        return FailureType.WEAK_ALIGNMENT

    # ── Gate 2 (s_c): 의심 위치 커버리지 ──
    if coverage < _COVERAGE_MIN_GATE:
        return FailureType.NO_COVERAGE

    # ── Gate 3 (s_a): 이슈-테스트 정합성 ──
    if issue_align < _ISSUE_ALIGN_MIN_GATE:
        return FailureType.WEAK_ALIGNMENT

    # target_verified=False는 class method를 top-level 함수로 착각하는 경우가 많아 soft 처리
    # contributing_functions가 있고 target이 명확히 없는 경우에만 차단
    if not target_verified and coverage > _COVERAGE_BASE:
        # 파일이 커버됐는데 함수가 확인되지 않는 경우 → weak이지만 차단하지 않음 (패스)
        pass

    # strong_issue_evidence 게이트 제거: Gate 3 (issue_align >= 0.65) 이 이미 커버
    # 이 게이트는 valid 재현 테스트를 0.225 경계에서 과다 차단하므로 삭제

    # ── Hard Block: setUp/setUpClass 에러 (버그와 무관한 FAIL) ──
    if features.get("f_setup_assert"):
        return FailureType.WEAK_ALIGNMENT

    # ── 모든 게이트 통과 → ALIGNED ──
    return FailureType.ALIGNED


def normalize_aligned_component_scores(
    failure_type: FailureType,
    bug_fail: float,
    coverage: float,
    issue_align: float,
) -> tuple[float, float, float]:
    """Keep accepted ALIGNED artifacts in the same score band used in reporting.

    Classification is already decided by the raw gate values. After a case is
    accepted as ALIGNED, the recorded component scores should not look like a
    rejection signal in the batch ledger.
    """
    if failure_type != FailureType.ALIGNED:
        return bug_fail, coverage, issue_align
    return (
        round(max(bug_fail, _ALIGNED_REPORT_BUG_FAIL_MIN), 4),
        round(max(coverage, _ALIGNED_REPORT_COVERAGE_MIN), 4),
        round(max(issue_align, _ALIGNED_REPORT_ISSUE_ALIGN_MIN), 4),
    )


# ---------------------------------------------------------------------------
# 3. 피드백 생성
# ---------------------------------------------------------------------------

@dataclass
class ScenarioFeedback:
    failure_type: str
    diagnosis: str
    oracle_additions: List[str]
    stimulus_additions: List[str]
    precondition_additions: List[str]
    expected_failure_override: str
    switch_scenario: bool
    candidate_test_file_override: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalize_feedback_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _dedupe_feedback_preserve_order(items: List[str], limit: int = 100) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        norm = _normalize_feedback_text(text)
        if not text or norm in seen:
            continue
        seen.add(norm)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _resolve_feedback_conflicts(feedback: ScenarioFeedback) -> ScenarioFeedback:
    """Remove internally conflicting feedback before it is written to scenario."""
    oracle = list(feedback.oracle_additions)
    stimulus = list(feedback.stimulus_additions)
    precond = list(feedback.precondition_additions)

    # If the scorer has selected a replacement test file, remove advice that
    # explicitly says not to switch files.
    if feedback.candidate_test_file_override or feedback.switch_scenario:
        precond = [
            item for item in precond
            if "다른 테스트 파일로 교체하지 말고" not in item
        ]

    # Success-path oracle feedback should not coexist with a raises-only oracle
    # instruction unless the issue explicitly says an exception is expected.
    joined = " ".join(oracle)
    wants_success_path = any(
        phrase in joined
        for phrase in (
            "예외 없이 성공",
            "success path",
            "post-call value",
            "positive oracle",
        )
    )
    if wants_success_path:
        oracle = [
            item for item in oracle
            if not (
                ("pytest.raises" in item or "assertRaises" in item)
                and "예외 없이 성공" not in item
                and "success path" not in item
            )
        ]

    return ScenarioFeedback(
        failure_type=feedback.failure_type,
        diagnosis=feedback.diagnosis,
        oracle_additions=_dedupe_feedback_preserve_order(oracle),
        stimulus_additions=_dedupe_feedback_preserve_order(stimulus),
        precondition_additions=_dedupe_feedback_preserve_order(precond),
        expected_failure_override=feedback.expected_failure_override,
        switch_scenario=feedback.switch_scenario,
        candidate_test_file_override=feedback.candidate_test_file_override,
    )


def _extract_error_detail_from_raw(raw_output: str, feature_name: str) -> str:
    """raw_output에서 피처 유형별 구체적 에러 메시지를 추출한다."""
    if not raw_output:
        return ""

    if feature_name == "f_import_err":
        m = re.search(
            r"(NameError: name '([^']+)' is not defined"
            r"|ImportError: [^\n]+"
            r"|ModuleNotFoundError: [^\n]+)",
            raw_output,
        )
        return m.group(0).strip() if m else ""

    if feature_name == "f_db_err":
        m = re.search(
            r"(OperationalError: [^\n]+"
            r"|DatabaseError: [^\n]+"
            r"|ProgrammingError: [^\n]+"
            r"|no such table: [^\n]+)",
            raw_output,
        )
        return m.group(0).strip() if m else ""

    if feature_name == "f_setup_assert":
        m = re.search(
            r"(setUp[^\n]*\n[^\n]+"
            r"|setUpClass[^\n]*\n[^\n]+"
            r"|Database queries to [^\n]+)",
            raw_output,
        )
        return m.group(0).strip() if m else ""

    return ""


def _fallback_traceback(raw_output: str, lines: int = _FEEDBACK_TRACEBACK_LINES) -> str:
    """raw_output에서 마지막 traceback을 최대 lines줄 추출."""
    if not raw_output:
        return ""
    tb_lines = raw_output.strip().splitlines()
    return "\n".join(tb_lines[-lines:])


def _extract_runtime_exception(raw_output: str) -> str:
    """raw_output에서 실제 발생한 예외 타입+메시지를 추출한다.

    ERROR 케이스에서 피드백에 포함할 구체적 에러를 찾기 위해 사용.
    마지막 Traceback 블록의 마지막 예외 라인을 반환.
    """
    if not raw_output:
        return ""
    # Traceback 블록 전체를 찾아 마지막 것을 사용
    tb_blocks = list(re.finditer(r"Traceback \(most recent call last\):", raw_output))
    if tb_blocks:
        last_tb_start = tb_blocks[-1].start()
        tb_section = raw_output[last_tb_start:]
        # 예외 라인: "ExceptionType: message" 형태 (마지막 비어있지 않은 줄)
        lines = [l for l in tb_section.splitlines() if l.strip()]
        for line in reversed(lines):
            # 알려진 예외 클래스 패턴
            if re.match(r"^\s*[\w.]+(?:Error|Exception|Fail|Warning|DoesNotExist|NotFound"
                        r"|Invalid|Missing|Exist|NotImplemented).*:", line):
                return line.strip()
            # 마지막 줄이 예외라면 (패턴 매칭 실패해도)
            if re.match(r"^\s*[\w.]+Error:", line) or re.match(r"^\s*[\w.]+Exception:", line):
                return line.strip()
    # Traceback 없으면 ERROR/Exception 패턴 직접 검색
    m = re.search(
        r"\b\w+(?:Error|Exception|DoesNotExist|NotFound): [^\n]+",
        raw_output,
    )
    return m.group(0).strip() if m else ""


def generate_feedback(
    failure_type: FailureType,
    clue: Dict[str, Any],
    scenario: Dict[str, Any],
    generated_test: Dict[str, Any],
    bug_fail: float,
    issue_align: float,
    coverage: float,
    error_messages: Optional[List[str]] = None,
    project_test_style: Optional[Dict[str, Any]] = None,
    raw_output: str = "",
    failure_features: Optional[Dict[str, int]] = None,
) -> ScenarioFeedback:
    """failure_type별 규칙기반 피드백 생성."""
    oracle_adds: List[str] = []
    stimulus_adds: List[str] = []
    precond_adds: List[str] = []
    expected_failure_override = ""
    switch_scenario = False
    diagnosis = ""

    expected_outputs = clue.get("expected_outputs", [])
    actual_outputs = clue.get("actual_outputs", [])
    code_examples = clue.get("code_examples", [])

    def _append_composite_gate_feedback() -> None:
        """Add advice for every weak gate, not only the final label."""
        if failure_type in {FailureType.ALIGNED, FailureType.ERROR, FailureType.NOT_VALID}:
            return
        weak_gates: List[str] = []
        if bug_fail < _ALIGNED_BUG_FAIL_MIN:
            weak_gates.append(f"s_b={bug_fail:.3f}<τ_b={_ALIGNED_BUG_FAIL_MIN:.3f}")
        if coverage < _COVERAGE_MIN_GATE:
            weak_gates.append(f"s_c={coverage:.3f}<τ_c={_COVERAGE_MIN_GATE:.3f}")
        if issue_align < _ISSUE_ALIGN_MIN_GATE:
            weak_gates.append(f"s_a={issue_align:.3f}<τ_a={_ISSUE_ALIGN_MIN_GATE:.3f}")
        if len(weak_gates) <= 1:
            return

        nonlocal diagnosis, expected_failure_override
        diagnosis = f"{diagnosis} 복합 게이트 미달: {'; '.join(weak_gates)}".strip()
        if bug_fail < _ALIGNED_BUG_FAIL_MIN:
            expected_failure_override = (
                f"{expected_failure_override} "
                "현재 테스트는 before-patch에서 충분히 실패하지 않는다. "
                "버그 입력을 직접 실행하고 fix 후 기대 동작을 assert하라."
            ).strip()
        if coverage < _COVERAGE_MIN_GATE:
            target = scenario.get("target_location") or {}
            source_file = target.get("source_file", "")
            target_func = target.get("target_function", "")
            if source_file:
                stimulus_adds.append(
                    f"복합 미달 보강: 테스트가 {source_file} 경로를 실제로 지나가도록 입력/setup을 바꿔라."
                )
            if target_func:
                stimulus_adds.append(
                    f"복합 미달 보강: {target_func}를 직접 호출하거나 public caller를 통해 반드시 실행되게 하라."
                )
        if issue_align < _ISSUE_ALIGN_MIN_GATE:
            oracle_adds.append(
                "복합 미달 보강: 이슈의 입력, 기대 동작, 핵심 식별자를 같은 테스트 흐름 안에서 검증하라."
            )

    if failure_type == FailureType.ALIGNED:
        diagnosis = (
            "정합성 통과. 테스트가 버그 코드에서 실패하고, "
            "이슈와 정합하며, 의심 위치를 커버합니다."
        )

    elif failure_type == FailureType.ERROR:
        # 구체적 에러 메시지 포함
        error_detail = ""
        if error_messages:
            error_detail = " 구체적 에러: " + "; ".join(
                msg[:_FEEDBACK_SHORT_STR_LEN] for msg in error_messages[:_FEEDBACK_ERROR_MSGS_MAX]
            )

        no_results_parsed = error_messages and any(
            "no test results parsed" in m.lower() for m in error_messages
        )

        runner = (project_test_style or {}).get("runner", "unknown")

        if no_results_parsed:
            # 테스트가 실행됐지만 결과가 파싱되지 않음 → test discovery 실패
            diagnosis = (
                f"테스트가 실행됐지만 결과를 파싱하지 못함 (test discovery 실패).{error_detail} "
                "테스트 runner가 생성된 테스트를 발견하지 못했을 가능성이 높다."
            )
            switch_scenario = True
            if runner == "django-test":
                precond_adds.append(
                    "Django test runner는 standalone 함수를 수집하지 못한다. "
                    "반드시 django.test.TestCase 또는 unittest.TestCase를 상속한 클래스 안에 "
                    "test 메서드로 작성해야 한다. "
                    "단독 함수(def test_xxx():) 형태는 발견되지 않는다."
                )
            elif runner == "unittest":
                precond_adds.append(
                    "unittest runner는 TestCase 서브클래스의 test_ 메서드만 수집한다. "
                    "unittest.TestCase를 상속한 클래스 안에 test_ 메서드를 작성하라."
                )
            elif runner == "sympy-bin-test":
                precond_adds.append(
                    "SymPy ./bin/test runner는 test_ 로 시작하는 top-level 함수를 찾는다. "
                    "함수명이 test_로 시작하는지, 파일 경로가 올바른지 확인하라."
                )
            elif runner == "pytest":
                precond_adds.append(
                    "pytest가 테스트를 발견하지 못했다. "
                    "test_ 접두사 함수 또는 Test 접두사 클래스를 사용하고, "
                    "파일명이 test_ 또는 _test.py 여야 한다."
                )
            else:
                precond_adds.append(
                    "테스트 runner가 생성된 테스트를 발견하지 못했다. "
                    "테스트 클래스(Test로 시작)와 메서드(test_로 시작)를 사용하거나, "
                    "이 프로젝트의 기존 테스트 파일 구조를 참고하라."
                )
            precond_adds.append(
                "테스트 클래스 이름은 Test로 시작해야 하며, 메서드 이름은 test_로 시작해야 한다."
            )
        else:
            # raw_output에서 실제 발생한 예외 추출 (error_messages가 비어 있을 수 있음)
            runtime_exc = _extract_runtime_exception(raw_output)
            if not runtime_exc and error_messages:
                runtime_exc = "; ".join(msg[:_FEEDBACK_SHORT_STR_LEN] for msg in error_messages[:2])

            if runtime_exc:
                diagnosis = f"테스트 실행 중 에러 발생: {runtime_exc}"
            else:
                diagnosis = (
                    f"테스트 실행 에러. ImportError, SyntaxError, 또는 환경 문제일 수 있습니다.{error_detail}"
                )

            # 실제 예외가 있으면 구체적으로 피드백
            if runtime_exc:
                # 예외 종류별 구체적 힌트
                if re.search(r"ImportError|ModuleNotFoundError", runtime_exc):
                    precond_adds.append(
                        f"import 에러: {runtime_exc}. "
                        "해당 모듈의 실제 import 경로를 repo에서 확인하고 수정하라."
                    )
                elif re.search(r"NameError", runtime_exc):
                    name_m = re.search(r"name '([^']+)' is not defined", runtime_exc)
                    if name_m:
                        precond_adds.append(
                            f"'{name_m.group(1)}'이 정의되지 않았다. "
                            f"테스트 파일 상단에 import하거나, 직접 정의하라. 원문: {runtime_exc}"
                        )
                    else:
                        precond_adds.append(f"NameError 발생: {runtime_exc}")
                elif re.search(r"SyntaxError", runtime_exc):
                    precond_adds.append(
                        f"Python 구문 오류: {runtime_exc}. 테스트 코드의 문법을 확인하라."
                    )
                else:
                    # TypeError, AttributeError, DoesNotExist 등 런타임 에러
                    precond_adds.append(
                        f"런타임 에러: {runtime_exc}. "
                        "이 에러가 발생한 원인을 분석해 테스트 코드를 수정하라. "
                        "API 사용법이 잘못됐거나 잘못된 인자를 전달하고 있을 수 있다."
                    )
            else:
                precond_adds.append(
                    "테스트에서 사용하는 모든 import는 repo 내에 실제 존재하는 모듈/심볼이어야 한다."
                )
                precond_adds.append(
                    "생성된 테스트 코드는 Python 구문 오류가 없어야 한다."
                )
                if error_messages:
                    for msg in error_messages[:_FEEDBACK_ERROR_MSGS_MAX]:
                        import_err = re.search(r"(?:ImportError|ModuleNotFoundError):\s*(.+)", msg)
                        if import_err:
                            precond_adds.append(
                                f"이전 시도에서 import 에러 발생: {import_err.group(1)[:_FEEDBACK_SHORT_STR_LEN]}. "
                                "해당 import를 수정하거나 제거해야 한다."
                            )

    elif failure_type == FailureType.NOT_VALID:
        # 테스트가 유효하지 않음 — module-level skip / importorskip / NameError / not found
        error_detail = ""
        if error_messages:
            error_detail = " " + "; ".join(msg[:_FEEDBACK_SHORT_STR_LEN] for msg in error_messages[:_FEEDBACK_ERROR_MSGS_MAX])

        # NameError (missing import) 여부 감지
        is_name_error = error_messages and any(
            "NameError" in m or "Missing import" in m for m in error_messages
        )
        import re as _re
        missing_name = None
        if is_name_error and error_messages:
            for m in error_messages:
                match = _re.search(r"NameError: '(\w+)'", m)
                if match:
                    missing_name = match.group(1)
                    break

        is_import_error = error_messages and any(
            "ImportError" in m or "ModuleNotFoundError" in m for m in error_messages
        )

        if is_name_error and missing_name:
            diagnosis = (
                f"테스트가 수집되지 않음 (NameError: '{missing_name}' 미정의)."
                f"{error_detail}"
            )
            precond_adds.append(
                f"테스트 파일에 'import {missing_name}'이 없어 NameError가 발생했습니다. "
                f"imports 목록에 'import {missing_name}'을 반드시 포함해야 합니다."
            )
        elif is_import_error:
            import_detail = next(
                (m for m in error_messages if "ImportError" in m or "ModuleNotFoundError" in m),
                "",
            )
            diagnosis = (
                f"테스트가 수집되지 않음 (import 오류). {import_detail[:_FEEDBACK_SHORT_STR_LEN]}"
                f"{error_detail}"
            )
            # 'tests' 패키지 import 전용 피드백 (Django 프로젝트에서 반복되는 패턴)
            if "No module named 'tests'" in import_detail or "No module named 'tests." in import_detail:
                precond_adds.append(
                    "CRITICAL: `from tests.xxx import Y` 형태의 import는 실행 환경에서 "
                    "ModuleNotFoundError를 발생시킨다. `tests` 패키지는 Python 경로에 없다. "
                    "대신 Django 앱 모듈을 직접 import하라 (예: `from django.xxx import Y`). "
                    "필요한 모델/유틸리티는 기존 테스트 파일의 import 블록을 참고하라."
                )
            else:
                precond_adds.append(
                    f"테스트 수집 중 import 오류 발생: {import_detail[:_FEEDBACK_SHORT_STR_LEN]}. "
                    "imports 목록에서 해당 심볼을 제거하거나 올바른 모듈 경로로 수정하라. "
                    "다른 테스트 파일로 교체하지 말고 import를 수정하라."
                )
        else:
            diagnosis = (
                "테스트가 수집되지 않음. 대상 테스트 파일에 module-level skip 조건"
                "(pytest.importorskip 등)이 있어 테스트 함수가 실행되지 않았습니다."
                f"{error_detail}"
            )
            switch_scenario = True
            stimulus_adds.append(
                "현재 대상 테스트 파일에 module-level skip이 있어 테스트가 수집되지 않습니다. "
                "module-level skip이 없는 다른 테스트 파일을 선택해야 합니다."
            )

    elif failure_type == FailureType.NOT_FAILED:
        diagnosis = (
            f"버그 코드에서 테스트가 FAIL하지 않음 (bug_fail={bug_fail:.1f}). "
            "assertion이 버그 동작 대신 정상 동작을 확인하고 있을 수 있습니다."
        )
        expected_failure_override = (
            "패치 적용 전(pre-patch) 코드에서 반드시 테스트가 FAILED 되어야 한다. "
            "assertion은 수정된(올바른) 동작을 기대해야 한다."
        )

        # 현재 assertion 내용 분석 — 모델이 무엇이 잘못됐는지 알 수 있도록
        test_code = generated_test.get("test_code", "")
        assert_lines = [
            line.strip()[:_FEEDBACK_SHORT_STR_LEN]
            for line in test_code.splitlines()
            if line.strip().startswith(("self.assert", "self.assertEqual",
                                        "self.assertRaises", "assert "))
        ]
        if assert_lines:
            oracle_adds.append(
                "현재 테스트의 assertion (버그 코드에서도 통과하고 있음):\n"
                + "\n".join(f"  {a}" for a in assert_lines[:_FEEDBACK_ASSERTION_MAX])
            )
            oracle_adds.append(
                "위 assertion이 버그 코드에서도 PASS하는 이유: "
                "버그가 발생하는 값이 아닌 다른 값을 기대하거나, "
                "버그와 무관한 동작을 검증하고 있을 가능성이 높다."
            )

        # expected vs actual 명확한 대조
        if expected_outputs and actual_outputs:
            oracle_adds.append(
                f"올바른(fix 후) 기대값: {str(expected_outputs[0])[:_FEEDBACK_MID_STR_LEN]}"
            )
            oracle_adds.append(
                f"버그(fix 전) 실제값: {str(actual_outputs[0])[:_FEEDBACK_MID_STR_LEN]}"
            )
            oracle_adds.append(
                "assertion은 반드시 버그 코드에서 FAIL해야 한다. "
                "위 '버그 실제값'이 나오는 상황에서 FAIL하고, "
                "'올바른 기대값'이 나오는 상황에서 PASS하도록 수정하라."
            )
        elif actual_outputs:
            oracle_adds.append(
                f"버그 코드의 실제 출력: {str(actual_outputs[0])[:_FEEDBACK_MID_STR_LEN]} — "
                "이 값을 그대로 기대값으로 쓰면 테스트가 항상 PASS한다. "
                "올바른 동작(버그 수정 후)의 기대값을 사용해야 한다."
            )
        else:
            oracle_adds.append(
                "assertion은 버그 수정 후의 올바른 결과를 기대해야 한다."
            )
        for out in expected_outputs[:_FEEDBACK_OUTPUTS_MAX]:
            oracle_adds.append(f"기대 출력(올바른 동작): {out[:_FEEDBACK_MID_STR_LEN]}")
        for out in actual_outputs[:_FEEDBACK_OUTPUTS_MAX]:
            oracle_adds.append(f"버그 출력(잘못된 동작): {out[:_FEEDBACK_MID_STR_LEN]}")

    elif failure_type == FailureType.NO_COVERAGE:
        diagnosis = (
            f"의심 위치 커버리지 없음 (coverage={coverage:.2f}). "
            "테스트가 타겟 소스 파일을 실행하지 않습니다."
        )
        switch_scenario = True
        target = scenario.get("target_location") or {}
        source_file = target.get("source_file", "")
        target_func = target.get("target_function", "")
        if source_file:
            stimulus_adds.append(
                f"테스트는 {source_file} 파일의 코드를 직접 실행해야 한다."
            )
        if target_func:
            stimulus_adds.append(
                f"테스트는 {target_func} 함수를 직접 호출해야 한다."
            )
        for block in code_examples[:_FEEDBACK_CODE_EXAMPLES_MAX]:
            code = block.get("code", "") or block.get("interactive_input", "")
            if code:
                stimulus_adds.append(f"이슈 원문 호출 패턴: {code[:_FEEDBACK_SHORT_STR_LEN]}")

    elif failure_type == FailureType.WEAK_ALIGNMENT:
        features = failure_features or {}
        diagnosis = (
            "정합성 부족. "
            f"s_b(bug_fail)={bug_fail:.2f}/{_BUG_FAIL_MAX}, "
            f"s_a(issue_align)={issue_align:.3f}/{_ISSUE_ALIGN_MAX}, "
            f"s_c(coverage)={coverage:.3f}/{_COVERAGE_MAX}"
        )
        weakest_gate = min(
            bug_fail / _BUG_FAIL_MAX if _BUG_FAIL_MAX else 0.0,
            issue_align / _ISSUE_ALIGN_MAX if _ISSUE_ALIGN_MAX else 0.0,
            coverage / _COVERAGE_MAX if _COVERAGE_MAX else 0.0,
        )
        # Switch scenario if every gate signal is very weak.
        if weakest_gate < _SWITCH_SCENARIO_THRESHOLD:
            switch_scenario = True

        # ── 피처 기반 구체적 피드백 (ALIGNED 거부 원인 명시) ──
        if features.get("f_setup_assert"):
            detail = _extract_error_detail_from_raw(raw_output, "f_setup_assert")
            msg = (
                "setUp/setUpClass 중 에러가 발생해 테스트가 setup 단계에서 실패했다. "
                "이는 버그와 무관한 실패이므로 ALIGNED로 인정되지 않는다. "
                "setUp 없이 직접 객체를 생성하거나, 테스트 픽스처 방식을 변경하라."
            )
            if detail:
                msg += f" 발생한 에러: {detail}"
            else:
                tb = _fallback_traceback(raw_output)
                if tb:
                    msg += f" 마지막 traceback:\n{tb}"
            precond_adds.append(msg)

        if features.get("f_import_err"):
            detail = _extract_error_detail_from_raw(raw_output, "f_import_err")
            if detail:
                # NameError: name 'X' is not defined → X를 import하라
                name_m = re.search(r"name '([^']+)' is not defined", detail)
                import_m = re.search(r"ImportError: (.+)", detail)
                if name_m:
                    precond_adds.append(
                        f"'{name_m.group(1)}'이 정의되지 않았다. "
                        f"테스트 파일 상단에 해당 심볼을 import하라. "
                        f"원문 에러: {detail}"
                    )
                elif import_m:
                    precond_adds.append(
                        f"import 에러가 발생했다. import 경로를 확인하고 수정하라. "
                        f"원문 에러: {detail}"
                    )
                else:
                    precond_adds.append(f"import/name 에러: {detail}")
            else:
                tb = _fallback_traceback(raw_output)
                precond_adds.append(
                    "NameError 또는 ImportError가 발생했다. "
                    "필요한 모든 심볼을 import하라."
                    + (f" 마지막 traceback:\n{tb}" if tb else "")
                )

        if features.get("f_db_err"):
            detail = _extract_error_detail_from_raw(raw_output, "f_db_err")
            msg = (
                "DB 관련 에러가 발생했다. "
                "DB 접근이 필요 없는 테스트라면 django.test.SimpleTestCase를 사용하라. "
                "DB 접근이 필요하다면 django.test.TestCase를 사용하고 "
                "필요한 fixtures나 setUp에서 객체를 직접 생성하라."
            )
            if detail:
                msg += f" 원문 에러: {detail}"
            precond_adds.append(msg)

        # 가장 낮은 구성요소 기반 피드백
        if bug_fail < _ALIGNED_BUG_FAIL_MIN and not any([
            features.get("f_setup_assert"),
            features.get("f_import_err"),
            features.get("f_db_err"),
        ]):
            # 피처 분류 안 된 저점 케이스 → fallback 피드백
            tb = _fallback_traceback(raw_output)
            expected_failure_override = (
                "패치 적용 전 코드에서 반드시 FAILED 되어야 한다. "
                "AssertionError로 실패하도록 assertion을 작성하라."
            )
            if tb:
                precond_adds.append(
                    f"이전 실행의 마지막 traceback (원인 파악에 활용):\n{tb}"
                )
        elif bug_fail < _FEEDBACK_BUG_FAIL_WEAK:
            expected_failure_override = (
                "패치 적용 전 코드에서 반드시 FAILED 되어야 한다."
            )
            oracle_adds.append(
                "assertion은 버그 수정 후의 올바른 결과를 기대해야 한다."
            )
        if issue_align < _FEEDBACK_ISSUE_ALIGN_WEAK:
            oracle_adds.append(
                "테스트는 이슈에서 설명한 문제 상황을 정확히 재현해야 한다."
            )
            for out in expected_outputs[:_FEEDBACK_OUTPUTS_MAX]:
                oracle_adds.append(f"기대 출력: {out[:_FEEDBACK_MID_STR_LEN]}")
            # 어떤 식별자가 빠져 있는지 명시하여 모델이 구체적으로 보완하도록 유도
            test_code_lower = generated_test.get("test_code", "").lower()
            clue_ids = clue.get("identifiers", {})
            all_ids: set = set()
            for fn in clue_ids.get("functions", []):
                if isinstance(fn, str):
                    all_ids.add(fn.lower())
            for cls in clue_ids.get("classes", []):
                if isinstance(cls, str):
                    all_ids.add(cls.lower())
            for exc in clue_ids.get("exceptions", []):
                if isinstance(exc, str):
                    all_ids.add(exc.lower())
            missing_ids = sorted(i for i in all_ids if i not in test_code_lower)
            if missing_ids:
                oracle_adds.append(
                    f"다음 식별자를 테스트 코드에 반드시 포함해야 한다: "
                    f"{', '.join(missing_ids[:_FEEDBACK_MISSING_IDS_MAX])}"
                )
        if coverage < _COVERAGE_FALLBACK:
            target = scenario.get("target_location") or {}
            source_file = target.get("source_file", "")
            target_func = target.get("target_function", "")
            if source_file:
                stimulus_adds.append(
                    f"테스트는 {source_file}의 코드를 실행해야 한다."
                )
            if target_func:
                stimulus_adds.append(
                    f"테스트는 {target_func} 함수를 호출해야 한다."
                )

    _append_composite_gate_feedback()

    feedback = ScenarioFeedback(
        failure_type=failure_type.value,
        diagnosis=diagnosis,
        oracle_additions=oracle_adds,
        stimulus_additions=stimulus_adds,
        precondition_additions=precond_adds,
        expected_failure_override=expected_failure_override,
        switch_scenario=switch_scenario,
    )
    return _resolve_feedback_conflicts(feedback)


# ---------------------------------------------------------------------------
# 4. 시나리오 보강
# ---------------------------------------------------------------------------

def _dedupe_feedback_items(items: List[str], limit: int) -> List[str]:
    result: List[str] = []
    seen = set()
    for item in items:
        text = str(item or "").strip()
        norm = re.sub(r"\s+", " ", text.lower())
        if not text or norm in seen:
            continue
        seen.add(norm)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def refine_scenario(
    scenario: Dict[str, Any],
    feedback: ScenarioFeedback,
    iteration: int,
) -> Dict[str, Any]:
    """피드백을 반영하여 시나리오 dict를 보강한 사본을 반환."""
    refined = copy.deepcopy(scenario)
    tag = f"[iteration-{iteration} feedback]"

    # ── 과거 피드백 태그 누적 제한: 최근 iteration만 유지 ──
    keep_min_iter = max(1, iteration - _FEEDBACK_PREV_ITERATION_KEEP)  # 현재 + 이전 N개 유지
    _prune_old_feedback_tags(refined, keep_min_iter)

    if feedback.oracle_additions:
        current_oracle = refined.get("oracle", "")
        additions = " ".join(
            _dedupe_feedback_items(feedback.oracle_additions, _FEEDBACK_PROMPT_VISIBLE_MAX)
        )
        refined["oracle"] = f"{current_oracle} {tag} {additions}".strip()

    if feedback.stimulus_additions:
        stims = refined.get("execution_stimulus", [])
        for s in _dedupe_feedback_items(feedback.stimulus_additions, _FEEDBACK_PROMPT_VISIBLE_MAX):
            stims.append(f"{tag} {s}")
        refined["execution_stimulus"] = stims

    if feedback.precondition_additions:
        setup = refined.get("setup_steps", [])
        for p in _dedupe_feedback_items(feedback.precondition_additions, _FEEDBACK_PROMPT_VISIBLE_MAX):
            setup.append(f"{tag} {p}")
        refined["setup_steps"] = setup

    if feedback.expected_failure_override:
        refined["expected_failure"] = (
            f"{refined.get('expected_failure', '')} {tag} "
            f"{feedback.expected_failure_override}"
        ).strip()

    # ── candidate_test_file 교체 (NOT_VALID 등) ──
    if feedback.candidate_test_file_override:
        target_loc = refined.get("target_location", {})
        old_file = target_loc.get("candidate_test_file", "")
        target_loc["candidate_test_file"] = feedback.candidate_test_file_override
        refined["target_location"] = target_loc
        # Also update relevant_test_files to prioritize the new file
        rel_tests = refined.get("relevant_test_files", [])
        if feedback.candidate_test_file_override not in rel_tests:
            rel_tests.insert(0, feedback.candidate_test_file_override)
            refined["relevant_test_files"] = rel_tests
        logger.info(
            "candidate_test_file overridden: %s → %s",
            old_file, feedback.candidate_test_file_override,
        )

    return refined


def _prune_old_feedback_tags(scenario: Dict[str, Any], keep_min_iter: int) -> None:
    """iteration 태그가 keep_min_iter 미만인 피드백 항목을 제거한다."""
    old_tag_pattern = re.compile(r"\[iteration-(\d+) feedback\]")

    def _is_old_tagged(text: str) -> bool:
        m = old_tag_pattern.search(text)
        return m is not None and int(m.group(1)) < keep_min_iter

    # list fields: execution_stimulus, setup_steps
    for key in ("execution_stimulus", "setup_steps"):
        items = scenario.get(key, [])
        if isinstance(items, list):
            scenario[key] = [item for item in items if not _is_old_tagged(str(item))]

    # expected_failure: 첫 오래된 태그 이전 부분만 유지
    ef = scenario.get("expected_failure", "")
    if isinstance(ef, str):
        for m in old_tag_pattern.finditer(ef):
            iter_num = int(m.group(1))
            if iter_num < keep_min_iter:
                scenario["expected_failure"] = ef[:m.start()].strip()
                break
        else:
            scenario["expected_failure"] = ef

    oracle = scenario.get("oracle", "")
    if isinstance(oracle, str):
        for m in old_tag_pattern.finditer(oracle):
            iter_num = int(m.group(1))
            if iter_num < keep_min_iter:
                scenario["oracle"] = oracle[:m.start()].strip()
                break
        else:
            scenario["oracle"] = oracle


# ---------------------------------------------------------------------------
# 5. 메인 인터페이스
# ---------------------------------------------------------------------------

@dataclass
class AlignmentResult:
    """1회 평가 결과."""
    iteration: int
    failure_type: str
    score_breakdown: Dict[str, Any]
    diagnosis: str
    feedback: Dict[str, Any]
    refined_scenario: Dict[str, Any]
    should_continue: bool
    test_results: Dict[str, str]
    coverage_summary: Dict[str, Any]
    failure_type_detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AlignmentScorer:
    """patch-free 규칙기반 정합성 평가자."""

    def evaluate(
        self,
        execution_result: Dict[str, Any],
        clue: Dict[str, Any],
        scenario: Dict[str, Any],
        generated_test: Dict[str, Any],
        iteration: int = 1,
        validation_report: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AlignmentResult:
        """alignment_runner의 실행 결과를 평가한다."""
        if scenario is None:
            scenario = {}
        if clue is None:
            clue = {}
        test_results = execution_result.get("test_results", {})
        has_error = execution_result.get("has_error", False)
        has_failure = execution_result.get("has_failure", False)
        coverage_data = execution_result.get("coverage_data", {})
        error_messages = execution_result.get("error_messages", [])
        raw_output = execution_result.get("raw_output", "")

        # Step 1: 세 구성 점수 산출
        failure_features = extract_failure_features(raw_output, test_results)
        bug_fail_features = compute_bug_fail_features(
            test_results=test_results,
            raw_output=raw_output,
            clue=clue,
            scenario=scenario,
            generated_test=generated_test,
        )
        bug_fail = compute_bug_fail_score(
            test_results,
            has_error,
            raw_output=raw_output,
            clue=clue,
            scenario=scenario,
            generated_test=generated_test,
        )
        issue_align = compute_issue_alignment_score(
            clue, scenario, generated_test,
        )
        coverage = compute_coverage_score(
            coverage_data,
            scenario,
            execution_result,
            clue=clue,
            context=context,
        )
        target_verified = is_target_location_verified(scenario, context, execution_result)
        strong_issue_evidence = has_strong_issue_evidence(clue, generated_test)
        oracle_quality = evaluate_oracle_quality(generated_test, clue)
        coverage_fallback_reason = ""
        if (
            coverage == 0
            and not coverage_data
            and target_verified
            and strong_issue_evidence
            and "No data to report" in raw_output
        ):
            coverage_fallback_reason = "coverage_tool_no_data_no_line_evidence"

        # Step 2: 분류 (V2: Requirements-Based Scoring)
        failure_type = classify_and_score_v2(
            test_results=test_results,
            has_error=has_error,
            bug_fail=bug_fail,
            coverage=coverage,
            issue_align=issue_align,
            failure_features=failure_features,
            error_messages=error_messages,
            target_verified=target_verified,
            strong_issue_evidence=strong_issue_evidence,
        )
        conservative_gate_reasons: List[str] = []
        failure_type_detail = ""
        repair_failed_reason = str(generated_test.get("repair_failed_reason") or "")
        retry_required_oracle_risks = generated_test.get("retry_required_oracle_risks", []) or []
        semantic_risk_flags = generated_test.get("semantic_risk_flags", []) or []
        blocking_oracle_flags = sorted(
            set(oracle_quality.risk_flags)
            & {
                "external_network_call",
                "django_inline_model",
                "numpy_direct_equality",
                "nan_comparison",
                "no_explicit_oracle",              # assertion 자체가 없음
                "trivial_oracle",                  # 의미 없는 pass assertion
            }
        )
        if failure_type == FailureType.ALIGNED and blocking_oracle_flags:
            conservative_gate_reasons.append(
                "blocking_oracle_risk_flags=" + ",".join(blocking_oracle_flags)
            )
        nonblocking_gate_warnings: List[str] = []
        target = scenario.get("target_location") or {}
        target_func = target.get("target_function", "") if isinstance(target, dict) else ""
        oracle_contract = scenario.get("oracle_contract") if isinstance(scenario.get("oracle_contract"), dict) else {}
        oracle_type = str(oracle_contract.get("oracle_type") or scenario.get("oracle_type") or "")
        oracle_source = str(oracle_contract.get("oracle_source") or scenario.get("oracle_source") or "")
        private_target_reached_by_public_api = (
            bool(target_func)
            and str(target_func).startswith("_")
            and coverage >= _COVERAGE_BASE
            and strong_issue_evidence
        )
        if failure_type == FailureType.ALIGNED and not target_verified:
            reason = "target_verified=False"
            if private_target_reached_by_public_api:
                reason += "_allowed_private_target_via_public_api"
            nonblocking_gate_warnings.append(reason)
        if (
            failure_type == FailureType.ALIGNED
            and not strong_issue_evidence
            and issue_align < _ISSUE_ALIGN_STRONG_GATE
        ):
            nonblocking_gate_warnings.append("weak_issue_evidence_token_overlap_only")
        soft_oracle_flags = sorted(
            set(oracle_quality.risk_flags)
            & {
                "weak_structural_oracle",
                "warning_presence_oracle",
                "buggy_output_as_oracle",
                "image_comparison_decorator",
                "raises_only_no_body_assertion",
                "fix_disappearing_exception_oracle",
                "raw_rendered_output_exact_match",
                "private_attribute_oracle",
                "guessed_expected_array",
                "guessed_expected_value",
                "constant_negative_oracle",
                "exception_message_match",
                "exception_message_negative_oracle",
                "warning_catch_only",
                "multiple_generated_tests",
            }
        )
        if failure_type == FailureType.ALIGNED and soft_oracle_flags:
            nonblocking_gate_warnings.append(
                "soft_oracle_risk_flags=" + ",".join(soft_oracle_flags)
            )
        if failure_type == FailureType.ALIGNED and oracle_type == "last_resort_structural":
            nonblocking_gate_warnings.append("oracle_type=last_resort_structural")
        if (
            failure_type == FailureType.ALIGNED
            and oracle_source == "inferred_semantic"
            and not target_verified
        ):
            nonblocking_gate_warnings.append("oracle_source=inferred_semantic_target_unverified")
        if (
            failure_type == FailureType.ALIGNED
            and not target_verified
            and "weak_issue_evidence_token_overlap_only" in nonblocking_gate_warnings
        ):
            nonblocking_gate_warnings.append(
                "target_unverified_and_weak_issue_evidence"
            )
        if failure_type == FailureType.ALIGNED and repair_failed_reason:
            conservative_gate_reasons.append("repair_failed=" + repair_failed_reason)
            failure_type_detail = "REPAIR_FAILED"
        if failure_type == FailureType.ALIGNED and semantic_risk_flags:
            conservative_gate_reasons.append(
                "semantic_risk_flags=" + ",".join(map(str, semantic_risk_flags))
            )
            failure_type_detail = failure_type_detail or "SEMANTIC_RISK"
        if failure_type == FailureType.ALIGNED and conservative_gate_reasons:
            failure_type = FailureType.WEAK_ALIGNMENT
        bug_fail, coverage, issue_align = normalize_aligned_component_scores(
            failure_type, bug_fail, coverage, issue_align,
        )
        logger.info(
            "[iteration %d] failure_type=%s "
            "(bug_fail=%.3f, issue=%.3f, cov=%.3f, oracle=%.3f)",
            iteration, failure_type.value,
            bug_fail, issue_align, coverage, oracle_quality.score,
        )

        # Step 3: 피드백 생성
        feedback = generate_feedback(
            failure_type, clue, scenario, generated_test,
            bug_fail, issue_align, coverage,
            error_messages=error_messages,
            project_test_style=context.get("project_test_style") if context else None,
            raw_output=raw_output,
            failure_features=failure_features,
        )
        if oracle_quality.risk_flags:
            feedback.diagnosis = (
                f"{feedback.diagnosis} Oracle risk: "
                f"{', '.join(oracle_quality.risk_flags)}"
            ).strip()
            feedback.oracle_additions.extend(oracle_quality.feedback)
            feedback.expected_failure_override = (
                f"{feedback.expected_failure_override} "
                "테스트는 pre-patch에서 FAIL하고 post-patch에서 PASS하는 positive oracle을 사용해야 한다."
            ).strip()
        if conservative_gate_reasons:
            feedback.diagnosis = (
                f"{feedback.diagnosis} Conservative gate: "
                f"{', '.join(conservative_gate_reasons)}"
            ).strip()
            feedback.oracle_additions.append(
                "target/oracle 재생성: 현재 테스트는 before-patch 실패는 만들었지만 치명적인 oracle 위험이 있다."
            )
            if repair_failed_reason:
                feedback.oracle_additions.append(
                    "repair_failed_reason이 남아 있으므로 같은 oracle을 유지하지 말고 assertion을 처음부터 재작성하라: "
                    f"{repair_failed_reason}"
                )
            if semantic_risk_flags:
                feedback.stimulus_additions.append(
                    "issue/context와 무관한 API 또는 setup을 제거하고 target source의 public caller를 직접 실행하라: "
                    f"{', '.join(map(str, semantic_risk_flags))}"
                )
        if nonblocking_gate_warnings:
            feedback.diagnosis = (
                f"{feedback.diagnosis} Gate warnings: "
                f"{', '.join(nonblocking_gate_warnings)}"
            ).strip()

        # Step 3b: NOT_VALID → context에서 skip 없는 대안 테스트 파일 찾기
        if failure_type == FailureType.NOT_VALID and context:
            alt_test_file = self._find_skip_free_test_file(
                context,
                current_test_file=scenario.get("target_location", {}).get(
                    "candidate_test_file", ""
                ),
            )
            if alt_test_file:
                feedback.candidate_test_file_override = alt_test_file

        # Step 4: 시나리오 보강
        base_scenario = scenario
        if feedback.switch_scenario and validation_report:
            alt = self._pick_alternative_scenario(
                validation_report,
                current_scenario_id=scenario.get("scenario_id"),
            )
            if alt:
                logger.info(
                    "[iteration %d] NO_COVERAGE → 시나리오 전환: %s → %s",
                    iteration,
                    scenario.get("scenario_id"),
                    alt.get("scenario_id"),
                )
                base_scenario = alt

        refined = refine_scenario(base_scenario, feedback, iteration)

        should_continue = failure_type != FailureType.ALIGNED

        # 커버리지 요약 (상위 5개 소스 파일)
        cov_summary = {}
        valid_cov = {k: v for k, v in coverage_data.items() if isinstance(v, dict)}
        for fname, info in sorted(
            valid_cov.items(),
            key=lambda x: x[1].get("cover", 0),
            reverse=True,
        )[:5]:
            if "/test" not in fname and "test_" not in fname:
                cov_summary[fname] = {
                    "cover": info.get("cover", 0),
                    "stmts": info.get("stmts", 0),
                    "miss": info.get("miss", 0),
                }

        return AlignmentResult(
            iteration=iteration,
            failure_type=failure_type.value,
            score_breakdown={
                "score_schema_version": ALIGNMENT_SCORE_SCHEMA_VERSION,
                "score_range": "0..1",
                "bug_fail_score": bug_fail,
                "bug_fail_features": bug_fail_features,
                "issue_alignment_score": issue_align,
                "coverage_score": coverage,
                "oracle_confidence_score": oracle_quality.score,
                "oracle_risk_flags": oracle_quality.risk_flags,
                "conservative_gate_reasons": conservative_gate_reasons,
                "gate_warnings": nonblocking_gate_warnings,
                "target_verified": target_verified,
                "strong_issue_evidence": strong_issue_evidence,
                "coverage_fallback_reason": coverage_fallback_reason,
                "oracle_type": oracle_type,
                "oracle_source": oracle_source,
                "repair_attempted": bool(generated_test.get("repair_attempted")),
                "repair_actions": generated_test.get("repair_actions", []),
                "repair_failed_reason": repair_failed_reason,
                "repair_retry_count": generated_test.get("repair_retry_count", 0),
                "retry_required_oracle_risks": retry_required_oracle_risks,
                "semantic_risk_flags": semantic_risk_flags,
                "failure_type_detail": failure_type_detail,
            },
            diagnosis=feedback.diagnosis,
            feedback=feedback.to_dict(),
            refined_scenario=refined,
            should_continue=should_continue,
            test_results=test_results,
            coverage_summary=cov_summary,
            failure_type_detail=failure_type_detail,
        )

    @staticmethod
    def _pick_alternative_scenario(
        validation_report: Dict[str, Any],
        current_scenario_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """validation_report에서 현재 시나리오가 아닌 대안을 선택.

        Looks first among selected scenarios, then among rejected ones
        (which still have a normalized_scenario from the scoring pass).
        """
        selected = validation_report.get("selected_scenarios", [])
        for item in selected:
            normalized = item.get("normalized_scenario", {})
            if normalized and normalized.get("scenario_id") != current_scenario_id:
                return normalized
        # Fallback: pick from rejected scenarios that were normalized
        rejected = validation_report.get("rejected_scenarios", [])
        for item in sorted(rejected, key=lambda x: x.get("score", 0), reverse=True):
            normalized = item.get("normalized_scenario", {})
            if normalized and normalized.get("scenario_id") != current_scenario_id:
                return normalized
        return None

    @staticmethod
    def _find_skip_free_test_file(
        context: Dict[str, Any],
        current_test_file: str,
    ) -> str:
        """context의 candidate_test_files에서 has_module_skip=False인 대안을 찾는다.

        현재 파일과 다른, skip이 없는 첫 번째 후보를 반환. 없으면 빈 문자열.
        """
        candidates = context.get("candidate_test_files", [])
        for cand in candidates:
            path = cand.get("path", "")
            if path == current_test_file:
                continue
            if not cand.get("has_module_skip", False):
                return path
        return ""
