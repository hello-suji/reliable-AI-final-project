# LLM 기반 이슈 재현 테스트의 신뢰성 평가

신뢰할 수 있는 인공지능 팀 프로젝트  
서울시립대학교 일반대학원 인공지능학과  
홍수지(G202548008), 강태우(G202648001)

## 프로젝트 개요

이 프로젝트는 GitHub 이슈를 바탕으로 LLM이 생성한 버그 재현 테스트를 신뢰할 수 있는지 평가한다.

기존 재현 테스트 평가는 주로 패치 적용 전에는 실패하고 패치 적용 후에는 통과하는
F-to-P(Fail-to-Pass) 조건을 사용한다. 그러나 F-to-P를 만족하더라도 잘못된 assertion, 실행 환경 오류,
잘못된 입력 또는 이슈와 무관한 코드 경로 때문에 실패했을 수 있다.

본 프로젝트는 단순 F-to-P 여부를 넘어 다음 세 가지 증거를 함께 확인한다.

1. **버그 재현 여부**: 생성 테스트가 패치 전 코드에서 의미 있는 실패를 만드는가?
2. **코드 커버리지**: 생성 테스트가 이슈와 관련된 의심 코드 위치를 실제로 실행하는가?
3. **이슈 정합성**: 테스트 입력, assertion, 실패 결과가 이슈 설명의 핵심 조건과 일치하는가?

세 조건을 모두 만족한 테스트만 `ALIGNED`로 분류한다. 이는 일부 유효한 테스트를 제외할 수 있더라도,
이슈와 무관한 테스트를 신뢰 가능한 테스트로 잘못 채택하지 않는 soundness 중심 설계이다.

## 신뢰할 수 있는 AI 관점

프로젝트는 수업에서 다룬 testing, coverage, correctness specification, verification의 사고방식을
LLM 생성 테스트 평가에 적용한다.

이슈 재현 평가를 Hoare logic의 형태로 해석하면 다음과 같다.

```text
{P} C {Q}
```

- `P`: 이슈 설명에서 추출한 재현 조건과 입력 조건
- `C`: LLM이 생성한 테스트 실행
- `Q`: 이슈에서 설명한 기대 실패 또는 오류 증상

핵심 검증 질문은 다음과 같다.

> 이슈 조건 P에서 생성 테스트 C를 실행했을 때, 이슈에서 설명한 실패 Q가 실제로 발생하는가?

평가 결과에는 최종 라벨뿐 아니라 실행 결과, 점수, 커버리지 및 실패 원인을 함께 저장하여 판단 근거를
검토할 수 있도록 한다.

## 평가 파이프라인

```text
GitHub Issue
  |
  v
Issue Specification Extraction
  |
  v
Code Context Retrieval
  |
  v
Test Scenario Generation and Validation
  |
  v
Reproduction Test Generation
  |
  v
Before-Patch Execution
  |
  v
Strict Alignment Gates
  |
  +-- Bug Fail Gate
  +-- Coverage Gate
  +-- Issue Alignment Gate
  |
  v
ALIGNED Tests Only
  |
  v
Final F-to-P Evaluation
```

### 엄격한 정합성 게이트

| Gate | 검증 내용 | 실패 시 라벨 |
| --- | --- | --- |
| Bug Fail Gate | 패치 전 코드에서 이슈 재현 실패를 만드는지 확인 | `NOT_FAILED`, `ERROR`, `NOT_VALID` |
| Coverage Gate | 이슈 관련 의심 코드 위치를 실행하는지 확인 | `NO_COVERAGE` |
| Issue Alignment Gate | 입력, oracle, 실패 증상이 이슈 명세와 정합한지 확인 | `WEAK_ALIGNMENT` |

`ALIGNED`는 세 게이트를 모두 통과한 경우만 의미한다. 별도의 relaxed 승격 기준은 사용하지 않는다.

### 최종 라벨

| Label | 의미 |
| --- | --- |
| `ALIGNED` | 버그 실패, 관련 코드 커버리지, 이슈 정합성을 모두 만족 |
| `NO_COVERAGE` | 테스트가 의심 코드 위치를 충분히 실행하지 못함 |
| `WEAK_ALIGNMENT` | 테스트 실패와 이슈 설명의 정합성이 부족함 |
| `NOT_FAILED` | 패치 전 코드에서 테스트가 실패하지 않음 |
| `ERROR` | 테스트 실행 또는 환경 오류 발생 |
| `NOT_VALID` | 생성된 테스트가 유효하지 않거나 수집되지 않음 |

## 실험 설정

- Dataset: TDD-Bench-Verified 일부 샘플
- Sample size: 23 instances from 12 repositories
- Model: `Qwen/Qwen2.5-Coder-32B-Instruct`
- Alignment iterations: 최대 5회
- Alignment policy: strict three-gate policy
- Final evaluation: `ALIGNED` 테스트만 패치 전/후 실행

본 실험은 전체 449개 인스턴스 성능을 주장하기 위한 실험이 아니라, 제안한 신뢰성 평가 시스템을
검증하기 위한 소규모 프로토타입 실험이다.

## 실험 결과

최신 strict 스몰배치 결과는 `results/smallbatch/20260610_101413/`에 저장되어 있다.

| Metric | Result |
| --- | ---: |
| Total instances | 23 |
| ALIGNED | 10 |
| Resolved | 9 |
| Final evaluation count | 10 |
| ALIGNED rate | 43.5% |
| Resolve rate | 39.1% |
| Average iterations | 3.48 |
| Average prompt tokens | 14,258 |
| Average completion tokens | 1,185 |
| Average total tokens | 15,443 |
| Average patch-line coverage | 77.2% |

### 실패 유형 분포

| Label | Count | Rate |
| --- | ---: | ---: |
| `ALIGNED` | 10 | 43.5% |
| `NO_COVERAGE` | 7 | 30.4% |
| `NOT_FAILED` | 4 | 17.4% |
| `NOT_VALID` | 2 | 8.7% |
| `ERROR` | 0 | 0.0% |
| `WEAK_ALIGNMENT` | 0 | 0.0% |

### 결과 해석

- 23개 생성 테스트 중 10개만 엄격한 정합성 게이트를 모두 통과했다.
- ALIGNED 10개 중 9개가 최종 F-to-P 평가에서 resolved로 판정되었다.
- 가장 빈번한 제외 사유는 `NO_COVERAGE`였다. 테스트가 실패하더라도 관련 버그 경로를 실행하지 않으면
  신뢰 가능한 재현 테스트로 채택하기 어렵다는 점을 보여준다.
- `NOT_FAILED`와 `NOT_VALID` 결과는 LLM 생성 테스트가 항상 실행 가능한 버그 재현 테스트가 되지는
  않는다는 점을 보여준다.

## 결과 파일 구조

```text
results/smallbatch/20260610_101413/
├── batch_summary.json
└── <instance_id>/
    ├── clue.json
    ├── context.json
    ├── scenario.json
    ├── scenario_validation.json
    ├── generated_test.json
    ├── generated_test.patch
    ├── generated_test_rendered.py
    ├── alignment_execution.json
    ├── alignment_result.json
    ├── execution_result.json
    └── final_evaluation.json
```

- `clue.json`: 이슈에서 추출한 명세와 단서
- `context.json`: 관련 소스 및 테스트 코드 문맥
- `scenario.json`: 생성된 재현 시나리오
- `generated_test.json`: LLM이 생성한 재현 테스트
- `alignment_execution.json`: 패치 전 실행 결과와 커버리지
- `alignment_result.json`: 엄격한 정합성 게이트 판정과 근거
- `final_evaluation.json`: ALIGNED 테스트의 패치 전/후 평가 결과
- `batch_summary.json`: 스몰배치 전체 집계

## 소스 코드 구조

| Path | 역할 |
| --- | --- |
| `src/issue_parser/issue_clues.py` | 이슈 명세 및 단서 추출 |
| `src/context_builder/code_context.py` | 관련 코드 문맥 수집 |
| `src/scenario/scenario_generator.py` | 테스트 시나리오 생성 |
| `src/scenario/scenario_validator.py` | 시나리오 검증 |
| `src/generator/repro_test_generator.py` | 재현 테스트 생성 |
| `src/executor/alignment_runner.py` | 패치 전 테스트 실행 및 커버리지 수집 |
| `src/alignment/alignment_scorer.py` | 엄격한 정합성 게이트 판정 |
| `src/evaluator/final_evaluator.py` | 최종 F-to-P 평가 |
| `src/pipeline/run_batch.py` | 배치 실행 및 결과 집계 |

## 한계

- 소규모 샘플 실험이므로 전체 TDD-Bench-Verified 성능으로 일반화할 수 없다.
- 이슈 정합성 점수는 규칙 기반 증거와 이슈 텍스트에 의존하므로 복잡한 의미 관계를 완전히 포착하지 못할 수 있다.
- 커버리지 부족은 테스트 품질 문제뿐 아니라 결함 위치 추정 오류에서 발생할 수 있다.
- LLM 생성은 비결정적이므로 동일 인스턴스를 다시 실행할 때 결과가 달라질 수 있다.
- 현재 구현은 Python 프로젝트와 해당 벤치마크의 Docker 실행 환경을 중심으로 검증되었다.

## 향후 과제

- 이슈 설명의 정보 충분성 평가
- 호출 그래프와 실행 추적을 활용한 결함 위치 추정 개선
- 라벨 및 판정 근거를 비교할 수 있는 설명 가능한 대시보드
- 반복 실행을 통한 생성 안정성 및 결과 변동성 분석
- Java, JavaScript 등 다른 언어와 테스트 프레임워크 지원

