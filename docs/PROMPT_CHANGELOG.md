# 프롬프트 변경 기록

프롬프트를 버전별로 무엇을, 왜 바꿨고, 결과가 어땠는지 기록한다.

## 기록 규칙

- 프롬프트 문구를 바꾸면 그 파일의 `PROMPT_VERSION`을 한 단계 올린다.
- 버전을 올린 PR에서 이 문서의 해당 표에 한 줄을 추가한다.
- 각 칸에는 다음을 적는다.
  - **바꾼 것**: 프롬프트에서 달라진 내용
  - **이유**: 왜 바꿨는지. 관련 이슈가 있으면 번호를 함께 적는다
  - **결과**: 바꾼 뒤 확인한 수치나 관찰. 확인 전이면 비워 두고 나중에 채운다
- 모델만 바꾸고 문구가 같으면 버전을 올리지 않아도 된다. 올렸다면 바꾼 것에 모델 변경을 적는다.

## 프롬프트 목록

| 프롬프트 | 파일 | PROMPT_VERSION |
|---|---|---|
| 문제 생성 | `src/problem/nodes/generate_problem.py` | `generate_problem` |
| 지문 모순 검사 | `src/problem/nodes/check_contradiction.py` | `check_contradiction` |
| 검증용 코드 생성 | `src/problem/nodes/generate_ref_code.py` | `generate_ref_code` |
| 비공개 테스트케이스 생성 | `src/problem/nodes/generate_testcase.py` | `generate_testcase` |
| 정답 코드·힌트 생성 | `src/problem/nodes/generate_solution_code.py` | `generate_solution_code` |
| 채점 키워드 생성 | `src/problem/nodes/generate_nl_keyword.py` | `generate_nl_keyword` |
| 배틀 문제 생성 | `src/problem/battle/nodes.py` | `generate_battle_problem` |
| 자연어 풀이 채점 | `src/problem/evaluation.py` | `evaluate_nl_solution` |