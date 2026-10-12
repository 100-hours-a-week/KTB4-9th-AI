"""외부 데이터셋 문제를 우리 GeneratedProblem 형식으로 옮기는 프롬프트.

난이도와 카테고리는 mapping.py가 정하므로 모델에게 맡기지 않는다.
모델은 영어 지문을 한국어로 옮기고 우리 스키마에 맞는 나머지 필드를 채운다.
"""

CONVERT_PROMPT = """\
당신은 영어 알고리즘 문제를 한국어 코딩 테스트 문제로 옮기는 번역가다.

[원본 문제]
제목: {name}

지문:
{description}

예시 입출력:
{examples}

실행 제한: {time_limit_s}초, {memory_limit_mb}MB

[요청]
난이도: {difficulty} (이미 정해졌다. difficulty에 그대로 적는다)
카테고리: {category} (이미 정해졌다. category에 그대로 적는다)

[규칙]
- 원본의 문제 의미를 바꾸지 않는다. 조건, 제약, 입출력 형식을 그대로 옮긴다.
- problem_title은 {max_title}자 이하, problem_content는 {max_content}자 이하로 쓴다.
- 원본 지문이 길면 핵심 조건을 빠뜨리지 않는 선에서 줄인다.
- 수식 기호(≤, ≥, ⋅, …)는 한국어 문장에 맞게 풀어 쓴다.
- problem_content에는 입력 형식과 출력 형식을 적지 않는다.
  그 둘은 input_format, output_format에 따로 적는다.
- problem_examples는 원본 예시를 그대로 옮긴다. 값을 바꾸지 않는다.
- input_constraints에는 입력 항목(scope: INPUT)과
  출력(scope: OUTPUT, target: output)을 모두 적는다.
  원본 지문에 적힌 범위를 읽어 min_value, max_value를 채운다.
- data_type이 숫자형(INT, LONG, FLOAT, DOUBLE)일 때만
  min_value, max_value를 숫자로 적는다.
- 정수는 INT, 범위가 크면 LONG, 실수는 DOUBLE을 쓴다.
  FLOAT는 명시적으로 단정밀도가 필요할 때만 쓴다.
- 숫자형이 아니면(STRING, CHAR, BOOLEAN) min_value, max_value를 비워 둔다.
- 값들이 서로 달라야 하면 special_conditions에 "서로 다른 값"이라고 적는다.
- execution_limits는 {languages} 네 언어를 모두 적는다.
  원본의 실행 제한을 언어마다 같은 값으로 적는다.
- category_select_reason에는 이 카테고리로 판단한 근거를
  {max_reason}자 이하의 한 문장으로 적는다.
- algorithm_core에는 이야기 설정을 빼고, 이 문제만의 조건과 계산 대상을
  한 문장으로 적는다. 자료구조·알고리즘 이름만 적지 않는다.
  같은 알고리즘을 쓰는 다른 문제와 구분되는 조건을 반드시 포함한다.
  나쁜 예: "BFS로 최단 거리를 구한다"
  좋은 예: "벽을 최대 한 번 부술 수 있다는 조건에서 BFS로 최단 거리를 구한다"
"""


def render_examples(public_tests: dict) -> str:
    """원본 예시 입출력을 프롬프트에 넣을 문자열로 만든다."""
    lines = []
    pairs = zip(public_tests["input"], public_tests["output"], strict=True)
    for i, (stdin, expected) in enumerate(pairs, start=1):
        lines.append(f"예시 {i}")
        lines.append(f"입력:\n{stdin.strip()}")
        lines.append(f"출력:\n{expected.strip()}")
        lines.append("")
    return "\n".join(lines)
