# Architecture Decision Records

설계에서 여러 선택지 중 하나를 고른 결정과 그 이유를 기록한다.
작은 수정이나 버그 수정은 적지 않는다. 프롬프트 수정 이력은 [PROMPT_CHANGELOG](../PROMPT_CHANGELOG.md)에 적는다.

## 작성 규칙

- 결정 하나당 문서 하나를 `NNNN-결정-요약.md` 이름으로 만든다.
- 상황, 선택지, 결정, 이유, 결과 순서로 적는다.
- 결정이 바뀌면 기존 문서를 지우지 않고, 새 ADR을 쓴 뒤 기존 문서의 상태를 `대체됨`으로 바꾼다.

## 목록

| ADR | 결정 | 상태 |
|---|---|---|
| [ADR-0001](0001-duplicate-check-by-algorithm-core-embedding.md) | 핵심 풀이(algorithm_core) 임베딩 유사도로 중복 문제를 거른다 | 채택 |