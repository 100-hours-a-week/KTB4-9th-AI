# 파이썬 코딩 컨벤션 (Python Coding Convention)

---

## 1. 들여쓰기 (Indentation)

- 공백(Space) 4칸을 기본으로 사용한다. 탭(Tab)은 사용하지 않는다.

## 2. 줄 길이와 줄바꿈

- 한 줄은 최대 **88자**를 넘지 않도록 제한한다. (Ruff 포매터 기준)
- 줄이 길어질 경우 괄호를 이용해 다음 줄로 나누는 것을 권장하며, 이어지는 줄은 4칸 들여쓴다.

```python
some_function_with_a_long_name(
    argument_1, argument_2, argument_3,
    argument_4, argument_5
)
```

## 3. 공백 규칙

- 모듈 레벨 함수·클래스 정의 사이는 빈 줄 2줄로 구분한다.
- 클래스 내부 메서드 사이는 빈 줄 1줄로 구분한다.
- 괄호·중괄호·대괄호 바로 안쪽에는 공백을 넣지 않는다.
- 쉼표, 콜론, 세미콜론 앞에는 공백을 넣지 않고, 뒤에는 공백을 넣는다.
- 이항 연산자(`=`, `+`, `==` 등) 앞뒤에는 공백을 넣는다.

```python
class MyClass:
    pass

def my_function():
    pass
```

## 4. 임포트 (Import)

- 모든 임포트는 파일 최상단에 모아서 작성한다.
- 다음 3개 그룹 순서로 작성하고, 그룹 사이는 빈 줄로 구분한다. (Ruff Isort 규격 자동 적용)
    1. 표준 라이브러리
    2. 서드파티 라이브러리
    3. 로컬(프로젝트 내부) 모듈
- 하나의 `import` 문에는 모듈 하나만 작성한다.

```python
import os
import sys

import numpy as np
import pandas as pd

import mymodule
```

## 5. 네이밍 규칙

| 대상 | 규칙 | 예시 |
| --- | --- | --- |
| 변수, 함수, 메서드 | snake_case (소문자 + 밑줄) | `my_variable`, `my_function()` |
| 클래스 | PascalCase(CamelCase) | `MyClass`, `MyDerivedClass` |
| 상수 | UPPER_SNAKE_CASE, 모듈 레벨에서만 정의 | `MAX_SIZE`, `PI` |
| 모듈명 | 영어 소문자 | `my_module` |
| protected 멤버 | 밑줄 1개로 시작 | `_value` |
| private 멤버 | 밑줄 2개로 시작 | `__value` |
| 글로벌 변수 | 모듈 내부 전용, `g_` 접두어 권장 | `g_config` |
| 클래스 변수 | 일반 변수와 동일하게 snake_case 사용 | `count` |

## 6. 문자열

- 일반적인 문자열에는 큰따옴표(`"`)를 사용한다. (Ruff 포매터 기준)

## 7. 주석과 Docstring

- 한 줄 주석: `#` 뒤에 공백을 두고 간결하게 작성한다.
- 블록(여러 줄) 주석: 각 줄 앞에 `#`과 공백을 두어 정렬한다.
- Docstring: 함수·클래스·모듈 설명은 `"""`로 감싸 작성하며, 설명/Parameters/Returns를 포함한다. Docstring 내부의 코드 예시도 Ruff 포맷팅 규칙을 따른다.

```python
def add(a: int, b: int) -> int:
    """
    두 숫자의 합을 반환합니다.

    Parameters:
        a (int): 첫 번째 숫자
        b (int): 두 번째 숫자

    Returns:
        int: 두 숫자의 합
    """
    return a + b
```

## 8. 함수 작성 규칙

- 함수 정의 시 타입 힌트(typing)를 사용해 인자·반환 타입을 명시한다.
- 참/거짓 비교 시 `if x is True:` 대신 `if x:` 또는 `if not x:`와 같이 변수 자체를 조건문에 사용한다. (`None` 비교 시에만 `is None` / `is not None` 명시)
- 한 줄로 표현 가능한 조건문은 삼항 연산자를 사용한다.
    
    ```python
    result = "짝수" if x % 2 == 0 else "홀수"
    ```

## 9. 자료구조

- 빈 리스트와 딕셔너리를 초기화할 때는 속도와 가독성을 위해 `[]`, `{}` 리터럴 표현식을 권장한다.
- 간단한 리스트 생성은 리스트 컴프리헨션(list comprehension)을 활용한다.
    
    ```python
    squares = [n ** 2 for n in range(10)]
    ```

## 10. 클래스 작성 규칙

- 클래스명은 CamelCase(PascalCase)로 작성한다.
- 메서드명은 snake_case로 작성하고, 메서드 사이는 빈 줄 1줄로 구분한다.
- 타입 검사는 `type()` 대신 `isinstance()`를 사용한다.
- 생성자(`__init__`)와 인스턴스 표현을 위한 `__str__` 또는 `__repr__`을 적극 활용한다. (소멸자 `__del__`은 메모리 누수 위험으로 사용을 금지한다.)

## 11. 파일 입출력

- 파일을 열 때는 반드시 `with` 문을 사용해 자동으로 닫히도록 한다.
    
    ```python
    with open("data.txt", "r", encoding="utf-8") as f:
        content = f.read()
    ```
- 로그를 남길 때는 날짜와 시간을 반드시 포함한다.
