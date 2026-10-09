-- fewshot_seeds 적재
--
-- 80개 조합(16 카테고리 × 5 난이도)에 조합당 1건씩 넣는다.
-- generated_problems에서 골라 복사하며, 원본 행은 수정하지 않는다.
--
-- LV1  : 사람이 고른 16건. 모바일에서 짧게 읽히도록 지문을 다듬었다.
--        풀이가 단순한 것을 우선하고, 그중 지문이 짧은 것을 골랐다.
-- LV2~5: 조합마다 파이썬 정답 코드 줄 수가 중간값인 것을 자동 선택.
--        가장 짧은 코드는 그 난이도치곤 쉽고, 가장 긴 코드는 과한 문제일 수
--        있어 중간을 택한다. 지문은 손대지 않는다.

BEGIN;

-- ── LV1 (16건) ────────────────────────────────────────────────

INSERT INTO fewshot_seeds (
    id, source, source_ref, category, difficulty,
    problem_title, problem_content, input_format, output_format,
    input_constraints, execution_limits, problem_examples,
    is_active, created_at
)
SELECT DISTINCT ON (g.category)
    gen_random_uuid(), 'PROMOTED', g.id::text, g.category, g.difficulty,
    g.problem_title, v.content, g.input_format, g.output_format,
    g.input_constraints, g.execution_limits, g.problem_examples,
    true, now()
FROM generated_problems g
JOIN (VALUES
('ARRAY','양옆의 합보다 큰 원소',
E'길이가 N인 정수 배열 A가 주어집니다.\n\n양옆에 이웃이 모두 있는 원소 중에서, 자기 값이 양옆 두 원소의 합보다 큰 것이 몇 개인지 구하는 프로그램을 작성하세요.'),

('BACKTRACKING','목표 합을 만드는 K개의 수',
E'자연수 N개로 이루어진 수열과 두 정수 K, T가 주어집니다.\n\n이 수열에서 서로 다른 위치의 원소 K개를 골랐을 때, 고른 수들의 합이 정확히 T가 되는 경우의 수를 구하는 프로그램을 작성하세요.\n\n고른 위치의 구성이 같으면 순서가 달라도 같은 경우로 봅니다.'),

('BINARY_SEARCH','정렬된 수열에서 숫자 찾기',
E'오름차순으로 정렬된 정수 N개와 찾으려는 정수 X가 주어집니다.\n\nX가 수열에 있으면 처음 등장하는 위치의 인덱스를, 없으면 -1을 출력하는 프로그램을 작성하세요.\n\n인덱스는 0부터 셉니다.'),

('BRUTE_FORCE','자릿수 합 맞추기',
E'양의 정수 N과 K가 주어집니다.\n\n1 이상 N 이하의 자연수 중에서 각 자릿수의 합이 정확히 K가 되는 수가 몇 개인지 구하는 프로그램을 작성하세요.\n\n예를 들어 N이 20, K가 5이면 5와 14로 2개입니다.'),

('DP','토끼의 계단 오르기',
E'토끼가 N개의 계단을 오르려고 합니다. 한 번에 1개 또는 2개의 계단을 오를 수 있습니다.\n\n바닥에서 출발해 N번째 계단에 도달하는 방법이 몇 가지인지 구하는 프로그램을 작성하세요.\n\n방법의 수가 클 수 있으므로 1,000,000,007로 나눈 나머지를 출력합니다.'),

('GRAPH','직접 연결된 컴퓨터의 수',
E'1번부터 N번까지 번호가 매겨진 컴퓨터 N대가 M개의 양방향 케이블로 연결되어 있습니다.\n\n1번 컴퓨터에 바이러스가 침투했습니다. 1번과 케이블로 직접 연결된 컴퓨터가 몇 대인지 구하는 프로그램을 작성하세요.'),

('GREEDY','마감 전 과제 해결하기',
E'민우에게는 처리해야 할 과제 N개가 있고, 남은 시간은 T분입니다.\n\n각 과제에 필요한 시간이 분 단위로 주어질 때, T분 안에 완료할 수 있는 과제의 최대 개수를 구하는 프로그램을 작성하세요.\n\n한 번 시작한 과제는 중단할 수 없습니다.'),

('HASH','유일한 사탕 찾기',
E'사탕 선물 세트에 N개의 사탕이 들어 있습니다.\n\n사탕 이름 목록이 주어질 때, 세트 전체에서 정확히 한 번만 등장하는 사탕이 몇 종류인지 구하는 프로그램을 작성하세요.'),

('HEAP','가장 저렴한 선물 K개',
E'어느 쇼핑몰에서 친구들을 위해 N개의 선물 후보 중 가장 가격이 저렴한 K개의 선물을 골라 구매하려고 합니다.\n\n각 선물의 가격이 주어졌을 때, 가장 저렴한 K개 선물의 가격 총합을 구하는 프로그램을 작성하세요.'),

('IMPLEMENTATION','트랙 위의 로봇',
E'1번부터 N번까지 번호가 매겨진 N개의 칸으로 이루어진 트랙이 있습니다. 로봇은 처음에 K번 칸에 있습니다.\n\n''L''은 왼쪽으로 한 칸, ''R''은 오른쪽으로 한 칸 이동하는 명령입니다. 명령 문자열이 주어질 때, 모든 명령을 순서대로 수행한 뒤 로봇이 있는 칸의 번호를 구하는 프로그램을 작성하세요.\n\n트랙 밖으로 나가는 명령은 무시하고 제자리에 머뭅니다.'),

('MATH','다음 소수 찾기',
E'양의 정수 N이 주어집니다. N 이상인 자연수 중에서 가장 작은 소수(Prime Number)를 구하는 프로그램을 작성하세요.\n\n소수란 1보다 큰 자연수 중에서 1과 자기 자신만을 약수로 가지는 수를 의미합니다.'),

('SORTING','단어 길이순 정렬',
E'서로 다른 영어 단어 N개가 주어집니다.\n\n이 단어들을 아래 규칙에 따라 정렬하여 출력하는 프로그램을 작성하세요.\n\n1. 길이가 짧은 단어가 먼저 옵니다.\n2. 길이가 같으면 사전 순으로 앞선 단어가 먼저 옵니다.'),

('STACK_QUEUE','연속 문자 정리하기',
E'알파벳 소문자로 이루어진 문자열 S가 주어집니다.\n\n같은 문자가 두 개 연속으로 붙어 있으면 그 둘을 제거합니다. 제거할 수 있는 짝이 없을 때까지 반복한 뒤 남은 문자열을 출력하는 프로그램을 작성하세요.\n\n모두 제거되어 빈 문자열이 되면 EMPTY를 출력합니다.'),

('STRING','홀수 짝수 문자 재배치',
E'알파벳 대소문자로 이루어진 문자열 S가 주어집니다.\n\n홀수 번째 문자들을 순서대로 나열한 뒤, 그 뒤에 짝수 번째 문자들을 순서대로 이어 붙인 문자열을 구하는 프로그램을 작성하세요.\n\n문자의 위치는 첫 번째 문자를 1번으로 셉니다.'),

('TREE','리프 노드의 개수 세기',
E'1번부터 N번까지 번호가 매겨진 정점 N개로 이루어진 트리가 주어집니다. 1번 정점이 루트입니다.\n\n2번부터 N번까지 각 정점의 부모 번호가 순서대로 주어질 때, 자식이 하나도 없는 리프 노드의 개수를 구하는 프로그램을 작성하세요.'),

('TWO_POINTER','공통 숫자 찾기',
E'오름차순으로 정렬된 두 수열이 주어집니다. 첫 번째는 N개, 두 번째는 M개이며 각 수열 안에 중복은 없습니다.\n\n두 수열에 모두 등장하는 숫자가 몇 개인지 구하는 프로그램을 작성하세요.')
) AS v(cat, title, content)
  ON g.category::text = v.cat
 AND g.problem_title = v.title
 AND g.difficulty = 'LV1'
ORDER BY g.category, length(g.problem_content);

-- ── LV2~LV5 (64건) ────────────────────────────────────────────

WITH scored AS (
    SELECT g.*,
           array_length(string_to_array(
               (SELECT c->>'content'
                  FROM jsonb_array_elements(g.solution_codes) c
                 WHERE c->>'language' = 'PYTHON'
                 LIMIT 1),
               E'\n'), 1) AS code_lines
      FROM generated_problems g
     WHERE g.difficulty <> 'LV1'
),
ranked AS (
    SELECT *,
           row_number() OVER (
               PARTITION BY category, difficulty
               ORDER BY code_lines NULLS LAST, created_at
           ) AS rn,
           count(*) OVER (PARTITION BY category, difficulty) AS total
      FROM scored
)
INSERT INTO fewshot_seeds (
    id, source, source_ref, category, difficulty,
    problem_title, problem_content, input_format, output_format,
    input_constraints, execution_limits, problem_examples,
    is_active, created_at
)
SELECT gen_random_uuid(), 'PROMOTED', id::text, category, difficulty,
       problem_title, problem_content, input_format, output_format,
       input_constraints, execution_limits, problem_examples,
       true, now()
  FROM ranked
 WHERE rn = (total + 1) / 2;

COMMIT;