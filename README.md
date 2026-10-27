# COSMOS AI Server (`KTB4-9th-AI`)

COSMOS 코딩 테스트 플랫폼의 AI 문제 생성, 다단계 자동 검증(정적·모순·Judge0 의미 검증), 배틀 문제 생성 및 서술형 풀이를 담당하는 서버 저장소다.

---

## 주요 문서 바로가기

- **[COSMOS AI 모델 평가 및 LLMOps 사용 가이드 (`docs/LLMOPS_GUIDE.md`)](docs/LLMOPS_GUIDE.md)**
  - GitHub Actions 원클릭 모델 평가(`LLMOps Evaluation & Release`) 실행 방법
  - 로컬 PC(Mac) + 클라우드 GPU(RunPod / Modal / Colab) 수동 평가 가이드
  - Langfuse 트레이스 그래프·점수·프롬프트 버전 관리 활용법
  - MLflow 실험 비교 및 AWS SSM 기반 무중단 모델 승격·롤백 절차

---

## 빠른 시작 (로컬 개발 및 평가)

```bash
# 1. 의존성 설치
uv sync --extra dev

# 2. 로컬 PostgreSQL 기동 및 마이그레이션
docker compose up -d
uv run alembic upgrade head

# 3. 운영 기준 Few-shot 시드 데이터 80건 적재 (최초 1회)
uv run python -m src.db.seed_fewshot

# 4. FastAPI 서버 기동
uv run uvicorn src.main:app --reload --port 8000
```

모델 평가 하네스(`evals.generate` → `evals.score` → `evals.summarize`) 및 GitHub Actions 자동화 파이프라인의 상세 사용법은 **[`docs/LLMOPS_GUIDE.md`](docs/LLMOPS_GUIDE.md)**를 참고한다.
