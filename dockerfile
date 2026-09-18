FROM python:3.11-slim

# 파이썬 출력 버퍼링 비활성화 및 바이트코드(.pyc) 미생성
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 1. 의존성 먼저 복사 및 설치 (도커 레이어 캐시 활용)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 2. 애플리케이션 코드 복사
COPY . .

# FastAPI 기본 포트 (내부 8000)
EXPOSE 8000

# Uvicorn 기동 (0.0.0.0 바인딩)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]