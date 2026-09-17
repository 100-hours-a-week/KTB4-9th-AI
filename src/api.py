from fastapi import FastAPI

# from src.shared.config import get_settings

app = FastAPI()

# settings = get_settings()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
