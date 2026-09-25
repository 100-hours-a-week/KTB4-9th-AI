import httpx
import pytest

from src.client import judge0


@pytest.fixture
def judge0_available() -> None:
    """judge0에 실제로 제출하는 테스트용. judge0가 떠 있지 않으면 건너뛴다."""
    try:
        httpx.get(f"{judge0.JUDGE0_URL}/about", timeout=1.0).raise_for_status()
    except httpx.HTTPError as error:
        pytest.skip(f"judge0에 연결할 수 없음 ({judge0.JUDGE0_URL}): {error!r}")
