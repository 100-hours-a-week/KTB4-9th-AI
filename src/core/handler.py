import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.core.exception import CosmosError
from src.enum import ErrorCode
from src.schema import BaseResponse

logger = logging.getLogger(__name__)

# 검증 실패한 필드 이름 → 더 구체적인 코드
_FIELD_ERROR_CODES = {
    "difficulty": ErrorCode.INVALID_DIFFICULTY,
    "requested_difficulty": ErrorCode.INVALID_DIFFICULTY,
    "category": ErrorCode.INVALID_CATEGORY,
    "requested_category": ErrorCode.INVALID_CATEGORY,
    "language": ErrorCode.UNSUPPORTED_LANGUAGE,
}


def _body(message: str, code: ErrorCode | None) -> dict:
    return BaseResponse(success=False, message=message, error_code=code).model_dump(
        by_alias=True
    )


def _resolve_error_code(error: dict) -> ErrorCode:
    """검증 오류 하나를 ErrorCode로 옮긴다.

    필드가 아예 없으면 누락이고, 값이 있는데 틀렸으면 필드별 코드를 쓴다.
    """
    if error.get("type") == "missing":
        return ErrorCode.MISSING_REQUIRED_FIELD
    # loc은 ("body", "difficulty") 형태라 마지막 항목이 필드 이름이다
    location = [part for part in error.get("loc", ()) if isinstance(part, str)]
    field = location[-1] if location else ""
    return _FIELD_ERROR_CODES.get(field, ErrorCode.MISSING_REQUIRED_FIELD)


async def handle_cosmos_error(request: Request, exc: CosmosError) -> JSONResponse:
    logger.warning(
        "%s: %s", type(exc).__name__, exc.message, extra={"path": request.url.path}
    )
    return JSONResponse(status_code=exc.status, content=_body(exc.message, exc.code))


async def handle_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    errors = exc.errors()
    first = errors[0] if errors else {}
    code = _resolve_error_code(first)
    message = first.get("msg", "요청 형식이 올바르지 않습니다")
    logger.info(
        "요청 검증 실패: %s",
        message,
        extra={"path": request.url.path, "loc": str(first.get("loc", ""))},
    )
    return JSONResponse(status_code=422, content=_body(message, code))


async def handle_http_exception(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """404, 503 등 FastAPI가 직접 올리는 예외도 같은 본문 형태로 맞춘다.

    4xx는 ErrorCode 목록에 대응하는 값이 없으므로 errorCode를 비운다.
    INTERNAL_SERVER_ERROR를 넣으면 클라이언트가 서버 장애로 오해한다.
    """
    code = ErrorCode.INTERNAL_SERVER_ERROR if exc.status_code >= 500 else None
    if exc.status_code >= 500:
        logger.error("%s: %s", exc.status_code, exc.detail)
    return JSONResponse(
        status_code=exc.status_code, content=_body(str(exc.detail), code)
    )


async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """예상하지 못한 예외. 원인 추적이 필요하므로 스택을 남긴다."""
    logger.error(
        "처리되지 않은 예외: %s", exc, exc_info=True, extra={"path": request.url.path}
    )
    return JSONResponse(
        status_code=500,
        content=_body("서버 내부 오류가 발생했습니다", ErrorCode.INTERNAL_SERVER_ERROR),
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(CosmosError, handle_cosmos_error)
    app.add_exception_handler(RequestValidationError, handle_validation_error)
    app.add_exception_handler(StarletteHTTPException, handle_http_exception)
    app.add_exception_handler(Exception, handle_unexpected_error)
