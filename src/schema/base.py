from pydantic import BaseModel, ConfigDict

from src.core.exception import ErrorCode


def to_camel(string: str) -> str:
    parts = string.split("_")
    return parts[0] + "".join(word.capitalize() for word in parts[1:])


class CamelBaseModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )


class BaseResponse(CamelBaseModel):
    success: bool = True
    message: str | None = None
    error_code: ErrorCode | None = None
