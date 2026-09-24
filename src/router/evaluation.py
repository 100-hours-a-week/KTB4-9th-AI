from fastapi import APIRouter

from src.schema.evaluation import EvaluationRequest, EvaluationResponse

router = APIRouter()


@router.post("/api/llm/evaluation")
async def evaluate_nl_solution(request: EvaluationRequest) -> EvaluationResponse:
    return EvaluationResponse()
