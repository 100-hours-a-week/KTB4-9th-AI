from fastapi import APIRouter

from src.problem import evaluation as evaluation_service
from src.schema.evaluation import EvaluationRequest, EvaluationResponse

router = APIRouter()


@router.post("/api/llm/evaluation")
async def evaluate_nl_solution(request: EvaluationRequest) -> EvaluationResponse:
    evaluation = await evaluation_service.evaluate(request)
    return EvaluationResponse(
        score=evaluation.score,
        llm_feedback=evaluation.feedback,
        keywords=evaluation_service.align_keywords(
            request.solution_keywords, evaluation.keywords
        ),
    )
