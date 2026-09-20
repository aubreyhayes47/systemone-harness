from __future__ import annotations

import asyncio
from datetime import date

from fastapi import FastAPI, HTTPException

from .backends.base import ChoiceBackend
from .schema import ChoiceAnswer, ModelList, ModelMetadata, SystemOneRequest, SystemOneResponse, Usage


def create_app(backend: ChoiceBackend) -> FastAPI:
    """Create the deliberately small local TypeSafe-compatible HTTP surface."""

    app = FastAPI(title="systemone-local", version="0.1.0")

    @app.get("/health")
    async def health():
        return {"status": "ready", "model": backend.name}

    @app.get("/v1/models", response_model=ModelList)
    async def models() -> ModelList:
        return ModelList(
            models=[
                ModelMetadata(
                    name=backend.name,
                    description=backend.description,
                    release_date=date.today().isoformat(),
                )
            ]
        )

    @app.post("/v1/systemone", response_model=SystemOneResponse)
    async def system_one(request: SystemOneRequest) -> SystemOneResponse:
        if request.model != backend.name:
            raise HTTPException(
                status_code=422,
                detail=f"Loaded model is {backend.name!r}; received {request.model!r}",
            )

        answers: dict[str, ChoiceAnswer] = {}
        input_tokens = 0
        for question_id, question in request.questions.items():
            try:
                result = await asyncio.to_thread(
                    backend.choice,
                    request.state,
                    question.instructions,
                    question.criteria,
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

            probabilities = dict(result.probabilities)
            confidence = probabilities.get(result.choice)
            if confidence is None:
                raise HTTPException(status_code=500, detail="Backend returned an inconsistent choice result")
            answers[question_id] = ChoiceAnswer(
                choice=result.choice,
                confidence=confidence,
                probabilities=probabilities,
            )
            input_tokens += result.input_tokens

        return SystemOneResponse(
            model=backend.name,
            answers=answers,
            usage=Usage(input_tokens=input_tokens, output_tokens=0),
        )

    return app
