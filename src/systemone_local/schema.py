from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator


class ChoiceQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["choice"]
    instructions: str | dict[str, JsonValue] | list[JsonValue] | None = None
    criteria: dict[str, str | dict[str, JsonValue] | list[JsonValue] | None] = Field(min_length=2)

    @field_validator("criteria")
    @classmethod
    def bounded_options(cls, value):
        if len(value) > 26:
            raise ValueError("v0.1 supports at most 26 choice options")
        if any(not key for key in value):
            raise ValueError("choice option IDs must not be empty")
        return value


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: str | dict[str, JsonValue] | list[JsonValue]
    model: str = Field(min_length=1)
    questions: dict[str, ChoiceQuestion] = Field(min_length=1)

    @field_validator("questions")
    @classmethod
    def nonempty_question_ids(cls, value):
        if any(not key for key in value):
            raise ValueError("question IDs must not be empty")
        return value


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    confidence: float
    probabilities: dict[str, float]


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int = 0


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, ChoiceAnswer]
    usage: Usage


class ModelMetadata(BaseModel):
    name: str
    description: str
    release_date: str


class ModelList(BaseModel):
    models: list[ModelMetadata]
