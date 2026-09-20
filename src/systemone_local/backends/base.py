from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol


@dataclass(frozen=True, slots=True)
class ChoiceResult:
    """One choice decision returned by a local model backend."""

    choice: str
    probabilities: Mapping[str, float]
    input_tokens: int


class ChoiceBackend(Protocol):
    """The entire internal backend contract for v0.1."""

    name: str
    description: str

    def choice(
        self,
        state: str | dict[str, Any] | list[Any],
        instructions: Any,
        criteria: Mapping[str, Any],
    ) -> ChoiceResult: ...
