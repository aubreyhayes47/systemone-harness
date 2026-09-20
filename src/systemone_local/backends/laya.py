from __future__ import annotations

import math
import threading
from collections.abc import Mapping
from typing import Any

from .base import ChoiceResult


class LayaBackend:
    """Thin native adapter around Laya's own typed-decision runtime."""

    def __init__(self, agent, *, name: str, description: str | None = None) -> None:
        self.agent = agent
        self.name = name
        self.description = description or "Local Laya System One decision model."
        self._lock = threading.Lock()

    @classmethod
    def load(
        cls,
        source: str = "convaiinnovations/laya",
        *,
        subfolder: str | None = None,
        device: str = "auto",
    ) -> "LayaBackend":
        try:
            import laya
        except ImportError as exc:  # pragma: no cover - exercised by installations
            raise RuntimeError(
                "Laya inference requires the 'laya' extra: pip install -e '.[local,laya]'"
            ) from exc

        agent = laya.load(
            source,
            subfolder=subfolder,
            device=None if device == "auto" else device,
        )
        return cls(agent, name=source)

    def choice(
        self,
        state: str | dict[str, Any] | list[Any],
        instructions: Any,
        criteria: Mapping[str, Any],
    ) -> ChoiceResult:
        if not 2 <= len(criteria) <= 26:
            raise ValueError("choice criteria must contain 2-26 options")
        question = {
            "action": {
                "type": "choice",
                "instructions": instructions,
                "criteria": dict(criteria),
            }
        }
        with self._lock:
            result = self.agent.predict(state, question)

        try:
            answer = result["answers"]["action"]
            selected = answer["choice"]
            probabilities = {key: float(value) for key, value in answer["probabilities"].items()}
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Laya returned an invalid choice response") from exc

        expected = set(criteria)
        if selected not in expected or set(probabilities) != expected:
            raise ValueError("Laya returned choices that do not match the request")
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in probabilities.values()):
            raise ValueError("Laya returned invalid probabilities")
        if abs(sum(probabilities.values()) - 1.0) > 0.01:
            raise ValueError("Laya probabilities do not sum to approximately 1")

        usage = result.get("usage") or {}
        input_tokens = int(usage.get("input_tokens") or 0)
        if input_tokens < 0:
            raise ValueError("Laya returned invalid token usage")
        return ChoiceResult(selected, probabilities, input_tokens)
