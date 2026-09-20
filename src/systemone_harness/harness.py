from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping
import os

from typesafe_sdk import Choice, TypeSafeClient


@dataclass(frozen=True, slots=True)
class Action:
    id: str
    description: str | None = None
    payload: Any = None

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise ValueError("Action.id must be non-empty.")


@dataclass(frozen=True, slots=True)
class Decision:
    action: Action
    probabilities: Mapping[str, float]
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def confidence(self) -> float | None:
        return self.probabilities.get(self.action.id)


class SystemOneHarness:
    """Minimal state + legal-actions -> action decision wrapper.

    The harness deliberately delegates protocol details to TypeSafe's official SDK.
    It can therefore point either at TypeSafe or at any local server implementing the
    same System One HTTP contract.
    """

    def __init__(
        self,
        client: TypeSafeClient,
        *,
        model: str | None = None,
        default_instructions: str | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.default_instructions = default_instructions or (
            "Choose the currently available action that best advances the stated goal "
            "given the current state. Choose only from the supplied actions."
        )

    @classmethod
    def from_env(
        cls,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 10.0,
    ) -> "SystemOneHarness":
        resolved_base_url = base_url or os.getenv("TYPESAFE_BASE_URL")
        resolved_model = model or os.getenv("TYPESAFE_DEFAULT_MODEL")

        # The official SDK requires a key even for a custom endpoint. For localhost,
        # use a harmless placeholder unless the caller explicitly supplied a key.
        resolved_key = api_key or os.getenv("TYPESAFE_API_KEY")
        if resolved_key is None and resolved_base_url:
            if resolved_base_url.startswith(("http://127.0.0.1", "http://localhost")):
                resolved_key = "local"

        client = TypeSafeClient(
            api_key=resolved_key,
            base_url=resolved_base_url,
            model=resolved_model,
            timeout=timeout,
        )
        return cls(client, model=resolved_model)

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "SystemOneHarness":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def list_models(self):
        return self.client.models.list().models

    def decide(
        self,
        *,
        state: Any,
        actions: Iterable[Action],
        goal: str | None = None,
        instructions: str | None = None,
        model: str | None = None,
    ) -> Decision:
        action_list = list(actions)
        if not action_list:
            raise ValueError("At least one action is required.")

        by_id = {action.id: action for action in action_list}
        if len(by_id) != len(action_list):
            raise ValueError("Action IDs must be unique.")

        criteria = {
            action.id: action.description if action.description is not None else action.id
            for action in action_list
        }

        model_state = (
            {"goal": goal, "state": state}
            if goal is not None
            else state
        )

        response = self.client.system_one(
            state=model_state,
            questions={
                "action": Choice(
                    instructions=instructions or self.default_instructions,
                    criteria=criteria,
                )
            },
            model=model or self.model,
        )

        answer = response.choices.get("action")
        if answer is None:
            raise RuntimeError("Backend did not return a Choice answer named 'action'.")

        selected = by_id.get(answer.choice)
        if selected is None:
            raise RuntimeError(
                f"Backend selected unknown action {answer.choice!r}; "
                f"expected one of {sorted(by_id)}."
            )

        return Decision(
            action=selected,
            probabilities=dict(answer.probabilities),
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
