from dataclasses import dataclass

import pytest

from systemone_harness import Action, SystemOneHarness


@dataclass
class FakeUsage:
    input_tokens: int | None = 12
    output_tokens: int | None = 0


@dataclass
class FakeChoice:
    choice: str = "retreat"
    probabilities: dict[str, float] = None

    def __post_init__(self):
        if self.probabilities is None:
            self.probabilities = {"attack": 0.1, "retreat": 0.9}


class FakeResponse:
    model = "fake-system-one"
    usage = FakeUsage()
    choices = {"action": FakeChoice()}


class FakeModels:
    def list(self):
        return type("R", (), {"models": ("fake-system-one",)})()


class FakeClient:
    def __init__(self):
        self.calls = []
        self.models = FakeModels()
        self.closed = False

    def system_one(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse()

    def close(self):
        self.closed = True


def test_decide_maps_choice_back_to_original_action():
    client = FakeClient()
    harness = SystemOneHarness(client)

    attack = Action("attack", "Attack.")
    retreat = Action("retreat", "Retreat.", payload={"dx": -1})

    result = harness.decide(
        goal="Survive.",
        state={"health": 2},
        actions=[attack, retreat],
    )

    assert result.action is retreat
    assert result.action.payload == {"dx": -1}
    assert result.confidence == pytest.approx(0.9)
    assert client.calls[0]["state"] == {
        "goal": "Survive.",
        "state": {"health": 2},
    }


def test_duplicate_action_ids_are_rejected():
    harness = SystemOneHarness(FakeClient())

    with pytest.raises(ValueError, match="unique"):
        harness.decide(
            state={},
            actions=[Action("same"), Action("same")],
        )


def test_empty_action_set_is_rejected():
    harness = SystemOneHarness(FakeClient())

    with pytest.raises(ValueError, match="At least one"):
        harness.decide(state={}, actions=[])
