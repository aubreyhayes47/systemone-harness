from fastapi.testclient import TestClient

from systemone_local.backends.base import ChoiceResult
from systemone_local.server import create_app


class FakeBackend:
    name = "fake-system-one"
    description = "Fake test backend"

    def choice(self, state, instructions, criteria):
        keys = list(criteria)
        probabilities = {key: 0.0 for key in keys}
        probabilities[keys[-1]] = 1.0
        return ChoiceResult(keys[-1], probabilities, 11)


def client():
    return TestClient(create_app(FakeBackend()))


def test_models_matches_official_sdk_shape():
    response = client().get("/v1/models")
    assert response.status_code == 200
    model = response.json()["models"][0]
    assert model["name"] == "fake-system-one"
    assert model["description"]
    assert len(model["release_date"]) == 10


def test_systemone_choice_response_matches_official_shape():
    response = client().post(
        "/v1/systemone",
        json={
            "model": "fake-system-one",
            "state": {"health": 2},
            "questions": {
                "action": {
                    "type": "choice",
                    "instructions": "What next?",
                    "criteria": {"attack": "Attack", "retreat": "Retreat"},
                }
            },
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body == {
        "model": "fake-system-one",
        "answers": {
            "action": {
                "type": "choice",
                "choice": "retreat",
                "confidence": 1.0,
                "probabilities": {"attack": 0.0, "retreat": 1.0},
            }
        },
        "usage": {"input_tokens": 11, "output_tokens": 0},
    }


def test_multiple_choice_questions_are_deliberately_serial_and_accounted():
    response = client().post(
        "/v1/systemone",
        json={
            "model": "fake-system-one",
            "state": "state",
            "questions": {
                "one": {"type": "choice", "criteria": {"a": None, "b": None}},
                "two": {"type": "choice", "criteria": {"x": None, "y": None}},
            },
        },
    )
    assert response.status_code == 200
    assert response.json()["usage"]["input_tokens"] == 22


def test_unsupported_question_types_fail_cleanly():
    response = client().post(
        "/v1/systemone",
        json={
            "model": "fake-system-one",
            "state": "state",
            "questions": {
                "urgency": {"type": "score", "criteria": ["low", "high"]}
            },
        },
    )
    assert response.status_code == 422


def test_wrong_model_is_rejected():
    response = client().post(
        "/v1/systemone",
        json={
            "model": "other",
            "state": "state",
            "questions": {
                "action": {"type": "choice", "criteria": {"a": None, "b": None}}
            },
        },
    )
    assert response.status_code == 422
