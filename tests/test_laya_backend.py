from systemone_local.backends.laya import LayaBackend


class FakeAgent:
    def predict(self, state, questions):
        assert questions["action"]["type"] == "choice"
        return {
            "answers": {
                "action": {
                    "choice": "retreat",
                    "probabilities": {"attack": 0.2, "retreat": 0.8},
                }
            },
            "usage": {"input_tokens": 42, "output_tokens": 0},
        }


def test_laya_native_choice_is_normalized():
    backend = LayaBackend(FakeAgent(), name="laya")
    result = backend.choice({}, "Choose", {"attack": "Attack", "retreat": "Retreat"})
    assert result.choice == "retreat"
    assert result.probabilities == {"attack": 0.2, "retreat": 0.8}
    assert result.input_tokens == 42
