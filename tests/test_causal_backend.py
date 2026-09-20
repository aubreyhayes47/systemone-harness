from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from systemone_local.backends.causal import CausalBackend


class ByteTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant: "

    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))


class FixedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask, use_cache, return_dict, logits_to_keep=None):
        width = 1 if logits_to_keep else input_ids.shape[1]
        logits = torch.zeros((1, width, 256), device=input_ids.device)
        logits[0, -1, ord("A")] = 1.0
        logits[0, -1, ord("B")] = 3.0
        return SimpleNamespace(logits=logits)


def test_direct_choice_uses_last_position_candidate_logits():
    backend = CausalBackend(FixedModel(), ByteTokenizer(), name="fake")
    result = backend.choice(
        {"health": 3},
        "What should happen next?",
        {"attack": "Attack", "retreat": "Retreat"},
    )

    assert result.choice == "retreat"
    assert result.probabilities["retreat"] > result.probabilities["attack"]
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    assert result.input_tokens > 0


def test_boundary_retokenization_is_rejected():
    class Unstable(ByteTokenizer):
        def encode(self, text, add_special_tokens=False):
            ids = super().encode(text, add_special_tokens=add_special_tokens)
            if text.endswith("A"):
                return ids[:-2] + [255]
            return ids

    backend = CausalBackend(FixedModel(), Unstable(), name="fake")
    with pytest.raises(ValueError, match="single-token stable"):
        backend.choice({}, "Choose", {"a": "A", "b": "B"})


def test_choice_option_limit_is_explicit():
    backend = CausalBackend(FixedModel(), ByteTokenizer(), name="fake")
    criteria = {str(i): str(i) for i in range(27)}
    with pytest.raises(ValueError, match="2-26"):
        backend.choice({}, "Choose", criteria)
