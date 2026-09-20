# systemone-harness

A deliberately small **System One** action harness plus a local inference runtime.

The application side stays simple:

```text
state + goal + currently legal actions
                  ↓
            System One model
                  ↓
       selected action + probabilities
```

The harness uses TypeSafe's official Python SDK as the client contract. The local
runtime implements the two endpoints that contract needs:

```text
GET  /v1/models
POST /v1/systemone
```

v0.1 deliberately supports **`choice` only**. It does not contain a planner, memory
system, action executor, model registry, dynamic loading, quantization manager, or
KV-cache optimizer.

## Harness

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

```python
from systemone_harness import Action, SystemOneHarness

with SystemOneHarness.from_env() as harness:
    decision = harness.decide(
        goal="Reach safety without dying.",
        state={"health": 4, "enemy_distance": 2.5, "has_cover": True},
        actions=[
            Action("attack", "Attack the nearby enemy."),
            Action("retreat", "Move behind nearby cover."),
            Action("wait", "Do nothing for one control step."),
        ],
    )

print(decision.action.id)
print(decision.probabilities)
```

## Local causal model

Install the local server and causal backend dependencies:

```bash
pip install -e '.[local,causal]'
```

Serve one model:

```bash
systemone-local \
  --backend causal \
  --model Qwen/Qwen3.5-4B \
  --device auto
```

Then point the official SDK/harness at it:

```bash
export TYPESAFE_BASE_URL=http://127.0.0.1:8000
export TYPESAFE_API_KEY=local
export TYPESAFE_DEFAULT_MODEL=Qwen/Qwen3.5-4B
```

The causal backend does exactly one inference operation per choice question:

1. render a compact state/question/options prompt with the model's native chat template;
2. verify `A`, `B`, `C`, ... are distinct single tokens **at that exact prompt boundary**;
3. run one model forward pass with no generation;
4. read only those candidate logits at the final position;
5. softmax the candidate logits and map the winner back to the requested option ID.

There is no autoregressive decode loop and `output_tokens` is zero.

## Local Laya

```bash
pip install -e '.[local,laya]'

systemone-local \
  --backend laya \
  --model convaiinnovations/laya \
  --device auto
```

Optional Laya checkpoints can use `--subfolder multilingual` or
`--subfolder typed-decisions`. Laya uses its native `agent.predict()` path rather
than being forced through causal-LM token scoring.

## CLI harness

`examples/decision.json` contains a complete request. Run:

```bash
systemone-harness examples/decision.json
```

## Application loop

```python
while not env.done:
    decision = harness.decide(
        goal=env.goal,
        state=env.observe(),
        actions=env.legal_actions(),
    )
    env.execute(decision.action)
```

The environment owns observation, legal-action generation, execution, history, and
termination. The model owns the choice.

## What v0.1 intentionally does not do

Multiple choice questions are accepted but evaluated independently. There is no
shared-prefix KV caching yet because the intended control loop normally asks one
action question per step. There is also no persistent cross-step cache, score/noul
support, model pull/registry layer, quantization management, or hardware-specific
optimization. Those should be added only after profiling the minimal engine.

## Tests

```bash
pip install -e '.[dev]'
pytest -q
```

The causal tests compare behavior at the candidate-logit boundary with a tiny fake
model, including an adversarial tokenizer that changes tokenization at the answer
boundary. The HTTP tests verify the TypeSafe-shaped response. When `typesafe-sdk`
is installed, an end-to-end test starts the local server and calls it through
TypeSafe's official `TypeSafeClient`.
