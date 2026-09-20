# systemone-harness

A deliberately tiny harness for **System One control loops**.

It does exactly one interesting thing:

```text
state + goal + currently legal actions
                  ↓
            System One model
                  ↓
       selected action + probabilities
```

It uses TypeSafe's **official Python SDK** as the client/protocol contract. Today the
endpoint can be TypeSafe's hosted service. Later, a local runtime only needs to
implement the same `/v1/systemone` and `/v1/models` API shape.

The harness does **not** plan, execute actions, manage memory, or run an agent framework.
Your application owns the loop.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For TypeSafe's hosted service:

```bash
export TYPESAFE_API_KEY="..."
```

For a future local server:

```bash
export TYPESAFE_BASE_URL="http://127.0.0.1:8000"
export TYPESAFE_API_KEY="local"
export TYPESAFE_DEFAULT_MODEL="laya"
```

The official SDK currently requires an API-key value even when a custom `base_url`
is used, so a local server can simply ignore the placeholder `local` bearer token.

## Python

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

## CLI

`examples/decision.json`:

```json
{
  "goal": "Reach safety without dying.",
  "state": {
    "health": 4,
    "enemy_distance": 2.5,
    "has_cover": true
  },
  "actions": [
    {"id": "attack", "description": "Attack the nearby enemy."},
    {"id": "retreat", "description": "Move behind nearby cover."},
    {"id": "wait", "description": "Do nothing for one control step."}
  ]
}
```

Run:

```bash
systemone-harness examples/decision.json
```

Output is JSON containing the selected action, option probabilities, model, and token
usage reported by the backend.

## The application loop

The intended integration is intentionally boring:

```python
while not env.done:
    state = env.observe()
    actions = env.legal_actions()

    decision = harness.decide(
        goal=env.goal,
        state=state,
        actions=actions,
    )

    env.execute(decision.action)
```

That is the entire harness philosophy.

## Local-runtime boundary

The next project layer does not need to change this harness. It needs to serve the
official SDK contract:

- `POST /v1/systemone`
- `GET /v1/models`

Behind that HTTP boundary, we can later load Laya, SemIf/Qwen, Kev, or other System One
models with model-specific runtimes.

That keeps environment code independent from inference code.
