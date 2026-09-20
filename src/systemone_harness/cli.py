from __future__ import annotations

import argparse
import json
from pathlib import Path

from .harness import Action, SystemOneHarness


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Choose one legal action using a System One model."
    )
    parser.add_argument("request", type=Path, help="JSON request file")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--model")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    raw = json.loads(args.request.read_text())
    actions = [
        Action(
            id=item["id"],
            description=item.get("description"),
            payload=item.get("payload"),
        )
        for item in raw["actions"]
    ]

    with SystemOneHarness.from_env(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        timeout=args.timeout,
    ) as harness:
        decision = harness.decide(
            goal=raw.get("goal"),
            state=raw["state"],
            actions=actions,
            instructions=raw.get("instructions"),
        )

    print(
        json.dumps(
            {
                "action": {
                    "id": decision.action.id,
                    "description": decision.action.description,
                    "payload": decision.action.payload,
                },
                "probabilities": decision.probabilities,
                "confidence": decision.confidence,
                "model": decision.model,
                "usage": {
                    "input_tokens": decision.input_tokens,
                    "output_tokens": decision.output_tokens,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
