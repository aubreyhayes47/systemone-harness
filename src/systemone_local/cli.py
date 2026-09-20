from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve one local System One model.")
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local model path")
    parser.add_argument("--backend", choices=["causal", "laya"], required=True)
    parser.add_argument("--revision")
    parser.add_argument("--subfolder", help="Laya checkpoint subfolder")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install local server dependencies with: pip install -e '.[local]'") from exc

    if args.backend == "causal":
        if args.subfolder:
            parser.error("--subfolder is only valid with --backend laya")
        from .backends.causal import CausalBackend

        backend = CausalBackend.load(
            args.model,
            revision=args.revision,
            device=args.device,
            dtype=args.dtype,
            max_tokens=args.max_tokens,
        )
    else:
        if args.revision:
            parser.error("--revision is not yet supported by the minimal Laya loader")
        from .backends.laya import LayaBackend

        backend = LayaBackend.load(
            args.model,
            subfolder=args.subfolder,
            device=args.device,
        )

    from .server import create_app

    uvicorn.run(create_app(backend), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
