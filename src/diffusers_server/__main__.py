"""CLI entry point: `diffusers-server --model <hf_id> --port <p>`."""
import argparse
import logging
import sys

import uvicorn

from .server import ServerConfig, make_app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="diffusers-server",
        description="OpenAI-compatible HTTP server for HuggingFace diffusers (ROCm).",
    )
    parser.add_argument("--model", required=True,
                        help="Hugging Face model id (e.g. unsloth/FLUX.1-schnell-GGUF)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--served-model-name", default=None,
                        help="Name advertised in API responses. Defaults to --model.")
    parser.add_argument("--torch-dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda",
                        help="Torch device. 'cuda' is correct on ROCm too (HIP).")
    parser.add_argument("--gguf-file", default=None,
                        help="GGUF filename inside the HF repo (when --model is a GGUF repo)")
    parser.add_argument("--base-repo", default=None,
                        help="Base HF repo for non-transformer pipeline parts (GGUF flow)")
    parser.add_argument("--log-level", default="info",
                        choices=["debug", "info", "warning", "error"])

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = ServerConfig(
        model_id=args.model,
        served_model_name=args.served_model_name or args.model,
        torch_dtype=args.torch_dtype,
        device=args.device,
        gguf_filename=args.gguf_file,
        base_repo=args.base_repo,
    )

    app = make_app(config)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
