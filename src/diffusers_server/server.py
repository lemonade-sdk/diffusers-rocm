"""FastAPI app for diffusers-server.

Exposes OpenAI-compatible /v1/images/generations and /health. The pipeline
is loaded synchronously at startup; the process exits non-zero if loading
fails (Lemonade's router catches this and reports failure to the caller).
"""
import asyncio
import base64
import io
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional

from fastapi import FastAPI, HTTPException

from .models import (
    HealthResponse,
    ImageData,
    ImageGenerationRequest,
    ImageGenerationResponse,
)
from .pipeline_loader import ModelSpec, default_inference_kwargs, load_pipeline, resolve

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    model_id: str
    served_model_name: str
    torch_dtype: str = "bfloat16"
    device: str = "cuda"
    gguf_filename: Optional[str] = None
    base_repo: Optional[str] = None


def make_app(config: ServerConfig) -> FastAPI:
    state: dict = {"pipeline": None, "spec": None, "config": config}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        spec = resolve(config.model_id, config.gguf_filename, config.base_repo)
        state["spec"] = spec
        logger.info("Loading pipeline: model=%s spec=%s", config.model_id, spec)
        t0 = time.time()
        state["pipeline"] = load_pipeline(
            config.model_id, spec, config.torch_dtype, config.device
        )
        logger.info("Pipeline loaded in %.1fs", time.time() - t0)
        yield
        state["pipeline"] = None

    app = FastAPI(title="diffusers-server", lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok" if state["pipeline"] is not None else "loading",
            model=config.served_model_name,
            pipeline_loaded=state["pipeline"] is not None,
        )

    async def _generate(req: ImageGenerationRequest) -> ImageGenerationResponse:
        pipe = state["pipeline"]
        spec: ModelSpec = state["spec"]
        if pipe is None:
            raise HTTPException(status_code=503, detail="pipeline not loaded yet")

        try:
            width, height = (int(x) for x in req.size.lower().split("x"))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"invalid size {req.size!r}") from e

        kwargs = default_inference_kwargs(spec)
        if req.num_inference_steps is not None:
            kwargs["num_inference_steps"] = req.num_inference_steps
        if req.guidance_scale is not None:
            kwargs["guidance_scale"] = req.guidance_scale

        # Seeded generator for reproducibility
        if req.seed is not None:
            import torch
            kwargs["generator"] = torch.Generator(device=config.device).manual_seed(req.seed)

        if req.negative_prompt is not None:
            kwargs["negative_prompt"] = req.negative_prompt

        logger.info("Generating n=%d size=%dx%d kwargs=%s", req.n, width, height,
                    {k: v for k, v in kwargs.items() if k != "generator"})

        t0 = time.time()
        # pipe(...) is blocking and GPU-bound. Run it on a thread so the event
        # loop keeps serving /health and other concurrent requests.
        result = await asyncio.to_thread(
            pipe,
            prompt=req.prompt,
            num_images_per_prompt=req.n,
            width=width,
            height=height,
            **kwargs,
        )
        logger.info("Generated in %.2fs", time.time() - t0)

        images = []
        for img in result.images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            images.append(ImageData(b64_json=b64))

        return ImageGenerationResponse(created=int(time.time()), data=images)

    @app.post("/v1/images/generations", response_model=ImageGenerationResponse)
    async def generations_v1(req: ImageGenerationRequest):
        return await _generate(req)

    # Lemonade registers endpoints under four prefixes; serve all of them
    # so the bundled service is callable from any of Lemonade's URL shapes.
    @app.post("/v0/images/generations", response_model=ImageGenerationResponse)
    async def generations_v0(req: ImageGenerationRequest):
        return await _generate(req)

    @app.post("/api/v1/images/generations", response_model=ImageGenerationResponse)
    async def generations_api_v1(req: ImageGenerationRequest):
        return await _generate(req)

    @app.post("/api/v0/images/generations", response_model=ImageGenerationResponse)
    async def generations_api_v0(req: ImageGenerationRequest):
        return await _generate(req)

    return app
