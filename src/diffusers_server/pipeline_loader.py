"""Pipeline loading: resolves a model id to a diffusers Pipeline instance.

Phase 1 supports FLUX.1-schnell (full and GGUF variants). Add more models by
extending KNOWN_MODELS or falling through to AutoPipelineForText2Image.
"""
import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    """How to load a given model id."""
    kind: str           # "flux-gguf" | "flux" | "auto"
    base_repo: Optional[str] = None       # base HF repo for non-transformer parts (GGUF)
    gguf_filename: Optional[str] = None   # filename within repo when kind=*-gguf


KNOWN_MODELS: dict[str, ModelSpec] = {
    "unsloth/FLUX.1-schnell-GGUF": ModelSpec(
        kind="flux-gguf",
        base_repo="black-forest-labs/FLUX.1-schnell",
        gguf_filename="flux1-schnell-Q4_K_M.gguf",
    ),
    "city96/FLUX.1-schnell-gguf": ModelSpec(
        kind="flux-gguf",
        base_repo="black-forest-labs/FLUX.1-schnell",
        gguf_filename="flux1-schnell-Q4_K_S.gguf",
    ),
    "black-forest-labs/FLUX.1-schnell": ModelSpec(kind="flux"),
}


def resolve(model_id: str, gguf_filename_override: Optional[str] = None,
            base_repo_override: Optional[str] = None) -> ModelSpec:
    """Resolve a model id to a ModelSpec.

    Falls back to kind=auto for unrecognized ids (uses AutoPipelineForText2Image).
    Explicit overrides let callers point at unknown GGUF repos without code changes.
    """
    spec = KNOWN_MODELS.get(model_id)
    if spec is None:
        if "gguf" in model_id.lower() or (gguf_filename_override is not None):
            spec = ModelSpec(
                kind="flux-gguf",
                base_repo=base_repo_override,
                gguf_filename=gguf_filename_override,
            )
        else:
            spec = ModelSpec(kind="auto")

    # Apply overrides
    if gguf_filename_override:
        spec.gguf_filename = gguf_filename_override
    if base_repo_override:
        spec.base_repo = base_repo_override
    return spec


def load_pipeline(model_id: str, spec: ModelSpec, torch_dtype_str: str = "bfloat16",
                  device: str = "cuda"):
    """Load and return a diffusers Pipeline ready for inference."""
    import torch
    from huggingface_hub import hf_hub_download

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[torch_dtype_str]

    if spec.kind == "flux-gguf":
        from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig

        if not spec.gguf_filename or not spec.base_repo:
            raise ValueError(
                f"flux-gguf needs gguf_filename + base_repo "
                f"(got {spec.gguf_filename=}, {spec.base_repo=}). "
                f"Pass --gguf-file and --base-repo or use a KNOWN_MODELS id."
            )

        logger.info("Downloading GGUF transformer %s/%s", model_id, spec.gguf_filename)
        gguf_path = hf_hub_download(repo_id=model_id, filename=spec.gguf_filename)

        logger.info("Loading FluxTransformer2DModel from %s", gguf_path)
        transformer = FluxTransformer2DModel.from_single_file(
            gguf_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
            torch_dtype=dtype,
        )

        logger.info("Loading FluxPipeline base from %s", spec.base_repo)
        pipe = FluxPipeline.from_pretrained(
            spec.base_repo,
            transformer=transformer,
            torch_dtype=dtype,
        )
    elif spec.kind == "flux":
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)
    else:
        from diffusers import AutoPipelineForText2Image
        pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=dtype)

    logger.info("Moving pipeline to device=%s", device)
    pipe.to(device)

    # Strix Halo has 128GB unified; full residency is fine. For smaller GPUs we
    # would call pipe.enable_model_cpu_offload() — leave that to a future flag.
    return pipe


def default_inference_kwargs(spec: ModelSpec) -> dict:
    """Per-model defaults for steps and guidance, used when caller doesn't specify."""
    if spec.kind.startswith("flux"):
        # FLUX.1-schnell is a 4-step distilled model, guidance=0
        return {"num_inference_steps": 4, "guidance_scale": 0.0}
    return {}
