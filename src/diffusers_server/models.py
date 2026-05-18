from typing import Optional, Literal
from pydantic import BaseModel, Field


class ImageGenerationRequest(BaseModel):
    """OpenAI-compatible /v1/images/generations request body."""

    prompt: str
    model: Optional[str] = None
    n: int = Field(default=1, ge=1, le=4)
    size: str = "1024x1024"
    response_format: Literal["b64_json", "url"] = "b64_json"

    # Diffusers extensions (not in OpenAI's schema)
    negative_prompt: Optional[str] = None
    num_inference_steps: Optional[int] = Field(default=None, ge=1, le=200)
    guidance_scale: Optional[float] = Field(default=None, ge=0.0, le=20.0)
    seed: Optional[int] = None


class ImageData(BaseModel):
    b64_json: Optional[str] = None
    url: Optional[str] = None
    revised_prompt: Optional[str] = None


class ImageGenerationResponse(BaseModel):
    created: int
    data: list[ImageData]


class HealthResponse(BaseModel):
    status: str
    model: Optional[str] = None
    pipeline_loaded: bool
