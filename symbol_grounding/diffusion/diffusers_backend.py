"""Diffusers backend for text-to-image generation (optional ControlNet)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class DiffusionConfig:
    """Configuration for diffusers-based generation."""

    model_id: str = "runwayml/stable-diffusion-v1-5"
    controlnet_model_id: Optional[str] = None
    use_controlnet: bool = False
    device: str = "auto"
    height: int = 512
    width: int = 512
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    seed: int = 0
    negative_prompt: Optional[str] = None
    controlnet_conditioning_scale: float = 1.0
    cache_dir: Optional[str] = None
    local_files_only: bool = False
    attention_slicing: bool = False
    cpu_offload: bool = False


def _resolve_device(device: str) -> str:
    device = device.lower()
    if device == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    if device in {"cuda", "gpu"}:
        return "cuda"
    if device.startswith("cuda"):
        return device
    if device == "cpu":
        return "cpu"
    return device


def _torch_dtype_for_device(device: str):
    import torch

    if device.startswith("cuda"):
        return torch.float16
    return torch.float32


def _load_pipeline(config: DiffusionConfig):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyTorch is required to run diffusers pipelines.") from exc

    try:
        import diffusers  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError("diffusers is required. Install diffusers, transformers, accelerate, safetensors.") from exc

    device = _resolve_device(config.device)
    if config.device != "auto" and device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested but not available. "
            "Use --device cpu to run on CPU (slow), or install CUDA-enabled PyTorch."
        )

    torch_dtype = _torch_dtype_for_device(device)

    try:
        if config.use_controlnet or config.controlnet_model_id:
            from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
            controlnet_id = config.controlnet_model_id or "lllyasviel/sd-controlnet-scribble"
            controlnet = ControlNetModel.from_pretrained(
                controlnet_id,
                torch_dtype=torch_dtype,
                cache_dir=config.cache_dir,
                local_files_only=config.local_files_only,
            )
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                config.model_id,
                controlnet=controlnet,
                torch_dtype=torch_dtype,
                cache_dir=config.cache_dir,
                local_files_only=config.local_files_only,
            )
        else:
            from diffusers import StableDiffusionPipeline

            pipe = StableDiffusionPipeline.from_pretrained(
                config.model_id,
                torch_dtype=torch_dtype,
                cache_dir=config.cache_dir,
                local_files_only=config.local_files_only,
            )

        try:
            from diffusers import DPMSolverMultistepScheduler

            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        except Exception:
            pass

        if config.attention_slicing:
            pipe.enable_attention_slicing()

        if config.cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        return pipe, device, torch_dtype
    except ImportError:
        raise
    except Exception as exc:
        raise RuntimeError(
            "Failed to load diffusers pipeline. Ensure models are available and "
            "you have access to the Hugging Face Hub (or use --local-files-only)."
        ) from exc


def generate_image(
    prompt: str,
    config: DiffusionConfig,
    control_image: Optional["object"] = None,
):
    """Generate an image with diffusers using the provided configuration."""
    if config.height % 8 != 0 or config.width % 8 != 0:
        raise ValueError("height and width must be multiples of 8 for Stable Diffusion.")

    pipe, device, _ = _load_pipeline(config)

    import torch

    generator = torch.Generator(device=device).manual_seed(config.seed)

    try:
        if config.use_controlnet or config.controlnet_model_id:
            if control_image is None:
                raise ValueError("ControlNet enabled but no control image was provided.")
            result = pipe(
                prompt=prompt,
                image=control_image,
                negative_prompt=config.negative_prompt,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                generator=generator,
                height=config.height,
                width=config.width,
                controlnet_conditioning_scale=config.controlnet_conditioning_scale,
            )
        else:
            result = pipe(
                prompt=prompt,
                negative_prompt=config.negative_prompt,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                generator=generator,
                height=config.height,
                width=config.width,
            )
        return result.images[0]
    except RuntimeError as exc:
        message = str(exc).lower()
        if "out of memory" in message:
            raise RuntimeError(
                "CUDA out of memory while generating. Try smaller --height/--width, "
                "fewer --steps, or enable --cpu-offload/--attention-slicing."
            ) from exc
        raise


def load_pipeline(config: DiffusionConfig):
    """Public helper to load the configured diffusers pipeline."""
    return _load_pipeline(config)


__all__ = ["DiffusionConfig", "generate_image", "load_pipeline"]
