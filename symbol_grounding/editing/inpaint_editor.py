"""Diffusers-based inpainting editor implementation."""
from __future__ import annotations

from dataclasses import dataclass

from .interface import EditRequest, EditResult, Editor

try:
    from PIL import Image  # type: ignore
    _PIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PIL_AVAILABLE = False


@dataclass
class InpaintEditor(Editor):
    """Inpainting editor using StableDiffusionInpaintPipeline."""

    def edit(self, request: EditRequest) -> EditResult:
        if not _PIL_AVAILABLE:
            raise ImportError("Pillow is required for inpainting.")

        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise ImportError("PyTorch is required to run diffusers pipelines.") from exc

        try:
            from diffusers import StableDiffusionInpaintPipeline
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "diffusers is required. Install diffusers, transformers, accelerate, safetensors."
            ) from exc

        device = _select_device(request.device)
        torch_dtype = _torch_dtype_for_device(device)

        try:
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                request.model_id,
                torch_dtype=torch_dtype,
                cache_dir=request.cache_dir,
                local_files_only=request.local_files_only,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load inpainting pipeline. Check model IDs, network access, or use --local-files-only."
            ) from exc

        pipe.to(device)

        image = Image.open(request.image_path).convert("RGB")
        if request.mask_image is not None:
            mask = request.mask_image.convert("RGB")
        elif request.mask_path:
            mask = Image.open(request.mask_path).convert("RGB")
        else:
            raise ValueError("mask_path or mask_image must be provided for inpainting.")

        if request.width and request.height:
            image = image.resize((request.width, request.height))
            mask = mask.resize((request.width, request.height))
        else:
            request.width, request.height = image.size

        generator = torch.Generator(device=device).manual_seed(request.seed)
        if request.base_prompt:
            prompt = f"{request.base_prompt}, {request.edit_prompt}"
        else:
            prompt = request.edit_prompt

        try:
            result = pipe(
                prompt=prompt,
                image=image,
                mask_image=mask,
                negative_prompt=request.negative_prompt,
                num_inference_steps=request.steps,
                guidance_scale=request.guidance_scale,
                height=request.height,
                width=request.width,
                generator=generator,
            )
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" in message:
                raise RuntimeError(
                    "CUDA out of memory. Try smaller --height/--width or fewer --steps."
                ) from exc
            raise

        return EditResult(image=result.images[0], seed=request.seed)


def _select_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda"):
        return requested if torch.cuda.is_available() else "cpu"
    return requested


def _torch_dtype_for_device(device: str):
    import torch

    if device.startswith("cuda"):
        return torch.float16
    return torch.float32


__all__ = ["InpaintEditor"]
