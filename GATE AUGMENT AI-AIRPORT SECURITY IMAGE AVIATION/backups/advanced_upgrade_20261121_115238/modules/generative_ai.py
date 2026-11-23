from __future__ import annotations

from importlib.util import find_spec

import numpy as np
from PIL import Image


def diffusion_availability() -> tuple[bool, str]:
    missing = [name for name in ("torch", "diffusers") if find_spec(name) is None]
    if missing:
        return False, f"Diffusion is unavailable: install optional packages {', '.join(missing)}."
    return True, "Diffusers and PyTorch are installed; model weights may still need to be downloaded."


def generate_with_diffusion(image: np.ndarray, *, model_id: str, prompt: str,
                            strength: float = 0.25, seed: int | None = None) -> np.ndarray:
    """Run a real Diffusers image-to-image pipeline, downloading model weights if needed."""
    if not model_id.strip():
        raise ValueError("Enter a Hugging Face Diffusers model ID.")
    if not 0.0 < strength <= 1.0:
        raise ValueError("Diffusion strength must be greater than 0 and at most 1.")
    try:
        import torch
        from diffusers import StableDiffusionImg2ImgPipeline
    except ImportError as exc:
        raise RuntimeError("Diffusion requires optional packages. Install torch and diffusers to enable this feature.") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipeline = None
    try:
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id, torch_dtype=dtype,
        )
        pipeline = pipeline.to(device)
        if hasattr(pipeline, "enable_attention_slicing"):
            pipeline.enable_attention_slicing()
        height, width = image.shape[:2]
        # Stable Diffusion expects dimensions divisible by 8.
        width = max(64, (width // 8) * 8)
        height = max(64, (height // 8) * 8)
        source = Image.fromarray(image).resize((width, height), Image.Resampling.LANCZOS)
        generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
        result = pipeline(
            prompt=prompt, image=source, strength=float(strength),
            guidance_scale=5.0, generator=generator,
        ).images[0]
        del pipeline
        return np.asarray(result.convert("RGB"), dtype=np.uint8)
    except RuntimeError as exc:
        if pipeline is not None:
            del pipeline
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if "out of memory" in str(exc).casefold():
            raise RuntimeError(
                "Insufficient GPU memory for this diffusion model. Try a smaller model, "
                "lower-resolution input, or CPU inference."
            ) from exc
        raise RuntimeError(f"Could not run Diffusers model '{model_id}': {exc}") from exc
    except (OSError, ValueError) as exc:
        if pipeline is not None:
            del pipeline
        raise RuntimeError(f"Could not load or run Diffusers model '{model_id}': {exc}") from exc
