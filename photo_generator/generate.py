import os
import argparse
import numpy as np
import torch
import json
import utils
import logging
from PIL import Image
from datetime import datetime
from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DESCRIPTION = "Product photoshoot"

HF_TOKEN = os.getenv("HF_TOKEN")

MIN_IMAGE_SIZE = int(os.getenv("MIN_IMAGE_SIZE", "512"))
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "2048"))
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE") == "1"
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD") == "1"

MODEL = os.getenv(
    "MODEL",
    "https://huggingface.co/Lykon/DreamShaper/blob/main/DreamShaperXL_Turbo_dpmppSdeKarras_half_pruned_6.safetensors",
)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_pipeline(model_name):
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=torch.float16,
    )
    pipeline = (
        StableDiffusionXLPipeline.from_single_file
        if MODEL.endswith(".safetensors")
        else StableDiffusionXLPipeline.from_pretrained
    )

    pipe = pipeline(
        model_name,
        vae=vae,
        torch_dtype=torch.float16,
        custom_pipeline="lpw_stable_diffusion_xl",
        use_safetensors=True,
        add_watermarker=False,
        use_auth_token=HF_TOKEN,
        variant="fp16",
    )

    pipe.to(device)
    return pipe


def generate(
    prompt,
    negative_prompt="",
    seed=0,
    custom_width=1024,
    custom_height=1024,
    guidance_scale=7.0,
    num_inference_steps=30,
    sampler="DPM++ 2M SDE Karras",
    aspect_ratio_selector="1024 x 1024",
    use_upscaler=False,
    upscaler_strength=0.55,
    upscale_by=1.5,
):
    generator = utils.seed_everything(seed)

    width, height = utils.aspect_ratio_handler(
        aspect_ratio_selector,
        custom_width,
        custom_height,
    )

    width, height = utils.preprocess_image_dimensions(width, height)

    backup_scheduler = pipe.scheduler
    pipe.scheduler = utils.get_scheduler(pipe.scheduler.config, sampler)

    if use_upscaler:
        upscaler_pipe = StableDiffusionXLImg2ImgPipeline(**pipe.components)
    metadata = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "resolution": f"{width} x {height}",
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "seed": seed,
        "sampler": sampler,
    }

    if use_upscaler:
        new_width = int(width * upscale_by)
        new_height = int(height * upscale_by)
        metadata["use_upscaler"] = {
            "upscale_method": "nearest-exact",
            "upscaler_strength": upscaler_strength,
            "upscale_by": upscale_by,
            "new_resolution": f"{new_width} x {new_height}",
        }
    else:
        metadata["use_upscaler"] = None
    logger.info(json.dumps(metadata, indent=4))

    try:
        if use_upscaler:
            latents = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                output_type="latent",
            ).images
            upscaled_latents = utils.upscale(latents, "nearest-exact", upscale_by)
            images = upscaler_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=upscaled_latents,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                strength=upscaler_strength,
                generator=generator,
                output_type="pil",
            ).images
        else:
            images = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                output_type="pil",
            ).images

        return images, metadata
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        raise
    finally:
        if use_upscaler:
            del upscaler_pipe
        pipe.scheduler = backup_scheduler
        utils.free_memory()


if torch.cuda.is_available():
    pipe = load_pipeline(MODEL)
    logger.info("Loaded on Device!")
else:
    pipe = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--prompt", type=str, help="The prompt for image generation")
    parser.add_argument("--negative_prompt", type=str, default="", help="The negative prompt")
    parser.add_argument("--seed", type=int, default=0, help="The seed for reproducibility")
    parser.add_argument("--custom_width", type=int, default=1024, help="The width of the image")
    parser.add_argument("--custom_height", type=int, default=1024, help="The height of the image")
    parser.add_argument("--guidance_scale", type=float, default=7.0, help="The guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="The number of inference steps")
    parser.add_argument("--sampler", type=str, default="DPM++ 2M SDE Karras", help="The sampler to use")
    parser.add_argument("--aspect_ratio_selector", type=str, default="1024 x 1024", help="The aspect ratio selector")
    parser.add_argument("--use_upscaler", action="store_true", help="Whether to use the upscaler")
    parser.add_argument("--upscaler_strength", type=float, default=0.55, help="The strength of the upscaler")
    parser.add_argument("--upscale_by", type=float, default=1.5, help="The upscale factor")

    args = parser.parse_args()

    images, metadata = generate(
        args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        custom_width=args.custom_width,
        custom_height=args.custom_height,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        sampler=args.sampler,
        aspect_ratio_selector=args.aspect_ratio_selector,
        use_upscaler=args.use_upscaler,
        upscaler_strength=args.upscaler_strength,
        upscale_by=args.upscale_by,
        )
    
    for i, image in enumerate(images):
        image.save(f"output_{i}.png")