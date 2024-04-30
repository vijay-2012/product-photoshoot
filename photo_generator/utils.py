import gc
import os
import random
import numpy as np
import json
import torch
from PIL import Image, PngImagePlugin
from datetime import datetime
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
)

MAX_SEED = np.iinfo(np.int32).max

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def seed_everything(seed: int) -> torch.Generator:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator

def parse_aspect_ratio(aspect_ratio: str) -> Optional[Tuple[int, int]]:
    if aspect_ratio == "Custom":
        return None
    width, height = aspect_ratio.split(" x ")
    return int(width), int(height)

def aspect_ratio_handler(aspect_ratio: str, custom_width: int, custom_height: int) -> Tuple[int, int]:
    if aspect_ratio == "Custom":
        return custom_width, custom_height
    else:
        width, height = parse_aspect_ratio(aspect_ratio)
        return width, height

def get_scheduler(scheduler_config: Dict, name: str) -> Optional[Callable]:
    scheduler_factory_map = {
        "DPM++ 2M Karras": lambda: DPMSolverMultistepScheduler.from_config(scheduler_config, use_karras_sigmas=True),
        "DPM++ SDE Karras": lambda: DPMSolverSinglestepScheduler.from_config(scheduler_config, use_karras_sigmas=True),
        "DPM++ 2M SDE Karras": lambda: DPMSolverMultistepScheduler.from_config(scheduler_config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++"),
        "Euler": lambda: EulerDiscreteScheduler.from_config(scheduler_config),
        "Euler a": lambda: EulerAncestralDiscreteScheduler.from_config(scheduler_config),
        "DDIM": lambda: DDIMScheduler.from_config(scheduler_config),
    }
    return scheduler_factory_map.get(name, lambda: None)()

def free_memory() -> None:
    torch.cuda.empty_cache()
    gc.collect()

def common_upscale(samples: torch.Tensor, width: int, height: int, upscale_method: str) -> torch.Tensor:
    return torch.nn.functional.interpolate(samples, size=(height, width), mode=upscale_method)

def upscale(samples: torch.Tensor, upscale_method: str, scale_by: float) -> torch.Tensor:
    width = round(samples.shape[3] * scale_by)
    height = round(samples.shape[2] * scale_by)
    return common_upscale(samples, width, height, upscale_method)

def preprocess_image_dimensions(width, height):
    if width % 8 != 0:
        width = width - (width % 8)
    if height % 8 != 0:
        height = height - (height % 8)
    return width, height

def save_image(image, metadata, output_dir):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    filename = f"image_{current_time}.png"
    filepath = os.path.join(output_dir, filename)

    metadata_str = json.dumps(metadata)
    info = PngImagePlugin.PngInfo()
    info.add_text("metadata", metadata_str)
    image.save(filepath, "PNG", pnginfo=info)
    return filepath

def is_google_colab():
    try:
        import google.colab
        return True
    except:
        return False