# Product Photoshoot with Stable Diffusion XL

This repository contains a Python script, `generate.py`, that leverages the Stable Diffusion XL model to generate high-quality product images using text prompts. The script is designed to be flexible and customizable, allowing you to adjust various parameters such as image resolution, guidance scale, and sampling method.

## Features

- **Text-to-Image Generation**: Generate photorealistic product images from textual descriptions using the powerful Stable Diffusion XL model.
- **Customizable Parameters**: Adjust image resolution, guidance scale, number of inference steps, and sampling method to fine-tune the generation process.
- **Upscaling**: Optionally upscale the generated images using various upscaling methods and strengths.
- **Negative Prompting**: Provide negative prompts to guide the model away from undesired image characteristics.
- **Reproducibility**: Ensure consistent results across multiple runs by specifying a seed value.

## Usage

1. Clone the repository and navigate to the project directory.
2. Set up the required environment variables (e.g., `HF_TOKEN`, `MIN_IMAGE_SIZE`, `MAX_IMAGE_SIZE`, `USE_TORCH_COMPILE`, `ENABLE_CPU_OFFLOAD`, `OUTPUT_DIR`, `MODEL`).
3. You can use the provided *script.sh* to run the script with predefined or customized arguments:


```bash
    bash script.sh
```
4. You can customize the prompt and other parameters in the *script.sh*.
5. The generated images and their metadata will be saved in the directory

## Evaluation
The generated product images have been evaluated using the Kernel Inception Distance (KID) score, which is a widely used metric for assessing the quality and diversity of generated images. The calculated KID score for the images in the *generated_product_images* folder is 0.03768, indicating that the generated images are of high quality and diverse.

## Dependencies
The script relies on the following Python libraries:

- diffusers
- numpy
- torch
- Pillow

Make sure to install these dependencies before running the script.