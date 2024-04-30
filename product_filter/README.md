# Product Image Filter

This repository contains a Python script that filters images based on product detection using the DINO (Grounding DINO) model. The script takes an input directory containing images, and it separates the images into two directories: one for images containing relevant products, and another for non-relevant images.

## Requirements

- Python 3.6+
- PyTorch
- OpenCV
- Transformers
- segment_anything_hq

1. Clone the repository:

```bash
    git clone https://github.com/vijay-2012/product-photoshoot.git
```

2. You can install the required packages using the following command:
``` bash
    pip install -r requirements.txt
```
3. Download and place the segment-anything-hq model under *pretrained_checkpoint* fodler.
```bash
    mkdir pretrained_checkpoint
    wget https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth
    mv sam_hq_vit_l.pth pretrained_checkpoint
```
## Usage

1. Place your input images in the products_unfiltered directory.
2. Run the script:
```bash
    bash script.sh
```
This will execute the *apply_filter.py* script, which filters the images in the *products_unfiltered* directory based on the specified labels and confidence threshold. The relevant images will be saved in the *products_filtered* directory, while the non-relevant images will be saved in the *non_revelent_products* directory.

You can customize the input directory, target directories, labels, and confidence threshold by modifying the *script.sh* file or passing the appropriate arguments to the *apply_filter.py* script.

## Repository Structure

**product_segment.py:** Contains the functions for segmenting the detected products using SAM and enhancing the segmented regions.

**product_filter.py:** Implements the product detection and filtering pipeline, integrating object detection and image segmentation.

**apply_filter.py:** The main script that processes the input images and saves the filtered and enhanced images to the respective directories.

**script.sh:** A bash script that simplifies the execution of the application by providing default arguments.

**requirements.txt:** Lists the required Python packages for the application.
