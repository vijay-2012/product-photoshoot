import os
from glob import glob
import argparse
import cv2
import numpy as np
import shutil
from product_filter import filter_product


def main(input_dir, labels, target_dir, non_relevant_dir, threshold):
    
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(non_relevant_dir, exist_ok=True)

    input_dir_img_paths = []
    jpg_img_paths = sorted(glob(os.path.join(input_dir, '*.jpg')))
    png_img_paths = sorted(glob(os.path.join(input_dir, '*.png')))
    input_dir_img_paths.extend(jpg_img_paths)
    input_dir_img_paths.extend(png_img_paths)
    for img_path in input_dir_img_paths:
        
        result_image = filter_product(img_path, labels, threshold)
        img_name = os.path.basename(img_path)
        if result_image is None:
            
            shutil.copyfile(img_path, os.path.join(non_relevant_dir, img_name))
        else:
            new_image_name = os.path.join(target_dir, img_name)
            cv2.imwrite(new_image_name, result_image)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter images based on product detection.')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the input directory containing images')
    parser.add_argument('--target_dir', type=str, required=True, help='Path to the target directory for relevant images')
    parser.add_argument('--non_relevant_dir', type=str, required=True, help='Path to the directory for non-relevant images')
    parser.add_argument('--labels', type=str, nargs='+', required=True, help='List of labels for product detection')
    parser.add_argument('--threshold', type=float, default=0.3, required=False, help='Confidence threshold for product detection')

    args = parser.parse_args()

    main(args.input_dir, args.labels, args.target_dir, args.non_relevant_dir, args.threshold)