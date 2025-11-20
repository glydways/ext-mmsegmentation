import os
import shutil
import json
from glob import glob
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
import argparse
import random

# USAGE: python raindrops_on_windshield.py --val_subs D3
# Alternatively, you can use a random split by not specifying --val_subs. It will randomly pick 1 subfolders as validation.

RAW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/download/raindrops_on_windshield_raw'))
OUT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raindrops_on_windshield'))


def binarize(mask_path):
    mask = np.array(Image.open(mask_path).convert('L'))  # No resize, just grayscale
    mask = (mask > 127).astype(np.uint8)  # 0 or 1
    return Image.fromarray(mask)


def expand_subs(subs):
    """Expand any parent folders to all their subfolders until there are only images inside of it."""
    expanded = []
    for s in subs:
        path = os.path.join(RAW_ROOT, 'images', s)
        if os.path.isdir(path):
            children = [os.path.join(s, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            if children:
                expanded.extend(expand_subs(children))
            else:
                expanded.append(s)
        else:
            expanded.append(s)
    return expanded

def process_and_copy_files(split_name, data_folders_paths):
    """process images and masks in the data_folders_paths, and process and copy to OUT_ROOT/split_name/img and OUT_ROOT/split_name/label
    - for images, make a copy
    - for masks, do binarization and save as png"""
    img_out = os.path.join(OUT_ROOT, split_name, 'img')
    label_out = os.path.join(OUT_ROOT, split_name, 'label')
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(label_out, exist_ok=True)

    for data_folder_path in tqdm(data_folders_paths, desc=f'Processing {split_name}'):
        img_path = os.path.join(RAW_ROOT, 'images', data_folder_path)
        mask_path = os.path.join(RAW_ROOT, 'masks', data_folder_path)
        # Process images
        for fname in tqdm(os.listdir(img_path), desc=f'Processing images in {data_folder_path}'):
            if fname.endswith('.png'):
                src = os.path.join(img_path, fname)
                dst = os.path.join(img_out, f"{os.path.splitext(fname)[0]}_img.png")
                # print(f'Copying {src} to {dst}')
                shutil.copy2(src, dst)
        # Process masks
        for fname in tqdm(os.listdir(mask_path), desc=f'Processing masks in {data_folder_path}'):
            if fname.endswith('.png'):
                src = os.path.join(mask_path, fname)
                dst = os.path.join(label_out, f"{os.path.splitext(fname)[0]}_label.png")
                # print(f'Binarizing {src} and saving to {dst}')
                binarize(src).save(dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_subs', type=str, default='', help='Comma-separated list of subfolders to use as validation (e.g., D3 or D3_0,D3_1)')
    args = parser.parse_args()

    all_subfolders = [d for d in os.listdir(os.path.join(RAW_ROOT, 'images')) if os.path.isdir(os.path.join(RAW_ROOT, 'images', d))]
    print(all_subfolders) # ['D3', 'D1', 'D2', '8o', 'v2', 'D4', '2o', '1o']

    if args.val_subs:
        # collect all subfolders and their files in the val_subs
        val_subs = [s.strip() for s in args.val_subs.split(',') if s.strip()]
        train_subs = [s for s in all_subfolders if s not in val_subs]
        val_subs = expand_subs(val_subs)
        train_subs = expand_subs(train_subs)

        print(val_subs) # ['D3/D3_2', 'D3/D3_1', 'D3/D3_0']
        print(train_subs) # ['D1', 'D2', '8o', 'v2', 'D4/D4_0', 'D4/D4_1', '2o', '1o']
    else:
        print("Using random split.")
        val_subs = random.sample(all_subfolders, 1)
        train_subs = [s for s in all_subfolders if s not in val_subs]
        val_subs = expand_subs(val_subs)
        train_subs = expand_subs(train_subs)
        print(val_subs)
        print(train_subs)

    process_and_copy_files('train', train_subs)
    process_and_copy_files('val', val_subs)

    print('Done!')

if __name__ == '__main__':
    main()
