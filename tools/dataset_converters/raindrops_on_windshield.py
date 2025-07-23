import os
import shutil
import json
from glob import glob
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
import argparse

# USAGE: python raindrops_on_windshield.py --val_subs D3
# Alternatively, you can use a random split by not specifying --val_subs
# However, this is not recommended as the images inside each subfolder  are series and not random. There might be some data leakage.

RAW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/download/raindrops_on_windshield_raw'))
OUT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raindrops_on_windshield'))

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def copy_image(src, dst):
    shutil.copy2(src, dst)

def save_mask_from_polygon(img_shape, polygons, out_path):
    mask = Image.new('L', img_shape, 0)
    draw = ImageDraw.Draw(mask)
    for poly in polygons:
        x = poly['all_points_x']
        y = poly['all_points_y']
        points = list(zip(x, y))
        draw.polygon(points, outline=1, fill=1)
    mask.save(out_path)

def binarize(mask_path):
    mask = np.array(Image.open(mask_path))
    mask = (mask > 127).astype(np.uint8)  # 0 or 1
    return Image.fromarray(mask)


def collect_images(subfolders):
    """Recursively collect all (subfolder, relative_path, filename) for images in the given subfolders."""
    items = []
    for sub in subfolders:
        img_dir = os.path.join(RAW_ROOT, 'images', sub)
        for img_path in glob(os.path.join(img_dir, '**', '*.png'), recursive=True):
            rel_dir = os.path.relpath(os.path.dirname(img_path), img_dir)
            fname = os.path.basename(img_path)
            items.append((sub, rel_dir, fname))
    return items

def expand_val_subs(val_subs):
    """Expand any parent folders to all their subfolders if needed."""
    expanded = []
    for s in val_subs:
        path = os.path.join(RAW_ROOT, 'images', s)
        if os.path.isdir(path):
            children = [os.path.join(s, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            if children:
                expanded.extend(children)
            else:
                expanded.append(s)
        else:
            expanded.append(s)
    return expanded

def process(items, split_name):
    img_out = os.path.join(OUT_ROOT, split_name, 'img')
    label_out = os.path.join(OUT_ROOT, split_name, 'label')
    ensure_dir(img_out)
    ensure_dir(label_out)

    for sub, rel_dir, fname in tqdm(items, desc=f'Processing {split_name}'):
        img_dir = os.path.join(RAW_ROOT, 'images', sub)
        img_path = os.path.join(img_dir, rel_dir, fname) if rel_dir != '.' else os.path.join(img_dir, fname)
        mask_dir = os.path.join(RAW_ROOT, 'masks', sub)
        mask_path = os.path.join(mask_dir, rel_dir, fname) if rel_dir != '.' else os.path.join(mask_dir, fname)
        if rel_dir == '.':
            out_base = f"{fname[:-4]}"
        else:
            out_base = f"{rel_dir.replace(os.sep, '_')}_{fname[:-4]}"
        img_dst = os.path.join(img_out, f'{out_base}_img.png')
        label_dst = os.path.join(label_out, f'{out_base}_label.png')
        copy_image(img_path, img_dst)
        if os.path.exists(mask_path):
            binarize(mask_path).save(label_dst)
        else:
            json_path = os.path.join(RAW_ROOT, 'json', f'{sub}.json')
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    anno = json.load(f)
                meta = anno['_via_img_metadata']
                key = None
                for k, v in meta.items():
                    if v['filename'] == fname:
                        key = k
                        break
                if key is not None:
                    regions = meta[key]['regions']
                    polygons = [r['shape_attributes'] for r in regions if r['shape_attributes']['name'] == 'polygon']
                    with Image.open(img_path) as im:
                        img_shape = im.size  # (width, height)
                    save_mask_from_polygon(img_shape, polygons, label_dst)
                else:
                    print(f'Warning: No annotation for {fname} in {json_path}')
            else:
                print(f'Warning: No mask or annotation for {fname}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_subs', type=str, default='', help='Comma-separated list of subfolders to use as validation (e.g., D3 or D3_0,D3_1)')
    args = parser.parse_args()

    all_subfolders = [d for d in os.listdir(os.path.join(RAW_ROOT, 'images')) if os.path.isdir(os.path.join(RAW_ROOT, 'images', d))]
    if args.val_subs:
        val_subs = [s.strip() for s in args.val_subs.split(',') if s.strip()]
        val_subs = expand_val_subs(val_subs)
        train_subs = [s for s in all_subfolders if s not in val_subs]
    else:
        np.random.shuffle(all_subfolders)
        n_val = int(len(all_subfolders) * 0.2)
        val_subs = all_subfolders[:n_val]
        train_subs = all_subfolders[n_val:]

    print(f"Train subfolders: {train_subs}")
    print(f"Val subfolders: {val_subs}")

    train_items = collect_images(train_subs)
    val_items = collect_images(val_subs)
    print(f"Train images: {len(train_items)} | Val images: {len(val_items)}")

    process(train_items, 'train')
    process(val_items, 'val')
    print('Done!')

if __name__ == '__main__':
    main()
