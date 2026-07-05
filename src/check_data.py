"""Validate that a dataset folder matches the format train.py / main.py expect.

    python check_data.py --data-root /path/to/dataset --class-info /path/to/class_info.csv

Expected format (derived from the released inference code + checkpoint):
  - images: RGB .png/.jpg
  - masks:  single-channel PNG, pixel value == class index (0=background,
            1..N-1 per class_info.csv `pixel_value`); 89 classes total
  - layout: train/{images,masks} (+ optional val/{images,masks}), OR flat images/ + masks/
"""

import argparse
from collections import Counter

import cv2
import numpy as np

from dataset import resolve_splits
from utils.visualization import load_class_info

EXPECTED_CLASSES = 89


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True)
    p.add_argument('--class-info', default='./images/class_info.csv')
    p.add_argument('--val-split', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--sample', type=int, default=50, help='masks to scan for class values')
    args = p.parse_args()

    num_classes, class_names = load_class_info(args.class_info)
    print(f'class_info: {num_classes} classes'
          + ('' if num_classes == EXPECTED_CLASSES
             else f'  (WARNING: released checkpoint has {EXPECTED_CLASSES})'))

    splits = resolve_splits(args.data_root, args.val_split, args.seed)
    for name, pairs in splits.items():
        print(f'{name}: {len(pairs)} image/mask pairs')

    all_pairs = splits['train'] + splits['val']
    seen = Counter()
    bad_range, resized, aspect_warn = [], 0, []
    for img_path, mask_path in all_pairs[:args.sample]:
        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            bad_range.append((str(mask_path), 'unreadable'))
            continue
        ih, iw = img.shape[:2]
        mh, mw = mask.shape[:2]
        if (ih, iw) != (mh, mw):
            resized += 1
            # image & mask get resized to the same square at train/infer time; only
            # a different aspect ratio would actually distort the alignment
            if abs(iw / ih - mw / mh) > 0.02:
                aspect_warn.append((img_path.name, f'{iw}x{ih}', f'{mw}x{mh}'))
        vals = np.unique(mask)
        seen.update(int(v) for v in vals)
        if vals.max() >= num_classes:
            bad_range.append((mask_path.name, f'max class {int(vals.max())} >= {num_classes}'))

    present = sorted(seen)
    n = min(args.sample, len(all_pairs))
    print(f'\nscanned {n} masks')
    print(f'class indices present: {len(present)} distinct, '
          f'range [{present[0]}..{present[-1]}]' if present else 'no mask values read')
    if resized:
        print(f'{resized}/{n} pairs have image/mask at different resolution '
              f'-> resized to {args.input if hasattr(args, "input") else 512}px (fine, same aspect ratio)')
    if aspect_warn:
        print(f'\nWARNING image/mask ASPECT RATIO differs (would distort!): {aspect_warn[:5]}')
    if bad_range:
        print(f'\nWARNING problems (first few): {bad_range[:5]}')
    if not aspect_warn and not bad_range:
        print('\nOK: format looks consistent with train.py / main.py.')


if __name__ == '__main__':
    main()
