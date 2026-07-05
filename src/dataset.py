"""Dataset discovery for the MSK lower-limb segmentation data (no torch dependency).

Handles the DataON release layout, where each case/plane is a sub-folder:

    <root>/images/<case>/<plane>/000000.jpg   (RGB CT slice)
    <root>/masks /<case>/<plane>/000000.png   (single-channel class-index mask)

`resolve_splits` returns {'train': [(img, mask), ...], 'val': [...]} from either
a predefined train/test split or a flat images/+masks/ folder (auto-split).
Images and masks are paired by their sub-path + filename stem (recursively), so
nested case/plane folders line up automatically.
"""

import os
import random
from pathlib import Path

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')
IMAGE_DIRNAMES = ('images', 'image', 'imgs', 'img', 'inputs')
MASK_DIRNAMES = ('masks', 'mask', 'labels', 'label', 'annotations', 'gt', 'targets')
# how a mask filename may differ from its image: masks/<stem><suffix>.png etc.
MASK_STEM_SUFFIXES = ('', '_mask', '_label', '_gt', '_seg', '_segmentation')


def _find_subdir(parent, candidates):
    """Return the first existing sub-directory matching one of candidates (case-insensitive)."""
    if not parent.is_dir():
        return None
    lut = {p.name.lower(): p for p in parent.iterdir() if p.is_dir()}
    for name in candidates:
        if name in lut:
            return lut[name]
    return None


def _match_mask(image_path, images_dir, masks_dir):
    """Find the mask for an image. Mirrors the image's sub-path under masks_dir first
    (handles nested plane/subject folders, e.g. images/axial/x.jpg -> masks/axial/x.png),
    then falls back to a flat stem search. Same stem, .png preferred, _mask suffix ok."""
    rel_parent = image_path.parent.relative_to(images_dir)
    stem = image_path.stem
    search_dirs = [masks_dir / rel_parent, masks_dir]  # mirrored first, then flat
    for d in search_dirs:
        for stem_suffix in MASK_STEM_SUFFIXES:
            for ext in ('.png',) + IMG_EXTS:
                cand = d / f'{stem}{stem_suffix}{ext}'
                if cand.exists():
                    return cand
    return None


def discover_pairs(images_dir, masks_dir):
    """List (image_path, mask_path) pairs (recursing into sub-folders, following
    symlinked case dirs), skipping images without a matching mask."""
    images = []
    for root, _dirs, files in os.walk(images_dir, followlinks=True):
        root = Path(root)
        images += [root / f for f in files if os.path.splitext(f)[1].lower() in IMG_EXTS]
    images.sort()
    pairs, missing = [], []
    for img in images:
        m = _match_mask(img, images_dir, masks_dir)
        (pairs if m else missing).append((img, m) if m else img)
    return pairs, missing


def _carve_val(pairs, val_split, seed):
    pairs = sorted(pairs, key=lambda p: str(p[0]))
    rng = random.Random(seed)
    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(pairs) * val_split)))
    val_idx = set(idx[:n_val])
    return {
        'train': [pairs[i] for i in range(len(pairs)) if i not in val_idx],
        'val': [pairs[i] for i in range(len(pairs)) if i in val_idx],
    }


def resolve_splits(root, val_split, seed):
    """Return {'train': [pairs], 'val': [pairs]} from either
    a pre-split layout (root/train/{images,masks}, root/test|val/...) or a flat
    layout (root/{images,masks}) auto-split by val_split."""
    root = Path(root)
    splits = {}
    # 1) pre-split: root/<split>/<images|masks>
    for split, aliases in (('train', ('train', 'training')),
                           ('val', ('val', 'valid', 'validation', 'test'))):
        sdir = _find_subdir(root, aliases)
        if sdir is None:
            continue
        idir, mdir = _find_subdir(sdir, IMAGE_DIRNAMES), _find_subdir(sdir, MASK_DIRNAMES)
        if idir and mdir:
            pairs, missing = discover_pairs(idir, mdir)
            if missing:
                print(f'[{split}] warning: {len(missing)} image(s) had no matching mask, skipped')
            if pairs:
                splits[split] = pairs
    if 'train' in splits:
        if 'val' not in splits:  # train present but no val -> carve one out
            splits = _carve_val(splits['train'], val_split, seed)
        return splits

    # 2) flat: root/<images|masks>
    idir, mdir = _find_subdir(root, IMAGE_DIRNAMES), _find_subdir(root, MASK_DIRNAMES)
    if idir and mdir:
        pairs, missing = discover_pairs(idir, mdir)
        if missing:
            print(f'warning: {len(missing)} image(s) had no matching mask, skipped')
        if not pairs:
            raise FileNotFoundError(f'no image/mask pairs found under {idir} / {mdir}')
        return _carve_val(pairs, val_split, seed)

    raise FileNotFoundError(
        f'could not find a dataset under {root}. Expected either '
        f'{root}/train/images + {root}/train/masks (+ optional test/), '
        f'or a flat {root}/images + {root}/masks.')
