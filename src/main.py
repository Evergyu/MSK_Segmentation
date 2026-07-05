import os
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import segmentation_models_pytorch as smp
from utils.visualization import create_color_map, save_plot, load_class_info

DATA_ROOT = './images'
CHECKPOINT = './checkpoints/checkpoint.pth'
OUTPUT_DIR = './results'
CLASS_INFO_CSV = './images/class_info.csv'

ARCH = 'upernet'
ENCODER = 'timm-efficientnet-b1'
BATCH_SIZE = 1
INPUT_SIZE = (512, 512)


class SegmentationModel(torch.nn.Module):
    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

    def forward(self, image):
        image = (image - self.mean) / self.std
        return self.model(image)


def find_images(data_root):
    data_root = Path(data_root)
    exts = ('*.jpg', '*.jpeg', '*.png')
    image_files = sorted(f for e in exts for f in data_root.glob(e))
    if not image_files:  # look under images/, recursing into case/plane sub-folders
        base = (data_root / 'images') if (data_root / 'images').is_dir() else data_root
        image_files = sorted(f for e in exts for f in base.rglob(e))
    print(f"Found {len(image_files)} images")
    return image_files


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(OUTPUT_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    image_files = find_images(DATA_ROOT)
    num_classes, class_names = load_class_info(CLASS_INFO_CSV)

    print(f"Loading {ARCH} ({ENCODER}) with {num_classes} classes")
    model = SegmentationModel(arch=ARCH, encoder_name=ENCODER, in_channels=3, out_classes=num_classes)
    
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    color_map = create_color_map(num_classes, csv_path=CLASS_INFO_CSV)
    print(f"Start visualization -> {save_dir}")
    
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(image_files), BATCH_SIZE), desc="Visualizing"):
            batch_end = min(batch_start + BATCH_SIZE, len(image_files))
            batch_files = image_files[batch_start:batch_end]
            
            batch_images, batch_orig_images, batch_orig_shapes = [], [], []
            
            for img_path in batch_files:
                orig_image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
                orig_h, orig_w = orig_image.shape[:2]
                
                image = cv2.resize(orig_image, INPUT_SIZE)
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                
                batch_images.append(image)
                batch_orig_images.append(orig_image)
                batch_orig_shapes.append((orig_h, orig_w))
            
            batch_images = torch.stack(batch_images).to(device)
            logits = model(batch_images)
            preds = logits.softmax(dim=1).argmax(dim=1).cpu().numpy()
            
            for i, img_path in enumerate(batch_files):
                orig_h, orig_w = batch_orig_shapes[i]
                # prediction runs at fixed 512x512, then is upscaled back to the
                # image's native resolution (nearest-neighbour keeps class indices)
                pred_mask = cv2.resize(preds[i].astype('float32'), (orig_w, orig_h),
                                      interpolation=cv2.INTER_NEAREST).astype('int')

                # keep case/plane in the output name so nested slices don't collide
                try:
                    rel = img_path.relative_to(DATA_ROOT)
                    stem = '_'.join(rel.with_suffix('').parts)
                except ValueError:
                    stem = img_path.stem
                save_name = save_dir / f"{stem}_visualization.png"
                save_plot(batch_orig_images[i], pred_mask, class_names, color_map, save_name)

    print(f"Done. {len(image_files)} images saved.")


if __name__ == '__main__':
    main()