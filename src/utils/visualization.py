import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def _read_class_csv(csv_path):
    # utf-8-sig strips the BOM the dataset's class_info.csv ships with
    return pd.read_csv(csv_path, encoding='utf-8-sig')


def _class_name(row):
    """Build a display name from whatever columns the CSV provides.
    The dataset ships Structure + Side (e.g. 'Pelvis' + 'Left'); older files
    may use a single 'name' column."""
    if 'name' in row and pd.notna(row['name']):
        return str(row['name'])
    if 'Structure' in row and pd.notna(row['Structure']):
        side = str(row['Side']) if 'Side' in row and pd.notna(row['Side']) else ''
        return f"{row['Structure']} ({side})".strip() if side else str(row['Structure'])
    return None


def load_class_info(csv_path):
    if os.path.exists(csv_path):
        df = _read_class_csv(csv_path)
        # order by pixel_value when present so index i maps to the right structure
        if 'pixel_value' in df.columns:
            num_classes = int(df['pixel_value'].max()) + 1
            names = ['Background'] + [f"Class_{i}" for i in range(1, num_classes)]
            for _, row in df.iterrows():
                idx = int(row['pixel_value'])
                nm = _class_name(row)
                if 0 < idx < num_classes and nm:
                    names[idx] = nm
        else:
            num_classes = len(df) + 1
            names = ['Background'] + [(_class_name(r) or f"Class_{i}")
                                      for i, (_, r) in enumerate(df.iterrows(), 1)]
        print(f"Loaded {len(df)} classes (+1 background = {num_classes} total)")
        return num_classes, names
    print("CSV not found. Using defaults.")
    return 3, ['Background', 'Class1', 'Class2']


def create_color_map(num_classes, csv_path=None):
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]

    if csv_path and os.path.exists(csv_path):
        print(f"Loading color map from {csv_path}")
        df = _read_class_csv(csv_path)
        if all(col in df.columns for col in ['pixel_value', 'R', 'G', 'B']):
            for _, row in df.iterrows():
                idx = int(row['pixel_value'])
                if 0 < idx < num_classes:
                    colors[idx] = [int(row['R']), int(row['G']), int(row['B'])]
        else:
            print("CSV columns mismatch. Using random colors.")
    else:
        print("Color CSV not found. Using random colors.")

    return colors


def save_plot(image, pred_mask, class_names, color_map, save_path, dpi=150):
    num_classes = len(color_map)
    pred_mask = np.clip(pred_mask, 0, num_classes - 1)
    
    pred_colored = color_map[pred_mask]
    overlay = (pred_colored * 0.5 + image * 0.5).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, img, title in zip(axes, [image, pred_colored, overlay], 
                               ['Original Image', 'Prediction', 'Overlay']):
        ax.imshow(img)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.axis('off')

    unique_classes = np.unique(pred_mask.flatten())
    patches = [mpatches.Patch(color=color_map[i]/255.0, label=f"{class_names[i]}") 
               for i in unique_classes if i < len(class_names)]
    
    if patches:
        fig.legend(handles=patches, loc='lower center', ncol=min(len(patches), 8), 
                   bbox_to_anchor=(0.5, 0.02), fontsize=10, frameon=False)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)