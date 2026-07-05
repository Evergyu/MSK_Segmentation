"""DDP training for the Lower-Limb Musculoskeletal CT Segmentation dataset.

Dataset: "Musculoskeletal CT Segmentation Dataset of Lower Limbs for High-fidelity
Human Digital Twin" (DataON, https://doi.org/10.22711/idr/1134).
89 classes = 88 musculoskeletal structures (44 per limb x left/right) + background;
multi-planar 2D CT slices (axial / coronal / sagittal) resized to 512x512.

Launch (single node, uses all visible GPUs — e.g. 8):
    python train.py --data-root /path/to/dataset --class-info /path/to/class_info.csv

Limit GPU count with --gpus N. Launching via torchrun also works:
    torchrun --nproc_per_node=8 train.py --data-root ...

Dataset layout (auto-detected; masks are single-channel PNGs of class indices):
    data_root/
    ├── train/images/*.png|jpg   train/masks/*.png   (subject-level split)
    └── test/images/*.png|jpg    test/masks/*.png    (or val/; a flat
                                                      images/+masks/ is auto-split)

The paper benchmarks five smp architectures via --arch: unet, fpn, pspnet, manet,
upernet (UPerNet is the released checkpoint, the best performer). Setup follows the
paper: Dice+CE loss (lambda=0.5), 512x512 input.

Paper training config (arch=upernet, encoder=timm-efficientnet-b1, classes=89):
    Adam lr=2e-4 betas=(0.9,0.999) weight_decay=1e-4
    CosineAnnealingLR (eta_min=1e-6) after 5 warmup epochs, 100 epochs, batch 64
    best checkpoint kept by test IoU (early stopping, patience 20, in the paper)
Note: the paper's reported IoU/DSC/Prec/Rec are MACRO (per-class mean); its stored
metrics dict labels them micro/macro/weighted. This script logs both foreground
micro and macro each epoch and stores the full metrics dict in the checkpoint:
    {epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict,
     test_iou, metrics}
"""

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

import segmentation_models_pytorch as smp
from dataset import resolve_splits
from utils.visualization import load_class_info


def parse_args():
    p = argparse.ArgumentParser(description='MSK segmentation DDP training')
    p.add_argument('--data-root', type=str, required=True,
                   help='dataset root: either train/ + val/ (each with images/ + masks/), '
                        'or a flat images/ + masks/ that gets auto-split by --val-split')
    p.add_argument('--val-split', type=float, default=0.1,
                   help='val fraction when the data is not already split into train/val')
    p.add_argument('--class-info', type=str, default='./images/class_info.csv')
    p.add_argument('--arch', type=str, default='upernet',
                   help='smp architecture; paper benchmarks: unet, fpn, pspnet, manet, upernet')
    p.add_argument('--encoder', type=str, default='timm-efficientnet-b1')
    p.add_argument('--input-size', type=int, default=512)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--warmup-epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=8,
                   help='per-GPU batch size (paper: 64 global = 8/GPU on 8 GPUs)')
    p.add_argument('--lr', type=float, default=2e-4, help='initial lr (paper: 2e-4)')
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--eta-min', type=float, default=1e-6)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--dice-lambda', type=float, default=0.5,
                   help='lambda in L = lambda*Dice + (1-lambda)*CE (paper uses 0.5)')
    p.add_argument('--amp', action='store_true', default=True)
    p.add_argument('--no-amp', dest='amp', action='store_false')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output-dir', type=str, default='./checkpoints')
    p.add_argument('--resume', type=str, default=None,
                   help='path to a checkpoint.pth to resume from')
    p.add_argument('--val-interval', type=int, default=1)
    p.add_argument('--gpus', type=int, default=None,
                   help='number of GPUs to use (default: all visible)')
    p.add_argument('--master-port', type=int, default=29500)
    return p.parse_args()


class SegmentationModel(torch.nn.Module):
    """Same wrapper as main.py so checkpoints stay drop-in compatible."""

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


class MSKDataset(Dataset):
    """Images are RGB; masks are single-channel PNGs whose pixel value is the class
    index (0 = background, 1..N-1 per class_info.csv pixel_value) — the exact layout
    the released main.py / checkpoint expects."""

    def __init__(self, pairs, input_size, num_classes, train=False):
        if not pairs:
            raise FileNotFoundError('empty split (no image/mask pairs)')
        self.pairs = pairs
        self.input_size = input_size
        self.num_classes = num_classes
        self.train = train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise RuntimeError(f'failed to read {img_path} / {mask_path}')

        size = (self.input_size, self.input_size)
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)

        if self.train and random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask.astype(np.int64)).clamp_(0, self.num_classes - 1)
        return image, mask


def setup_distributed(local_rank, world_size, master_port):
    if local_rank is None:  # launched via torchrun
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
    else:  # launched via plain `python` -> mp.spawn
        os.environ.setdefault('MASTER_ADDR', 'localhost')
        os.environ.setdefault('MASTER_PORT', str(master_port))
        dist.init_process_group(backend='nccl', rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), local_rank, dist.get_world_size()


def build_scheduler(optimizer, args):
    cosine_epochs = max(args.epochs - args.warmup_epochs, 1)
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_epochs, eta_min=args.eta_min)


def compute_metrics(tp, fp, fn, tn):
    """Replicate the metric dict stored in the released checkpoint:
    {metric: {micro, macro, weighted, per_class: [(value, support), ...]}}
    tp/fp/fn/tn: per-class pixel counts summed over the val set (C,)."""
    eps = 1e-12
    support = (tp + fn).double()
    present = support > 0
    weights = support / support.sum().clamp(min=eps)

    def summarize(per_class, micro_val):
        macro = per_class[present].mean().item() if present.any() else 0.0
        weighted = (per_class * weights).sum().item()
        per = [(per_class[c].item() if present[c] else 0.0, support[c].item())
               for c in range(len(per_class))]
        return {'micro': micro_val, 'macro': macro, 'weighted': weighted, 'per_class': per}

    tp_d, fp_d, fn_d = tp.double(), fp.double(), fn.double()
    iou_pc = tp_d / (tp_d + fp_d + fn_d).clamp(min=eps)
    prec_pc = tp_d / (tp_d + fp_d).clamp(min=eps)
    rec_pc = tp_d / (tp_d + fn_d).clamp(min=eps)
    dsc_pc = 2 * tp_d / (2 * tp_d + fp_d + fn_d).clamp(min=eps)

    iou_micro = (tp_d.sum() / (tp_d + fp_d + fn_d).sum().clamp(min=eps)).item()
    prec_micro = (tp_d.sum() / (tp_d + fp_d).sum().clamp(min=eps)).item()
    rec_micro = (tp_d.sum() / (tp_d + fn_d).sum().clamp(min=eps)).item()
    dsc_micro = (2 * tp_d.sum() / (2 * tp_d + fp_d + fn_d).sum().clamp(min=eps)).item()

    return {
        'iou': summarize(iou_pc, iou_micro),
        'precision': summarize(prec_pc, prec_micro),
        'recall': summarize(rec_pc, rec_micro),
        'dsc': summarize(dsc_pc, dsc_micro),
    }


@torch.no_grad()
def validate(model, loader, num_classes, device):
    model.eval()
    totals = torch.zeros(4, num_classes, dtype=torch.long)
    for images, masks in loader:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        preds = model(images).argmax(dim=1)
        tp, fp, fn, tn = smp.metrics.get_stats(
            preds, masks, mode='multiclass', num_classes=num_classes)
        totals[0] += tp.sum(0).cpu()
        totals[1] += fp.sum(0).cpu()
        totals[2] += fn.sum(0).cpu()
        totals[3] += tn.sum(0).cpu()
    totals = totals.to(device)  # nccl all_reduce needs GPU tensors
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    totals = totals.cpu()
    metrics = compute_metrics(*totals)
    metrics['_nobg'] = micro_excluding_background(*totals)  # paper's headline metric
    return metrics


def micro_excluding_background(tp, fp, fn, tn, eps=1e-12):
    """Micro-averaged metrics over foreground classes only (class 0 = background dropped),
    matching the paper's evaluation protocol."""
    tp, fp, fn = tp[1:].double().sum(), fp[1:].double().sum(), fn[1:].double().sum()
    return {
        'iou': (tp / (tp + fp + fn).clamp(min=eps)).item(),
        'precision': (tp / (tp + fp).clamp(min=eps)).item(),
        'recall': (tp / (tp + fn).clamp(min=eps)).item(),
        'dsc': (2 * tp / (2 * tp + fp + fn).clamp(min=eps)).item(),
    }


def main_worker(local_rank, world_size, args):
    rank, local_rank, world_size = setup_distributed(local_rank, world_size, args.master_port)
    device = torch.device(f'cuda:{local_rank}')
    is_main = rank == 0

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)
    torch.backends.cudnn.benchmark = True

    num_classes, class_names = load_class_info(args.class_info)
    if is_main:
        print(f'world_size={world_size}  classes={num_classes}  '
              f'global_batch={args.batch_size * world_size}')

    splits = resolve_splits(args.data_root, args.val_split, args.seed)
    if is_main:
        print(f"dataset: {len(splits['train'])} train / {len(splits['val'])} val pairs")
    train_set = MSKDataset(splits['train'], args.input_size, num_classes, train=True)
    val_set = MSKDataset(splits['val'], args.input_size, num_classes, train=False)
    train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=True)
    val_sampler = DistributedSampler(val_set, shuffle=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, sampler=val_sampler,
                            num_workers=max(args.num_workers // 2, 1), pin_memory=True)

    model = SegmentationModel(arch=args.arch, encoder_name=args.encoder,
                              in_channels=3, out_classes=num_classes).to(device)
    # UPerNet leaves part of the encoder unused, so the DDP reducer needs this
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    ce_loss = torch.nn.CrossEntropyLoss()
    dice_loss = smp.losses.DiceLoss(mode='multiclass', from_logits=True)

    start_epoch, best_iou = 0, 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.module.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        best_iou = ckpt.get('test_iou', 0.0)
        if is_main:
            print(f'resumed from {args.resume} (epoch {start_epoch}, iou {best_iou:.4f})')

    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        # linear warmup for the first warmup_epochs, then cosine (stepped below)
        if epoch < args.warmup_epochs:
            warm = (epoch + 1) / args.warmup_epochs
            for g in optimizer.param_groups:
                g['lr'] = args.lr * warm

        model.train()
        train_sampler.set_epoch(epoch)
        running = torch.zeros(2, device=device)  # loss sum, batch count

        iterator = train_loader
        if is_main:
            iterator = tqdm(train_loader, desc=f'epoch {epoch + 1}/{args.epochs}', ncols=100)
        for images, masks in iterator:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', enabled=args.amp):
                logits = model(images)
                loss = (args.dice_lambda * dice_loss(logits, masks)
                        + (1 - args.dice_lambda) * ce_loss(logits, masks))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running[0] += loss.detach()
            running[1] += 1
            if is_main:
                iterator.set_postfix(loss=f'{loss.item():.4f}',
                                     lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        if epoch >= args.warmup_epochs:
            scheduler.step()

        dist.all_reduce(running, op=dist.ReduceOp.SUM)
        if is_main:
            print(f'epoch {epoch + 1}: train_loss={(running[0] / running[1]).item():.4f}')

        if (epoch + 1) % args.val_interval == 0 or epoch + 1 == args.epochs:
            metrics = validate(model, val_loader, num_classes, device)
            # primary metric = paper's micro-average over foreground classes (bg excluded)
            val_iou = metrics['_nobg']['iou']
            if is_main:
                print(f"epoch {epoch + 1}: [fg] iou={val_iou:.4f} "
                      f"dsc={metrics['_nobg']['dsc']:.4f} "
                      f"prec={metrics['_nobg']['precision']:.4f} "
                      f"rec={metrics['_nobg']['recall']:.4f} | "
                      f"[incl-bg] iou micro={metrics['iou']['micro']:.4f} "
                      f"macro={metrics['iou']['macro']:.4f}")
                state = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'test_iou': val_iou,
                    'metrics': metrics,
                }
                torch.save(state, output_dir / 'last.pth')
                if val_iou > best_iou:
                    best_iou = val_iou
                    torch.save(state, output_dir / 'checkpoint.pth')
                    print(f'new best iou {best_iou:.4f} -> {output_dir / "checkpoint.pth"}')
            dist.barrier()

    if is_main:
        print(f'done. best val iou: {best_iou:.4f}')
    dist.destroy_process_group()


def main():
    args = parse_args()
    if 'LOCAL_RANK' in os.environ:  # torchrun already spawned one process per GPU
        main_worker(None, None, args)
    else:
        world_size = args.gpus or torch.cuda.device_count()
        if world_size < 1:
            raise RuntimeError('no CUDA devices visible')
        torch.multiprocessing.spawn(main_worker, args=(world_size, args), nprocs=world_size)


if __name__ == '__main__':
    main()
