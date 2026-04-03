"""Evaluation script for multi-organ abdominal MRI segmentation (CHAOS).

Loads a trained checkpoint and reports:
  - Per-organ Dice and IoU on the test set
  - Mean foreground Dice and IoU
  - Prediction grid (MRI slice / ground truth / prediction side-by-side)

TTA: horizontal flip — averages softmax probabilities before argmax.

Usage:
    cd src
    python evaluate.py --checkpoint ../results/best_model.pth \
                       --data-dir /path/to/CHAOS_Train_Sets/Train_Sets \
                       --output-dir ../results
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset import build_dataloaders, NUM_CLASSES, ORGAN_NAMES
from model import create_model


# Colour palette for organ visualisation (RGB)
PALETTE = np.array([
    [0,   0,   0  ],   # 0 background    — black
    [255, 80,  80 ],   # 1 liver         — red
    [80,  180, 80 ],   # 2 right kidney  — green
    [80,  80,  255],   # 3 left kidney   — blue
    [255, 200, 80 ],   # 4 spleen        — orange
], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(
    preds:   torch.Tensor,
    targets: torch.Tensor,
    eps:     float = 1e-6,
) -> dict[str, dict[str, float]]:
    """Per-organ Dice and IoU on non-empty targets only."""
    results = {}
    for c in range(1, NUM_CLASSES):
        pred_c   = (preds   == c).float()
        target_c = (targets == c).float()

        intersection = (pred_c * target_c).sum(dim=(1, 2))
        union_dice   = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))
        union_iou    = union_dice - intersection

        dice = (2.0 * intersection + eps) / (union_dice + eps)
        iou  = (intersection + eps)       / (union_iou  + eps)

        present = target_c.sum(dim=(1, 2)) > 0
        if present.any():
            results[ORGAN_NAMES[c]] = {
                "dice": dice[present].mean().item(),
                "iou":  iou[present].mean().item(),
            }
        else:
            results[ORGAN_NAMES[c]] = {"dice": 0.0, "iou": 0.0}

    return results


# ---------------------------------------------------------------------------
# TTA inference
# ---------------------------------------------------------------------------
def predict_with_tta(
    model: torch.nn.Module, images: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Horizontal-flip TTA — averages softmax probs, returns argmax class map."""
    images = images.to(device)
    with torch.no_grad():
        probs = (
            torch.softmax(model(images), dim=1) +
            torch.softmax(torch.flip(model(torch.flip(images, dims=[-1])),
                                     dims=[-1]), dim=1)
        ) / 2.0
    return probs.argmax(dim=1)


# ---------------------------------------------------------------------------
# Colourised mask
# ---------------------------------------------------------------------------
def colorise(mask: np.ndarray) -> np.ndarray:
    """Convert integer class mask [H, W] to RGB image [H, W, 3]."""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        rgb[mask == c] = PALETTE[c]
    return rgb


# ---------------------------------------------------------------------------
# Prediction grid
# ---------------------------------------------------------------------------
def save_prediction_grid(
    images: torch.Tensor, targets: torch.Tensor, preds: torch.Tensor,
    output_path: Path, n_samples: int = 6,
) -> None:
    """Save a grid of N samples: MRI slice | ground truth | prediction."""
    n = min(n_samples, images.size(0))
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), tight_layout=True)
    if n == 1:
        axes = axes[np.newaxis, :]

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for i in range(n):
        img  = images[i].cpu() * std + mean
        img  = img[0].numpy()
        img  = np.clip(img, 0, 1)
        gt   = colorise(targets[i].cpu().numpy())
        pred = colorise(preds[i].cpu().numpy())

        axes[i, 0].imshow(img,  cmap="gray"); axes[i, 0].set_title("MRI slice")
        axes[i, 1].imshow(gt);                axes[i, 1].set_title("Ground truth")
        axes[i, 2].imshow(pred);              axes[i, 2].set_title("Prediction")
        for ax in axes[i]:
            ax.axis("off")

    handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[c] / 255)
               for c in range(1, NUM_CLASSES)]
    labels  = [ORGAN_NAMES[c] for c in range(1, NUM_CLASSES)]
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02))

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Prediction grid saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate CHAOS multi-organ MRI segmentation model")
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--data-dir",    type=str, default=None)
    parser.add_argument("--output-dir",  type=str, default="../results")
    parser.add_argument("--batch-size",  type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--no-tta",      action="store_true",
                        help="Disable test-time augmentation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    print(f"Loaded checkpoint — epoch {ckpt.get('epoch', '?')} | "
          f"val Dice: {ckpt.get('val_dice', 0):.4f}")

    model = create_model(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    _, _, test_loader = build_dataloaders(
        data_root   = args.data_dir,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        seed        = args.seed,
    )

    use_tta = not args.no_tta
    print(f"TTA: {'enabled' if use_tta else 'disabled'}")

    all_metrics: dict[str, list[float]] = {
        organ: [] for organ in ORGAN_NAMES.values() if organ != "background"
    }
    grid_images = grid_targets = grid_preds = None

    for batch_idx, (images, masks) in enumerate(test_loader):
        masks = masks.to(device)
        if use_tta:
            preds = predict_with_tta(model, images, device)
        else:
            with torch.no_grad():
                preds = model(images.to(device)).argmax(dim=1)

        for organ, vals in compute_metrics(preds, masks).items():
            if vals["dice"] > 0:
                all_metrics[organ].append(vals["dice"])

        if batch_idx == 0:
            grid_images  = images
            grid_targets = masks.cpu()
            grid_preds   = preds.cpu()

    # Print results
    print("\n" + "=" * 40)
    print(f"{'Organ':<15} {'Dice':>8} {'IoU':>8}")
    print("-" * 40)
    mean_dices = []
    for organ, scores in all_metrics.items():
        d = float(np.mean(scores)) if scores else 0.0
        mean_dices.append(d)
        print(f"{organ:<15} {d:>8.4f}")
    print("-" * 40)
    print(f"{'Mean (organs)':<15} {np.mean(mean_dices):>8.4f}")
    print("=" * 40)

    if grid_images is not None:
        save_prediction_grid(
            grid_images, grid_targets, grid_preds,
            output_path=output_dir / "predictions.png",
        )


if __name__ == "__main__":
    main()
