"""Training loop for multi-organ abdominal MRI segmentation (U-Net + ResNet34).

Loss:       Weighted CrossEntropy + multi-class Dice
Optimiser:  Adam with differential learning rates (encoder < decoder)
Scheduler:  CosineAnnealingLR
Checkpoint: saved on highest mean foreground Dice (organs only)

Usage:
    cd src
    python train.py
    python train.py --epochs 50 --batch-size 8 --output-dir ../results
"""

import argparse
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import build_dataloaders, NUM_CLASSES, ORGAN_NAMES
from model import create_model, get_parameter_groups, count_parameters


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
class MultiClassDiceLoss(nn.Module):
    """Soft multi-class Dice loss (foreground classes only)."""

    def __init__(self, num_classes: int = NUM_CLASSES, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)   # [B, C, H, W]
        dice_losses = []
        for c in range(1, self.num_classes):   # skip background
            pred_c   = probs[:, c]
            target_c = (targets == c).float()
            intersection = (pred_c * target_c).sum(dim=(1, 2))
            union        = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))
            dice         = (2.0 * intersection + self.eps) / (union + self.eps)
            dice_losses.append((1.0 - dice).mean())
        return torch.stack(dice_losses).mean()


class CombinedLoss(nn.Module):
    """CrossEntropy (with class weights) + multi-class Dice."""

    def __init__(self, class_weights: torch.Tensor | None = None):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = MultiClassDiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, targets) + self.dice(logits, targets)


# ---------------------------------------------------------------------------
# Dice metric
# ---------------------------------------------------------------------------
def dice_per_organ(
    logits:  torch.Tensor,
    targets: torch.Tensor,
    eps:     float = 1e-6,
) -> dict[str, float]:
    """Per-organ Dice on non-empty targets only."""
    preds = logits.argmax(dim=1)
    scores = {}
    for c in range(1, NUM_CLASSES):
        pred_c   = (preds   == c).float()
        target_c = (targets == c).float()
        intersection = (pred_c * target_c).sum(dim=(1, 2))
        union        = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))
        per_sample   = (2.0 * intersection + eps) / (union + eps)
        present = target_c.sum(dim=(1, 2)) > 0
        if present.any():
            scores[ORGAN_NAMES[c]] = per_sample[present].mean().item()
    return scores


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module, loader: DataLoader,
    criterion: nn.Module, optimiser: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimiser.zero_grad()
        loss = criterion(model(images), masks)
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def validate(
    model: nn.Module, loader: DataLoader,
    criterion: nn.Module, device: torch.device,
) -> tuple[float, float, dict[str, float]]:
    model.eval()
    total_loss  = 0.0
    organ_scores: dict[str, list[float]] = {
        name: [] for name in ORGAN_NAMES.values() if name != "background"
    }
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            total_loss += criterion(logits, masks).item() * images.size(0)
            for organ, score in dice_per_organ(logits, masks).items():
                organ_scores[organ].append(score)

    per_organ = {o: float(np.mean(s)) if s else 0.0
                 for o, s in organ_scores.items()}
    mean_dice = float(np.mean(list(per_organ.values())))
    return total_loss / len(loader.dataset), mean_dice, per_organ


# ---------------------------------------------------------------------------
# Checkpoint + plots
# ---------------------------------------------------------------------------
def save_checkpoint(model: nn.Module, path: Path, metadata: dict) -> None:
    torch.save({"state_dict": model.state_dict(), **metadata}, path)
    print(f"  Checkpoint saved → {path}")


def plot_training_curves(
    train_losses: list[float], val_losses: list[float],
    val_dices: list[float], output_path: Path,
) -> None:
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), tight_layout=True)
    ax1.plot(epochs, train_losses, label="train loss")
    ax1.plot(epochs, val_losses,   label="val loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Training and validation loss")
    ax1.legend(); ax1.grid(linestyle=":")
    ax2.plot(epochs, val_dices, color="tab:green", label="mean foreground Dice")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Dice")
    ax2.set_title("Validation Dice (organs only)")
    ax2.set_ylim(0, 1); ax2.legend(); ax2.grid(linestyle=":")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Training curves saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train U-Net on CHAOS multi-organ MRI segmentation")
    parser.add_argument("--data-dir",     type=str,   default=None)
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch-size",   type=int,   default=8)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--encoder-lr",   type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers",  type=int,   default=2)
    parser.add_argument("--output-dir",   type=str,   default="../results")
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, _ = build_dataloaders(
        data_root   = args.data_dir,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        seed        = args.seed,
    )

    model = create_model(num_classes=NUM_CLASSES).to(device)
    total, trainable = count_parameters(model)
    print(f"Parameters — total: {total:,} | trainable: {trainable:,}")

    # Per-organ class weights tuned to organ size and difficulty:
    # liver (large but low contrast) and spleen (small) get higher weights
    class_weights = torch.tensor([1.0, 8.0, 5.0, 5.0, 10.0], device=device)
    # indices:                     bg   liv  rkid  lkid  spl

    criterion = CombinedLoss(class_weights=class_weights)
    encoder_params, decoder_params = get_parameter_groups(model)
    optimiser = torch.optim.Adam([
        {"params": encoder_params, "lr": args.encoder_lr},
        {"params": decoder_params, "lr": args.lr},
    ], weight_decay=args.weight_decay)
    print(f"LR — encoder: {args.encoder_lr:.0e} | decoder: {args.lr:.0e}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=1e-7
    )

    train_losses, val_losses, val_dices = [], [], []
    best_dice  = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimiser, device)
        val_loss, val_dice, per_organ = validate(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dices.append(val_dice)

        organ_str = " | ".join(f"{k[:3]}: {v:.3f}" for k, v in per_organ.items())
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train: {train_loss:.4f} | val: {val_loss:.4f} | "
              f"Dice: {val_dice:.4f} | {organ_str}")

        if val_dice > best_dice:
            best_dice  = val_dice
            best_epoch = epoch
            save_checkpoint(model, output_dir / "best_model.pth", {
                "epoch": epoch, "val_dice": val_dice, "val_loss": val_loss,
                "per_organ": per_organ, "lr": args.lr,
                "batch_size": args.batch_size, "seed": args.seed,
            })

        save_checkpoint(model, output_dir / "last_model.pth", {
            "epoch": epoch, "val_dice": val_dice, "val_loss": val_loss,
            "per_organ": per_organ, "lr": args.lr,
            "batch_size": args.batch_size, "seed": args.seed,
        })

    print(f"\nBest mean foreground Dice: {best_dice:.4f} at epoch {best_epoch}")
    plot_training_curves(train_losses, val_losses, val_dices,
                         output_path=output_dir / "training_curves.png")


if __name__ == "__main__":
    main()
