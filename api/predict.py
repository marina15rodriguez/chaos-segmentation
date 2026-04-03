"""Inference pipeline for CHAOS multi-organ MRI segmentation API.

Accepts a list of DICOM file paths (one T2SPIR series), runs 2.5D inference
with horizontal-flip TTA on all slices, and returns:
  - A colour-coded prediction grid PNG (base64-encoded)
  - Per-organ presence summary

The model is loaded once at startup and reused across requests.
"""

import io
import base64
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont

# Import from src — api/ is run from project root so src/ is on the path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataset import load_dicom_slice, NUM_CLASSES, ORGAN_NAMES, IMAGENET_MEAN, IMAGENET_STD
import segmentation_models_pytorch as smp
import torch.nn as nn


def _create_model_resnet34(num_classes: int = NUM_CLASSES) -> nn.Module:
    """Create U-Net with ResNet34 encoder — matches the v1 checkpoint."""
    return smp.Unet(
        encoder_name    = "resnet34",
        encoder_weights = None,       # weights loaded from checkpoint
        in_channels     = 3,
        classes         = num_classes,
        activation      = None,
    )

# ---------------------------------------------------------------------------
# Colour palette (same as evaluate.py)
# ---------------------------------------------------------------------------
PALETTE = np.array([
    [0,   0,   0  ],   # 0 background
    [255, 80,  80 ],   # 1 liver
    [80,  180, 80 ],   # 2 right kidney
    [80,  80,  255],   # 3 left kidney
    [255, 200, 80 ],   # 4 spleen
], dtype=np.uint8)

ORGAN_COLORS = {name: tuple(PALETTE[i]) for i, name in ORGAN_NAMES.items()}

# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------
_model = None
_device = None


def load_model(checkpoint_path: str) -> None:
    """Load the model checkpoint once at startup."""
    global _model, _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model  = _create_model_resnet34(num_classes=NUM_CLASSES).to(_device)
    ckpt    = torch.load(checkpoint_path, map_location=_device, weights_only=False)
    _model.load_state_dict(ckpt["state_dict"])
    _model.eval()
    print(f"Model loaded from {checkpoint_path} on {_device}")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess_slice(
    dcm_paths: list[Path], slice_idx: int
) -> torch.Tensor:
    """Load a single slice and return normalised tensor [1,3,H,W].

    Uses greyscale repeated 3× to match the v1 (ResNet34) checkpoint.
    """
    arr   = load_dicom_slice(dcm_paths[slice_idx])
    img   = Image.fromarray(arr, mode="L")
    img   = TF.resize(img, [256, 256], interpolation=Image.BILINEAR)
    image = TF.to_tensor(img)                                # [1, H, W]
    image = image.repeat(3, 1, 1)                            # [3, H, W]
    image = TF.normalize(image, IMAGENET_MEAN, IMAGENET_STD)
    return image.unsqueeze(0)                                # [1, 3, H, W]


# ---------------------------------------------------------------------------
# TTA inference
# ---------------------------------------------------------------------------
def predict_slice(tensor: torch.Tensor) -> np.ndarray:
    """Run TTA inference on a single [1,3,H,W] tensor. Returns [H,W] class map."""
    tensor = tensor.to(_device)
    with torch.no_grad():
        probs = (
            torch.softmax(_model(tensor), dim=1) +
            torch.softmax(
                torch.flip(_model(torch.flip(tensor, dims=[-1])), dims=[-1]),
                dim=1)
        ) / 2.0
    return probs.argmax(dim=1).squeeze(0).cpu().numpy()   # [H, W]


# ---------------------------------------------------------------------------
# Colourised overlay
# ---------------------------------------------------------------------------
def colorise(mask: np.ndarray) -> np.ndarray:
    """Convert integer class mask [H,W] → RGB [H,W,3]."""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        rgb[mask == c] = PALETTE[c]
    return rgb


def make_overlay(gray: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend greyscale CT slice with colour mask overlay."""
    gray_rgb = np.stack([gray, gray, gray], axis=-1)   # [H, W, 3]
    color    = colorise(mask)
    bg       = mask == 0
    overlay  = gray_rgb.copy()
    overlay[~bg] = (
        (1 - alpha) * gray_rgb[~bg] + alpha * color[~bg]
    ).astype(np.uint8)
    return overlay


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------
def run_prediction(zip_bytes: bytes) -> dict:
    """Run full-series inference from a ZIP of DICOM files.

    Args:
        zip_bytes: raw bytes of the uploaded ZIP file

    Returns:
        dict with:
            'grid_png'     : base64-encoded PNG prediction grid
            'organs_found' : list of organ names detected in the series
            'n_slices'     : total number of slices processed
    """
    # Extract ZIP to temp directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            zf.extractall(tmp_path)

        # Find all DICOM files (sorted)
        dcm_paths = sorted(tmp_path.rglob("*.dcm"))
        if not dcm_paths:
            raise ValueError("No .dcm files found in the uploaded ZIP.")

        # Run inference on all slices
        all_masks   = []
        all_grays   = []
        organs_seen = set()

        for i, dcm_path in enumerate(dcm_paths):
            tensor = preprocess_slice(dcm_paths, i)
            mask   = predict_slice(tensor)
            gray   = load_dicom_slice(dcm_path)

            all_masks.append(mask)
            all_grays.append(gray)

            for c in range(1, NUM_CLASSES):
                if (mask == c).any():
                    organs_seen.add(ORGAN_NAMES[c])

        # Select up to 8 most informative slices (most organ pixels)
        organ_counts = [np.sum(m > 0) for m in all_masks]
        top_indices  = sorted(
            range(len(organ_counts)),
            key=lambda i: organ_counts[i],
            reverse=True
        )[:8]
        top_indices  = sorted(top_indices)   # restore slice order

        # Build prediction grid
        grid_img = _build_grid(
            [all_grays[i]  for i in top_indices],
            [all_masks[i]  for i in top_indices],
            [i             for i in top_indices],
            len(dcm_paths),
        )

        # Encode to base64 PNG
        buf = io.BytesIO()
        grid_img.save(buf, format="PNG")
        grid_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "grid_png":     grid_b64,
        "organs_found": sorted(list(organs_seen)),
        "n_slices":     len(dcm_paths),
    }


def _build_grid(
    grays:       list[np.ndarray],
    masks:       list[np.ndarray],
    slice_indices: list[int],
    total_slices:  int,
) -> Image.Image:
    """Build a 2-row grid: top = original slices, bottom = overlays."""
    n      = len(grays)
    cell_w = 256
    cell_h = 256
    pad    = 4
    label_h = 20

    grid_w = n * (cell_w + pad) + pad
    grid_h = 2 * (cell_h + label_h + pad) + pad + 60   # +60 for legend

    grid = Image.new("RGB", (grid_w, grid_h), color=(30, 30, 30))

    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(grid)

    for col, (gray, mask, sidx) in enumerate(zip(grays, masks, slice_indices)):
        x = pad + col * (cell_w + pad)

        # Row 1: original slice
        gray_img = Image.fromarray(gray).convert("RGB").resize((cell_w, cell_h))
        y1 = pad
        draw.text((x, y1), f"Slice {sidx}/{total_slices-1}", fill=(200, 200, 200), font=font)
        grid.paste(gray_img, (x, y1 + label_h))

        # Row 2: overlay
        overlay = make_overlay(
            np.array(Image.fromarray(gray).resize((cell_w, cell_h))),
            np.array(Image.fromarray(mask.astype(np.uint8)).resize(
                (cell_w, cell_h), Image.NEAREST)),
        )
        overlay_img = Image.fromarray(overlay)
        y2 = pad + cell_h + label_h + pad
        draw.text((x, y2), "Prediction", fill=(200, 200, 200), font=font)
        grid.paste(overlay_img, (x, y2 + label_h))

    # Legend at bottom
    legend_y = grid_h - 50
    draw.text((pad, legend_y), "Organs: ", fill=(255, 255, 255), font=font)
    lx = pad + 60
    for c in range(1, NUM_CLASSES):
        color = tuple(PALETTE[c])
        draw.rectangle([lx, legend_y + 2, lx + 14, legend_y + 14], fill=color)
        draw.text((lx + 18, legend_y), ORGAN_NAMES[c], fill=(255, 255, 255), font=font)
        lx += 120

    return grid
