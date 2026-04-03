"""Data loading and augmentation for multi-organ MRI segmentation.

Dataset: CHAOS (Combined Healthy Abdominal Organ Segmentation)
  - MRI scans (T2SPIR) in DICOM format
  - Organ masks as PNG files (one per slice, same index as DICOM)
  - 4 organs encoded by pixel intensity:
      0   = background
      63  = liver
      126 = right kidney
      189 = left kidney
      252 = spleen

Layout:
    <data_root>/MR/<patient_id>/T2SPIR/
        DICOM_anon/  IMG-0002-000XX.dcm
        Ground/      IMG-0002-000XX.png

Split strategy: patient-level 70 / 15 / 15 (train / val / test).
All slices from a patient go to the same split — prevents data leakage.
"""

import random
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision.transforms as T

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_SIZE = 256          # CHAOS MR slices are 256×256 natively

NUM_CLASSES = 5           # background + 4 organs
ORGAN_NAMES = {
    0: "background",
    1: "liver",
    2: "right_kidney",
    3: "left_kidney",
    4: "spleen",
}

# Pixel intensity → class index mapping
INTENSITY_TO_CLASS = {
    0:   0,   # background
    63:  1,   # liver
    126: 2,   # right kidney
    189: 3,   # left kidney
    252: 4,   # spleen
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15


# ---------------------------------------------------------------------------
# DICOM loading
# ---------------------------------------------------------------------------
def load_dicom_slice(dcm_path: Path) -> np.ndarray:
    """Load a DICOM MRI slice and return a uint8 array [0, 255].

    MRI pixel values are normalised per-slice (no HU conversion needed).
    """
    dcm   = pydicom.dcmread(str(dcm_path))
    array = dcm.pixel_array.astype(np.float32)
    array = (array - array.min()) / (array.max() - array.min() + 1e-6)
    return (array * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Mask loading
# ---------------------------------------------------------------------------
def load_mask(png_path: Path) -> np.ndarray:
    """Load a PNG mask and convert intensity values to class indices [0..4]."""
    mask_raw = np.array(Image.open(png_path).convert("L"), dtype=np.uint8)
    mask_cls = np.zeros_like(mask_raw, dtype=np.uint8)
    for intensity, cls in INTENSITY_TO_CLASS.items():
        mask_cls[mask_raw == intensity] = cls
    return mask_cls


# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------
def collect_samples(data_root: Path) -> list[dict]:
    """Build a list of (slice, mask) samples.

    Each sample dict contains:
        'dcm_paths'  : list of all DCM paths in this series (for 2.5D context)
        'slice_idx'  : index of this slice within dcm_paths
        'mask_path'  : Path to matching PNG mask
        'patient_id' : str, for patient-level splitting
    """
    mr_dir  = data_root / "MR"
    samples = []

    for patient_dir in sorted(mr_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        patient_id = patient_dir.name

        t2_dir = patient_dir / "T2SPIR"
        if not t2_dir.exists():
            continue

        dicom_dir  = t2_dir / "DICOM_anon"
        ground_dir = t2_dir / "Ground"

        if not dicom_dir.exists() or not ground_dir.exists():
            continue

        dcm_files  = sorted(dicom_dir.glob("*.dcm"))
        mask_files = sorted(ground_dir.glob("*.png"))

        # Store full series list per sample for 2.5D neighbour access
        for slice_idx, mask_path in enumerate(mask_files):
            samples.append({
                "dcm_paths":  dcm_files,       # all slices in this series
                "slice_idx":  slice_idx,
                "mask_path":  mask_path,
                "patient_id": patient_id,
            })

    return samples


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CHAOSDataset(Dataset):
    """Loads (image, mask) pairs for multi-organ MRI segmentation.

    Uses 2.5D input: stacks slices [n-1, n, n+1] as 3 channels so the model
    sees spatial context between adjacent slices. For boundary slices (first/last),
    the missing neighbour is replaced by repeating the edge slice.

    Each __getitem__ returns:
        image : FloatTensor [3, H, W]  — 3 consecutive MRI slices as channels
        mask  : LongTensor  [H, W]     — integer class labels {0..4} for slice n
    """

    def __init__(self, samples: list[dict], augment: bool = False):
        self.samples = samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item      = self.samples[idx]
        dcm_paths = item["dcm_paths"]
        n         = item["slice_idx"]
        n_total   = len(dcm_paths)

        # 2.5D: load slices n-1, n, n+1 (clamp at boundaries)
        prev_idx = max(0, n - 1)
        next_idx = min(n_total - 1, n + 1)

        slice_prev = load_dicom_slice(dcm_paths[prev_idx])
        slice_curr = load_dicom_slice(dcm_paths[n])
        slice_next = load_dicom_slice(dcm_paths[next_idx])

        # Stack as PIL images for joint transforms (use current slice as reference)
        img_prev = Image.fromarray(slice_prev, mode="L")
        img_curr = Image.fromarray(slice_curr, mode="L")
        img_next = Image.fromarray(slice_next, mode="L")

        # Load PNG mask → class indices → PIL
        mask_array = load_mask(item["mask_path"])
        mask = Image.fromarray(mask_array, mode="L")

        # Joint spatial transforms — apply same transform to all 3 slices + mask
        img_prev, img_curr, img_next, mask = self._joint_transform(
            img_prev, img_curr, img_next, mask)

        # Stack 3 slices as channels → [3, H, W]
        ch_prev = TF.to_tensor(img_prev)   # [1, H, W]
        ch_curr = TF.to_tensor(img_curr)
        ch_next = TF.to_tensor(img_next)
        image   = torch.cat([ch_prev, ch_curr, ch_next], dim=0)  # [3, H, W]
        image   = TF.normalize(image, IMAGENET_MEAN, IMAGENET_STD)

        # Mask: integer class labels [H, W]
        mask = torch.from_numpy(np.array(mask)).long()            # [H, W] {0..4}

        return image, mask

    def _joint_transform(
        self,
        img_prev: Image.Image,
        img_curr: Image.Image,
        img_next: Image.Image,
        mask:     Image.Image,
    ) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
        """Apply identical spatial transforms to all 3 slices and the mask."""

        def resize_bilinear(img):
            return TF.resize(img, [IMAGE_SIZE, IMAGE_SIZE], interpolation=Image.BILINEAR)

        def resize_nearest(img):
            return TF.resize(img, [IMAGE_SIZE, IMAGE_SIZE], interpolation=Image.NEAREST)

        img_prev = resize_bilinear(img_prev)
        img_curr = resize_bilinear(img_curr)
        img_next = resize_bilinear(img_next)
        mask     = resize_nearest(mask)

        if self.augment:
            # Random horizontal flip — same decision for all
            if random.random() > 0.5:
                img_prev = TF.hflip(img_prev)
                img_curr = TF.hflip(img_curr)
                img_next = TF.hflip(img_next)
                mask     = TF.hflip(mask)

            # Random rotation — same angle for all
            angle    = random.uniform(-10, 10)
            img_prev = TF.rotate(img_prev, angle, interpolation=Image.BILINEAR)
            img_curr = TF.rotate(img_curr, angle, interpolation=Image.BILINEAR)
            img_next = TF.rotate(img_next, angle, interpolation=Image.BILINEAR)
            mask     = TF.rotate(mask,     angle, interpolation=Image.NEAREST)

            # Brightness/contrast jitter — apply same params to all 3 slices
            jitter = T.ColorJitter(brightness=0.2, contrast=0.2)
            params = jitter.get_params(
                jitter.brightness, jitter.contrast, jitter.saturation, jitter.hue)
            img_prev = TF.adjust_brightness(img_prev, params[1])
            img_curr = TF.adjust_brightness(img_curr, params[1])
            img_next = TF.adjust_brightness(img_next, params[1])
            img_prev = TF.adjust_contrast(img_prev, params[2])
            img_curr = TF.adjust_contrast(img_curr, params[2])
            img_next = TF.adjust_contrast(img_next, params[2])

        return img_prev, img_curr, img_next, mask


# ---------------------------------------------------------------------------
# Patient-level split
# ---------------------------------------------------------------------------
def patient_split(
    samples:    list[dict],
    train_frac: float = TRAIN_FRAC,
    val_frac:   float = VAL_FRAC,
    seed:       int   = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split samples at the patient level to prevent data leakage."""
    patient_ids = sorted(set(s["patient_id"] for s in samples))
    rng = random.Random(seed)
    rng.shuffle(patient_ids)

    n_train = int(len(patient_ids) * train_frac)
    n_val   = int(len(patient_ids) * val_frac)

    train_ids = set(patient_ids[:n_train])
    val_ids   = set(patient_ids[n_train: n_train + n_val])
    test_ids  = set(patient_ids[n_train + n_val:])

    return (
        [s for s in samples if s["patient_id"] in train_ids],
        [s for s in samples if s["patient_id"] in val_ids],
        [s for s in samples if s["patient_id"] in test_ids],
    )


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------
def build_dataloaders(
    data_root:   Path | None = None,
    batch_size:  int = 8,
    num_workers: int = 2,
    seed:        int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test DataLoaders with patient-level split."""
    if data_root is None:
        data_root = Path(__file__).resolve().parents[1] / "data"
    else:
        data_root = Path(data_root)

    all_samples = collect_samples(data_root)
    if not all_samples:
        raise RuntimeError(f"No samples found under {data_root}/MR/")

    train_samples, val_samples, test_samples = patient_split(
        all_samples, seed=seed
    )

    n_patients = len(set(s["patient_id"] for s in all_samples))
    print(f"Patients : {n_patients}")
    print(f"Slices   : {len(all_samples)} total")
    print(f"Split    — train: {len(train_samples)} | "
          f"val: {len(val_samples)} | test: {len(test_samples)}")

    train_ds = CHAOSDataset(train_samples, augment=True)
    val_ds   = CHAOSDataset(val_samples,   augment=False)
    test_ds  = CHAOSDataset(test_samples,  augment=False)

    loader_kws = dict(batch_size=batch_size, num_workers=num_workers,
                      pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kws)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kws)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kws)

    return train_loader, val_loader, test_loader
