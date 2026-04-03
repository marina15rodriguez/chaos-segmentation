# CHAOS — Multi-Organ Abdominal MRI Segmentation

U-Net segmentation of 4 abdominal organs (liver, right kidney, left kidney, spleen)
from MRI scans, trained on the CHAOS dataset.

## Task

Multi-class segmentation of abdominal organs from T2SPIR MRI slices in DICOM format.
Each pixel is assigned one of 5 classes:

| Class | Organ        | Mask intensity |
|-------|--------------|----------------|
| 0     | Background   | 0              |
| 1     | Liver        | 63             |
| 2     | Right kidney | 126            |
| 3     | Left kidney  | 189            |
| 4     | Spleen       | 252            |

## Dataset

**CHAOS (Combined Healthy Abdominal Organ Segmentation)** — available on Kaggle:

Search: `chaos-combined-ct-mr-healthy-abdominal-organ`

Expected layout:
```
data/
  MR/
    <patient_id>/
      T2SPIR/
        DICOM_anon/   IMG-0002-000XX.dcm
        Ground/       IMG-0002-000XX.png
```

## Architecture

- **Encoder**: ResNet34 pretrained on ImageNet
- **Decoder**: U-Net decoder with skip connections
- **Output**: 5-channel logits → softmax → argmax class per pixel
- **Input**: MRI slice normalised per-slice to [0, 255], repeated 3× for RGB

## Training

| Setting         | Value                                    |
|-----------------|------------------------------------------|
| Loss            | CrossEntropy (w=5 for organs) + multi-class Dice |
| Optimiser       | Adam, differential LRs                   |
| Encoder LR      | 5e-5                                     |
| Decoder LR      | 1e-3                                     |
| Scheduler       | CosineAnnealingLR                        |
| Batch size      | 8                                        |
| Image size      | 256 × 256                                |
| Split           | Patient-level 70/15/15                   |

Run training on Kaggle:
```python
!git clone https://github.com/marina15rodriguez/chaos-segmentation.git
%cd chaos-segmentation
!pip install -q -r requirements.txt

!python src/train.py \
    --data-dir /kaggle/input/.../CHAOS_Train_Sets/Train_Sets \
    --epochs 50 \
    --batch-size 8 \
    --output-dir /kaggle/working/results
```

## Evaluation

```bash
python src/evaluate.py \
    --checkpoint results/best_model.pth \
    --data-dir /path/to/CHAOS_Train_Sets/Train_Sets \
    --output-dir results
```

Outputs per-organ Dice and IoU on the test set, plus a colour-coded prediction grid.

## Key ML Concepts

- **Intensity → class mapping**: mask PNGs use raw intensities {63, 126, 189, 252}
  which are remapped to contiguous class indices {1, 2, 3, 4} for CrossEntropyLoss.
- **CrossEntropyLoss**: enforces that each pixel belongs to exactly one class via
  softmax — correct for multi-class, unlike BCE.
- **Class weights**: foreground organs upweighted ×5 to counteract background dominance.
- **Per-slice normalisation**: MRI has no standard intensity scale (unlike CT HU),
  so each slice is normalised independently to [0, 255].
- **Patient-level split**: all slices from a patient go to the same split,
  preventing data leakage between train and val.
- **TTA**: horizontal flip — averages softmax distributions before argmax
  for smoother organ boundaries.
