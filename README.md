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

## Model Iterations

### v1 — Baseline
Uniform foreground class weight ×5, horizontal flip + rotation ±10° augmentation.

| Organ        | Dice   |
|--------------|--------|
| Liver        | 0.7008 |
| Right kidney | 0.8966 |
| Left kidney  | 0.8017 |
| Spleen       | 0.5609 |
| **Mean**     | **0.7400** |

Kidneys performed well out of the box. Liver and spleen were the weakest organs.
Spleen struggled because it is the smallest organ — fewer pixels means the uniform
weight ×5 was not enough to force the model to learn its boundaries.

### v2 — Per-organ class weights + stronger augmentation (failed)
Two changes motivated by the v1 results:

1. **Per-organ class weights**: instead of a single weight for all foreground organs,
   each organ gets a weight proportional to its difficulty and size:
   - Background: 1.0
   - Liver: 8.0 — large but low-contrast, easy to miss boundaries
   - Right/left kidney: 5.0 — already well-segmented, keep moderate weight
   - Spleen: 10.0 — smallest organ, needs the strongest signal

2. **Stronger augmentation**: with only 20 patients the model sees limited spatial
   variety. Added vertical flip, increased rotation from ±10° to ±15°, and random
   scale/crop (zoom 80–100%) to simulate different FOV and patient positioning.

**Result — worse across all organs:**

| Organ        | Dice   |
|--------------|--------|
| Liver        | 0.6907 |
| Right kidney | 0.8932 |
| Left kidney  | 0.7884 |
| Spleen       | 0.5420 |
| **Mean**     | **0.7286** |

Why it failed: the very high spleen weight (10×) pushed the loss to focus almost
exclusively on spleen pixels during early training, making it harder for the model
to learn the other organs simultaneously. The additional augmentations (vertical flip,
scale/crop) introduced too much spatial distortion for such a small dataset — with
only 14 training patients, the model didn't have enough examples to generalise across
the new transformations without overfitting.

### v3 — ResNet50 encoder (revert augmentation + weights to v1)
Since both v2 changes hurt performance, they were reverted to v1. The only modification
is upgrading the encoder from **ResNet34 to ResNet50**.

**Why ResNet50?** ResNet34 uses basic residual blocks (two 3×3 convolutions each).
ResNet50 uses bottleneck blocks (1×1 → 3×3 → 1×1 convolutions), giving it:
- ~25M parameters vs ~21M in ResNet34 — more capacity to learn subtle texture differences
- Deeper feature hierarchy — better at distinguishing low-contrast boundaries like
  the liver edge against surrounding tissue and the small spleen

This gives the model more representational power without changing the training dynamics
(same loss, same augmentation, same learning rates).

**Result — liver improved, spleen still low:**

| Organ        | Dice   |
|--------------|--------|
| Liver        | 0.7365 |
| Right kidney | 0.8919 |
| Left kidney  | 0.7464 |
| Spleen       | 0.5571 |
| **Mean**     | **0.7330** |

Liver improved (0.70 → 0.74) but left kidney regressed slightly and spleen barely
moved. The deeper encoder helped with large organs but not with the spleen — the
bottleneck is not model capacity but the loss function's inability to focus on small
organ overlap directly.

### v4 — Dice-only loss (drop CrossEntropy)
The combined CrossEntropy + Dice loss lets the model optimise pixel-level accuracy
(via CE) at the expense of overlap (via Dice). For small organs like the spleen,
CE can be satisfied by correctly classifying the many background pixels, even if the
spleen boundary is wrong.

Switching to **pure Dice loss** removes the CE term entirely. Every training step
now directly maximises the Dice overlap per organ — there is no shortcut of getting
easy background pixels right. This should force the model to learn the spleen and
liver boundaries more precisely.

**Result — overfitting:**

| Organ        | Dice   |
|--------------|--------|
| Liver        | 0.6748 |
| Right kidney | 0.8729 |
| Left kidney  | 0.7437 |
| Spleen       | 0.5668 |
| **Mean**     | **0.7145** |

Val Dice reached 0.876 during training but test Dice dropped to 0.71 — a clear sign
of overfitting. Pure Dice loss without the regularising effect of CrossEntropy is too
unstable on a dataset of only 20 patients. The model memorised the training patients
rather than learning generalisable organ boundaries.

### Interim summary after v1–v4

After four iterations, v1 achieved the best test generalisation. The spleen score
(0.56) is a dataset size limitation — with only 3 test patients, a single difficult
spleen case has a large impact on the mean. The combined CE+Dice loss proved the most
stable training signal across all experiments.

| Version | Change | Mean Dice |
|---------|--------|-----------|
| v1 | ResNet34, CE+Dice, weights ×5 | **0.7400** |
| v2 | Per-organ weights + stronger augmentation | 0.7286 |
| v3 | ResNet50 encoder | 0.7330 |
| v4 | Dice-only loss | 0.7145 |

### v5 — 2.5D input + EfficientNet-B4 encoder

Two architectural improvements targeting the root causes of poor spleen/liver scores:

**2.5D input (3 consecutive slices as channels)**
Previously each slice was segmented independently, with no information about
neighbouring slices. Organs like the spleen are only visible in a subset of slices
and their boundaries are easier to locate when the model can see what the slice above
and below look like. Instead of repeating the greyscale 3×, we stack slices
`[n-1, n, n+1]` as the 3 input channels. For the first and last slice, the missing
neighbour is replaced by repeating the edge slice. This is called 2.5D because it is
a 2D model that sees limited 3D context — a good compromise between full 3D
segmentation (much more memory and data) and pure 2D.

**EfficientNet-B4 encoder (replaces ResNet34)**
ResNet uses plain residual blocks. EfficientNet-B4 uses:
- **Compound scaling**: simultaneously scales network depth, width and input
  resolution in a balanced way, giving better accuracy per parameter.
- **Squeeze-and-Excitation (SE) blocks**: after each convolutional block, SE
  recalibrates channel-wise feature responses by learning which channels are most
  informative. This is especially helpful for small structures like the spleen,
  where specific feature channels need to be amplified over background noise.

EfficientNet-B4 has ~19M parameters vs ~21M for ResNet34 but consistently
outperforms it on fine-grained segmentation tasks due to the SE blocks.

## Key ML Concepts

- **Intensity → class mapping**: mask PNGs use raw intensities {63, 126, 189, 252}
  which are remapped to contiguous class indices {1, 2, 3, 4} for CrossEntropyLoss.
- **CrossEntropyLoss**: enforces that each pixel belongs to exactly one class via
  softmax — correct for multi-class, unlike BCE.
- **Per-organ class weights**: each organ is weighted individually based on size and
  difficulty. Small organs (spleen) need higher weights so the loss penalises missing
  them more strongly than missing a large, easy-to-find organ.
- **Per-slice normalisation**: MRI has no standard intensity scale (unlike CT HU),
  so each slice is normalised independently to [0, 255].
- **Patient-level split**: all slices from a patient go to the same split,
  preventing data leakage between train and val.
- **2.5D input**: instead of repeating the greyscale channel 3×, slices `[n-1, n, n+1]`
  are stacked as 3 channels. The model sees spatial continuity between slices without
  the memory cost of full 3D convolutions.
- **EfficientNet-B4 encoder**: squeeze-and-excitation blocks recalibrate channel
  responses per feature map, amplifying informative channels for small structures.
- **TTA**: horizontal flip — averages softmax distributions before argmax
  for smoother organ boundaries.

## Limitations

- **Dataset size**: only 20 patients (T2SPIR MRI). With 3 test patients, a single
  difficult case heavily influences per-organ scores — especially for the spleen
  which is the smallest organ and most sensitive to patient variation.
  Adding the T1DUAL sequences (same patients, different MRI contrast) or the CT
  subset would increase training diversity.
- **2D/2.5D only**: full 3D segmentation (processing the entire volume as a 3D tensor)
  would give better results but requires significantly more GPU memory and data.
- **T2SPIR only**: the model is trained on one MRI sequence. Performance may degrade
  on T1-weighted or other MRI contrasts without fine-tuning.
