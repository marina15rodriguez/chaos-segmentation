# CHAOS — Multi-Organ Abdominal MRI Segmentation

U-Net segmentation of 4 abdominal organs (liver, right kidney, left kidney, spleen)
from MRI scans, trained on the CHAOS dataset.

## Task

We have 20 MRI scans of healthy abdomens (T2SPIR sequence). Each scan is a 3D volume
stored as a series of 2D DICOM slices — think of it as a stack of cross-sectional
images through the abdomen, one slice per file. For each scan, we also have one PNG
mask per slice, where each pixel's intensity encodes which organ (if any) is at that
location.

The goal: train a model that, given a new MRI slice it has never seen, correctly
predicts which organ each pixel belongs to.

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

## Data Pipeline (`dataset.py`)

**Loading DICOMs**: each `.dcm` file is a 2D greyscale image. Unlike CT scans (which
use a standardised scale called Hounsfield Units where water is always 0 HU and air
is always -1000 HU), MRI pixel values have no fixed physical meaning — they depend on
the scanner settings and the patient. We therefore normalise each slice independently
to [0, 255] so all slices are on the same scale before feeding them to the model.

**Loading masks**: the PNG files store organ labels as raw pixel intensities
{0, 63, 126, 189, 252}. These arbitrary values have no meaning to PyTorch's loss
function. We remap them to contiguous integers {0, 1, 2, 3, 4} — one index per class
— which is the format CrossEntropyLoss expects.

**Patient-level split**: we have 20 patients × ~30 slices each = ~600 slices total.
If we split randomly by slice, the same patient could appear in both train and
validation — the model would be evaluated on slices of patients it already saw during
training, making the validation score artificially high. Instead we split by patient:
14 train / 3 val / 3 test. All slices from a patient always go to the same set.
This is called patient-level split and it prevents **data leakage**.

**Augmentation**: during training we randomly flip and rotate the image and mask
together — the exact same transformation is applied to both so they stay aligned.
If we only flipped the image but not the mask, the model would receive contradictory
supervision (liver on the left in the image but labelled on the right in the mask).
We never apply colour jitter to the mask — that is an image-only transform.

**What the model receives**: image tensor `[3, 256, 256]` (the greyscale slice
repeated 3 times to match the 3-channel RGB format that the pretrained ImageNet
encoder expects) and mask tensor `[256, 256]` with integer class labels.

## Architecture (`model.py`)

A **U-Net** with a pretrained **ResNet34 encoder**.

**U-Net** is an encoder-decoder architecture with skip connections:
- **Encoder** (downsampling path): applies convolutions and pooling to progressively
  reduce the spatial resolution (e.g. 256×256 → 128×128 → 64×64 → …) while
  increasing the number of feature channels. Each layer learns increasingly abstract
  representations — early layers detect edges and textures, deep layers detect
  organ-level shapes.
- **Decoder** (upsampling path): progressively restores the spatial resolution back
  to the original image size, producing a prediction map at full resolution.
- **Skip connections**: at each resolution level, the encoder's feature map is
  concatenated to the decoder's feature map. This is why the architecture is called
  U-Net (it looks like a U). Without skip connections, fine spatial detail (exact
  organ boundaries) would be lost in the deep encoder layers and the decoder could
  not recover it.

**Why pretrained ResNet34?** Training a U-Net from scratch on 20 patients would
overfit immediately — there is not enough data for the model to learn meaningful
features from random initialisation. ResNet34 was pretrained on ImageNet (1.2M
images across 1000 categories) so it already knows how to detect edges, textures
and shapes. We reuse these features and only need the training data to teach the
decoder how to map them to organ labels. This is called **transfer learning**.

**Differential learning rates**: we use two different learning rates for the two
parts of the model. The encoder gets LR=5e-5 (small) because its pretrained ImageNet
weights are already good — large updates would destroy them. The decoder gets LR=1e-3
(larger) because it is randomly initialised and needs to learn faster. This technique
is called **differential learning rates**.

**Output**: `[5, 256, 256]` — 5 channels (one per class) of raw scores called
**logits**. No activation is applied here — the loss function handles that internally.

## Training (`train.py`)

**CrossEntropyLoss**: takes the 5-channel logits and the integer mask. Internally it
applies **softmax** to the logits, which turns the raw scores into probabilities that
sum to 1 across the 5 classes for each pixel. For example, a pixel might become
[0.02, 0.71, 0.10, 0.12, 0.05] — 71% liver, 12% left kidney, etc. The loss then
penalises how much probability was assigned to the wrong class. The key property of
softmax is that the 5 class probabilities always sum to 1, so the classes
**compete** against each other — making the liver score higher automatically lowers
all other scores for that pixel. This correctly models the fact that each pixel
belongs to exactly one organ.

**Why not BCE (Binary Cross Entropy)?** BCE treats each output channel as an
independent yes/no question. A pixel could score 80% liver AND 80% spleen
simultaneously, which is physically impossible and gives contradictory gradients
during training. CE with softmax prevents this.

**Dice loss**: the Dice coefficient measures the overlap between prediction and ground
truth: `Dice = 2 × |pred ∩ target| / (|pred| + |target|)`. It is 1 when prediction
and ground truth perfectly overlap and 0 when they don't overlap at all. We add a
Dice loss term (1 - Dice) to the CE loss so the model directly optimises the metric
we care about. Without Dice loss, CE alone can be satisfied by correctly classifying
the many background pixels without ever learning the organ boundaries precisely.

**Class weights [1, 5, 5, 5, 5]**: organs occupy far fewer pixels than background
in a typical MRI slice — most of the image is background tissue. Without weighting,
the CE loss is dominated by background pixels and the model learns to predict
all-background (which scores low loss but is useless). We multiply the loss
contribution of each organ pixel by 5, forcing the model to pay more attention to
getting the organ regions right.

**CosineAnnealingLR**: the learning rate follows a cosine curve from 1e-3 down to
near zero across 50 epochs. This ensures the model takes large steps early (fast
learning) and small steps later (fine-tuning). We chose this over ReduceLROnPlateau
(which halves the LR whenever val Dice stops improving) because in earlier experiments
ReduceLROnPlateau collapsed the LR to ~1e-7 within 30 epochs before the model had
finished learning, causing val Dice to stay at 0.

## Evaluation (`evaluate.py`)

**Dice coefficient**: for each organ separately, we compute
`Dice = 2 × (pred ∩ target) / (pred + target)` on the test set. We only include
slices where the organ actually appears in the ground truth — on a slice where the
liver is not visible, a correct all-background prediction would score near 0 Dice
(numerator = 0, denominator ≈ small predicted area), which would unfairly penalise
a correct prediction.

**TTA (Test-Time Augmentation)**: "inference" means running the trained model on new
data to get predictions — as opposed to "training" where the model is updating its
weights. During inference the model's weights are frozen; we are just using it as a
function that maps an input slice to a segmentation mask.

TTA improves predictions by running inference multiple times with slightly different
versions of the same input and averaging the results. Here we run inference twice:
once on the original slice, and once on the horizontally flipped slice. The flipped
slice's prediction is flipped back to the original orientation, then we average the
two **softmax probability maps** (before taking argmax). Averaging makes the
probabilities more reliable — if the model is uncertain about a pixel on the boundary
of the liver, one run might say 60% liver and the other 70% liver; averaging gives
65% and argmax still picks liver. Near boundaries where the model is genuinely
uncertain, the two runs can reinforce each other and produce a smoother, more
confident boundary.

**Colour grid**: the integer prediction map is visualised as a colour overlay on top
of the original greyscale MRI slice — liver=red, right kidney=green, left
kidney=blue, spleen=orange — so you can visually inspect where the model is correct
and where it fails.

## Docker (recommended)

The easiest way to run the API — no Python environment or checkpoint setup needed.

**Pull and run:**
```powershell
docker pull marina15rodriguez/chaos-segmentation:v1
docker run -p 8000:8000 marina15rodriguez/chaos-segmentation:v1
```

Then open `http://localhost:8000` in your browser.

The image includes the trained model checkpoint and all dependencies. It runs on CPU only (no GPU required).

If port 8000 is already in use, map to a different port:
```powershell
docker run -p 8001:8000 marina15rodriguez/chaos-segmentation:v1
```

---

## Local API

A FastAPI web interface lets you upload a DICOM series and visualise the organ
segmentation predictions interactively.

**Setup:**
```powershell
cd C:\Users\marin\projects\chaos-segmentation
python -m pip install fastapi uvicorn python-multipart segmentation-models-pytorch pydicom torch torchvision Pillow numpy
```

**Run:**
```powershell
cd api
python -m uvicorn main:app --port 8000
```

Open `http://127.0.0.1:8000` in your browser.

**Usage:**
1. Zip a T2SPIR DICOM series folder:
   ```powershell
   Compress-Archive -Path "path\to\MR\<patient_id>\T2SPIR\DICOM_anon\*" -DestinationPath "series.zip"
   ```
2. Drag and drop `series.zip` onto the interface
3. Click **Run segmentation**
4. The interface shows the 8 most informative slices with colour-coded organ overlays and lists the detected organs

The API loads the v1 checkpoint (`results/best_model.pth`) at startup. Make sure the
checkpoint is downloaded from Kaggle and placed at that path before starting the server.

## Model Iterations

### v1 — Baseline
ResNet34 encoder, CrossEntropy + Dice loss, uniform foreground class weight ×5,
horizontal flip + rotation ±10° augmentation.

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

**Result — liver improved, spleen still at ceiling:**

| Organ        | Dice   |
|--------------|--------|
| Liver        | 0.7614 |
| Right kidney | 0.8609 |
| Left kidney  | 0.7328 |
| Spleen       | 0.5765 |
| **Mean**     | **0.7329** |

Liver improved significantly (0.70 → 0.76) thanks to the 2.5D context and better
encoder features. However the spleen remains near 0.57 across all versions. This
confirms the spleen score is a data limitation — with only 3 test patients, a single
difficult spleen appearance dominates the score. The right kidney also regressed
slightly, likely because EfficientNet-B4 features are tuned differently than ResNet34.

### Final comparison

| Version | Key change | Liver | R.Kidney | L.Kidney | Spleen | Mean |
|---------|-----------|-------|----------|----------|--------|------|
| v1 | ResNet34, CE+Dice | 0.701 | 0.897 | 0.802 | 0.561 | **0.740** |
| v2 | Per-organ weights + augmentation | 0.691 | 0.893 | 0.788 | 0.542 | 0.729 |
| v3 | ResNet50 encoder | 0.737 | 0.892 | 0.746 | 0.557 | 0.733 |
| v4 | Dice-only loss | 0.675 | 0.873 | 0.744 | 0.567 | 0.715 |
| v5 | 2.5D + EfficientNet-B4 | 0.761 | 0.861 | 0.733 | 0.577 | 0.733 |

**Best overall**: v1 (highest mean Dice 0.740, most stable across all organs).
**Best liver**: v5 (0.761 — 2.5D context and SE blocks help large organ boundaries).
**Spleen**: consistently ~0.56–0.58 across all versions — a dataset size ceiling.

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
