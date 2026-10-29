# Facade-Renderer

Facade-Renderer is an open-source Pix2Pix-style Conditional GAN (cGAN) project focused on façade translation with the CMP Facades dataset (blueprint/label map → building photo).

This repository documents a practical, from-scratch PyTorch implementation path for learning adversarial training stability in image-to-image translation.

---

## What This Project Builds

Unlike a standard GAN that maps random noise to arbitrary images, this project uses a **conditional setup**:

- **Input:** A specific architectural blueprint / semantic façade segmentation map (Domain B).
- **Output:** The corresponding photorealistic building façade (Domain A).

The model is trained to generate the *correct* building for each input layout, preserving window locations, door frames, balconies, and wall textures.

---

## Project Structure

```text
Facade-Renderer/
├── train_facade_renderer.ipynb # Standalone Jupyter Notebook ready for Kaggle GPU
├── dashboard.py        # Interactive Web Dashboard for testing and inference
├── dataset.py          # CMP Facades dataset, slicing, augmentation & DataLoader
├── models.py           # U-Net 256 Generator & 70x70 PatchGAN Discriminator
├── train.py            # Training loop, loss computation, visual logging & checkpointing
├── test.py             # Inference, evaluation metrics (L1, PSNR), and side-by-side export
├── download_data.py    # Kagglehub & direct dataset downloader to D:\Datasets
├── requirements.txt    # Python dependencies
└── README.md           # Documentation
```

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/Facade-Renderer.git
   cd Facade-Renderer
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Dataset Setup

The project uses the **CMP Facades dataset** (`vikramtiwari/pix2pix-dataset`), configured to download into `D:\Datasets`.

### Option 1: Automatic Download via `download_data.py`

Run the included downloader script:
```bash
python download_data.py
```

Or for bandwidth-constrained environments, download only the CMP Facades subset (~30 MB) directly:
```bash
python download_data.py --direct
```

### Option 2: Using Python with `kagglehub` directly

```python
import os
os.environ["KAGGLEHUB_CACHE"] = r"D:\Datasets"
import kagglehub

# Download latest version into D:\Datasets
path = kagglehub.dataset_download("vikramtiwari/pix2pix-dataset")
print("Path to dataset files:", path)
```

The dataset files are structured as:
```text
D:\Datasets\facades\
├── train/     # 400 composite paired images (512x256)
├── val/       # 100 composite paired images (512x256)
└── test/      # 106 composite paired images (512x256)
```

---

## Training Pipeline Overview

### 1) Dataset and DataLoader Configuration
The CMP Facades paired samples store input/target side-by-side in a single $512 \times 256$ image:
- **Slice:** Sliced into left (photo) and right (blueprint) halves during loading.
- **Direction:** Configured to `--direction BtoA` (blueprint $\to$ photo).
- **Augmentation:** Random jitter (scaling to $286 \times 286$ followed by synchronous random crop to $256 \times 256$) and random horizontal flips.
- **Normalization:** Normalized to $[-1, 1]$ matching Generator `Tanh` output.
- **Batch Size:** Small batch size (`1` to `4`) for stable PatchGAN convergence.

### 2) U-Net Generator
The architecture uses a symmetrical U-Net with skip connections:
- **Encoder:** 8 downsampling blocks (Conv $4\times4$, stride 2, LeakyReLU 0.2, BatchNorm).
- **Decoder:** 7 upsampling blocks (ConvTranspose $4\times4$, stride 2, ReLU, BatchNorm, Dropout 0.5 on early layers).
- **Skip Connections:** Connect encoder layer $i$ to decoder layer $n-i$, preserving spatial alignment so façade structures (windows, doors, trims) remain firmly anchored.
- **Activation:** Final layer utilizes `Tanh` to constrain output within $[-1, 1]$.

### 3) PatchGAN Discriminator
The discriminator operates as a $70 \times 70$ PatchGAN:
- **Conditional Input:** Receives a 6-channel tensor combining the condition blueprint and the target image:
  $$\text{Input}_D = \text{Concat}(\text{Condition}, \text{Target})$$
- **Patch Grid:** Outputs a $\approx 30 \times 30$ grid of raw logits, penalizing high-frequency blur and enforcing crisp edge details.

### 4) Loss Function Balancing
The Generator objective combines adversarial game-play with pixel-level reconstruction:
$$\mathcal{L}_G = \mathcal{L}_{\text{cGAN}} + \lambda \mathcal{L}_{L1}$$
Where:
- $\mathcal{L}_{\text{cGAN}} = \mathbb{E}[\log(1 - D(x, G(x)))]$ (implemented via `BCEWithLogitsLoss`)
- $\mathcal{L}_{L1} = \mathbb{E}[\|y - G(x)\|_1]$
- $\lambda = 100.0$ (ensures structural fidelity while the discriminator enforces photorealism).

---

## How to Run

### Training

To begin training on CMP Facades:
```bash
python train.py --epochs 200 --batch_size 1 --lr 0.0002 --beta1 0.5 --lambda_l1 100.0
```

Key arguments:
- `--data_dir`: Path to dataset (auto-detected from `D:\Datasets\facades` if omitted).
- `--output_dir`: Path to store checkpoints and visual samples (default: `outputs`).
- `--direction`: `BtoA` (Blueprint $\to$ Photo, default) or `AtoB`.
- `--sample_interval`: Number of epochs between saving visual progression grids (default: `5`).
- `--checkpoint_interval`: Number of epochs between saving model weights (default: `10`).

Visual samples will be automatically saved to `outputs/samples/sample_epoch_XXX.png` showing:
`[ Condition Blueprint | Ground Truth | Generated Building ]`.

### Training on Kaggle (Jupyter Notebook)

A completely self-contained notebook is provided: [`train_facade_renderer.ipynb`](file:///D:/VSCodeProjects/Facade-Renderer/train_facade_renderer.ipynb).

1. Upload `train_facade_renderer.ipynb` to [Kaggle](https://www.kaggle.com/code).
2. Attach dataset `vikramtiwari/pix2pix-dataset` under **Input** (or let the notebook automatically download it).
3. Under **Notebook options**, select **Accelerator: GPU P100** or **GPU T4**.
4. Click **Run All**. The notebook:
   - Detects GPU acceleration.
   - Loads the CMP Facades dataset.
   - Shows inline visual comparison plots every 10 epochs.
   - Plots adversarial and L1 loss curves.
   - Saves final checkpoints into `/kaggle/working/checkpoints/` for direct download.

---

### Interactive Web Dashboard (`dashboard.py`)

For intuitive testing, evaluation, and interactive visual demonstration, launch the included dashboard:

```bash
python dashboard.py
```

Then navigate to `http://127.0.0.1:5000` in your web browser.

**Dashboard Features:**
- **Model Selector:** Auto-detects checkpoints in `outputs/checkpoints/` or accepts custom `.pth` paths.
- **Test Set Gallery:** Browse and select any sample from the 106 test facade pairs with instant side-by-side display of the input blueprint, real ground truth, and generated facade.
- **Custom Upload:** Drag and drop any custom blueprint, architectural layout, or mask.
- **Quantitative Metrics:** Live readout of inference latency (ms), L1 MAE error, and PSNR (dB).
- **One-Click Download:** Save the synthesized façade image directly.

---

### CLI Evaluation & Inference

Evaluate on the test set:
```bash
python test.py --checkpoint outputs/checkpoints/generator_latest.pth
```

Run inference on a single blueprint image:
```bash
python test.py --checkpoint outputs/checkpoints/generator_latest.pth --input_image path/to/blueprint.png
```

Evaluation metrics computed:
- **Mean L1 Error** (Pixel-wise mean absolute error)
- **PSNR** (Peak Signal-to-Noise Ratio in dB)

---

## Common Failure Modes & Stabilization

### 1. Discriminator Dominates Too Early (Mode Collapse)
- **Symptom:** Generator outputs become repetitive gray smudges and gradients vanish.
- **Solution:** Balanced learning rate (`0.0002`), Adam momentum `beta1 = 0.5`, `beta2 = 0.999`, and batch size $\le 4$.

### 2. Missing Conditional Input in Discriminator
- **Symptom:** Discriminator judges whether an image looks like *any* building rather than *this specific blueprint*.
- **Solution:** Concatenate the blueprint with the candidate photo along channel dimension ($6$ channels total).

### 3. Using L2 Instead of L1 Loss
- **Symptom:** L2 loss averages plausible textures, producing blurry, hazy walls and soft edges.
- **Solution:** L1 (MAE) loss encourages sharper transitions and well-defined window borders.
