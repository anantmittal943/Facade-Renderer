# Facade-Renderer

Facade-Renderer is an open-source Pix2Pix-style Conditional GAN (cGAN) project focused on façade translation with the CMP Facades dataset (blueprint/label map → building photo).

This repository is intended as a practical guide for learning adversarial training stability and implementing image-to-image translation from first principles in PyTorch.

## What This Project Builds

Unlike a standard GAN that maps random noise to arbitrary images, this project uses a **conditional setup**:

- **Input:** a specific blueprint/semantic façade map
- **Output:** the corresponding photoreal façade

The model is trained to generate the *correct* building for each input, not just any plausible building image.

## Recommended Training Pipeline

### 1) Dataset and DataLoader Configuration

Use CMP Facades paired samples where input/target are side-by-side in a single image.

- Slice each sample into left/right halves during loading
- Resize both halves to `256x256`
- Normalize pixels to `[-1, 1]` (compatible with generator `Tanh` output)
- Keep batch size small (`1` to `4`) for stability and efficient PatchGAN training

### 2) U-Net Generator

Use a U-Net generator (not a plain encoder-decoder):

- Downsample to a bottleneck
- Upsample back to full resolution
- Add skip connections from encoder layer `i` to decoder layer `n-i`

These skip connections preserve spatial alignment so façade structures (windows, doors, edges) remain in the correct locations.

### 3) PatchGAN Discriminator

Use a PatchGAN discriminator that predicts a grid (`N x N`, often around `30 x 30`) instead of one scalar.

Each output value judges whether a local patch (commonly `70 x 70`) is real or fake. This pushes the generator to produce sharper high-frequency details (textures, edges, brick patterns).

### 4) Loss Function Balancing

Train the generator with:

- **Adversarial loss** (`L_cGAN`)
- **L1 reconstruction loss** (`L_L1`)

Combined objective:

`L_G = L_cGAN + λ * L_L1`

Use a high reconstruction weight (`λ ≈ 100`) to preserve structure while still learning realism.

## Common Failure Modes (and Fixes)

### Discriminator Dominates Too Early (Mode Collapse)

Symptoms: generator outputs become repetitive/gray and stop improving.

Fixes:

- Use the same learning rate for G and D (typically `0.0002`)
- Use Adam with `beta1 = 0.5` (instead of default `0.9`)

### Missing Conditional Input in Discriminator

The discriminator must see the **input blueprint concatenated with**:

- real target image (for real pairs)
- generated image (for fake pairs)

If not, D only learns “looks like a building” instead of “matches this specific blueprint.”

### Using L2 Instead of L1 for Reconstruction

L2 (MSE) often causes blurry results in Pix2Pix-style tasks. Prefer **L1 (MAE)** for sharper, cleaner façade boundaries.
