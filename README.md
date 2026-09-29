# Facade-Renderer

Facade-Renderer is an open-source Pix2Pix-style Conditional GAN (cGAN) project focused on façade translation with the CMP Facades dataset (blueprint/label map → building photo).

This repository documents a practical, from-scratch PyTorch implementation path for learning adversarial training stability in image-to-image translation.

## What This Project Builds

Unlike a standard GAN that maps random noise to arbitrary images, this project uses a **conditional setup**:

- **Input:** a specific blueprint/semantic façade map
- **Output:** the corresponding photoreal façade

The model is trained to generate the *correct* building for each input, not just any plausible building image.

## Training Pipeline Overview

### 1) Dataset and DataLoader Configuration

The CMP Facades paired samples store input/target side-by-side in a single image.

- Slice each sample into left/right halves during loading
- Resize both halves to `256x256`
- Normalize pixels to `[-1, 1]` (compatible with generator `Tanh` output)
- The documented configuration uses small batch sizes (`1` to `4`) for stability and efficient PatchGAN training

### 2) U-Net Generator

The project architecture uses a U-Net generator rather than a plain encoder-decoder:

- Downsample to a bottleneck
- Upsample back to full resolution
- Add skip connections from encoder layer `i` to decoder layer `n-i`

These skip connections preserve spatial alignment so façade structures (windows, doors, edges) remain in the correct locations.

### 3) PatchGAN Discriminator

The discriminator follows the PatchGAN design and predicts a grid (`N x N`, often around `30 x 30`) instead of one scalar.

Each output value judges whether a local patch (commonly `70 x 70`) is real or fake. This pushes the generator to produce sharper high-frequency details (textures, edges, brick patterns).

### 4) Loss Function Balancing

The generator objective combines:

- **Adversarial loss** (`L_cGAN`)
- **L1 reconstruction loss** (`L_L1`)

Combined objective:

`L_G = L_cGAN + λ * L_L1`

A high reconstruction weight (`λ ≈ 100`) preserves structure while still learning realism.

## Common Failure Modes

### Discriminator Dominates Too Early (Mode Collapse)

Symptoms: generator outputs become repetitive/gray and stop improving.

Typical stabilization settings in this setup:

- Same learning rate for G and D (typically `0.0002`)
- Adam with `beta1 = 0.5` (instead of default `0.9`)

### Missing Conditional Input in Discriminator

In the conditional formulation, the discriminator input is the **input blueprint concatenated with**:

- real target image (for real pairs)
- generated image (for fake pairs)

If not, D only learns “looks like a building” instead of “matches this specific blueprint.”

### Using L2 Instead of L1 for Reconstruction

L2 (MSE) often causes blurry results in Pix2Pix-style tasks, while L1 (MAE) is associated with sharper, cleaner façade boundaries.
