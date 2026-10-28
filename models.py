"""
Facade-Renderer: Models
Implements U-Net Generator and PatchGAN Discriminator for Pix2Pix cGAN.
"""

import torch
import torch.nn as nn


def weights_init_normal(m):
    """
    Initializes weights according to the original Pix2Pix paper:
    Normal(mean=0.0, std=0.02) for Conv/ConvTranspose,
    Normal(mean=1.0, std=0.02) and Constant(0.0) for BatchNorm.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class UNetDown(nn.Module):
    """Downsampling block: Conv -> [BatchNorm] -> LeakyReLU"""

    def __init__(self, in_channels: int, out_channels: int, normalize: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=False))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UNetUp(nn.Module):
    """Upsampling block: ReLU -> ConvTranspose -> BatchNorm -> [Dropout]"""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip_input: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        # Concatenate skip connection along channel dimension
        return torch.cat((x, skip_input), dim=1)


class UNetGenerator(nn.Module):
    """
    U-Net 256 Generator with skip connections.
    Maps a 3-channel input condition (e.g. blueprint/segmentation mask)
    to a 3-channel output image (building facade).
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()

        # Encoder (Downsampling)
        self.down1 = UNetDown(in_channels, 64, normalize=False)  # 256 -> 128
        self.down2 = UNetDown(64, 128)                           # 128 -> 64
        self.down3 = UNetDown(128, 256)                          # 64 -> 32
        self.down4 = UNetDown(256, 512)                          # 32 -> 16
        self.down5 = UNetDown(512, 512)                          # 16 -> 8
        self.down6 = UNetDown(512, 512)                          # 8 -> 4
        self.down7 = UNetDown(512, 512)                          # 4 -> 2
        self.down8 = nn.Sequential(                              # 2 -> 1 (Bottleneck)
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
        )

        # Decoder (Upsampling with skip connections)
        self.up1 = UNetUp(512, 512, dropout=0.5)                 # 1 -> 2 (concat with d7 => 1024)
        self.up2 = UNetUp(1024, 512, dropout=0.5)                # 2 -> 4 (concat with d6 => 1024)
        self.up3 = UNetUp(1024, 512, dropout=0.5)                # 4 -> 8 (concat with d5 => 1024)
        self.up4 = UNetUp(1024, 512)                             # 8 -> 16 (concat with d4 => 1024)
        self.up5 = UNetUp(1024, 256)                             # 16 -> 32 (concat with d3 => 512)
        self.up6 = UNetUp(512, 128)                              # 32 -> 64 (concat with d2 => 256)
        self.up7 = UNetUp(256, 64)                               # 64 -> 128 (concat with d1 => 128)

        # Final output layer: 128 -> 256, Tanh activation maps to [-1, 1]
        self.final = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # Decoder with skip connections
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


class PatchGANDiscriminator(nn.Module):
    """
    70x70 PatchGAN Discriminator.
    Evaluates whether each local 70x70 image patch is real or fake.
    Conditioned on input image: takes concat([condition, target], dim=1) -> 6 channels.
    Outputs ~30x30 spatial grid of raw logits.
    """

    def __init__(self, in_channels: int = 6):
        super().__init__()

        def discriminator_block(in_f: int, out_f: int, stride: int = 2, normalize: bool = True):
            layers = [nn.Conv2d(in_f, out_f, kernel_size=4, stride=stride, padding=1, bias=not normalize)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            return layers

        self.model = nn.Sequential(
            # Layer 1: 256 -> 128
            *discriminator_block(in_channels, 64, stride=2, normalize=False),
            # Layer 2: 128 -> 64
            *discriminator_block(64, 128, stride=2),
            # Layer 3: 64 -> 32
            *discriminator_block(128, 256, stride=2),
            # Layer 4: 32 -> 31
            *discriminator_block(256, 512, stride=1),
            # Layer 5: 31 -> 30 raw logits output
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, condition: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Concatenate condition and target along channel dimension
        img_input = torch.cat((condition, target), dim=1)
        return self.model(img_input)
