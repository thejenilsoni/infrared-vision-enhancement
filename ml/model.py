from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as functional


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, channels), channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(inputs + self.block(inputs))


class EncoderStage(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            input_channels, output_channels, 4, stride=2, padding=1
        )
        self.residual = ResidualBlock(output_channels)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.residual(self.projection(inputs))


class DecoderStage(nn.Module):
    def __init__(self, input_channels: int, skip_channels: int, output_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            input_channels, output_channels, 4, stride=2, padding=1
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(output_channels + skip_channels, output_channels, 3, padding=1),
            nn.GroupNorm(min(8, output_channels), output_channels),
            nn.SiLU(inplace=True),
            ResidualBlock(output_channels),
        )

    def forward(self, inputs: Tensor, skip: Tensor) -> Tensor:
        output = self.upsample(inputs)
        if output.shape[-2:] != skip.shape[-2:]:
            output = functional.interpolate(
                output, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        return self.fusion(torch.cat([output, skip], dim=1))


class InfraredColorizationUNet(nn.Module):
    """Residual U-Net mapping one normalized infrared channel to RGB."""

    def __init__(self, base_channels: int = 48) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, 5, padding=2),
            nn.SiLU(inplace=True),
            ResidualBlock(base_channels),
        )
        self.encoder_1 = EncoderStage(base_channels, base_channels * 2)
        self.encoder_2 = EncoderStage(base_channels * 2, base_channels * 4)
        self.encoder_3 = EncoderStage(base_channels * 4, base_channels * 8)
        self.bottleneck = nn.Sequential(
            EncoderStage(base_channels * 8, base_channels * 8),
            ResidualBlock(base_channels * 8),
            ResidualBlock(base_channels * 8),
        )
        self.decoder_3 = DecoderStage(
            base_channels * 8, base_channels * 8, base_channels * 8
        )
        self.decoder_2 = DecoderStage(
            base_channels * 8, base_channels * 4, base_channels * 4
        )
        self.decoder_1 = DecoderStage(
            base_channels * 4, base_channels * 2, base_channels * 2
        )
        self.decoder_0 = DecoderStage(
            base_channels * 2, base_channels, base_channels
        )
        self.head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, infrared: Tensor) -> Tensor:
        skip_0 = self.stem(infrared)
        skip_1 = self.encoder_1(skip_0)
        skip_2 = self.encoder_2(skip_1)
        skip_3 = self.encoder_3(skip_2)
        latent = self.bottleneck(skip_3)
        output = self.decoder_3(latent, skip_3)
        output = self.decoder_2(output, skip_2)
        output = self.decoder_1(output, skip_1)
        output = self.decoder_0(output, skip_0)
        return self.head(output)


def structural_similarity_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Differentiable global SSIM loss, averaged over batch and channels."""
    c1, c2 = 0.01**2, 0.03**2
    mu_prediction = functional.avg_pool2d(prediction, 11, 1, 5)
    mu_target = functional.avg_pool2d(target, 11, 1, 5)
    sigma_prediction = (
        functional.avg_pool2d(prediction * prediction, 11, 1, 5)
        - mu_prediction.square()
    )
    sigma_target = (
        functional.avg_pool2d(target * target, 11, 1, 5) - mu_target.square()
    )
    covariance = (
        functional.avg_pool2d(prediction * target, 11, 1, 5)
        - mu_prediction * mu_target
    )
    similarity = (
        (2 * mu_prediction * mu_target + c1) * (2 * covariance + c2)
    ) / (
        (mu_prediction.square() + mu_target.square() + c1)
        * (sigma_prediction + sigma_target + c2)
    )
    return 1 - similarity.mean()


class ReconstructionLoss(nn.Module):
    def __init__(self, structural_weight: float = 0.35) -> None:
        super().__init__()
        self.structural_weight = structural_weight

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        pixel_loss = functional.smooth_l1_loss(prediction, target)
        structural_loss = structural_similarity_loss(prediction, target)
        return pixel_loss + self.structural_weight * structural_loss
