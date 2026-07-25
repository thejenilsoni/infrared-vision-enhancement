from __future__ import annotations

import random
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional
from torch.utils.data import Dataset
from torchvision.transforms import functional as transforms

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


class PairedInfraredDataset(Dataset[tuple[Tensor, Tensor]]):
    """Loads aligned pairs from `<root>/infrared` and `<root>/visible`."""

    def __init__(
        self,
        root: str | Path,
        crop_size: int = 256,
        augment: bool = True,
    ) -> None:
        self.root = Path(root)
        self.crop_size = crop_size
        self.augment = augment
        infrared_directory = self.root / "infrared"
        visible_directory = self.root / "visible"

        infrared = {
            path.stem: path
            for path in infrared_directory.iterdir()
            if path.suffix.lower() in SUPPORTED_SUFFIXES
        }
        visible = {
            path.stem: path
            for path in visible_directory.iterdir()
            if path.suffix.lower() in SUPPORTED_SUFFIXES
        }
        common = sorted(infrared.keys() & visible.keys())
        if not common:
            raise ValueError(
                f"No aligned image stems found below {self.root}. "
                "Expected infrared/ and visible/ directories."
            )
        self.pairs = [(infrared[stem], visible[stem]) for stem in common]

    def __len__(self) -> int:
        return len(self.pairs)

    def _aligned_crop(self, infrared: Tensor, visible: Tensor) -> tuple[Tensor, Tensor]:
        _, height, width = infrared.shape
        pad_height = max(0, self.crop_size - height)
        pad_width = max(0, self.crop_size - width)
        if pad_height or pad_width:
            padding = [0, 0, pad_width, pad_height]
            infrared = functional.pad(infrared, padding, mode="reflect")
            visible = functional.pad(visible, padding, mode="reflect")
            _, height, width = infrared.shape

        top = random.randint(0, height - self.crop_size)
        left = random.randint(0, width - self.crop_size)
        slices = (..., slice(top, top + self.crop_size), slice(left, left + self.crop_size))
        return infrared[slices], visible[slices]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        infrared_path, visible_path = self.pairs[index]
        with Image.open(infrared_path) as source:
            infrared = transforms.to_tensor(source.convert("L"))
        with Image.open(visible_path) as target:
            visible = transforms.to_tensor(target.convert("RGB"))

        if visible.shape[-2:] != infrared.shape[-2:]:
            visible = functional.interpolate(
                visible.unsqueeze(0),
                size=infrared.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        infrared, visible = self._aligned_crop(infrared, visible)
        if self.augment and random.random() < 0.5:
            infrared = torch.flip(infrared, dims=(-1,))
            visible = torch.flip(visible, dims=(-1,))
        return infrared, visible
