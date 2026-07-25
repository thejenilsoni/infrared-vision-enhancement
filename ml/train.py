from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

from .dataset import PairedInfraredDataset
from .model import InfraredColorizationUNet, ReconstructionLoss


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the paired infrared-to-visible reconstruction model."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/training"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def peak_signal_to_noise_ratio(prediction: Tensor, target: Tensor) -> float:
    error = torch.mean((prediction - target).square()).clamp_min(1e-10)
    return float(10 * torch.log10(1 / error))


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: AdamW | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    scores: list[float] = []

    for infrared, visible in loader:
        infrared = infrared.to(device, non_blocking=True)
        visible = visible.to(device, non_blocking=True)
        if optimizer:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type, enabled=device.type == "cuda"
        ):
            prediction = model(infrared)
            loss = criterion(prediction, visible)

        if optimizer and scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        losses.append(float(loss.detach()))
        scores.append(peak_signal_to_noise_ratio(prediction.detach(), visible))
    return float(np.mean(losses)), float(np.mean(scores))


def main() -> None:
    arguments = parse_arguments()
    seed_everything(arguments.seed)
    arguments.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_dataset = PairedInfraredDataset(
        arguments.data, crop_size=arguments.crop_size, augment=True
    )
    validation_dataset = PairedInfraredDataset(
        arguments.data, crop_size=arguments.crop_size, augment=False
    )
    validation_size = max(1, round(len(training_dataset) * 0.1))
    training_size = len(training_dataset) - validation_size
    if training_size < 1:
        raise ValueError("At least two aligned pairs are required for training")
    indices = torch.randperm(
        len(training_dataset),
        generator=torch.Generator().manual_seed(arguments.seed),
    ).tolist()
    validation_indices = indices[:validation_size]
    training_indices = indices[validation_size:]
    training = Subset(training_dataset, training_indices)
    validation = Subset(validation_dataset, validation_indices)

    loader_options = {
        "batch_size": arguments.batch_size,
        "num_workers": arguments.workers,
        "pin_memory": device.type == "cuda",
    }
    training_loader = DataLoader(training, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation, shuffle=False, **loader_options)

    model = InfraredColorizationUNet().to(device)
    criterion = ReconstructionLoss()
    optimizer = AdamW(
        model.parameters(), lr=arguments.learning_rate, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=arguments.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_score = -float("inf")
    history: list[dict[str, float | int]] = []

    for epoch in range(1, arguments.epochs + 1):
        started = time.perf_counter()
        training_loss, training_psnr = run_epoch(
            model, training_loader, criterion, device, optimizer, scaler
        )
        validation_loss, validation_psnr = run_epoch(
            model, validation_loader, criterion, device
        )
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": round(training_loss, 6),
            "train_psnr": round(training_psnr, 4),
            "validation_loss": round(validation_loss, 6),
            "validation_psnr": round(validation_psnr, 4),
            "seconds": round(time.perf_counter() - started, 2),
        }
        history.append(record)
        print(json.dumps(record))

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": record,
            "arguments": vars(arguments),
        }
        torch.save(checkpoint, arguments.output / "latest.pt")
        if validation_psnr > best_score:
            best_score = validation_psnr
            torch.save(checkpoint, arguments.output / "best.pt")
        (arguments.output / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
