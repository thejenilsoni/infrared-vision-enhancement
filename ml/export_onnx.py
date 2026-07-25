from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import InfraredColorizationUNet


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a trained model to ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/iris-colorizer.onnx"))
    parser.add_argument("--opset", type=int, default=18)
    arguments = parser.parse_args()

    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=True)
    model = InfraredColorizationUNet()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)

    example = torch.rand(1, 1, 256, 256)
    torch.onnx.export(
        model,
        example,
        arguments.output,
        input_names=["infrared"],
        output_names=["colorized"],
        dynamic_axes={
            "infrared": {0: "batch", 2: "height", 3: "width"},
            "colorized": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=arguments.opset,
        dynamo=True,
    )
    print(f"Exported {arguments.output}")


if __name__ == "__main__":
    main()
