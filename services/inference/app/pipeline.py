from __future__ import annotations

import base64
import io
import math
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from PIL import Image, ImageFilter, ImageOps

MAX_PIXELS = 32_000_000
DISCLOSURE = (
    "Perceptual colorization is an interpretation aid and does not reconstruct "
    "ground-truth visible-spectrum color. Inspect the uncertainty map before "
    "using enhanced details for decisions."
)


@dataclass(frozen=True)
class ProcessingConfig:
    color_map: str = "thermal"
    denoise_strength: int = 38
    detail_strength: int = 62
    gamma: float = 0.92
    hotspot_percentile: int = 91

    def validated(self) -> "ProcessingConfig":
        if self.color_map not in {"thermal", "inferno", "viridis", "grayscale"}:
            raise ValueError(f"Unsupported color map: {self.color_map}")
        if not 0 <= self.denoise_strength <= 100:
            raise ValueError("denoise_strength must be between 0 and 100")
        if not 0 <= self.detail_strength <= 100:
            raise ValueError("detail_strength must be between 0 and 100")
        if not 0.5 <= self.gamma <= 1.5:
            raise ValueError("gamma must be between 0.5 and 1.5")
        if not 75 <= self.hotspot_percentile <= 99:
            raise ValueError("hotspot_percentile must be between 75 and 99")
        return self


@dataclass(frozen=True)
class Metrics:
    entropy: float
    contrast: float
    sharpness: float
    edge_density: float
    dynamic_range: float

    def to_camel_dict(self) -> dict[str, float]:
        return {
            "entropy": self.entropy,
            "contrast": self.contrast,
            "sharpness": self.sharpness,
            "edgeDensity": self.edge_density,
            "dynamicRange": self.dynamic_range,
        }


@dataclass(frozen=True)
class Region:
    id: int
    label: str
    confidence: float
    x: float
    y: float
    width: float
    height: float
    mean_intensity: float

    def to_camel_dict(self) -> dict[str, int | str | float]:
        values = asdict(self)
        values["meanIntensity"] = values.pop("mean_intensity")
        return values


@dataclass(frozen=True)
class Analysis:
    request_id: str
    width: int
    height: int
    bit_depth: int
    enhanced_image: str
    colorized_image: str
    uncertainty_map: str
    regions: list[Region]
    input_metrics: Metrics
    output_metrics: Metrics
    processing_time_ms: float

    def to_payload(self) -> dict[str, object]:
        return {
            "requestId": self.request_id,
            "width": self.width,
            "height": self.height,
            "bitDepth": self.bit_depth,
            "enhancedImage": self.enhanced_image,
            "colorizedImage": self.colorized_image,
            "uncertaintyMap": self.uncertainty_map,
            "regions": [region.to_camel_dict() for region in self.regions],
            "inputMetrics": self.input_metrics.to_camel_dict(),
            "outputMetrics": self.output_metrics.to_camel_dict(),
            "processingTimeMs": self.processing_time_ms,
            "disclosure": DISCLOSURE,
        }


COLOR_STOPS: dict[str, tuple[tuple[float, tuple[int, int, int]], ...]] = {
    "thermal": (
        (0.0, (3, 3, 22)),
        (0.20, (24, 21, 87)),
        (0.42, (50, 76, 154)),
        (0.62, (195, 45, 114)),
        (0.80, (247, 111, 55)),
        (1.0, (255, 239, 127)),
    ),
    "inferno": (
        (0.0, (0, 0, 4)),
        (0.20, (66, 10, 104)),
        (0.42, (147, 38, 103)),
        (0.64, (221, 81, 58)),
        (0.82, (252, 165, 10)),
        (1.0, (252, 255, 164)),
    ),
    "viridis": (
        (0.0, (68, 1, 84)),
        (0.25, (59, 82, 139)),
        (0.50, (33, 145, 140)),
        (0.75, (94, 201, 98)),
        (1.0, (253, 231, 37)),
    ),
    "grayscale": (
        (0.0, (0, 0, 0)),
        (1.0, (255, 255, 255)),
    ),
}

UNCERTAINTY_STOPS = (
    (0.0, (7, 18, 38)),
    (0.35, (24, 126, 157)),
    (0.62, (70, 222, 175)),
    (0.80, (255, 222, 71)),
    (1.0, (240, 58, 88)),
)


def load_grayscale(content: bytes) -> tuple[np.ndarray, int]:
    if not content:
        raise ValueError("The uploaded file is empty")

    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    try:
        with Image.open(io.BytesIO(content)) as source:
            source.load()
            width, height = source.size
            if width * height > MAX_PIXELS:
                raise ValueError("Image exceeds the 32 megapixel safety limit")
            bit_depth = 16 if source.mode in {"I", "I;16", "I;16B", "I;16L"} else 8
            image = ImageOps.exif_transpose(source).convert("F")
            array = np.asarray(image, dtype=np.float32)
    except (OSError, ValueError) as error:
        raise ValueError("Unsupported or corrupted image") from error

    finite = np.isfinite(array)
    if not finite.any():
        raise ValueError("Image contains no finite pixel data")
    fill = float(np.median(array[finite]))
    return np.where(finite, array, fill), bit_depth


def robust_normalize(array: np.ndarray) -> np.ndarray:
    low, high = np.percentile(array, (1.0, 99.5))
    if high - low < 1e-8:
        minimum, maximum = float(array.min()), float(array.max())
        low, high = minimum, maximum
    if high - low < 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _as_u8(array: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)


def enhance(array: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    normalized = robust_normalize(array)
    image = Image.fromarray(_as_u8(normalized), mode="L")

    if config.denoise_strength > 5:
        size = 5 if config.denoise_strength >= 70 else 3
        denoised = image.filter(ImageFilter.MedianFilter(size=size))
        alpha = config.denoise_strength / 130.0
        image = Image.blend(image, denoised, alpha=min(alpha, 0.76))

    cutoff = max(0, min(4, round(config.denoise_strength / 35)))
    image = ImageOps.autocontrast(image, cutoff=cutoff)

    if config.detail_strength > 0:
        image = image.filter(
            ImageFilter.UnsharpMask(
                radius=0.8 + config.detail_strength / 42.0,
                percent=70 + config.detail_strength * 2,
                threshold=max(1, 8 - config.detail_strength // 15),
            )
        )

    enhanced = np.asarray(image, dtype=np.float32) / 255.0
    gamma_corrected = np.power(
        np.clip(enhanced, 0.0, 1.0), 1.0 / config.gamma
    )
    return np.clip(gamma_corrected, 0.0, 1.0).astype(np.float32)


def apply_color_map(
    grayscale: np.ndarray,
    stops: Iterable[tuple[float, tuple[int, int, int]]],
) -> np.ndarray:
    stop_list = tuple(stops)
    positions = np.array([item[0] for item in stop_list], dtype=np.float32)
    colors = np.array([item[1] for item in stop_list], dtype=np.float32)
    flat = np.clip(grayscale, 0.0, 1.0).reshape(-1)
    channels = [
        np.interp(flat, positions, colors[:, channel]) for channel in range(3)
    ]
    rgb = np.stack(channels, axis=1).reshape((*grayscale.shape, 3))
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def estimate_uncertainty(original: np.ndarray, enhanced: np.ndarray) -> np.ndarray:
    source = Image.fromarray(_as_u8(original), mode="L")
    local_mean = np.asarray(
        source.filter(ImageFilter.GaussianBlur(radius=2.2)), dtype=np.float32
    ) / 255.0
    squared = Image.fromarray(_as_u8(np.square(original)), mode="L")
    local_square_mean = np.asarray(
        squared.filter(ImageFilter.GaussianBlur(radius=2.2)), dtype=np.float32
    ) / 255.0

    variance = np.maximum(local_square_mean - np.square(local_mean), 0.0)
    texture_uncertainty = robust_normalize(np.sqrt(variance))
    saturation = np.clip(
        np.maximum(0.08 - original, original - 0.92) / 0.08, 0.0, 1.0
    )
    reconstruction_delta = np.abs(enhanced - original)
    uncertainty = (
        0.48 * texture_uncertainty
        + 0.32 * saturation
        + 0.20 * robust_normalize(reconstruction_delta)
    )
    return np.clip(uncertainty, 0.0, 1.0).astype(np.float32)


def _components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[tuple[int, int, int, int, int]] = []

    for start_y, start_x in zip(*np.nonzero(mask)):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        min_x = max_x = int(start_x)
        min_y = max_y = int(start_y)
        area = 0

        while stack:
            y, x = stack.pop()
            area += 1
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and mask[ny, nx]
                    and not visited[ny, nx]
                ):
                    visited[ny, nx] = True
                    stack.append((ny, nx))

        components.append((min_x, min_y, max_x, max_y, area))
    return components


def detect_regions(
    enhanced: np.ndarray, percentile: int, max_regions: int = 12
) -> list[Region]:
    scale = max(1, math.ceil(max(enhanced.shape) / 600))
    sampled = enhanced[::scale, ::scale]
    threshold = float(np.percentile(sampled, percentile))
    mask = sampled >= threshold
    minimum_area = max(6, int(mask.size * 0.00012))
    components = [
        component
        for component in _components(mask)
        if component[4] >= minimum_area
    ]
    components.sort(key=lambda item: item[4], reverse=True)

    height, width = sampled.shape
    regions: list[Region] = []
    for index, (min_x, min_y, max_x, max_y, area) in enumerate(
        components[:max_regions], start=1
    ):
        crop = sampled[min_y : max_y + 1, min_x : max_x + 1]
        mean_intensity = float(crop.mean())
        contrast = max(0.0, (mean_intensity - threshold) / max(1 - threshold, 0.01))
        confidence = float(np.clip(0.58 + 0.34 * contrast, 0.58, 0.94))
        regions.append(
            Region(
                id=index,
                label="thermal anomaly",
                confidence=round(confidence, 4),
                x=round(min_x / width, 6),
                y=round(min_y / height, 6),
                width=round((max_x - min_x + 1) / width, 6),
                height=round((max_y - min_y + 1) / height, 6),
                mean_intensity=round(mean_intensity, 4),
            )
        )
    return regions


def calculate_metrics(array: np.ndarray) -> Metrics:
    normalized = np.clip(array, 0.0, 1.0)
    values = _as_u8(normalized)
    histogram = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram[histogram > 0] / histogram.sum()
    entropy = float(-(probabilities * np.log2(probabilities)).sum())

    horizontal = np.abs(np.diff(normalized, axis=1))
    vertical = np.abs(np.diff(normalized, axis=0))
    edge_density = float(
        (horizontal > 0.08).mean() * 0.5 + (vertical > 0.08).mean() * 0.5
    )
    sharpness = float(
        np.mean(horizontal) * 0.5 + np.mean(vertical) * 0.5
    )
    low, high = np.percentile(normalized, (1, 99))
    return Metrics(
        entropy=round(entropy, 4),
        contrast=round(float(np.std(normalized)), 4),
        sharpness=round(sharpness, 4),
        edge_density=round(edge_density, 4),
        dynamic_range=round(float(high - low), 4),
    )


def encode_png(array: np.ndarray) -> str:
    if array.ndim == 2:
        image = Image.fromarray(_as_u8(array), mode="L")
    else:
        image = Image.fromarray(array.astype(np.uint8), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def analyze(content: bytes, config: ProcessingConfig | None = None) -> Analysis:
    started = time.perf_counter()
    active_config = (config or ProcessingConfig()).validated()
    raw, bit_depth = load_grayscale(content)
    normalized = robust_normalize(raw)
    enhanced = enhance(raw, active_config)
    colorized = apply_color_map(enhanced, COLOR_STOPS[active_config.color_map])
    uncertainty = estimate_uncertainty(normalized, enhanced)
    uncertainty_rgb = apply_color_map(uncertainty, UNCERTAINTY_STOPS)
    regions = detect_regions(enhanced, active_config.hotspot_percentile)

    return Analysis(
        request_id=uuid.uuid4().hex[:8],
        width=int(raw.shape[1]),
        height=int(raw.shape[0]),
        bit_depth=bit_depth,
        enhanced_image=encode_png(enhanced),
        colorized_image=encode_png(colorized),
        uncertainty_map=encode_png(uncertainty_rgb),
        regions=regions,
        input_metrics=calculate_metrics(normalized),
        output_metrics=calculate_metrics(enhanced),
        processing_time_ms=round((time.perf_counter() - started) * 1000, 2),
    )
