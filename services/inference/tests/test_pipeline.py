from __future__ import annotations

import base64
import io
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT))

from app.pipeline import (  # noqa: E402
    ProcessingConfig,
    analyze,
    apply_color_map,
    calculate_metrics,
    detect_regions,
    enhance,
    robust_normalize,
)


def synthetic_infrared(width: int = 160, height: int = 96) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width]
    background = 0.10 + 0.38 * (x / width) + 0.12 * (y / height)
    warm_object = np.exp(
        -(((x - width * 0.68) / 13) ** 2 + ((y - height * 0.48) / 18) ** 2)
    )
    hot_object = np.exp(
        -(((x - width * 0.30) / 8) ** 2 + ((y - height * 0.72) / 8) ** 2)
    )
    noise = np.random.default_rng(42).normal(0, 0.018, (height, width))
    return np.clip(background + 0.42 * warm_object + 0.58 * hot_object + noise, 0, 1)


def png_bytes(array: np.ndarray) -> bytes:
    image = Image.fromarray(np.rint(array * 255).astype(np.uint8), mode="L")
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


class PipelineTests(unittest.TestCase):
    def test_normalization_is_bounded_and_robust_to_outlier(self) -> None:
        source = synthetic_infrared()
        source[0, 0] = 100
        normalized = robust_normalize(source)
        self.assertEqual(normalized.dtype, np.float32)
        self.assertGreaterEqual(float(normalized.min()), 0)
        self.assertLessEqual(float(normalized.max()), 1)
        self.assertGreater(float(np.median(normalized)), 0.1)

    def test_enhancement_preserves_shape_and_increases_dynamic_range(self) -> None:
        source = synthetic_infrared()
        output = enhance(source, ProcessingConfig())
        self.assertEqual(output.shape, source.shape)
        self.assertGreaterEqual(float(output.min()), 0)
        self.assertLessEqual(float(output.max()), 1)
        self.assertGreater(
            calculate_metrics(output).dynamic_range,
            calculate_metrics(source).dynamic_range,
        )

    def test_color_map_returns_rgb_image(self) -> None:
        source = np.linspace(0, 1, 80, dtype=np.float32).reshape(8, 10)
        output = apply_color_map(
            source,
            ((0.0, (0, 0, 0)), (1.0, (255, 128, 64))),
        )
        self.assertEqual(output.shape, (8, 10, 3))
        self.assertEqual(output.dtype, np.uint8)
        np.testing.assert_array_equal(output[0, 0], [0, 0, 0])
        np.testing.assert_array_equal(output[-1, -1], [255, 128, 64])

    def test_region_detector_finds_hot_objects(self) -> None:
        source = synthetic_infrared()
        regions = detect_regions(source, percentile=88)
        self.assertGreaterEqual(len(regions), 1)
        self.assertTrue(all(0 <= region.confidence <= 1 for region in regions))
        self.assertTrue(all(0 <= region.x < 1 for region in regions))
        self.assertTrue(all(region.width > 0 for region in regions))

    def test_end_to_end_analysis_returns_png_artifacts(self) -> None:
        result = analyze(png_bytes(synthetic_infrared()))
        payload = result.to_payload()
        self.assertEqual(payload["width"], 160)
        self.assertEqual(payload["height"], 96)
        self.assertEqual(payload["bitDepth"], 8)
        self.assertGreater(payload["processingTimeMs"], 0)
        for key in ("enhancedImage", "colorizedImage", "uncertaintyMap"):
            encoded = payload[key]
            self.assertTrue(encoded.startswith("data:image/png;base64,"))
            binary = base64.b64decode(encoded.split(",", 1)[1])
            self.assertGreater(len(binary), 100)

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ProcessingConfig(color_map="rainbow").validated()
        with self.assertRaises(ValueError):
            ProcessingConfig(gamma=3.0).validated()

    def test_constant_image_is_processed_without_nan(self) -> None:
        result = analyze(png_bytes(np.full((32, 48), 0.5, dtype=np.float32)))
        self.assertEqual(result.width, 48)
        self.assertTrue(np.isfinite(result.output_metrics.entropy))


if __name__ == "__main__":
    unittest.main()
