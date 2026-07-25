# Architecture

## Design goals

IRIS is structured around five requirements:

1. Preserve the original measurement and make transformations inspectable.
2. Produce a useful baseline without requiring a proprietary model or GPU.
3. Support learned color reconstruction when paired training data is available.
4. Quantify uncertainty and image-quality changes.
5. Remain deployable on isolated infrastructure.

## Runtime topology

The browser communicates only with a same-origin Next.js route. That server route forwards multipart requests to the inference service, so the internal service URL is never exposed to the browser.

| Component | Responsibility |
|---|---|
| Next.js console | Upload, parameter selection, comparison, overlays, export |
| Next.js API route | Same-origin request proxy and timeout enforcement |
| FastAPI service | Validation, orchestration, response serialization |
| Processing pipeline | Enhancement, color mapping, metrics, regions, uncertainty |
| Training package | Paired dataset loading, model training, checkpoints, ONNX export |

## Deterministic inference pipeline

### 1. Input normalization

The loader preserves 8-bit versus 16-bit metadata, applies EXIF orientation, replaces non-finite pixels, and rejects decompression bombs above 32 megapixels. Intensities are normalized using the 1st and 99.5th percentiles to reduce sensitivity to dead pixels and extreme sensor values.

### 2. Noise suppression

An adjustable median filter suppresses impulse noise. The output is blended with the input instead of replacing it completely, limiting the loss of small structures.

### 3. Contrast and detail restoration

Histogram autocontrast expands the useful range. An unsharp mask restores local edges, with radius, percentage, and threshold coupled to a single user-facing detail control. Gamma correction is applied last in normalized intensity space.

### 4. Color interpretation

The grayscale field is mapped through monotonic, piecewise-linear RGB palettes. The mapping is deterministic and does not claim to infer true material color.

### 5. Region proposals

Candidate thermal anomalies are extracted above an adjustable percentile. Connected components are computed on a bounded working resolution, filtered by relative area, sorted, and returned as normalized boxes. These are intentionally named `thermal anomaly` rather than assigned an unverified semantic class.

### 6. Uncertainty

The uncertainty estimator combines:

- local texture variance;
- saturation near the sensor range boundaries;
- the magnitude of the enhancement transformation.

This is an engineering confidence proxy, not a calibrated probability. A learned pipeline should replace or calibrate it against held-out reconstruction error.

## Learned reconstruction

The residual U-Net maps one normalized infrared channel to three visible channels. Four encoder stages progressively compress spatial context, residual blocks preserve stable gradient flow, and skip-connected decoder stages recover detail. The loss combines:

\[
\mathcal{L} = \mathcal{L}_{SmoothL1} + 0.35(1 - SSIM)
\]

The current training code uses AdamW, cosine learning-rate decay, automatic mixed precision, gradient clipping, deterministic splitting, validation PSNR, and best/latest checkpoint retention.

## Deployment

The web and inference services use independent non-root containers. The inference container exposes a health check; Compose waits for it before starting the web service. The system can run without external APIs after images and dependencies are packaged.

For production:

- terminate TLS at an ingress proxy;
- set upload and request limits at both ingress and application layers;
- mount model weights read-only;
- replace permissive development CORS with the production origin;
- emit structured request metrics without retaining source imagery by default;
- add authentication and authorization appropriate to the deployment.
