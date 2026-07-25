# Model and Pipeline Card

## Summary

IRIS contains two related paths:

- a deterministic enhancement and pseudocolor baseline available without model weights;
- a residual U-Net training implementation for paired infrared-to-visible reconstruction.

No pretrained learned weights are distributed in this repository. Results from the baseline must be described as enhancement and perceptual color interpretation, not learned true-color reconstruction.

## Intended uses

- Improve visual inspection of infrared images.
- Reveal low-contrast boundaries and thermal regions.
- Compare enhancement settings without hiding the source.
- Create a reproducible baseline for colorization research.
- Study whether enhancement improves a separately validated downstream detector.

## Out-of-scope uses

- Inferring the true visible color of an object from single-band infrared alone.
- Identifying people or sensitive attributes.
- Making autonomous safety-critical decisions.
- Treating generated texture as evidence of a physical feature.
- Comparing sensors without radiometric calibration.

## Inputs and outputs

Accepted baseline inputs are PNG, JPEG, TIFF, and WebP files up to 25 MB and 32 megapixels. TIFF inputs may contain 16-bit intensity data, which is normalized for display processing.

Outputs include:

- enhanced monochrome PNG;
- perceptually colorized PNG;
- uncertainty visualization;
- thermal anomaly proposals;
- input and output image metrics;
- processing duration and source dimensions.

## Important limitations

Infrared-to-visible colorization is underdetermined. Different visible surfaces can produce similar infrared intensity, and their relationship changes with wavelength, temperature, emissivity, illumination, sensor response, atmospheric conditions, and time of day.

The baseline uncertainty score is a heuristic derived from texture, saturation, and transformation magnitude. It is not a calibrated likelihood. Region confidences rank thermal salience; they are not probabilities that an object belongs to a semantic class.

A trained model can inherit geographic, climatic, sensor, and dataset biases. Strong results on aligned driving datasets do not establish performance on orbital or airborne imagery.

## Required evaluation before deployment

1. Create sensor-disjoint, geography-disjoint, and time-disjoint test sets.
2. Report PSNR and SSIM, but do not use them as the only quality criteria.
3. Measure perceptual metrics and downstream task performance.
4. Compare against grayscale, histogram equalization, and pseudocolor baselines.
5. Review high-confidence errors and low-uncertainty errors manually.
6. Evaluate haze, cloud, saturation, blur, low signal, and unseen sensor failure sets.
7. Benchmark latency and memory at intended image sizes.
8. Document the precise model checkpoint and data version used.

## Human oversight

The interface keeps the source visible, provides a comparison slider, labels the interpreted view, and exposes uncertainty. An operator should return to the source pixels whenever a generated or enhanced detail affects a decision.
