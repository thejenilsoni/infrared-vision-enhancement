import type { AnalysisResponse, AnalysisSettings } from "./types";

export async function analyzeImage(
  file: File,
  settings: AnalysisSettings
): Promise<AnalysisResponse> {
  const payload = new FormData();
  payload.append("file", file);
  payload.append("color_map", settings.colorMap);
  payload.append("denoise_strength", String(settings.denoiseStrength));
  payload.append("detail_strength", String(settings.detailStrength));
  payload.append("gamma", String(settings.gamma));
  payload.append("hotspot_percentile", String(settings.hotspotPercentile));

  const response = await fetch("/api/analyze", {
    method: "POST",
    body: payload
  });

  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? "The analysis service could not process this image.");
  }

  return response.json() as Promise<AnalysisResponse>;
}
