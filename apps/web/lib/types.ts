export type ColorMap = "thermal" | "inferno" | "viridis" | "grayscale";

export interface AnalysisSettings {
  colorMap: ColorMap;
  denoiseStrength: number;
  detailStrength: number;
  gamma: number;
  hotspotPercentile: number;
}

export interface Region {
  id: number;
  label: string;
  confidence: number;
  x: number;
  y: number;
  width: number;
  height: number;
  meanIntensity: number;
}

export interface ImageMetrics {
  entropy: number;
  contrast: number;
  sharpness: number;
  edgeDensity: number;
  dynamicRange: number;
}

export interface AnalysisResponse {
  requestId: string;
  width: number;
  height: number;
  bitDepth: number;
  enhancedImage: string;
  colorizedImage: string;
  uncertaintyMap: string;
  regions: Region[];
  inputMetrics: ImageMetrics;
  outputMetrics: ImageMetrics;
  processingTimeMs: number;
  disclosure: string;
}
