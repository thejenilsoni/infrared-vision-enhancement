"use client";

import { ChangeEvent, DragEvent, useEffect, useRef, useState } from "react";
import { Icon } from "@/components/Icon";
import { analyzeImage } from "@/lib/api";
import type {
  AnalysisResponse,
  AnalysisSettings,
  ColorMap,
  ImageMetrics
} from "@/lib/types";

const DEFAULT_SETTINGS: AnalysisSettings = {
  colorMap: "thermal",
  denoiseStrength: 38,
  detailStrength: 62,
  gamma: 0.92,
  hotspotPercentile: 91
};

const EMPTY_METRICS: ImageMetrics = {
  entropy: 0,
  contrast: 0,
  sharpness: 0,
  edgeDensity: 0,
  dynamicRange: 0
};

const colorMaps: Array<{ value: ColorMap; label: string; gradient: string }> = [
  {
    value: "thermal",
    label: "Thermal",
    gradient: "linear-gradient(90deg,#060327,#4145b8,#df3564,#ffce49)"
  },
  {
    value: "inferno",
    label: "Inferno",
    gradient: "linear-gradient(90deg,#05010a,#550f6d,#d5493f,#f8f27a)"
  },
  {
    value: "viridis",
    label: "Viridis",
    gradient: "linear-gradient(90deg,#440154,#31688e,#35b779,#fde725)"
  },
  {
    value: "grayscale",
    label: "Mono",
    gradient: "linear-gradient(90deg,#080b0d,#f5f7f8)"
  }
];

async function loadDemoFile(): Promise<File> {
  const image = new Image();
  image.src = "/demo-thermal.svg";
  await image.decode();

  const canvas = document.createElement("canvas");
  canvas.width = 1280;
  canvas.height = 800;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("This browser cannot prepare the demo image.");
  context.drawImage(image, 0, 0, canvas.width, canvas.height);

  const blob = await new Promise<Blob | null>((resolve) =>
    canvas.toBlob(resolve, "image/png", 0.96)
  );
  if (!blob) throw new Error("The demo image could not be prepared.");
  return new File([blob], "demo-thermal-scene.png", { type: "image/png" });
}

function Metric({
  label,
  value,
  suffix = ""
}: {
  label: string;
  value: number;
  suffix?: string;
}) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>
        {value.toFixed(2)}
        <small>{suffix}</small>
      </strong>
    </div>
  );
}

function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  unit = "",
  onChange
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  unit?: string;
  onChange: (value: number) => void;
}) {
  const percentage = ((value - min) / (max - min)) * 100;
  return (
    <label className="control">
      <span>
        {label}
        <b>
          {value}
          {unit}
        </b>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        style={{ "--progress": `${percentage}%` } as React.CSSProperties}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function BeforeAfter({
  before,
  after,
  regions
}: {
  before: string;
  after: string;
  regions: AnalysisResponse["regions"];
}) {
  const [position, setPosition] = useState(48);

  return (
    <div className="comparison">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={before} alt="Original infrared input" />
      <div className="after-layer" style={{ clipPath: `inset(0 0 0 ${position}%)` }}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={after} alt="Enhanced color interpretation" />
        {regions.map((region) => (
          <div
            className="region-box"
            key={region.id}
            style={{
              left: `${region.x * 100}%`,
              top: `${region.y * 100}%`,
              width: `${region.width * 100}%`,
              height: `${region.height * 100}%`
            }}
          >
            <span>
              {region.label} · {Math.round(region.confidence * 100)}%
            </span>
          </div>
        ))}
      </div>
      <div className="divider" style={{ left: `${position}%` }}>
        <span />
      </div>
      <input
        className="comparison-range"
        aria-label="Compare original and enhanced image"
        type="range"
        min="0"
        max="100"
        value={position}
        onChange={(event) => setPosition(Number(event.target.value))}
      />
      <div className="image-label label-left">Source IR</div>
      <div className="image-label label-right">Interpreted</div>
    </div>
  );
}

export default function Dashboard() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [sourceUrl, setSourceUrl] = useState("/demo-thermal.svg");
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [activeView, setActiveView] = useState<"comparison" | "uncertainty">(
    "comparison"
  );

  useEffect(() => {
    return () => {
      if (sourceUrl.startsWith("blob:")) URL.revokeObjectURL(sourceUrl);
    };
  }, [sourceUrl]);

  function updateSetting<Key extends keyof AnalysisSettings>(
    key: Key,
    value: AnalysisSettings[Key]
  ) {
    setSettings((current) => ({ ...current, [key]: value }));
  }

  function selectFile(selected: File | undefined) {
    if (!selected) return;
    if (!selected.type.startsWith("image/")) {
      setError("Choose a PNG, JPEG, TIFF, or WebP infrared image.");
      return;
    }
    if (selected.size > 25 * 1024 * 1024) {
      setError("The maximum image size is 25 MB.");
      return;
    }
    setError("");
    setResult(null);
    setFile(selected);
    setSourceUrl(URL.createObjectURL(selected));
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    selectFile(event.dataTransfer.files[0]);
  }

  async function runAnalysis() {
    setLoading(true);
    setError("");
    try {
      let input = file;
      if (!input) {
        input = await loadDemoFile();
      }
      setResult(await analyzeImage(input, settings));
      setActiveView("comparison");
    } catch (reason) {
      setError(
        reason instanceof Error ? reason.message : "Unexpected processing error."
      );
    } finally {
      setLoading(false);
    }
  }

  function handleFileInput(event: ChangeEvent<HTMLInputElement>) {
    selectFile(event.target.files?.[0]);
  }

  function downloadResult() {
    if (!result) return;
    const anchor = document.createElement("a");
    anchor.href = result.colorizedImage;
    anchor.download = `iris-${result.requestId}.png`;
    anchor.click();
  }

  const inputMetrics = result?.inputMetrics ?? EMPTY_METRICS;
  const outputMetrics = result?.outputMetrics ?? EMPTY_METRICS;
  const displayedImage =
    activeView === "uncertainty" && result
      ? result.uncertaintyMap
      : result?.colorizedImage;

  return (
    <main>
      <aside className="rail">
        <div className="brand-mark">
          <Icon name="aperture" />
        </div>
        <nav aria-label="Primary">
          <button className="nav-button active" title="Analyze">
            <Icon name="scan" />
          </button>
          <button className="nav-button" title="Collections">
            <Icon name="layers" />
          </button>
          <button className="nav-button" title="Detections">
            <Icon name="target" />
          </button>
          <button className="nav-button" title="Evaluation">
            <Icon name="activity" />
          </button>
        </nav>
        <div className="rail-status" title="Inference status">
          <span />
        </div>
      </aside>

      <section className="shell">
        <header>
          <div className="wordmark">
            <strong>IRIS</strong>
            <span>Infrared Interpretation Suite</span>
          </div>
          <div className="header-meta">
            <span className="system-status">
              <i /> PIPELINE READY
            </span>
            <span className="session">SESSION / 08F2-A7</span>
          </div>
        </header>

        <div className="content-grid">
          <section className="workspace">
            <div className="section-heading">
              <div>
                <p>VISUAL ANALYSIS</p>
                <h1>Interpretation workspace</h1>
              </div>
              <div className="view-tabs">
                <button
                  className={activeView === "comparison" ? "active" : ""}
                  onClick={() => setActiveView("comparison")}
                >
                  Comparison
                </button>
                <button
                  className={activeView === "uncertainty" ? "active" : ""}
                  onClick={() => setActiveView("uncertainty")}
                  disabled={!result}
                >
                  Uncertainty
                </button>
              </div>
            </div>

            <div
              className="viewport"
              onDragOver={(event) => event.preventDefault()}
              onDrop={handleDrop}
            >
              {activeView === "comparison" && result ? (
                <BeforeAfter
                  before={sourceUrl}
                  after={result.colorizedImage}
                  regions={result.regions}
                />
              ) : displayedImage ? (
                <div className="single-image">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={displayedImage} alt="Model uncertainty visualization" />
                  <div className="uncertainty-legend">
                    <span>Low uncertainty</span>
                    <i />
                    <span>High uncertainty</span>
                  </div>
                </div>
              ) : (
                <div className="source-preview">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={sourceUrl} alt="Infrared source preview" />
                  <div className="scan-line" />
                </div>
              )}

              <div className="viewport-corners" aria-hidden="true">
                <i />
                <i />
                <i />
                <i />
              </div>
            </div>

            <div className="image-strip">
              <div className="file-data">
                <Icon name="image" />
                <div>
                  <strong>{file?.name ?? "DEMO_THERMAL_SCENE.SVG"}</strong>
                  <span>
                    {result
                      ? `${result.width} × ${result.height} · ${result.bitDepth}-BIT`
                      : "REFERENCE SCENE · READY"}
                  </span>
                </div>
              </div>
              <button
                className="secondary-button"
                onClick={() => inputRef.current?.click()}
              >
                <Icon name="upload" /> Replace image
              </button>
              <input
                ref={inputRef}
                type="file"
                accept="image/png,image/jpeg,image/tiff,image/webp,image/svg+xml"
                hidden
                onChange={handleFileInput}
              />
            </div>

            <div className="metrics-row">
              <Metric label="Entropy" value={outputMetrics.entropy} />
              <Metric label="Contrast" value={outputMetrics.contrast} />
              <Metric label="Sharpness" value={outputMetrics.sharpness} />
              <Metric
                label="Edge density"
                value={outputMetrics.edgeDensity * 100}
                suffix="%"
              />
              <div className="gain">
                <span>Interpretability gain</span>
                <strong>
                  {result
                    ? `+${Math.max(
                        0,
                        ((outputMetrics.contrast - inputMetrics.contrast) /
                          Math.max(inputMetrics.contrast, 0.01)) *
                          100
                      ).toFixed(0)}%`
                    : "—"}
                </strong>
              </div>
            </div>

            {error && (
              <div className="error-banner" role="alert">
                <Icon name="info" /> {error}
              </div>
            )}
          </section>

          <aside className="control-panel">
            <div className="panel-heading">
              <div>
                <p>PROCESSING STACK</p>
                <h2>Enhancement controls</h2>
              </div>
              <Icon name="sliders" />
            </div>

            <div className="control-group">
              <h3>Color interpretation</h3>
              <div className="colormap-grid">
                {colorMaps.map((map) => (
                  <button
                    key={map.value}
                    className={settings.colorMap === map.value ? "active" : ""}
                    onClick={() => updateSetting("colorMap", map.value)}
                  >
                    <i style={{ background: map.gradient }} />
                    <span>{map.label}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className="control-group sliders">
              <h3>Image restoration</h3>
              <Slider
                label="Denoise"
                value={settings.denoiseStrength}
                min={0}
                max={100}
                unit="%"
                onChange={(value) => updateSetting("denoiseStrength", value)}
              />
              <Slider
                label="Detail recovery"
                value={settings.detailStrength}
                min={0}
                max={100}
                unit="%"
                onChange={(value) => updateSetting("detailStrength", value)}
              />
              <Slider
                label="Gamma"
                value={settings.gamma}
                min={0.5}
                max={1.5}
                step={0.01}
                onChange={(value) => updateSetting("gamma", value)}
              />
            </div>

            <div className="control-group">
              <h3>Region analysis</h3>
              <Slider
                label="Anomaly threshold"
                value={settings.hotspotPercentile}
                min={75}
                max={99}
                unit="%"
                onChange={(value) => updateSetting("hotspotPercentile", value)}
              />
              <div className="region-summary">
                <div>
                  <Icon name="target" />
                  <span>Regions detected</span>
                </div>
                <strong>{result?.regions.length ?? "—"}</strong>
              </div>
            </div>

            <div className="disclosure">
              <Icon name="shield" />
              <p>
                <strong>Interpretability safeguard</strong>
                Color output is a perceptual aid, not recovered visible-spectrum
                truth. Uncertainty is estimated for every result.
              </p>
            </div>

            <button
              className="process-button"
              disabled={loading}
              onClick={runAnalysis}
            >
              {loading ? (
                <>
                  <i className="spinner" /> Processing pipeline
                </>
              ) : (
                <>
                  <Icon name="scan" /> Run interpretation
                </>
              )}
            </button>
            <button
              className="export-button"
              disabled={!result}
              onClick={downloadResult}
            >
              <Icon name="download" /> Export interpreted image
            </button>
            {result && (
              <p className="runtime">
                REQUEST {result.requestId.toUpperCase()} ·{" "}
                {result.processingTimeMs.toFixed(0)} MS
              </p>
            )}
          </aside>
        </div>
      </section>
    </main>
  );
}
