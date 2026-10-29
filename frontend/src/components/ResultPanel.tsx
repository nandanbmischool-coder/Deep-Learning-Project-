import { Download, FileText } from "lucide-react";
import type { Prediction } from "../lib/api";
import { dt, ms, pct } from "../lib/utils";
import { Badge, Disclaimer, Notice, classTone } from "./ui";

function Panel({ src, title, dl }: { src: string; title: string; dl?: string }) {
  return (
    <figure className="min-w-0">
      <div className="grid aspect-[4/5] place-items-center overflow-hidden rounded bg-box">
        <img src={src} alt={title} className="max-h-full max-w-full object-contain" />
      </div>
      <figcaption className="mt-2 flex items-center justify-between text-xs text-white/70">
        {title}
        {dl && <a className="inline-flex items-center gap-1 text-[#7FD1D8] hover:underline" href={dl} download><Download className="h-3 w-3" aria-hidden />Save</a>}
      </figcaption>
    </figure>
  );
}

export default function ResultPanel({ p }: { p: Prediction }) {
  const probs = [["NORMAL", p.probabilities.NORMAL], ["PNEUMONIA", p.probabilities.PNEUMONIA]] as const;
  return (
    <div className="space-y-4">
      <div className="rounded-panel bg-ink p-4 text-white">
        <div className="grid gap-4 sm:grid-cols-3">
          <Panel src={p.assets.original} title="Original" />
          <Panel src={p.assets.heatmap} title="Grad-CAM heatmap" dl={`${p.assets.heatmap}?download=true`} />
          <Panel src={p.assets.overlay} title={`Overlay (target: ${p.target_class})`} dl={`${p.assets.overlay}?download=true`} />
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-[1fr_1.2fr]">
        <section className="card p-4">
          <p className="text-xs text-mute">Model prediction</p>
          <p className="mt-1 flex items-center gap-3 font-display text-3xl">{p.predicted_class === "PNEUMONIA" ? "Pneumonia pattern" : "Normal pattern"}
            <Badge tone={classTone(p.predicted_class)}>{p.predicted_class}</Badge></p>
          <div className="mt-4 space-y-3">
            {probs.map(([k, v]) => (
              <div key={k}>
                <div className="flex justify-between text-sm"><span>P({k})</span><span className="tabular-nums">{pct(v, 2)}</span></div>
                <div className="mt-1 h-2 rounded bg-paper" role="img" aria-label={`${k} probability ${pct(v, 1)}`}>
                  <div className="h-2 rounded" style={{ width: `${v * 100}%`, background: k === "PNEUMONIA" ? "#B4432E" : "#2F7D5B" }} />
                </div>
              </div>
            ))}
          </div>
          <p className="mt-3 text-xs text-mute">Scores are raw model probabilities (not calibrated). Pneumonia is predicted when P(PNEUMONIA) ≥ {p.threshold.toFixed(3)}, a threshold chosen on validation data.</p>
        </section>

        <section className="card p-4 text-sm">
          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5">
            <dt className="text-mute">Analysis ID</dt><dd className="font-mono text-xs">{p.analysis_id}</dd>
            <dt className="text-mute">Model version</dt><dd>{p.model_version}</dd>
            <dt className="text-mute">Inference time</dt><dd>{ms(p.inference_ms)} (+ {ms(p.gradcam_ms)} Grad-CAM)</dd>
            <dt className="text-mute">Image</dt><dd>{p.image_size.width} × {p.image_size.height}px</dd>
            <dt className="text-mute">Created</dt><dd>{dt(p.created_at)}</dd>
          </dl>
          <div className="mt-4 flex flex-wrap gap-2">
            <a className="btn-ghost" href={`/api/v1/predictions/${p.analysis_id}/report`} download><FileText className="h-4 w-4" aria-hidden />HTML report</a>
            <a className="btn-ghost" href={`/api/v1/predictions/${p.analysis_id}/report?format=json`} download>JSON</a>
          </div>
        </section>
      </div>

      {p.image_validation.status === "warning" && (
        <Notice tone="warn">
          <p className="font-medium">Input check warnings (heuristic)</p>
          <ul className="list-disc pl-5">{p.image_validation.warnings.map((w) => <li key={w}>{w}</li>)}</ul>
          <p className="mt-1 text-xs">{p.image_validation.note}</p>
        </Notice>
      )}
      {p.image_validation.gradcam_flat_map && <Notice tone="warn">Grad-CAM found no positive evidence for the target class, so the heatmap is empty. Nothing was drawn artificially.</Notice>}

      <Notice>
        <p className="font-medium">How to read the heatmap</p>
        <p>Grad-CAM weights the last convolutional feature maps by the gradient of the target class score. Warmer colours mark regions that pushed the model's score up. It explains <em>model activation</em>; it does not show that the model reasons like a clinician and does not prove a medical finding.</p>
      </Notice>
      <Disclaimer />
    </div>
  );
}
