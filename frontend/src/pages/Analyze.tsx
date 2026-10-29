import { ImagePlus, RefreshCw, Trash2, UploadCloud } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { api, toApiError, type ApiError, type Prediction } from "../lib/api";
import ResultPanel from "../components/ResultPanel";
import { Disclaimer, ErrorState, Notice, PageHeader } from "../components/ui";

const MAX_MB = 10;
const OK_TYPES = ["image/jpeg", "image/png", "image/webp"];

export function validateFile(f: File): string | null {
  if (!OK_TYPES.includes(f.type)) return "Choose a JPEG, PNG or WebP image.";
  if (f.size === 0) return "That file is empty.";
  if (f.size > MAX_MB * 1024 * 1024) return `File is larger than ${MAX_MB} MB.`;
  return null;
}

export default function Analyze() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [fileErr, setFileErr] = useState<string | null>(null);
  const [target, setTarget] = useState("");
  const [progress, setProgress] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);
  const [drag, setDrag] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);

  const pick = useCallback((f: File | undefined) => {
    if (!f) return;
    const e = validateFile(f);
    setFileErr(e); setError(null); setResult(null);
    if (e) { setFile(null); setPreview(null); return; }
    setFile(f); setPreview(URL.createObjectURL(f));
  }, []);

  const clear = () => { setFile(null); setPreview(null); setFileErr(null); setResult(null); setError(null); setProgress(null); if (input.current) input.current.value = ""; };

  const submit = async () => {
    if (!file) return; // never send a request without a valid image
    setBusy(true); setError(null); setProgress(0);
    const fd = new FormData();
    fd.append("file", file);
    if (target) fd.append("target_class", target);
    try {
      const r = await api.post<Prediction>("/predictions", fd, { onUploadProgress: (e) => setProgress(e.total ? Math.round((e.loaded / e.total) * 100) : null) });
      setResult(r.data);
    } catch (e) { setError(toApiError(e)); } finally { setBusy(false); setProgress(null); }
  };

  return (
    <>
      <PageHeader title="Analyze a chest X-ray" sub="Upload a frontal chest radiograph. The model returns a probability for each class and a Grad-CAM view of what influenced the score." />
      <div className="grid gap-6 lg:grid-cols-[minmax(0,360px)_1fr]">
        <section className="space-y-3">
          {!file ? (
            <div
              onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
              onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files[0]); }}
              className={`flex flex-col items-center rounded-panel border-2 border-dashed px-6 py-12 text-center ${drag ? "border-teal bg-teal-soft" : "border-line bg-white"}`}>
              <UploadCloud className="h-8 w-8 text-teal" aria-hidden />
              <p className="mt-3 font-medium">Drop an image here</p>
              <p className="text-xs text-mute">JPEG, PNG or WebP · up to {MAX_MB} MB</p>
              <button className="btn-primary mt-4" onClick={() => input.current?.click()}><ImagePlus className="h-4 w-4" aria-hidden />Choose file</button>
            </div>
          ) : (
            <div className="card p-3">
              <div className="grid aspect-[4/5] place-items-center overflow-hidden rounded bg-box"><img src={preview!} alt="Selected X-ray preview" className="max-h-full max-w-full object-contain" /></div>
              <p className="mt-2 truncate text-xs text-mute">{file.name} · {(file.size / 1024).toFixed(0)} KB</p>
              <div className="mt-2 flex gap-2">
                <button className="btn-ghost flex-1" onClick={() => input.current?.click()} disabled={busy}><RefreshCw className="h-4 w-4" aria-hidden />Replace</button>
                <button className="btn-ghost flex-1" onClick={clear} disabled={busy}><Trash2 className="h-4 w-4" aria-hidden />Remove</button>
              </div>
            </div>
          )}
          <input ref={input} type="file" accept={OK_TYPES.join(",")} className="sr-only" aria-label="Choose X-ray image" onChange={(e) => pick(e.target.files?.[0])} />
          {fileErr && <p role="alert" className="text-sm text-brick">{fileErr}</p>}

          <div>
            <label className="label" htmlFor="target">Explain class</label>
            <select id="target" className="input" value={target} onChange={(e) => setTarget(e.target.value)} disabled={busy}>
              <option value="">Predicted class (default)</option>
              <option value="PNEUMONIA">PNEUMONIA</option>
              <option value="NORMAL">NORMAL</option>
            </select>
          </div>
          <button className="btn-primary w-full" disabled={!file || busy} onClick={submit}>{busy ? "Analyzing…" : "Analyze image"}</button>
          {busy && (
            <div role="status" className="text-xs text-mute">
              {progress !== null && progress < 100 ? `Uploading ${progress}%` : "Running the model and Grad-CAM…"}
              <div className="mt-1 h-1.5 rounded bg-line"><div className="h-1.5 rounded bg-teal transition-all" style={{ width: `${progress ?? 100}%` }} /></div>
            </div>
          )}
          <Disclaimer />
        </section>

        <section className="min-w-0">
          {error?.code === "MODEL_NOT_TRAINED" ? (
            <Notice tone="warn"><p className="font-medium">Model not trained</p><p>No model is active, so no prediction can be made. An administrator can train one in the Training center.</p></Notice>
          ) : error ? <ErrorState error={error} /> : null}
          {result ? <ResultPanel p={result} /> : !error && (
            <div className="grid min-h-[320px] place-items-center rounded-panel border border-dashed border-line bg-white p-8 text-center text-sm text-mute">Results appear here: original, Grad-CAM heatmap and overlay, side by side.</div>
          )}
        </section>
      </div>
    </>
  );
}
