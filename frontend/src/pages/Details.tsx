import { ArrowLeft, Trash2 } from "lucide-react";
import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import ResultPanel from "../components/ResultPanel";
import { ErrorState, Modal, PageHeader, Spinner } from "../components/ui";
import { api, toApiError, type Prediction } from "../lib/api";
import { useGet } from "../lib/useApi";

export default function Details() {
  const { id } = useParams();
  const nav = useNavigate();
  const { data, error, loading, reload } = useGet<Prediction>(`/predictions/${id}`);
  const [confirm, setConfirm] = useState(false);
  const [err, setErr] = useState("");
  if (loading) return <Spinner />;
  if (error) return <><Link to="/app/history" className="mb-4 inline-flex items-center gap-1 text-sm underline"><ArrowLeft className="h-4 w-4" />Back to history</Link><ErrorState error={error} onRetry={reload} /></>;
  const p = data!;
  return (
    <>
      <Link to="/app/history" className="mb-3 inline-flex items-center gap-1 text-sm text-teal-strong underline"><ArrowLeft className="h-4 w-4" />Back to history</Link>
      <PageHeader title="Analysis details" sub={p.filename_hint ? `Uploaded as “${p.filename_hint}”` : undefined} actions={<button className="btn-ghost" onClick={() => setConfirm(true)}><Trash2 className="h-4 w-4 text-brick" aria-hidden />Delete</button>} />
      <ResultPanel p={p} />
      <section className="card mt-4 p-4 text-sm">
        <h3 className="text-lg">Preprocessing applied at inference</h3>
        <ol className="mt-2 list-decimal pl-5">{p.preprocessing?.steps.map((s) => <li key={s}>{s}</li>)}</ol>
        <p className="mt-2 text-xs text-mute">Augmentation: {p.preprocessing?.augmentation}. The original image is stored (re-encoded as PNG) and shown unmodified.</p>
        <h3 className="mt-4 text-lg">Limitations</h3>
        <ul className="mt-1 list-disc pl-5 text-mute">
          <li>The model was trained on one public paediatric dataset and may not generalise to other populations or scanners.</li>
          <li>Probabilities are uncalibrated scores; the decision threshold was chosen on validation data.</li>
          <li>Grad-CAM highlights regions that influenced the score; it is not a localisation of disease.</li>
        </ul>
      </section>
      {confirm && (
        <Modal title="Delete this analysis?" onClose={() => setConfirm(false)}>
          <p>This permanently removes the record and its stored images.</p>
          {err && <p role="alert" className="mt-2 text-brick">{err}</p>}
          <div className="mt-4 flex justify-end gap-2"><button className="btn-ghost" onClick={() => setConfirm(false)}>Cancel</button>
            <button className="btn-danger" onClick={async () => { try { await api.delete(`/predictions/${p.analysis_id}`); nav("/app/history"); } catch (e) { setErr(toApiError(e).message); } }}>Delete analysis</button></div>
        </Modal>
      )}
    </>
  );
}
