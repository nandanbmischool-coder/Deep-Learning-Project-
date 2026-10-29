import { Search, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, toApiError, type Page, type Prediction } from "../lib/api";
import { Badge, EmptyState, ErrorState, Modal, PageHeader, Spinner, classTone } from "../components/ui";
import { useGet } from "../lib/useApi";
import { dt, ms } from "../lib/utils";

export default function HistoryPage() {
  const [q, setQ] = useState("");
  const [qDebounced, setQd] = useState("");
  const [cls, setCls] = useState("");
  const [page, setPage] = useState(1);
  const [del, setDel] = useState<Prediction | null>(null);
  const [delErr, setDelErr] = useState("");
  useEffect(() => { const t = setTimeout(() => { setQd(q); setPage(1); }, 300); return () => clearTimeout(t); }, [q]);
  const url = `/predictions?page=${page}&page_size=10${qDebounced ? `&q=${encodeURIComponent(qDebounced)}` : ""}${cls ? `&predicted_class=${cls}` : ""}`;
  const { data, error, loading, reload } = useGet<Page<Prediction>>(url);
  const pages = data ? Math.max(1, Math.ceil(data.total / data.page_size)) : 1;

  const confirmDelete = async () => {
    if (!del) return;
    try { await api.delete(`/predictions/${del.analysis_id}`); setDel(null); setDelErr(""); reload(); } catch (e) { setDelErr(toApiError(e).message); }
  };

  return (
    <>
      <PageHeader title="Analysis history" sub="Only your own analyses are listed." />
      <div className="mb-4 flex flex-wrap gap-3">
        <div className="relative min-w-[220px] flex-1">
          <Search className="absolute left-3 top-2.5 h-4 w-4 text-mute" aria-hidden />
          <input className="input pl-9" placeholder="Search by analysis ID or file name" aria-label="Search analyses" value={q} onChange={(e) => setQ(e.target.value)} />
        </div>
        <select className="input !w-auto" aria-label="Filter by predicted class" value={cls} onChange={(e) => { setCls(e.target.value); setPage(1); }}>
          <option value="">All classes</option><option value="NORMAL">NORMAL</option><option value="PNEUMONIA">PNEUMONIA</option>
        </select>
      </div>
      {loading ? <Spinner /> : error ? <ErrorState error={error} onRetry={reload} /> : !data?.items.length ? (
        <EmptyState title={q || cls ? "No analyses match those filters" : "No analyses yet"} action={!q && !cls ? <Link className="btn-primary" to="/app/analyze">Analyze an X-ray</Link> : undefined} />
      ) : (
        <div className="card overflow-x-auto">
          <table className="w-full min-w-[720px] text-sm">
            <thead className="border-b border-line text-left text-xs text-mute"><tr><th className="p-3">Image</th><th>Analysis ID</th><th>Date</th><th>Prediction</th><th>Model</th><th>Latency</th><th /></tr></thead>
            <tbody>
              {data.items.map((p) => (
                <tr key={p.analysis_id} className="border-b border-line last:border-0">
                  <td className="p-3"><img src={p.assets.thumbnail} alt="" className="h-12 w-12 rounded bg-box object-cover" /></td>
                  <td className="font-mono text-xs">{p.analysis_id}</td><td>{dt(p.created_at)}</td>
                  <td><Badge tone={classTone(p.predicted_class)}>{p.predicted_class}</Badge></td>
                  <td className="max-w-[160px] truncate text-xs" title={p.model_version}>{p.model_version}</td><td>{ms(p.inference_ms)}</td>
                  <td className="space-x-2 whitespace-nowrap pr-3 text-right">
                    <Link className="btn-ghost !py-1" to={`/app/history/${p.analysis_id}`}>View</Link>
                    <button className="btn-ghost !p-1.5" aria-label={`Delete analysis ${p.analysis_id}`} onClick={() => setDel(p)}><Trash2 className="h-4 w-4 text-brick" /></button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {data && data.total > 0 && (
        <div className="mt-4 flex items-center justify-between text-sm">
          <span className="text-mute">{data.total} result{data.total === 1 ? "" : "s"}</span>
          <div className="flex items-center gap-2">
            <button className="btn-ghost" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>Previous</button>
            <span>Page {page} of {pages}</span>
            <button className="btn-ghost" disabled={page >= pages} onClick={() => setPage((p) => p + 1)}>Next</button>
          </div>
        </div>
      )}
      {del && (
        <Modal title="Delete this analysis?" onClose={() => setDel(null)}>
          <p>This permanently removes the record and its stored images ({del.analysis_id}).</p>
          {delErr && <p role="alert" className="mt-2 text-brick">{delErr}</p>}
          <div className="mt-4 flex justify-end gap-2"><button className="btn-ghost" onClick={() => setDel(null)}>Cancel</button><button className="btn-danger" onClick={confirmDelete}>Delete analysis</button></div>
        </Modal>
      )}
    </>
  );
}
