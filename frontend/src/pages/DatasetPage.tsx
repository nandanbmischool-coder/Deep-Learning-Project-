import { useState } from "react";
import { ChartCard, DatasetChart } from "../components/charts";
import { Badge, EmptyState, ErrorState, Notice, PageHeader, Spinner } from "../components/ui";
import { api, toApiError } from "../lib/api";
import { useGet } from "../lib/useApi";
import { dt } from "../lib/utils";

interface Status { available: boolean; counts: Record<string, Record<string, number>>; validation_running: boolean; validation_progress: { done: number; total: number } | null; validation_error: string | null; instructions: string; has_report: boolean }
interface Report {
  created_at: string; status: string; issues: string[]; counts: Record<string, Record<string, number>>; totals: Record<string, number>; imbalance_ratio: number | null;
  corrupt_count: number; corrupt_examples: string[]; empty_folders: string[];
  duplicates: { exact_groups: number; exact_groups_across_splits: number; exact_examples: string[][]; near_duplicate_pairs_total: number; near_duplicate_pairs_across_splits_examples: [string, string, number][]; near_duplicate_method: string };
  patient_group_overlap: Record<string, number>; leakage_note: string;
}

export default function DatasetPage() {
  const st = useGet<Status>("/dataset/status", { poll: 3000 });
  const rep = useGet<{ report: Report | null }>("/dataset/summary", { poll: st.data?.validation_running ? 3000 : 0 });
  const [err, setErr] = useState("");
  if (st.loading) return <Spinner />;
  if (st.error) return <ErrorState error={st.error} onRetry={st.reload} />;
  const s = st.data!, r = rep.data?.report;
  const chartData = Object.entries(s.counts).map(([split, c]) => ({ split, NORMAL: c.NORMAL ?? 0, PNEUMONIA: c.PNEUMONIA ?? 0 }));
  const start = async () => { setErr(""); try { await api.post("/dataset/validate"); st.reload(); } catch (e) { setErr(toApiError(e).message); } };

  return (
    <>
      <PageHeader title="Dataset" sub="Availability, class balance and integrity checks for the chest X-ray dataset." actions={<button className="btn-primary" disabled={!s.available || s.validation_running} onClick={start}>{s.validation_running ? `Validating… ${s.validation_progress ? `${s.validation_progress.done}/${s.validation_progress.total}` : ""}` : "Run validation"}</button>} />
      {err && <ErrorState error={err} />}
      {!s.available ? (
        <EmptyState title="Dataset not found"><p>{s.instructions}</p>
          <pre className="mt-3 overflow-x-auto rounded bg-ink p-3 text-left text-xs text-white">{`pip install kaggle\npython scripts/download_dataset.py`}</pre></EmptyState>
      ) : (
        <div className="grid gap-4 md:grid-cols-2">
          <ChartCard title="Images per split and class"><DatasetChart data={chartData} /></ChartCard>
          <section className="card p-4 text-sm">
            <h3 className="text-lg">Counts</h3>
            <table className="mt-2 w-full"><thead className="text-left text-xs text-mute"><tr><th>Split</th><th>NORMAL</th><th>PNEUMONIA</th></tr></thead>
              <tbody>{chartData.map((d) => <tr key={d.split} className="border-t border-line"><td className="py-1.5">{d.split}</td><td>{d.NORMAL}</td><td>{d.PNEUMONIA}</td></tr>)}</tbody></table>
            <p className="mt-2 text-xs text-mute">Quick counts by file extension (unverified). Run validation for readable-image counts.</p>
          </section>
        </div>
      )}
      {s.validation_error && <div className="mt-4"><ErrorState error={s.validation_error} /></div>}
      {r && (
        <section className="mt-4 space-y-4">
          <div className="card p-4 text-sm">
            <div className="flex flex-wrap items-center justify-between gap-2"><h3 className="text-lg">Validation report</h3><span><Badge tone={r.status === "ok" ? "moss" : "amber"}>{r.status}</Badge> <span className="text-xs text-mute">{dt(r.created_at)}</span></span></div>
            <dl className="mt-3 grid grid-cols-2 gap-3 md:grid-cols-4">
              <div><dt className="text-xs text-mute">Corrupt / unreadable</dt><dd className="font-display text-2xl">{r.corrupt_count}</dd></div>
              <div><dt className="text-xs text-mute">Exact duplicate groups (across splits)</dt><dd className="font-display text-2xl">{r.duplicates.exact_groups_across_splits}</dd></div>
              <div><dt className="text-xs text-mute">Near-duplicate pairs</dt><dd className="font-display text-2xl">{r.duplicates.near_duplicate_pairs_total}</dd></div>
              <div><dt className="text-xs text-mute">Imbalance ratio</dt><dd className="font-display text-2xl">{r.imbalance_ratio ? r.imbalance_ratio.toFixed(2) : "n/a"}</dd></div>
            </dl>
            <h4 className="mt-4 font-medium">Leakage checks</h4>
            <p className="text-xs text-mute">Shared patient groups: train∩val {r.patient_group_overlap.train_val}, train∩test {r.patient_group_overlap.train_test}, val∩test {r.patient_group_overlap.val_test}. {r.leakage_note}</p>
            <p className="mt-1 text-xs text-mute">{r.duplicates.near_duplicate_method}</p>
          </div>
          {r.issues.length > 0 && <Notice tone="warn"><ul className="list-disc pl-5">{r.issues.map((i) => <li key={i}>{i}</li>)}</ul></Notice>}
        </section>
      )}
      {!r && s.available && <div className="mt-4"><Notice>No validation report yet. Run validation to check for corrupt files, duplicates and leakage.</Notice></div>}
      <section className="card mt-4 p-4 text-sm">
        <h3 className="text-lg">Loading the dataset</h3>
        <p className="mt-1">{s.instructions}</p>
      </section>
    </>
  );
}
