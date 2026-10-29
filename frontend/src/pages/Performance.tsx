import { Link } from "react-router-dom";
import { ChartCard, ConfusionMatrix, LossChart, PrChart, RocChart } from "../components/charts";
import { EmptyState, ErrorState, Notice, PageHeader, Spinner } from "../components/ui";
import type { EvalReport, ModelVersion } from "../lib/api";
import { useGet } from "../lib/useApi";
import { dt, ms, num, pct } from "../lib/utils";

interface EvList { items: { id: number; model_version_id: number }[] }

export default function Performance() {
  const active = useGet<{ active: ModelVersion | null }>("/models/active");
  const evs = useGet<EvList>("/evaluations");
  const evId = evs.data?.items.find((e) => e.model_version_id === active.data?.active?.id)?.id;
  const ev = useGet<{ report: EvalReport }>(evId ? `/evaluations/${evId}` : null);

  if (active.loading || evs.loading || ev.loading) return <Spinner />;
  if (active.error) return <ErrorState error={active.error} onRetry={active.reload} />;
  const m = active.data?.active;
  if (!m) return <><PageHeader title="Model performance" /><EmptyState title="Model not trained" action={<Link className="btn-primary" to="/app/training">Go to Training center</Link>}>There is no active model yet, so there are no metrics to show.</EmptyState></>;
  const r = ev.data?.report;
  const ci = (k: string) => { const v = r?.confidence_intervals[k]; return Array.isArray(v) ? `[${v[0].toFixed(3)}, ${v[1].toFixed(3)}]` : "n/a"; };
  const cfg = (m.training_config ?? {}) as Record<string, unknown>;

  return (
    <>
      <PageHeader title="Model performance" sub={`${m.version} · evaluated on the held-out test split${r ? ` · ${dt(r.created_at)}` : ""}`} />
      {!r && <Notice tone="warn">Not evaluated. This model has no stored test evaluation; re-run training (the CLI evaluates automatically).</Notice>}
      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <section className="card p-4 text-sm lg:col-span-1">
          <h3 className="text-lg">Architecture</h3>
          <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
            <dt className="text-mute">Backbone</dt><dd>{m.architecture}</dd>
            <dt className="text-mute">Pretrained</dt><dd>{m.pretrained ? m.pretrained_source : "No — random initialisation"}</dd>
            <dt className="text-mute">Parameters</dt><dd>{m.parameter_count.toLocaleString()}</dd>
            <dt className="text-mute">Size</dt><dd>{m.model_size_mb} MB</dd>
            <dt className="text-mute">Best epoch</dt><dd>{m.best_epoch ?? "n/a"}</dd>
            <dt className="text-mute">Training time</dt><dd>{m.training_seconds ? `${(m.training_seconds / 60).toFixed(1)} min on ${m.device}` : "n/a"}</dd>
            <dt className="text-mute">Threshold</dt><dd>{m.threshold.toFixed(3)}</dd>
            <dt className="text-mute">Checkpoint</dt><dd className="break-all">{m.version}</dd>
          </dl>
          <p className="mt-2 text-xs text-mute">Threshold: {m.threshold_method}.</p>
          <h3 className="mt-4 text-lg">Training configuration</h3>
          <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
            {["epochs", "head_epochs", "batch_size", "lr", "finetune_lr", "fine_tune_mode", "patience", "seed", "val_fraction", "horizontal_flip"].map((k) => (<><dt key={`k${k}`} className="text-mute">{k}</dt><dd key={`v${k}`}>{String(cfg[k] ?? "n/a")}</dd></>))}
          </dl>
        </section>

        {r && (
          <section className="card p-4 lg:col-span-2">
            <h3 className="text-lg">Test metrics <span className="text-sm text-mute">(positive class: PNEUMONIA, n = {r.n_samples})</span></h3>
            <table className="mt-2 w-full text-sm">
              <thead className="text-left text-xs text-mute"><tr><th className="py-1">Metric</th><th>Value</th><th>95% bootstrap CI</th></tr></thead>
              <tbody>
                {([["Accuracy", r.metrics.accuracy, "accuracy"], ["Precision", r.metrics.precision, "precision"], ["Recall / sensitivity", r.metrics.recall, "recall"], ["Specificity", r.metrics.specificity, "specificity"], ["F1-score", r.metrics.f1, "f1"], ["ROC-AUC", r.metrics.roc_auc, "roc_auc"], ["PR-AUC", r.metrics.pr_auc, "pr_auc"]] as const).map(([l, v, k]) => (
                  <tr key={k} className="border-t border-line"><td className="py-1.5">{l}</td><td className="tabular-nums">{k === "roc_auc" || k === "pr_auc" ? num(v) : pct(v, 2)}</td><td className="tabular-nums text-mute">{ci(k)}</td></tr>
                ))}
              </tbody>
            </table>
            <p className="mt-2 text-xs text-mute">Latency {ms(r.latency.median_ms)} median ({r.latency.device}, batch 1) · Brier {num(r.calibration.brier_score)} · ECE {num(r.calibration.ece_10_bins)}. {r.calibration.note}</p>
          </section>
        )}
      </div>

      {r && (
        <div className="mt-4 grid gap-4 md:grid-cols-2">
          <ChartCard title="Training and validation loss"><LossChart history={m.history} kind="loss" /></ChartCard>
          <ChartCard title="Training and validation accuracy"><LossChart history={m.history} kind="acc" /></ChartCard>
          <ChartCard title="Confusion matrix" height={260}><ConfusionMatrix c={r.metrics.confusion} /></ChartCard>
          <section className="card p-4 text-sm">
            <h3 className="text-lg">Classification report</h3>
            <table className="mt-2 w-full"><thead className="text-left text-xs text-mute"><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
              <tbody>{Object.entries(r.per_class).map(([k, v]) => <tr key={k} className="border-t border-line"><td className="py-1.5">{k}</td><td>{num(v.precision)}</td><td>{num(v.recall)}</td><td>{num(v.f1)}</td><td>{v.support}</td></tr>)}</tbody></table>
            {r.split_summary && (<><h3 className="mt-4 text-lg">Dataset split</h3>
              <table className="mt-2 w-full"><thead className="text-left text-xs text-mute"><tr><th>Split</th><th>NORMAL</th><th>PNEUMONIA</th></tr></thead>
                <tbody>{Object.entries(r.split_summary.counts).map(([s, c]) => <tr key={s} className="border-t border-line"><td className="py-1">{s}</td><td>{c.NORMAL}</td><td>{c.PNEUMONIA}</td></tr>)}</tbody></table>
              <p className="mt-2 text-xs text-mute">{r.split_summary.method}. {r.split_summary.note}</p></>)}
          </section>
          <ChartCard title="ROC curve"><RocChart roc={r.curves.roc} auc={r.metrics.roc_auc} /></ChartCard>
          <ChartCard title="Precision–recall curve"><PrChart pr={r.curves.pr} ap={r.metrics.pr_auc} /></ChartCard>
        </div>
      )}
      <div className="mt-4"><Notice>Limitations: evaluated on a single public dataset; patient-level separation is verified only on a best-effort basis from file names; metrics do not imply clinical validity.</Notice></div>
    </>
  );
}
