import { Link } from "react-router-dom";
import { ChartCard, ComparisonChart, ConfusionMatrix, DatasetChart, DistributionChart, LossChart, NoData, PrChart, RocChart, VolumeChart } from "../components/charts";
import { Badge, EmptyState, ErrorState, Notice, PageHeader, Spinner, Stat, classTone } from "../components/ui";
import type { HistoryRow, Prediction } from "../lib/api";
import { useGet } from "../lib/useApi";
import { dt, ms, pct } from "../lib/utils";

interface Summary { operational: { total_analyses: number; normal: number; pneumonia: number; avg_inference_ms: number | null }; model: { active_version: string | null; evaluated: boolean; test_accuracy: number | null; test_roc_auc: number | null; evaluation_note: string | null } }
interface Charts {
  volume_over_time: { date: string; NORMAL: number; PNEUMONIA: number }[]; prediction_distribution: { name: string; value: number }[];
  dataset_distribution: { split: string; NORMAL: number; PNEUMONIA: number }[]; training_curves: HistoryRow[] | null;
  evaluation: { confusion: { tp: number; tn: number; fp: number; fn: number }; curves: { roc: { fpr: number[]; tpr: number[] } | null; pr: { precision: number[]; recall: number[] } | null } } | null;
  comparison: { rows: { architecture: string; status: string; metrics?: { accuracy: number; f1: number; roc_auc: number | null } }[] } | null;
}

export default function Dashboard() {
  const s = useGet<Summary>("/dashboard/summary");
  const a = useGet<{ items: Prediction[] }>("/dashboard/activity");
  const c = useGet<Charts>("/dashboard/charts");
  if (s.loading || c.loading) return <Spinner label="Loading dashboard" />;
  if (s.error) return <ErrorState error={s.error} onRetry={s.reload} />;
  const op = s.data!.operational, m = s.data!.model, ch = c.data;

  return (
    <>
      <PageHeader title="Dashboard" sub="Your activity, plus how the active model performed on held-out test images." />

      <h2 className="mb-2 text-xl">Your activity</h2>
      <p className="mb-3 text-xs text-mute">Counts of images you analysed. These are not disease prevalence and say nothing about model accuracy.</p>
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Total analyses" value={op.total_analyses} />
        <Stat label="Predicted normal" value={op.normal} tone="moss" />
        <Stat label="Predicted pneumonia" value={op.pneumonia} tone="brick" />
        <Stat label="Avg inference time" value={ms(op.avg_inference_ms)} hint="Recorded from your analyses" />
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <div className="lg:col-span-2"><ChartCard title="Analysis volume" sub="Last 30 days"><VolumeChart data={ch?.volume_over_time ?? []} /></ChartCard></div>
        <ChartCard title="Prediction distribution"><DistributionChart data={ch?.prediction_distribution ?? []} /></ChartCard>
      </div>

      <section className="card mt-4 p-4">
        <div className="flex items-center justify-between"><h3 className="text-lg">Recent analyses</h3><Link className="text-sm text-teal-strong underline" to="/app/history">View all</Link></div>
        {a.data?.items.length ? (
          <ul className="mt-2 divide-y divide-line">
            {a.data.items.map((p) => (
              <li key={p.analysis_id} className="flex items-center gap-3 py-2">
                <img src={p.assets.thumbnail} alt="" className="h-12 w-12 rounded bg-box object-cover" />
                <div className="min-w-0 flex-1"><Link to={`/app/history/${p.analysis_id}`} className="font-mono text-xs underline">{p.analysis_id}</Link><p className="text-xs text-mute">{dt(p.created_at)}</p></div>
                <Badge tone={classTone(p.predicted_class)}>{p.predicted_class}</Badge>
              </li>
            ))}
          </ul>
        ) : <div className="mt-3"><EmptyState title="No analyses yet" action={<Link className="btn-primary" to="/app/analyze">Analyze your first X-ray</Link>}>Upload a chest X-ray to see predictions and Grad-CAM explanations here.</EmptyState></div>}
      </section>

      <h2 className="mb-2 mt-10 text-xl">Model evaluation</h2>
      <p className="mb-3 text-xs text-mute">Computed on the held-out test split when training finished — independent of your uploads.</p>
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Active model" value={<span className="text-xl">{m.active_version ?? "None"}</span>} />
        <Stat label="Test accuracy" value={m.evaluated ? pct(m.test_accuracy) : "Not evaluated"} tone="teal" />
        <Stat label="Test ROC-AUC" value={m.evaluated ? (m.test_roc_auc?.toFixed(3) ?? "n/a") : "Not evaluated"} />
        <Stat label="Evaluation" value={<span className="text-base">{m.evaluated ? "Held-out test set" : "Pending"}</span>} />
      </div>
      {!m.evaluated && <div className="mt-3"><Notice tone="warn">{m.evaluation_note} <Link className="underline" to="/app/training">Open the Training center</Link>.</Notice></div>}

      <div className="mt-4 grid gap-4 md:grid-cols-2">
        <ChartCard title="Dataset class distribution"><DatasetChart data={ch?.dataset_distribution ?? []} /></ChartCard>
        <ChartCard title="Training and validation loss"><LossChart history={ch?.training_curves} kind="loss" /></ChartCard>
        <ChartCard title="Training and validation accuracy"><LossChart history={ch?.training_curves} kind="acc" /></ChartCard>
        <ChartCard title="Confusion matrix (test)" height={260}>{ch?.evaluation ? <ConfusionMatrix c={ch.evaluation.confusion} /> : <NoData>Not evaluated.</NoData>}</ChartCard>
        <ChartCard title="ROC curve (test)"><RocChart roc={ch?.evaluation?.curves.roc} auc={m.test_roc_auc} /></ChartCard>
        <ChartCard title="Precision–recall curve (test)"><PrChart pr={ch?.evaluation?.curves.pr} /></ChartCard>
        {ch?.comparison && <div className="md:col-span-2"><ChartCard title="Model comparison (test)" sub="Only architectures that finished a real comparison run appear here."><ComparisonChart rows={ch.comparison.rows} /></ChartCard></div>}
      </div>
    </>
  );
}
