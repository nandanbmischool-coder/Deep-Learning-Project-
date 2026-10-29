import { zodResolver } from "@hookform/resolvers/zod";
import { Play, Square } from "lucide-react";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { ChartCard, LossChart } from "../components/charts";
import { Badge, EmptyState, ErrorState, Notice, PageHeader, Spinner } from "../components/ui";
import { api, toApiError, type ModelVersion, type Page, type TrainingJob } from "../lib/api";
import { useAuth } from "../lib/auth";
import { useGet } from "../lib/useApi";
import { dt } from "../lib/utils";

const schema = z.object({
  architecture: z.enum(["densenet121", "resnet50", "mobilenet_v3_large"]),
  epochs: z.coerce.number().int().min(1).max(200),
  head_epochs: z.coerce.number().int().min(0).max(100),
  batch_size: z.coerce.number().int().min(2).max(256),
  learning_rate: z.coerce.number().gt(0).max(1),
  finetune_learning_rate: z.coerce.number().gt(0).max(1),
  fine_tune_mode: z.enum(["head_only", "last_block", "full"]),
  patience: z.coerce.number().int().min(1).max(50),
  seed: z.coerce.number().int().min(0),
  pretrained: z.boolean(),
  max_images_per_class: z.union([z.literal(""), z.coerce.number().int().min(8)]).optional(),
}).refine((v) => v.head_epochs <= v.epochs, { message: "Head epochs cannot exceed total epochs.", path: ["head_epochs"] });
type FormVals = z.input<typeof schema>;

const stateTone = (s: string) => (s === "completed" ? "moss" : s === "running" ? "teal" : s === "failed" ? "brick" : "amber") as "moss" | "teal" | "brick" | "amber";

export default function Training() {
  const { user } = useAuth();
  const isAdmin = user?.role === "admin";
  const jobs = useGet<Page<TrainingJob>>("/training/jobs", { poll: 4000 });
  const models = useGet<{ items: ModelVersion[] }>("/models");
  const running = jobs.data?.items.find((j) => j.state === "running");
  const [selected, setSelected] = useState<string | null>(null);
  const jobId = selected ?? running?.job_id ?? jobs.data?.items[0]?.job_id ?? null;
  const detail = useGet<TrainingJob>(jobId ? `/training/jobs/${jobId}` : null, { poll: 3000 });
  const [msg, setMsg] = useState<{ tone: "ok" | "err"; text: string } | null>(null);

  const f = useForm<FormVals>({ resolver: zodResolver(schema), defaultValues: { architecture: "densenet121", epochs: 12, head_epochs: 4, batch_size: 32, learning_rate: 0.001, finetune_learning_rate: 0.0001, fine_tune_mode: "last_block", patience: 4, seed: 42, pretrained: true, max_images_per_class: "" } });

  const start = f.handleSubmit(async (raw) => {
    setMsg(null);
    const v = schema.parse(raw);
    const body = { ...v, max_images_per_class: v.max_images_per_class === "" || v.max_images_per_class == null ? null : v.max_images_per_class };
    try { const r = await api.post<TrainingJob>("/training/jobs", body); setSelected(r.data.job_id); setMsg({ tone: "ok", text: "Training started." }); jobs.reload(); } catch (e) { setMsg({ tone: "err", text: toApiError(e).message }); }
  });
  const cancel = async (id: string) => { try { await api.post(`/training/jobs/${id}/cancel`); jobs.reload(); } catch (e) { setMsg({ tone: "err", text: toApiError(e).message }); } };
  const rescan = async () => { try { const r = await api.post<{ registered_or_checked: number }>("/models/sync"); setMsg({ tone: "ok", text: `Checkpoint folder scanned (${r.data.registered_or_checked} run${r.data.registered_or_checked === 1 ? "" : "s"} found).` }); models.reload(); } catch (e) { setMsg({ tone: "err", text: toApiError(e).message }); } };
  const activate = async (id: number) => { try { await api.post(`/models/${id}/activate`); models.reload(); } catch (e) { setMsg({ tone: "err", text: toApiError(e).message }); } };

  const d = detail.data, st = d?.progress?.status;
  const pctDone = st ? Math.min(100, ((Math.max(st.epoch - 1, 0) + (st.batches ? st.batch / st.batches : 0)) / Math.max(st.total_epochs, 1)) * 100) : 0;
  const num = (k: keyof FormVals, label: string, step?: string) => (
    <div><label className="label" htmlFor={k}>{label}</label><input id={k} type="number" step={step} className="input" disabled={!isAdmin} {...f.register(k)} />
      {f.formState.errors[k] && <p className="mt-1 text-xs text-brick">{String(f.formState.errors[k]?.message)}</p>}</div>
  );

  return (
    <>
      <PageHeader title="Training center" sub="Fine-tune a pretrained network on the chest X-ray dataset. Training runs as a separate process; progress below is read from the real run." />
      {!isAdmin && <div className="mb-4"><Notice>Only administrators can start, cancel or activate. You can follow progress and results.</Notice></div>}
      {msg && <div className="mb-4"><Notice tone={msg.tone === "ok" ? "ok" : "warn"}>{msg.text}</Notice></div>}
      <div className="grid gap-4 lg:grid-cols-[360px_1fr]">
        <form onSubmit={start} className="card space-y-3 p-4" noValidate>
          <h3 className="text-lg">New training run</h3>
          <div><label className="label" htmlFor="architecture">Model</label>
            <select id="architecture" className="input" disabled={!isAdmin} {...f.register("architecture")}><option value="densenet121">DenseNet121 (primary)</option><option value="resnet50">ResNet50 (optional)</option><option value="mobilenet_v3_large">MobileNetV3-Large (optional)</option></select></div>
          <div className="grid grid-cols-2 gap-3">{num("epochs", "Total epochs")}{num("head_epochs", "Head-only epochs")}{num("batch_size", "Batch size")}{num("patience", "Early-stop patience")}{num("learning_rate", "Head LR", "any")}{num("finetune_learning_rate", "Fine-tune LR", "any")}{num("seed", "Random seed")}{num("max_images_per_class", "Subset per class (optional)")}</div>
          <div><label className="label" htmlFor="fine_tune_mode">Fine-tuning</label>
            <select id="fine_tune_mode" className="input" disabled={!isAdmin} {...f.register("fine_tune_mode")}><option value="head_only">Head only (backbone frozen)</option><option value="last_block">Last dense block + head</option><option value="full">Entire network</option></select></div>
          <label className="flex items-center gap-2 text-sm"><input type="checkbox" disabled={!isAdmin} {...f.register("pretrained")} />Use ImageNet pretrained weights</label>
          <p className="text-xs text-mute">If pretrained weights cannot be downloaded the run fails with an explanation — it never falls back to random weights. Untick only for smoke tests.</p>
          <button className="btn-primary w-full" disabled={!isAdmin || !!running || f.formState.isSubmitting}><Play className="h-4 w-4" aria-hidden />{running ? "A job is running" : "Start training"}</button>
        </form>

        <div className="min-w-0 space-y-4">
          {jobs.loading ? <Spinner /> : jobs.error ? <ErrorState error={jobs.error} onRetry={jobs.reload} /> : !d ? (
            <EmptyState title="No training runs yet">Start a run to see live progress, logs and curves here.</EmptyState>
          ) : (
            <>
              <section className="card p-4 text-sm">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <h3 className="text-lg">{d.version}</h3>
                  <div className="flex items-center gap-2"><Badge tone={stateTone(d.state)}>{d.state}</Badge>
                    {d.state === "running" && isAdmin && <button className="btn-ghost !py-1" onClick={() => cancel(d.job_id)}><Square className="h-3.5 w-3.5" aria-hidden />Cancel</button>}</div>
                </div>
                {st && <p className="mt-2 text-xs text-mute">{st.phase} · epoch {st.epoch}/{st.total_epochs}{st.batches ? ` · batch ${st.batch}/${st.batches}` : ""} · {st.message}{st.best_val_loss != null ? ` · best val loss ${st.best_val_loss.toFixed(4)}` : ""}</p>}
                {d.state === "running" && <div className="mt-2 h-2 rounded bg-line" role="progressbar" aria-valuenow={Math.round(pctDone)} aria-valuemin={0} aria-valuemax={100}><div className="h-2 rounded bg-teal transition-all" style={{ width: `${pctDone}%` }} /></div>}
                {d.error && <p className="mt-2 text-brick" role="alert">{d.error}</p>}
              </section>
              <div className="grid gap-4 md:grid-cols-2">
                <ChartCard title="Loss" height={200}><LossChart history={d.progress?.history} kind="loss" /></ChartCard>
                <ChartCard title="Accuracy" height={200}><LossChart history={d.progress?.history} kind="acc" /></ChartCard>
              </div>
              <section className="card p-4"><h3 className="text-lg">Training log</h3>
                <pre className="mt-2 max-h-56 overflow-auto rounded bg-ink p-3 text-xs text-white/90">{d.progress?.log_tail.length ? d.progress.log_tail.join("\n") : "No log output yet."}</pre></section>
            </>
          )}
        </div>
      </div>

      <div className="mt-6 grid gap-4 lg:grid-cols-2">
        <section className="card p-4 text-sm"><h3 className="text-lg">Run history</h3>
          {!jobs.data?.items.length ? <p className="mt-2 text-mute">No runs yet.</p> : (
            <ul className="mt-2 divide-y divide-line">{jobs.data.items.map((j) => (
              <li key={j.job_id} className="flex items-center justify-between gap-2 py-2"><button className="min-w-0 truncate text-left underline" onClick={() => setSelected(j.job_id)}>{j.version}</button><span className="flex items-center gap-2 text-xs text-mute">{dt(j.created_at)} <Badge tone={stateTone(j.state)}>{j.state}</Badge></span></li>))}</ul>)}
        </section>
        <section className="card p-4 text-sm"><div className="flex items-center justify-between"><h3 className="text-lg">Checkpoints</h3>{isAdmin && <button className="btn-ghost !py-1" onClick={rescan}>Rescan folder</button>}</div>
          {!models.data?.items.length ? <p className="mt-2 text-mute">No trained models yet.</p> : (
            <ul className="mt-2 divide-y divide-line">{models.data.items.map((m) => (
              <li key={m.id} className="flex items-center justify-between gap-2 py-2"><span className="min-w-0 truncate">{m.version}{m.evaluated ? "" : " (not evaluated)"}</span>
                {m.is_active ? <Badge tone="moss">active</Badge> : isAdmin ? <button className="btn-ghost !py-1" onClick={() => activate(m.id)}>Activate</button> : null}</li>))}</ul>)}
        </section>
      </div>
    </>
  );
}
