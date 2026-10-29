import type { ReactNode } from "react";
import { Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Confusion, HistoryRow } from "../lib/api";
import { pct } from "../lib/utils";

export const COLORS = { NORMAL: "#2F7D5B", PNEUMONIA: "#B4432E", train: "#0E7C86", val: "#D98A1F", grid: "#E4E9E6", axis: "#5B6B70" };
const tick = { fontSize: 11, fill: COLORS.axis };

export function ChartCard({ title, sub, children, height = 240 }: { title: string; sub?: string; children: ReactNode; height?: number }) {
  return (
    <section className="card p-4">
      <h3 className="text-lg">{title}</h3>
      {sub && <p className="text-xs text-mute">{sub}</p>}
      <div className="mt-3" style={{ height }}>{children}</div>
    </section>
  );
}

export const NoData = ({ children }: { children: ReactNode }) => (
  <div className="flex h-full items-center justify-center rounded-panel bg-paper px-4 text-center text-sm text-mute">{children}</div>
);

export function VolumeChart({ data }: { data: { date: string; NORMAL: number; PNEUMONIA: number }[] }) {
  if (!data.some((d) => d.NORMAL + d.PNEUMONIA > 0)) return <NoData>No analyses in the last 30 days.</NoData>;
  return (
    <ResponsiveContainer>
      <AreaChart data={data}>
        <CartesianGrid stroke={COLORS.grid} vertical={false} />
        <XAxis dataKey="date" tick={tick} tickFormatter={(d: string) => d.slice(5)} interval={4} />
        <YAxis allowDecimals={false} tick={tick} width={28} />
        <Tooltip />
        <Legend />
        <Area type="monotone" dataKey="NORMAL" stackId="1" stroke={COLORS.NORMAL} fill={COLORS.NORMAL} fillOpacity={0.35} />
        <Area type="monotone" dataKey="PNEUMONIA" stackId="1" stroke={COLORS.PNEUMONIA} fill={COLORS.PNEUMONIA} fillOpacity={0.35} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export function DistributionChart({ data }: { data: { name: string; value: number }[] }) {
  if (!data.some((d) => d.value > 0)) return <NoData>No predictions yet.</NoData>;
  return (
    <ResponsiveContainer>
      <PieChart>
        <Pie data={data} dataKey="value" nameKey="name" innerRadius={55} outerRadius={85} paddingAngle={2}>
          {data.map((d) => <Cell key={d.name} fill={d.name === "PNEUMONIA" ? COLORS.PNEUMONIA : COLORS.NORMAL} />)}
        </Pie>
        <Tooltip />
        <Legend />
      </PieChart>
    </ResponsiveContainer>
  );
}

export function DatasetChart({ data }: { data: { split: string; NORMAL: number; PNEUMONIA: number }[] }) {
  if (!data.length) return <NoData>Dataset not found. See the Dataset page for setup instructions.</NoData>;
  return (
    <ResponsiveContainer>
      <BarChart data={data}>
        <CartesianGrid stroke={COLORS.grid} vertical={false} />
        <XAxis dataKey="split" tick={tick} />
        <YAxis tick={tick} width={40} />
        <Tooltip />
        <Legend />
        <Bar dataKey="NORMAL" fill={COLORS.NORMAL} />
        <Bar dataKey="PNEUMONIA" fill={COLORS.PNEUMONIA} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function LossChart({ history, kind }: { history: HistoryRow[] | null | undefined; kind: "loss" | "acc" }) {
  if (!history?.length) return <NoData>No training history yet — train a model first.</NoData>;
  const a = kind === "loss" ? "train_loss" : "train_acc";
  const b = kind === "loss" ? "val_loss" : "val_acc";
  return (
    <ResponsiveContainer>
      <LineChart data={history}>
        <CartesianGrid stroke={COLORS.grid} vertical={false} />
        <XAxis dataKey="epoch" tick={tick} />
        <YAxis tick={tick} width={40} domain={kind === "acc" ? [0, 1] : ["auto", "auto"]} />
        <Tooltip formatter={(v: number) => v.toFixed(4)} />
        <Legend />
        <Line type="monotone" dataKey={a} name={`train ${kind}`} stroke={COLORS.train} dot={false} strokeWidth={2} />
        <Line type="monotone" dataKey={b} name={`val ${kind}`} stroke={COLORS.val} dot={false} strokeWidth={2} />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function ConfusionMatrix({ c }: { c: Confusion }) {
  const total = c.tp + c.tn + c.fp + c.fn || 1;
  const cell = (label: string, v: number, good: boolean) => (
    <div className={`flex flex-col items-center justify-center rounded-panel p-4 ${good ? "bg-moss-soft" : "bg-brick-soft"}`}>
      <span className="font-display text-3xl">{v}</span>
      <span className="text-xs text-mute">{label} · {pct(v / total, 1)}</span>
    </div>
  );
  return (
    <div className="grid grid-cols-[auto_1fr_1fr] gap-2 text-center text-xs text-mute" role="table" aria-label="Confusion matrix">
      <div /><div>Predicted NORMAL</div><div>Predicted PNEUMONIA</div>
      <div className="self-center [writing-mode:vertical-rl] rotate-180">True NORMAL</div>{cell("TN", c.tn, true)}{cell("FP", c.fp, false)}
      <div className="self-center [writing-mode:vertical-rl] rotate-180">True PNEUMONIA</div>{cell("FN", c.fn, false)}{cell("TP", c.tp, true)}
    </div>
  );
}

export function RocChart({ roc, auc }: { roc: { fpr: number[]; tpr: number[] } | null | undefined; auc?: number | null }) {
  if (!roc) return <NoData>Not evaluated.</NoData>;
  const data = roc.fpr.map((x, i) => ({ fpr: x, tpr: roc.tpr[i] }));
  return (
    <ResponsiveContainer>
      <LineChart data={data}>
        <CartesianGrid stroke={COLORS.grid} />
        <XAxis dataKey="fpr" type="number" domain={[0, 1]} tick={tick} label={{ value: "False positive rate", fontSize: 11, dy: 14 }} />
        <YAxis type="number" domain={[0, 1]} tick={tick} width={34} />
        <Tooltip formatter={(v: number) => v.toFixed(3)} />
        <Line dataKey="tpr" name={`ROC${auc != null ? ` (AUC ${auc.toFixed(3)})` : ""}`} stroke={COLORS.train} dot={false} strokeWidth={2} isAnimationActive={false} />
        <Line data={[{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }]} dataKey="tpr" stroke="#9AA8AC" strokeDasharray="4 4" dot={false} name="chance" isAnimationActive={false} />
        <Legend />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function PrChart({ pr, ap }: { pr: { precision: number[]; recall: number[] } | null | undefined; ap?: number | null }) {
  if (!pr) return <NoData>Not evaluated.</NoData>;
  const data = pr.recall.map((x, i) => ({ recall: x, precision: pr.precision[i] })).sort((a, b) => a.recall - b.recall);
  return (
    <ResponsiveContainer>
      <LineChart data={data}>
        <CartesianGrid stroke={COLORS.grid} />
        <XAxis dataKey="recall" type="number" domain={[0, 1]} tick={tick} label={{ value: "Recall", fontSize: 11, dy: 14 }} />
        <YAxis type="number" domain={[0, 1]} tick={tick} width={34} />
        <Tooltip formatter={(v: number) => v.toFixed(3)} />
        <Line dataKey="precision" name={`PR${ap != null ? ` (AP ${ap.toFixed(3)})` : ""}`} stroke={COLORS.val} dot={false} strokeWidth={2} isAnimationActive={false} />
        <Legend />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function ComparisonChart({ rows }: { rows: { architecture: string; status: string; metrics?: { accuracy: number; f1: number; roc_auc: number | null } }[] }) {
  const data = rows.filter((r) => r.status === "completed" && r.metrics).map((r) => ({ name: r.architecture, accuracy: r.metrics!.accuracy, f1: r.metrics!.f1, roc_auc: r.metrics!.roc_auc ?? 0 }));
  if (!data.length) return <NoData>No completed comparison runs.</NoData>;
  return (
    <ResponsiveContainer>
      <BarChart data={data}>
        <CartesianGrid stroke={COLORS.grid} vertical={false} />
        <XAxis dataKey="name" tick={tick} />
        <YAxis domain={[0, 1]} tick={tick} width={34} />
        <Tooltip formatter={(v: number) => v.toFixed(3)} />
        <Legend />
        <Bar dataKey="accuracy" fill={COLORS.train} />
        <Bar dataKey="f1" fill={COLORS.val} />
        <Bar dataKey="roc_auc" fill="#24444F" />
      </BarChart>
    </ResponsiveContainer>
  );
}
