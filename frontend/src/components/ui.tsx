import { AlertTriangle, CheckCircle2, Info, Loader2 } from "lucide-react";
import type { ReactNode } from "react";
import { cn } from "../lib/utils";
import type { ApiError } from "../lib/api";

export function Spinner({ label = "Loading" }: { label?: string }) {
  return (
    <div role="status" className="flex items-center gap-2 py-10 text-sm text-mute">
      <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> {label}…
    </div>
  );
}

export function ErrorState({ error, onRetry }: { error: ApiError | Error | string; onRetry?: () => void }) {
  const msg = typeof error === "string" ? error : error.message;
  return (
    <div role="alert" className="flex items-start gap-3 rounded-panel border border-brick/30 bg-brick-soft p-4 text-sm">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-brick" aria-hidden />
      <div className="flex-1">
        <p className="font-medium text-brick">Something went wrong</p>
        <p className="mt-0.5">{msg}</p>
      </div>
      {onRetry && <button className="btn-ghost !py-1" onClick={onRetry}>Try again</button>}
    </div>
  );
}

export function EmptyState({ title, children, action }: { title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="rounded-panel border border-dashed border-line bg-white px-6 py-10 text-center">
      <p className="font-display text-lg">{title}</p>
      {children && <div className="mx-auto mt-1 max-w-md text-sm text-mute">{children}</div>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

export function Notice({ tone = "info", children }: { tone?: "info" | "warn" | "ok"; children: ReactNode }) {
  const t = { info: ["border-teal/30 bg-teal-soft", Info, "text-teal-strong"], warn: ["border-amber/40 bg-amber-soft", AlertTriangle, "text-amber"], ok: ["border-moss/30 bg-moss-soft", CheckCircle2, "text-moss"] }[tone] as [string, typeof Info, string];
  const Icon = t[1];
  return (
    <div className={cn("flex items-start gap-3 rounded-panel border p-3 text-sm", t[0])}>
      <Icon className={cn("mt-0.5 h-4 w-4 shrink-0", t[2])} aria-hidden />
      <div>{children}</div>
    </div>
  );
}

export function PageHeader({ title, sub, actions }: { title: string; sub?: string; actions?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-3">
      <div>
        <h1 className="text-3xl leading-tight">{title}</h1>
        {sub && <p className="mt-1 max-w-2xl text-sm text-mute">{sub}</p>}
      </div>
      {actions}
    </div>
  );
}

export function Stat({ label, value, hint, tone }: { label: string; value: ReactNode; hint?: string; tone?: "brick" | "moss" | "teal" }) {
  return (
    <div className="card p-4">
      <p className="text-xs text-mute">{label}</p>
      <p className={cn("mt-1 font-display text-3xl", tone === "brick" && "text-brick", tone === "moss" && "text-moss", tone === "teal" && "text-teal-strong")}>{value}</p>
      {hint && <p className="mt-1 text-xs text-mute">{hint}</p>}
    </div>
  );
}

export function Badge({ children, tone = "neutral" }: { children: ReactNode; tone?: "neutral" | "brick" | "moss" | "amber" | "teal" }) {
  const c = { neutral: "bg-paper text-ink border-line", brick: "bg-brick-soft text-brick border-brick/30", moss: "bg-moss-soft text-moss border-moss/30", amber: "bg-amber-soft text-amber border-amber/40", teal: "bg-teal-soft text-teal-strong border-teal/30" }[tone];
  return <span className={cn("inline-block rounded border px-2 py-0.5 text-xs font-medium", c)}>{children}</span>;
}

export const classTone = (c: string) => (c === "PNEUMONIA" ? "brick" : "moss") as "brick" | "moss";

export function Modal({ title, children, onClose }: { title: string; children: ReactNode; onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-ink/60 p-4" role="dialog" aria-modal="true" aria-label={title}>
      <div className="card w-full max-w-md p-5">
        <h2 className="text-xl">{title}</h2>
        <div className="mt-2 text-sm">{children}</div>
        <button className="sr-only" onClick={onClose}>Close</button>
      </div>
    </div>
  );
}

export const Disclaimer = () => (
  <p className="text-xs text-mute">
    Research and education use only. Not a clinically validated diagnostic device — never use it for medical decisions.
  </p>
);
