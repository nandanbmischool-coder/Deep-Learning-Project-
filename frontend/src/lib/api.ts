import axios, { AxiosError } from "axios";

export const api = axios.create({ baseURL: "/api/v1", withCredentials: true });

function csrfFromCookie(): string | undefined {
  return document.cookie.split("; ").find((c) => c.startsWith("pv_csrf="))?.split("=")[1];
}

api.interceptors.request.use((cfg) => {
  if (cfg.method && !["get", "head", "options"].includes(cfg.method)) {
    const t = csrfFromCookie();
    if (t) cfg.headers.set("X-CSRF-Token", t);
  }
  return cfg;
});

api.interceptors.response.use(
  (r) => r,
  (err: AxiosError) => {
    if (err.response?.status === 401 && !err.config?.url?.includes("/auth/")) {
      window.dispatchEvent(new Event("pv:unauthorized"));
    }
    return Promise.reject(err);
  },
);

export class ApiError extends Error {
  constructor(public code: string, message: string, public status: number, public details?: unknown) {
    super(message);
  }
}

export function toApiError(e: unknown): ApiError {
  if (axios.isAxiosError(e)) {
    const body = e.response?.data as { error?: { code: string; message: string; details?: unknown } } | undefined;
    if (body?.error) return new ApiError(body.error.code, body.error.message, e.response!.status, body.error.details);
    if (!e.response) return new ApiError("NETWORK", "Cannot reach the server. Is the backend running on port 8000?", 0);
    return new ApiError("HTTP_ERROR", e.message, e.response.status);
  }
  return new ApiError("UNKNOWN", e instanceof Error ? e.message : "Unexpected error", 0);
}

// ---------- types ----------
export interface User { id: number; email: string; full_name: string; role: "user" | "admin"; created_at: string }
export interface Prediction {
  analysis_id: string; predicted_class: "NORMAL" | "PNEUMONIA"; probabilities: { NORMAL: number; PNEUMONIA: number };
  threshold: number; target_class: string; model_version: string; inference_ms: number; gradcam_ms: number;
  image_size: { width: number; height: number }; image_validation: { status: string; warnings: string[]; note: string; checks: Record<string, number>; gradcam_flat_map?: boolean };
  assets: { original: string; heatmap: string; overlay: string; thumbnail: string }; created_at: string; disclaimer: string;
  preprocessing?: { steps: string[]; augmentation: string }; filename_hint?: string;
}
export interface Page<T> { items: T[]; total: number; page: number; page_size: number }
export interface Confusion { tp: number; tn: number; fp: number; fn: number }
export interface EvalReport {
  split: string; created_at: string; n_samples: number; threshold: number; threshold_method: string; positive_class: string;
  metrics: { accuracy: number; precision: number; recall: number; specificity: number; f1: number; roc_auc: number | null; pr_auc: number | null; confusion: Confusion };
  per_class: Record<string, { precision: number; recall: number; f1: number; support: number }>;
  confidence_intervals: Record<string, [number, number] | null | string>;
  curves: { roc: { fpr: number[]; tpr: number[] } | null; pr: { precision: number[]; recall: number[] } | null };
  calibration: { brier_score: number; ece_10_bins: number; note: string };
  latency: { median_ms: number; p95_ms: number; device: string };
  parameter_count: number; model_size_mb: number; split_summary?: { counts: Record<string, Record<string, number>>; method: string; note: string };
}
export interface HistoryRow { epoch: number; phase: string; train_loss: number; val_loss: number; train_acc: number; val_acc: number; val_auc: number | null }
export interface ModelVersion {
  id: number; version: string; architecture: string; pretrained: boolean; pretrained_source: string; parameter_count: number;
  model_size_mb: number; threshold: number; threshold_method: string; is_active: boolean; created_at: string; evaluated: boolean;
  training_config?: Record<string, unknown>; best_epoch?: number; training_seconds?: number; device?: string; torch?: string; history?: HistoryRow[];
}
export interface TrainingJob {
  job_id: string; architecture: string; state: string; config: Record<string, unknown>; error: string; version: string;
  created_at: string; started_at?: string; finished_at?: string;
  progress?: { status: { state: string; phase: string; epoch: number; total_epochs: number; batch: number; batches: number; message: string; best_val_loss: number | null } | null; history: HistoryRow[]; log_tail: string[] };
}
