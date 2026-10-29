export const pct = (v: number | null | undefined, d = 1) => (v == null ? "n/a" : `${(v * 100).toFixed(d)}%`);
export const num = (v: number | null | undefined, d = 3) => (v == null ? "n/a" : v.toFixed(d));
export const ms = (v: number | null | undefined) => (v == null ? "n/a" : `${v.toFixed(1)} ms`);
export const dt = (s?: string | null) => (s ? new Date(s).toLocaleString() : "n/a");
export const cn = (...c: (string | false | null | undefined)[]) => c.filter(Boolean).join(" ");
