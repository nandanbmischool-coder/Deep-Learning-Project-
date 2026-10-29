import { useCallback, useEffect, useRef, useState } from "react";
import { api, toApiError, type ApiError } from "./api";

/** GET helper with loading / error state and optional polling. */
export function useGet<T>(url: string | null, opts: { poll?: number } = {}) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(!!url);
  const alive = useRef(true);

  const load = useCallback(async () => {
    if (!url) return;
    try {
      const r = await api.get<T>(url);
      if (alive.current) { setData(r.data); setError(null); }
    } catch (e) {
      if (alive.current) setError(toApiError(e));
    } finally {
      if (alive.current) setLoading(false);
    }
  }, [url]);

  useEffect(() => {
    alive.current = true;
    setLoading(!!url);
    void load();
    const t = opts.poll ? window.setInterval(() => void load(), opts.poll) : undefined;
    return () => { alive.current = false; if (t) window.clearInterval(t); };
  }, [load, opts.poll, url]);

  return { data, error, loading, reload: load };
}
