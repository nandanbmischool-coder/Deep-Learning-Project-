import { useState } from "react";
import { Badge, ErrorState, Notice, PageHeader, Spinner } from "../components/ui";
import { api, toApiError, type ModelVersion, type Page, type User } from "../lib/api";
import { useAuth } from "../lib/auth";
import { useGet } from "../lib/useApi";
import { dt } from "../lib/utils";

interface Ready { ready: boolean; database: boolean; active_model: boolean; model_loaded_in_memory: boolean; dataset_present: boolean; environment: string; version: string }
interface Info { app_version: string; python: string; torch: string; torchvision: string; cuda_available: boolean; device_preference: string; max_upload_mb: number; database: string }
interface Audit { id: number; action: string; target: string; ip: string; created_at: string }

function Profile() {
  const { user, setUser } = useAuth();
  const [name, setName] = useState(user?.full_name ?? "");
  const [msg, setMsg] = useState("");
  const [cur, setCur] = useState(""); const [next, setNext] = useState(""); const [pwMsg, setPwMsg] = useState("");
  return (
    <section className="card space-y-6 p-4">
      <form className="space-y-3" onSubmit={async (e) => { e.preventDefault(); try { setUser((await api.patch<User>("/auth/me", { full_name: name })).data); setMsg("Profile saved."); } catch (x) { setMsg(toApiError(x).message); } }}>
        <h3 className="text-lg">Profile</h3>
        <div><label className="label" htmlFor="pn">Full name</label><input id="pn" className="input" value={name} onChange={(e) => setName(e.target.value)} minLength={2} required /></div>
        <p className="text-xs text-mute">{user?.email} · role: {user?.role}</p>
        <button className="btn-primary">Save profile</button>{msg && <span role="status" className="ml-3 text-sm">{msg}</span>}
      </form>
      <form className="space-y-3 border-t border-line pt-4" onSubmit={async (e) => { e.preventDefault(); try { await api.post("/auth/change-password", { current_password: cur, new_password: next }); setPwMsg("Password changed."); setCur(""); setNext(""); } catch (x) { setPwMsg(toApiError(x).message); } }}>
        <h3 className="text-lg">Change password</h3>
        <div><label className="label" htmlFor="cp">Current password</label><input id="cp" type="password" className="input" value={cur} onChange={(e) => setCur(e.target.value)} autoComplete="current-password" required /></div>
        <div><label className="label" htmlFor="np">New password</label><input id="np" type="password" className="input" value={next} onChange={(e) => setNext(e.target.value)} autoComplete="new-password" minLength={10} required /></div>
        <button className="btn-ghost">Change password</button>{pwMsg && <span role="status" className="ml-3 text-sm">{pwMsg}</span>}
      </form>
    </section>
  );
}

function System() {
  const r = useGet<Ready>("/ready", { poll: 10000 });
  const i = useGet<Info>("/system/info");
  const m = useGet<{ active: ModelVersion | null }>("/models/active");
  const row = (l: string, ok: boolean | undefined, okText = "OK", badText = "No") => <li className="flex items-center justify-between py-1.5"><span>{l}</span><Badge tone={ok ? "moss" : "amber"}>{ok ? okText : badText}</Badge></li>;
  return (
    <section className="card p-4 text-sm">
      <h3 className="text-lg">Application health</h3>
      {r.loading ? <Spinner /> : r.error ? <ErrorState error={r.error} /> : (
        <ul className="mt-2 divide-y divide-line">{row("Database", r.data?.database)}{row("Active model", r.data?.active_model, "Available", "Model not trained")}{row("Model loaded in memory", r.data?.model_loaded_in_memory, "Yes", "Not yet (loads on first use)")}{row("Dataset present", r.data?.dataset_present, "Found", "Not found")}</ul>)}
      {i.data && <dl className="mt-4 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-xs"><dt className="text-mute">App version</dt><dd>{i.data.app_version}</dd><dt className="text-mute">Python / PyTorch / torchvision</dt><dd>{i.data.python} / {i.data.torch} / {i.data.torchvision}</dd><dt className="text-mute">CUDA available</dt><dd>{String(i.data.cuda_available)} (preference: {i.data.device_preference})</dd><dt className="text-mute">Database</dt><dd>{i.data.database}</dd><dt className="text-mute">Max upload</dt><dd>{i.data.max_upload_mb} MB</dd></dl>}
      <h3 className="mt-5 text-lg">Model configuration</h3>
      {m.data?.active ? <p className="mt-1 text-xs text-mute">{m.data.active.version} · {m.data.active.architecture} · threshold {m.data.active.threshold.toFixed(3)} · {m.data.active.pretrained_source}</p> : <p className="mt-1 text-xs text-mute">Model not trained.</p>}
    </section>
  );
}

function Admin() {
  const a = useGet<Page<Audit>>("/admin/audit?page_size=20");
  const u = useGet<{ items: User[] }>("/admin/users");
  return (
    <>
      <section className="card p-4 text-sm"><h3 className="text-lg">Users</h3>
        <ul className="mt-2 divide-y divide-line">{u.data?.items.map((x) => <li key={x.id} className="flex justify-between py-1.5"><span>{x.full_name} <span className="text-xs text-mute">{x.email}</span></span><Badge tone={x.role === "admin" ? "teal" : "neutral"}>{x.role}</Badge></li>)}</ul></section>
      <section className="card p-4 text-sm"><h3 className="text-lg">Audit events</h3>
        <div className="overflow-x-auto"><table className="mt-2 w-full min-w-[520px] text-xs"><thead className="text-left text-mute"><tr><th>When</th><th>Action</th><th>Target</th><th>IP</th></tr></thead>
          <tbody>{a.data?.items.map((e) => <tr key={e.id} className="border-t border-line"><td className="py-1">{dt(e.created_at)}</td><td>{e.action}</td><td className="max-w-[180px] truncate">{e.target}</td><td>{e.ip}</td></tr>)}</tbody></table></div></section>
    </>
  );
}

export default function SettingsPage() {
  const { user } = useAuth();
  return (
    <>
      <PageHeader title="Settings" sub="Profile, system health and (for administrators) audit events." />
      <div className="grid gap-4 lg:grid-cols-2"><Profile /><System />{user?.role === "admin" ? <Admin /> : <Notice>Administration tools are visible to administrator accounts only.</Notice>}</div>
    </>
  );
}
