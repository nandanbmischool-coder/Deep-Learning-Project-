import { Activity, BarChart3, Database, FlaskConical, History, LayoutDashboard, LogOut, Menu, Microscope, Settings, X } from "lucide-react";
import { useState } from "react";
import { NavLink, Outlet, useNavigate } from "react-router-dom";
import type { ModelVersion } from "../lib/api";
import { useAuth } from "../lib/auth";
import { useGet } from "../lib/useApi";
import { cn } from "../lib/utils";

const NAV = [
  { to: "/app", label: "Dashboard", icon: LayoutDashboard, end: true },
  { to: "/app/analyze", label: "Analyze X-ray", icon: Microscope },
  { to: "/app/history", label: "History", icon: History },
  { to: "/app/performance", label: "Model performance", icon: BarChart3 },
  { to: "/app/dataset", label: "Dataset", icon: Database },
  { to: "/app/training", label: "Training center", icon: FlaskConical },
  { to: "/app/settings", label: "Settings", icon: Settings },
];

function ModelStatus() {
  const { data, error } = useGet<{ active: ModelVersion | null }>("/models/active", { poll: 30000 });
  const active = data?.active;
  const label = error ? "Server unreachable" : active ? `${active.architecture} · ${active.version.split("-").slice(1).join("-")}` : "Model not trained";
  const dot = error ? "bg-brick" : active ? "bg-moss" : "bg-amber";
  return (
    <div className="flex items-center gap-2 rounded-panel border border-line bg-white px-3 py-1.5 text-xs" title="Active model for new analyses">
      <span className={cn("h-2 w-2 rounded-full", dot)} aria-hidden />
      <span className="sr-only">Model status: </span>{label}
    </div>
  );
}

export default function Layout() {
  const { user, logout } = useAuth();
  const nav = useNavigate();
  const [open, setOpen] = useState(false);
  const [menu, setMenu] = useState(false);

  const sidebar = (
    <nav aria-label="Main" className="flex h-full flex-col bg-ink px-3 py-5 text-sm text-white/80">
      <div className="mb-6 flex items-center gap-2 px-2 font-display text-xl text-white"><Activity className="h-5 w-5 text-[#7FD1D8]" aria-hidden />PneumoVision</div>
      <ul className="space-y-0.5">
        {NAV.map(({ to, label, icon: Icon, end }) => (
          <li key={to}>
            <NavLink to={to} end={end} onClick={() => setOpen(false)}
              className={({ isActive }) => cn("flex items-center gap-3 rounded-panel px-3 py-2 transition-colors", isActive ? "bg-ink-3 text-white" : "hover:bg-ink-2")}>
              <Icon className="h-4 w-4" aria-hidden />{label}
            </NavLink>
          </li>
        ))}
      </ul>
      <p className="mt-auto px-3 text-xs leading-relaxed text-white/50">Research tool. Not a clinically validated medical device.</p>
    </nav>
  );

  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[240px_1fr]">
      <aside className="hidden lg:block lg:sticky lg:top-0 lg:h-screen">{sidebar}</aside>
      {open && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div className="absolute inset-0 bg-ink/60" onClick={() => setOpen(false)} aria-hidden />
          <div className="absolute inset-y-0 left-0 w-64">{sidebar}</div>
        </div>
      )}
      <div className="min-w-0">
        <header className="sticky top-0 z-30 flex items-center gap-3 border-b border-line bg-paper/95 px-4 py-2.5 backdrop-blur">
          <button className="btn-ghost !p-2 lg:hidden" onClick={() => setOpen((o) => !o)} aria-label="Toggle navigation">
            {open ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
          </button>
          <ModelStatus />
          <div className="relative ml-auto">
            <button className="btn-ghost" onClick={() => setMenu((m) => !m)} aria-haspopup="menu" aria-expanded={menu}>
              <span className="grid h-6 w-6 place-items-center rounded-full bg-teal text-xs text-white">{user?.full_name?.[0]?.toUpperCase()}</span>
              <span className="hidden sm:inline">{user?.full_name}</span>
            </button>
            {menu && (
              <div role="menu" className="card absolute right-0 mt-2 w-56 p-2 text-sm shadow-lg">
                <p className="px-2 py-1 text-xs text-mute">{user?.email} · {user?.role}</p>
                <button role="menuitem" className="flex w-full items-center gap-2 rounded px-2 py-2 hover:bg-paper" onClick={() => { setMenu(false); nav("/app/settings"); }}><Settings className="h-4 w-4" />Settings</button>
                <button role="menuitem" className="flex w-full items-center gap-2 rounded px-2 py-2 hover:bg-paper" onClick={async () => { await logout(); nav("/"); }}><LogOut className="h-4 w-4" />Sign out</button>
              </div>
            )}
          </div>
        </header>
        <main className="mx-auto max-w-6xl px-4 py-6 sm:px-6"><Outlet /></main>
      </div>
    </div>
  );
}
