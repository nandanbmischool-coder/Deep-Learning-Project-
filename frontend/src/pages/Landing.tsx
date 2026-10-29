import { Activity, BarChart3, Layers, ShieldCheck } from "lucide-react";
import { Link } from "react-router-dom";
import { useAuth } from "../lib/auth";

function Radiograph() {
  const ribs = Array.from({ length: 7 }, (_, i) => 70 + i * 28);
  return (
    <svg viewBox="0 0 400 480" role="img" aria-label="Illustration of a chest radiograph with a Grad-CAM heatmap over the lower left lung field" className="h-full w-full">
      <defs>
        <radialGradient id="lung" cx="50%" cy="45%" r="60%"><stop offset="0" stopColor="#3b4a50" /><stop offset="1" stopColor="#1b272c" /></radialGradient>
        <radialGradient id="cam" cx="50%" cy="50%" r="50%"><stop offset="0" stopColor="#ff3b1d" stopOpacity=".9" /><stop offset=".45" stopColor="#ffb020" stopOpacity=".55" /><stop offset="1" stopColor="#ffd84a" stopOpacity="0" /></radialGradient>
        <linearGradient id="bg" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor="#0d171c" /><stop offset="1" stopColor="#101e24" /></linearGradient>
      </defs>
      <rect width="400" height="480" fill="url(#bg)" />
      <ellipse cx="125" cy="235" rx="82" ry="150" fill="url(#lung)" /><ellipse cx="275" cy="235" rx="82" ry="150" fill="url(#lung)" />
      <path d="M200 60 V440" stroke="#cfd8db" strokeOpacity=".35" strokeWidth="14" strokeLinecap="round" />
      <path d="M170 250c0-45 25-70 55-62 30 8 45 40 38 88-6 40-35 70-62 62-22-7-31-45-31-88z" fill="#e9eef0" fillOpacity=".16" />
      {ribs.map((y) => (<g key={y} stroke="#e9eef0" strokeOpacity=".28" strokeWidth="3" fill="none" strokeLinecap="round"><path d={`M196 ${y}C150 ${y - 24} 90 ${y - 6} 52 ${y + 50}`} /><path d={`M204 ${y}C250 ${y - 24} 310 ${y - 6} 348 ${y + 50}`} /></g>))}
      <path d="M92 70c30-30 70-36 108-26 38-10 78-4 108 26" stroke="#e9eef0" strokeOpacity=".4" strokeWidth="5" fill="none" strokeLinecap="round" />
      <circle cx="108" cy="318" r="62" fill="url(#cam)" />
      <circle cx="108" cy="318" r="62" fill="none" stroke="#ffe08a" strokeOpacity=".6" strokeDasharray="3 5" />
      <text x="20" y="460" fill="#9fb1b6" fontSize="11" fontFamily="IBM Plex Sans, sans-serif">Illustration, not a real radiograph</text>
    </svg>
  );
}

const FEATURES = [
  { icon: Layers, title: "Transfer learning, trained for real", body: "DenseNet121 pretrained on ImageNet, head trained first, last dense block fine-tuned at a lower learning rate. If weights can’t be loaded, training stops instead of quietly using random ones." },
  { icon: Activity, title: "Grad-CAM from the model itself", body: "Heatmaps come from the actual gradients of the class score at the last convolutional block — shown beside the original and as an overlay." },
  { icon: BarChart3, title: "Metrics that are earned", body: "Accuracy, sensitivity, specificity, ROC-AUC and bootstrap intervals are computed on a held-out test split. Until then the app says “Not evaluated”." },
  { icon: ShieldCheck, title: "Built with safe defaults", body: "Content-checked uploads, cookie sessions with CSRF protection, per-user history, audit events and role-restricted training." },
];

export default function Landing() {
  const { user } = useAuth();
  return (
    <div className="bg-paper">
      <header className="bg-ink text-white">
        <nav className="mx-auto flex max-w-6xl items-center justify-between px-5 py-4" aria-label="Site">
          <span className="flex items-center gap-2 font-display text-xl"><Activity className="h-5 w-5 text-[#7FD1D8]" aria-hidden />PneumoVision AI</span>
          <div className="flex items-center gap-2">
            {user ? <Link className="btn-primary" to="/app">Open dashboard</Link> : (<><Link className="btn border border-white/30 hover:bg-white/10" to="/login">Sign in</Link><Link className="btn-primary" to="/register">Create account</Link></>)}
          </div>
        </nav>
        <div className="mx-auto grid max-w-6xl items-center gap-8 px-5 pb-14 pt-8 md:grid-cols-[1.1fr_0.9fr]">
          <div>
            <h1 className="text-4xl leading-[1.1] sm:text-5xl">See what the model saw when it called a chest X-ray normal or pneumonia.</h1>
            <p className="mt-5 max-w-xl text-white/75">PneumoVision AI fine-tunes a pretrained convolutional network on chest radiographs and pairs every prediction with a Grad-CAM view of the regions that drove it. A final-year academic project on explainable medical imaging.</p>
            <div className="mt-7 flex flex-wrap gap-3"><Link className="btn-primary" to={user ? "/app/analyze" : "/register"}>{user ? "Analyze an X-ray" : "Get started"}</Link><a className="btn border border-white/30 hover:bg-white/10" href="#how">How it works</a></div>
          </div>
          <div className="mx-auto aspect-[5/6] w-full max-w-sm overflow-hidden rounded-panel border border-white/10"><Radiograph /></div>
        </div>
      </header>

      <section className="mx-auto max-w-6xl px-5 py-14" aria-labelledby="features">
        <h2 id="features" className="text-3xl">What’s inside</h2>
        <div className="mt-6 grid gap-4 sm:grid-cols-2">
          {FEATURES.map(({ icon: Icon, title, body }) => (
            <article key={title} className="card p-5"><Icon className="h-5 w-5 text-teal" aria-hidden /><h3 className="mt-3 text-xl">{title}</h3><p className="mt-1 text-sm text-mute">{body}</p></article>
          ))}
        </div>
      </section>

      <section id="how" className="bg-white py-14" aria-labelledby="how-h">
        <div className="mx-auto max-w-6xl px-5">
          <h2 id="how-h" className="text-3xl">From upload to explanation</h2>
          <ol className="mt-6 grid gap-4 md:grid-cols-4">
            {[["Upload", "JPEG, PNG or WebP. The file’s content is verified, not just its extension, and re-encoded."], ["Preprocess", "RGB, resize to 224×224, ImageNet normalisation — identical to validation and test."], ["Predict", "Class probabilities and a decision threshold chosen on validation data."], ["Explain", "Grad-CAM heatmap and overlay generated from this exact forward and backward pass."]].map(([t, b], i) => (
              <li key={t} className="rounded-panel border border-line p-4"><span className="font-display text-2xl text-teal">{i + 1}</span><h3 className="text-lg">{t}</h3><p className="mt-1 text-sm text-mute">{b}</p></li>
            ))}
          </ol>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-5 py-12">
        <div className="rounded-panel border border-amber/40 bg-amber-soft p-5 text-sm">
          <h2 className="text-xl">Research use only</h2>
          <p className="mt-1">This is an academic research-support tool. It is not a clinically validated diagnostic device and must not be used to diagnose or treat anyone. Heatmaps show where the model’s score was influenced — not that its reasoning is clinically correct.</p>
        </div>
      </section>
      <footer className="border-t border-line py-6 text-center text-xs text-mute">PneumoVision AI · explainable pneumonia detection with transfer learning and Grad-CAM</footer>
    </div>
  );
}
