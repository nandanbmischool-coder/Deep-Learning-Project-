import { zodResolver } from "@hookform/resolvers/zod";
import { Activity } from "lucide-react";
import { forwardRef, useState } from "react";
import { useForm } from "react-hook-form";
import { Link, Navigate, useLocation, useNavigate } from "react-router-dom";
import { z } from "zod";
import { toApiError } from "../lib/api";
import { useAuth } from "../lib/auth";
import { Disclaimer } from "../components/ui";

const loginSchema = z.object({ email: z.string().email("Enter a valid email address."), password: z.string().min(1, "Enter your password.") });
const registerSchema = z.object({
  full_name: z.string().min(2, "Enter your full name."),
  email: z.string().email("Enter a valid email address."),
  password: z.string().min(10, "Use at least 10 characters.").regex(/[A-Za-z]/, "Include a letter.").regex(/\d/, "Include a digit."),
});

function Shell({ title, children, footer }: { title: string; children: React.ReactNode; footer: React.ReactNode }) {
  return (
    <div className="grid min-h-screen place-items-center bg-ink px-4">
      <div className="w-full max-w-sm">
        <Link to="/" className="mb-6 flex items-center justify-center gap-2 font-display text-2xl text-white"><Activity className="h-6 w-6 text-[#7FD1D8]" aria-hidden />PneumoVision AI</Link>
        <div className="card p-6">
          <h1 className="text-2xl">{title}</h1>
          <div className="mt-4">{children}</div>
        </div>
        <p className="mt-4 text-center text-sm text-white/70">{footer}</p>
        <div className="mt-4 text-center [&_p]:text-white/50"><Disclaimer /></div>
      </div>
    </div>
  );
}

const Field = forwardRef<HTMLInputElement, { label: string; error?: string } & React.InputHTMLAttributes<HTMLInputElement>>(
  function Field({ label, error, ...rest }, ref) {
    return (
      <div>
        <label className="label" htmlFor={rest.id}>{label}</label>
        <input ref={ref} className="input" aria-invalid={!!error} aria-describedby={error ? `${rest.id}-err` : undefined} {...rest} />
        {error && <p id={`${rest.id}-err`} className="mt-1 text-xs text-brick">{error}</p>}
      </div>
    );
  },
);

export function Login() {
  const { user, login } = useAuth();
  const nav = useNavigate();
  const loc = useLocation();
  const [err, setErr] = useState("");
  const f = useForm<z.infer<typeof loginSchema>>({ resolver: zodResolver(loginSchema) });
  if (user) return <Navigate to="/app" replace />;
  const from = (loc.state as { from?: string } | null)?.from ?? "/app";
  return (
    <Shell title="Sign in" footer={<>New here? <Link className="text-[#7FD1D8] underline" to="/register">Create an account</Link></>}>
      <form className="space-y-4" onSubmit={f.handleSubmit(async (v) => { setErr(""); try { await login(v.email, v.password); nav(from, { replace: true }); } catch (e) { setErr(toApiError(e).message); } })} noValidate>
        <Field id="email" label="Email" type="email" autoComplete="email" error={f.formState.errors.email?.message} {...f.register("email")} />
        <Field id="password" label="Password" type="password" autoComplete="current-password" error={f.formState.errors.password?.message} {...f.register("password")} />
        {err && <p role="alert" className="text-sm text-brick">{err}</p>}
        <button className="btn-primary w-full" disabled={f.formState.isSubmitting}>{f.formState.isSubmitting ? "Signing in…" : "Sign in"}</button>
      </form>
    </Shell>
  );
}

export function Register() {
  const { user, register } = useAuth();
  const nav = useNavigate();
  const [err, setErr] = useState("");
  const f = useForm<z.infer<typeof registerSchema>>({ resolver: zodResolver(registerSchema) });
  if (user) return <Navigate to="/app" replace />;
  return (
    <Shell title="Create your account" footer={<>Already registered? <Link className="text-[#7FD1D8] underline" to="/login">Sign in</Link></>}>
      <form className="space-y-4" onSubmit={f.handleSubmit(async (v) => { setErr(""); try { await register(v.email, v.full_name, v.password); nav("/app", { replace: true }); } catch (e) { setErr(toApiError(e).message); } })} noValidate>
        <Field id="full_name" label="Full name" autoComplete="name" error={f.formState.errors.full_name?.message} {...f.register("full_name")} />
        <Field id="email" label="Email" type="email" autoComplete="email" error={f.formState.errors.email?.message} {...f.register("email")} />
        <Field id="password" label="Password" type="password" autoComplete="new-password" error={f.formState.errors.password?.message} {...f.register("password")} />
        <p className="text-xs text-mute">The first account created on a fresh install becomes the administrator.</p>
        {err && <p role="alert" className="text-sm text-brick">{err}</p>}
        <button className="btn-primary w-full" disabled={f.formState.isSubmitting}>{f.formState.isSubmitting ? "Creating…" : "Create account"}</button>
      </form>
    </Shell>
  );
}
