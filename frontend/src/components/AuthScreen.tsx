"use client";

// Login / registration. Presentation + client-side validation only: the request itself is page.tsx's handleAuthSubmit.

import React, { useState } from 'react';
import { AlertCircle, ArrowRight, Check, Eye, EyeOff, Lock, Mail, RefreshCw, ShieldCheck, Sparkles, User, X, FileText, Target } from 'lucide-react';
import { isValidEmail } from '../lib/report';

interface Props {
  isLogin: boolean;
  setIsLogin: (v: boolean) => void;
  authName: string; setAuthName: (v: string) => void;
  authEmail: string; setAuthEmail: (v: string) => void;
  authPassword: string; setAuthPassword: (v: string) => void;
  authRole: string; setAuthRole: (v: string) => void;
  adminCode: string; setAdminCode: (v: string) => void;
  busy: boolean;
  error: string | null;
  notice: string | null;
  onDismiss: () => void;
  onSubmit: (e: React.FormEvent) => void;
}

const FEATURES = [
  { icon: Target, title: 'Role-aware questions', text: 'Each question is built for the role you choose, from a vetted knowledge base.' },
  { icon: FileText, title: 'Calibrated to your resume', text: 'Depth adapts to the experience your resume actually shows.' },
  { icon: ShieldCheck, title: 'Evidence-based evaluation', text: 'Every score is tied to what you answered, question by question.' },
];

type Touched = { name: boolean; email: boolean; password: boolean; code: boolean };

export default function AuthScreen(p: Props) {
  const [touched, setTouched] = useState<Touched>({ name: false, email: false, password: false, code: false });
  const [showPassword, setShowPassword] = useState(false);
  const touch = (k: keyof Touched) => setTouched((t) => ({ ...t, [k]: true }));

  const emailOk = isValidEmail(p.authEmail);
  const passwordOk = p.authPassword.length >= 1;
  const nameOk = p.authName.trim().length >= 2;
  const codeOk = p.authRole !== 'admin' || p.adminCode.trim().length > 0;
  const formOk = emailOk && passwordOk && (p.isLogin || (nameOk && codeOk));

  const errs = {
    name: touched.name && !p.isLogin && !nameOk ? 'Please enter your full name.' : '',
    email: touched.email && !emailOk ? 'Enter a valid email address.' : '',
    password: touched.password && !passwordOk ? 'Enter your password.' : '',
    code: touched.code && !codeOk ? 'The access code is required for administrator accounts.' : '',
  };

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (p.busy) return;
    if (!formOk) {
      setTouched({ name: true, email: true, password: true, code: true });
      return;
    }
    p.onSubmit(e);
  };

  const switchMode = (login: boolean) => {
    if (login === p.isLogin) return;
    p.setIsLogin(login);
    p.onDismiss();
    setTouched({ name: false, email: false, password: false, code: false });
  };

  return (
    <div className="sf-screen mx-auto grid w-full max-w-6xl items-center gap-10 py-6 lg:grid-cols-[1.08fr_1fr] lg:gap-16">
      {/* ------------------------------------------------------------ brand / hero panel */}
      <aside className="relative hidden lg:block" aria-hidden={false}>
        <div className="sf-stagger space-y-8">
          <div>
            <span className="sf-chip"><Sparkles size={12} className="text-indigo-300" /> AI technical interviews</span>
          </div>
          <h1 className="text-[2.9rem] font-semibold leading-[1.08] tracking-tight">
            <span className="sf-gradient-text">Interviews that adapt</span>
            <br />
            <span className="text-slate-300">to the person answering.</span>
          </h1>
          <p className="max-w-md text-[15px] leading-7 text-slate-400">
            A calm, one-to-one technical conversation. It listens to your answers, adjusts the depth, and finishes with a clear, evidence-based report.
          </p>
          <ul className="space-y-4">
            {FEATURES.map(({ icon: Icon, title, text }) => (
              <li key={title} className="flex items-start gap-3.5">
                <span className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border border-white/10 bg-white/[0.04] text-indigo-300"><Icon size={16} /></span>
                <span>
                  <span className="block text-[14px] font-medium text-slate-200">{title}</span>
                  <span className="block text-[13px] leading-6 text-slate-500">{text}</span>
                </span>
              </li>
            ))}
          </ul>

          {/* decorative preview of the product */}
          <div aria-hidden className="sf-float relative max-w-md">
            <div className="absolute -inset-6 -z-10 rounded-[2rem] bg-indigo-500/10 blur-3xl sf-glow-pulse" />
            <div className="sf-card p-4">
              <div className="mb-3 flex items-center justify-between text-[11px] text-slate-500">
                <span className="inline-flex items-center gap-1.5"><span className="h-1.5 w-1.5 rounded-full bg-indigo-400 iw-live-dot" /> In progress</span>
                <span>Question 3 of 10</span>
              </div>
              <div className="mb-3 flex gap-1">
                {Array.from({ length: 10 }).map((_, i) => <span key={i} className={`h-1 flex-1 rounded-full ${i < 2 ? 'bg-indigo-400/80' : i === 2 ? 'bg-indigo-400/40' : 'bg-white/[0.07]'}`} />)}
              </div>
              <div className="rounded-xl rounded-tl-sm border border-white/[0.06] bg-white/[0.03] px-3.5 py-2.5 text-[12.5px] leading-6 text-slate-300">
                How would you tell whether a model is overfitting, and what would you change first?
              </div>
              <div className="mt-2 ml-auto w-4/5 rounded-xl rounded-tr-sm border border-indigo-300/15 bg-indigo-400/[0.12] px-3.5 py-2.5 text-[12.5px] leading-6 text-indigo-50">
                I would compare training and validation error&hellip;
              </div>
            </div>
          </div>
        </div>
      </aside>

      {/* ------------------------------------------------------------ form card */}
      <section aria-label={p.isLogin ? 'Sign in' : 'Create account'} className="sf-enter mx-auto w-full max-w-md" style={{ animationDelay: '80ms' }}>
        <div className="sf-card p-6 sm:p-8">
          {/* mode switch */}
          <div role="tablist" aria-label="Account" className="relative mb-7 grid grid-cols-2 rounded-xl border border-white/[0.07] bg-black/20 p-1">
            <span aria-hidden className="absolute inset-y-1 left-1 w-[calc(50%-0.25rem)] rounded-lg bg-white/[0.08] shadow-[inset_0_1px_0_rgba(255,255,255,0.08)] transition-transform duration-300 ease-out"
              style={{ transform: p.isLogin ? 'translateX(0)' : 'translateX(100%)' }} />
            {([['Sign in', true], ['Create account', false]] as const).map(([label, login]) => (
              <button key={label} type="button" role="tab" aria-selected={p.isLogin === login} onClick={() => switchMode(login)}
                className={`sf-focus relative z-10 rounded-lg py-2 text-[13px] font-medium transition-colors duration-200 ${p.isLogin === login ? 'text-white' : 'text-slate-400 hover:text-slate-200'}`}>
                {label}
              </button>
            ))}
          </div>

          <div key={p.isLogin ? 'login' : 'register'} className="sf-swap">
            <h2 className="text-[1.65rem] font-semibold tracking-tight text-white">{p.isLogin ? 'Welcome back' : 'Create your account'}</h2>
            <p className="mt-1.5 text-[13.5px] leading-6 text-slate-400">
              {p.isLogin ? 'Sign in to continue to your interview.' : 'It takes a minute. Then upload your resume and begin.'}
            </p>
          </div>

          {p.notice && (
            <div role="status" className="sf-enter-soft mt-5 flex items-start gap-2.5 rounded-xl border border-emerald-400/20 bg-emerald-400/[0.07] px-3.5 py-3 text-[13px] text-emerald-100">
              <Check size={16} className="mt-0.5 shrink-0 text-emerald-300" /> <span className="flex-1">{p.notice}</span>
            </div>
          )}
          {p.error && (
            <div role="alert" className="sf-enter-soft mt-5 flex items-start gap-2.5 rounded-xl border border-rose-400/20 bg-rose-500/[0.07] px-3.5 py-3 text-[13px] text-rose-100">
              <AlertCircle size={16} className="mt-0.5 shrink-0 text-rose-300" /> <span className="flex-1">{p.error}</span>
              <button type="button" onClick={p.onDismiss} aria-label="Dismiss" className="sf-focus -m-1 rounded-md p-1 text-rose-200/70 hover:text-rose-100"><X size={14} /></button>
            </div>
          )}

          <form onSubmit={submit} noValidate className="mt-6 space-y-4">
            {!p.isLogin && (
              <div className="sf-enter-soft">
                <label htmlFor="auth-name" className="sf-label">Full name</label>
                <div className="relative">
                  <User size={15} className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-500" />
                  <input id="auth-name" type="text" autoComplete="name" value={p.authName} onChange={(e) => p.setAuthName(e.target.value)} onBlur={() => touch('name')}
                    aria-invalid={!!errs.name} aria-describedby={errs.name ? 'auth-name-err' : undefined} data-valid={touched.name && nameOk}
                    className="sf-input pl-10" placeholder="Jane Doe" />
                </div>
                {errs.name && <p id="auth-name-err" className="sf-enter-soft mt-1.5 text-[12px] text-rose-300">{errs.name}</p>}
              </div>
            )}

            <div>
              <label htmlFor="auth-email" className="sf-label">Email</label>
              <div className="relative">
                <Mail size={15} className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-500" />
                <input id="auth-email" type="email" autoComplete="email" value={p.authEmail} onChange={(e) => p.setAuthEmail(e.target.value)} onBlur={() => touch('email')}
                  aria-invalid={!!errs.email} aria-describedby={errs.email ? 'auth-email-err' : undefined} data-valid={touched.email && emailOk}
                  className="sf-input pl-10 pr-10" placeholder="you@email.com" />
                {emailOk && <Check size={15} className="sf-pop absolute right-3.5 top-1/2 -translate-y-1/2 text-emerald-300" aria-hidden />}
              </div>
              {errs.email && <p id="auth-email-err" className="sf-enter-soft mt-1.5 text-[12px] text-rose-300">{errs.email}</p>}
            </div>

            <div>
              <label htmlFor="auth-password" className="sf-label">Password</label>
              <div className="relative">
                <Lock size={15} className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-500" />
                <input id="auth-password" type={showPassword ? 'text' : 'password'} autoComplete={p.isLogin ? 'current-password' : 'new-password'} value={p.authPassword}
                  onChange={(e) => p.setAuthPassword(e.target.value)} onBlur={() => touch('password')}
                  aria-invalid={!!errs.password} aria-describedby={errs.password ? 'auth-password-err' : undefined}
                  className="sf-input pl-10 pr-11" placeholder="••••••••" />
                <button type="button" onClick={() => setShowPassword((v) => !v)} aria-label={showPassword ? 'Hide password' : 'Show password'} aria-pressed={showPassword}
                  className="sf-focus absolute right-2 top-1/2 -translate-y-1/2 rounded-lg p-1.5 text-slate-500 transition-colors hover:text-slate-200">
                  {showPassword ? <EyeOff size={15} /> : <Eye size={15} />}
                </button>
              </div>
              {errs.password && <p id="auth-password-err" className="sf-enter-soft mt-1.5 text-[12px] text-rose-300">{errs.password}</p>}
            </div>

            {!p.isLogin && (
              <fieldset className="sf-enter-soft">
                <legend className="sf-label">Account type</legend>
                <div className="grid grid-cols-2 gap-2.5">
                  {[['candidate', 'Candidate', 'Take an interview'], ['admin', 'Administrator', 'Review results']].map(([value, title, sub]) => (
                    <label key={value} className={`sf-hover-lift relative cursor-pointer rounded-xl border px-3.5 py-3 transition-colors duration-200 has-[:focus-visible]:ring-2 has-[:focus-visible]:ring-indigo-400/70 ${p.authRole === value ? 'border-indigo-400/50 bg-indigo-400/[0.09]' : 'border-white/[0.08] bg-white/[0.02]'}`}>
                      <input type="radio" name="account-type" value={value} checked={p.authRole === value} onChange={() => { p.setAuthRole(value); p.setAdminCode(''); }} className="sr-only" />
                      <span className="block text-[13px] font-medium text-slate-100">{title}</span>
                      <span className="block text-[11.5px] text-slate-500">{sub}</span>
                      {p.authRole === value && <Check size={14} className="sf-pop absolute right-3 top-3 text-indigo-300" aria-hidden />}
                    </label>
                  ))}
                </div>
              </fieldset>
            )}

            {!p.isLogin && p.authRole === 'admin' && (
              <div className="sf-enter-soft">
                <label htmlFor="auth-code" className="sf-label text-indigo-300">Administrator access code</label>
                <div className="relative">
                  <ShieldCheck size={15} className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-indigo-300/70" />
                  <input id="auth-code" type="text" value={p.adminCode} onChange={(e) => p.setAdminCode(e.target.value)} onBlur={() => touch('code')}
                    aria-invalid={!!errs.code} aria-describedby={errs.code ? 'auth-code-err' : undefined} className="sf-input pl-10" placeholder="Enter recruiter code" />
                </div>
                {errs.code && <p id="auth-code-err" className="sf-enter-soft mt-1.5 text-[12px] text-rose-300">{errs.code}</p>}
              </div>
            )}

            <button type="submit" disabled={p.busy} className="sf-btn sf-btn-primary group mt-2 w-full py-3">
              {p.busy ? <><RefreshCw size={15} className="sf-spin" /> {p.isLogin ? 'Signing in…' : 'Creating account…'}</>
                : <>{p.isLogin ? 'Sign in' : 'Create account'} <ArrowRight size={15} className="transition-transform duration-200 group-hover:translate-x-0.5" /></>}
            </button>
          </form>

          <p className="mt-6 text-center text-[13px] text-slate-500">
            {p.isLogin ? 'New here?' : 'Already registered?'}{' '}
            <button type="button" onClick={() => switchMode(!p.isLogin)} className="sf-link font-medium">{p.isLogin ? 'Create an account' : 'Sign in'}</button>
          </p>
        </div>
      </section>
    </div>
  );
}
