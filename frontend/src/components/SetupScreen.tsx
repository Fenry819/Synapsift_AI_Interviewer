"use client";

// Candidate setup: choose a role, add a resume, begin. Presentation only: starting is page.tsx's startSession.

import React, { useRef, useState } from 'react';
import { ArrowRight, Brain, BarChart3, Check, FileText, Layers, PenLine, Server, Upload, X, AlertCircle } from 'lucide-react';
import { formatFileSize, isPdfFile } from '../lib/report';
import { MAX_QUESTIONS } from '../lib/interviewProgress';

interface Props {
  userName: string;
  selectedRole: string;
  setSelectedRole: (v: string) => void;
  customRole: string;
  setCustomRole: (v: string) => void;
  file: File | null;
  setFile: (f: File | null) => void;
  onStart: () => void;
}

const ROLES = [
  { value: 'AI / Machine Learning Role', icon: Brain, text: 'Core ML concepts, evaluation and deep learning' },
  { value: 'Data Science / Applied ML Role', icon: BarChart3, text: 'Data preparation, modelling and validation' },
  { value: 'Advanced / Theoretical ML', icon: Layers, text: 'Probabilistic models, kernels and theory' },
  { value: 'Backend Engineering Intern', icon: Server, text: 'APIs, databases and systems fundamentals' },
  { value: 'Custom Role', icon: PenLine, text: 'Describe the role you are applying for' },
];

export default function SetupScreen(p: Props) {
  const [dragging, setDragging] = useState(false);
  const [fileError, setFileError] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  const isCustom = p.selectedRole === 'Custom Role';
  const roleDone = !isCustom || p.customRole.trim().length > 0;
  const fileDone = !!p.file;
  const ready = roleDone && fileDone;

  const accept = (f: File | null | undefined) => {
    if (!f) return;
    if (!isPdfFile(f)) { setFileError('Please choose a PDF file.'); return; }
    setFileError('');
    p.setFile(f);
  };

  const steps = [
    { label: 'Target role', done: roleDone },
    { label: 'Resume', done: fileDone },
    { label: 'Begin', done: false },
  ];
  const activeStep = !roleDone ? 0 : !fileDone ? 1 : 2;

  return (
    <div className="sf-screen mx-auto w-full max-w-3xl py-4">
      <div className="mb-8 text-center sf-enter-soft">
        <p className="sf-eyebrow">Your interview</p>
        <h1 className="mt-2 text-[2rem] font-semibold tracking-tight sm:text-[2.35rem]"><span className="sf-gradient-text">Let&rsquo;s set up your interview{p.userName ? `, ${p.userName.split(' ')[0]}` : ''}</span></h1>
        <p className="mx-auto mt-2.5 max-w-lg text-[14.5px] leading-7 text-slate-400">Pick the role you are interviewing for and add your resume. Questions are tailored to both.</p>
      </div>

      {/* progression */}
      <ol aria-label="Progress" className="mx-auto mb-7 flex max-w-md items-center">
        {steps.map((s, i) => (
          <li key={s.label} className="flex flex-1 items-center last:flex-none">
            <span className="flex items-center gap-2">
              <span className={`flex h-6 w-6 items-center justify-center rounded-full border text-[11px] font-medium transition-all duration-300 ${s.done ? 'border-indigo-400/60 bg-indigo-400/20 text-indigo-100' : i === activeStep ? 'border-indigo-300/60 text-indigo-200 shadow-[0_0_16px_-2px_rgba(129,140,248,0.7)]' : 'border-white/10 text-slate-500'}`}>
                {s.done ? <Check size={12} className="sf-pop" /> : i + 1}
              </span>
              <span className={`hidden text-[12px] transition-colors duration-300 sm:inline ${s.done || i === activeStep ? 'text-slate-200' : 'text-slate-500'}`}>{s.label}</span>
            </span>
            {i < steps.length - 1 && <span aria-hidden className="mx-3 h-px flex-1 overflow-hidden bg-white/10"><span className="block h-full origin-left bg-indigo-400/60 transition-transform duration-500" style={{ transform: s.done ? 'scaleX(1)' : 'scaleX(0)' }} /></span>}
          </li>
        ))}
      </ol>

      <div className="sf-card p-5 sm:p-7">
        {/* 1. role */}
        <fieldset>
          <legend className="mb-3 flex items-center gap-2 text-[13px] font-medium text-slate-200"><span className="sf-eyebrow">01</span> Target role</legend>
          <div role="radiogroup" aria-label="Target role" className="grid gap-2.5 sm:grid-cols-2">
            {ROLES.map(({ value, icon: Icon, text }) => {
              const selected = p.selectedRole === value;
              return (
                <label key={value} className={`sf-hover-lift group relative flex cursor-pointer items-start gap-3 rounded-xl border px-4 py-3.5 has-[:focus-visible]:ring-2 has-[:focus-visible]:ring-indigo-400/70 ${selected ? 'border-indigo-400/55 bg-indigo-400/[0.09] shadow-[0_0_32px_-14px_rgba(99,102,241,0.8)]' : 'border-white/[0.08] bg-white/[0.02]'} ${value === 'Custom Role' ? 'sm:col-span-2' : ''}`}>
                  <input type="radio" name="target-role" value={value} checked={selected} className="sr-only"
                    onChange={() => { p.setSelectedRole(value); if (value !== 'Custom Role') p.setCustomRole(''); }} />
                  <span className={`mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border transition-colors duration-200 ${selected ? 'border-indigo-300/40 bg-indigo-400/15 text-indigo-200' : 'border-white/10 bg-white/[0.03] text-slate-400 group-hover:text-slate-200'}`}><Icon size={16} /></span>
                  <span className="min-w-0 flex-1">
                    <span className="block text-[13.5px] font-medium text-slate-100">{value}</span>
                    <span className="block text-[12px] leading-5 text-slate-500">{text}</span>
                  </span>
                  {selected && <Check size={15} className="sf-pop mt-1 shrink-0 text-indigo-300" aria-hidden />}
                </label>
              );
            })}
          </div>
          {isCustom && (
            <div className="sf-enter-soft mt-3">
              <label htmlFor="custom-role" className="sf-label">Role title</label>
              <input id="custom-role" type="text" autoFocus value={p.customRole} onChange={(e) => p.setCustomRole(e.target.value)} placeholder="e.g. Cloud Security Architect" className="sf-input" />
            </div>
          )}
        </fieldset>

        <div className="sf-divider my-6" />

        {/* 2. resume */}
        <div>
          <p className="mb-3 flex items-center gap-2 text-[13px] font-medium text-slate-200"><span className="sf-eyebrow">02</span> Resume</p>
          {!p.file ? (
            <div
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => { e.preventDefault(); setDragging(false); accept(e.dataTransfer.files?.[0]); }}
              className={`group relative flex flex-col items-center justify-center gap-3 rounded-2xl border border-dashed px-6 py-10 text-center transition-all duration-300 has-[:focus-visible]:ring-2 has-[:focus-visible]:ring-indigo-400/70 ${dragging ? 'scale-[1.01] border-indigo-300/80 bg-indigo-400/[0.1] shadow-[0_0_40px_-12px_rgba(99,102,241,0.8)]' : 'border-white/15 bg-white/[0.02] hover:border-indigo-300/50 hover:bg-white/[0.04]'}`}
            >
              <input ref={inputRef} type="file" accept=".pdf,application/pdf" aria-label="Upload resume (PDF)" onChange={(e) => { accept(e.target.files?.[0]); e.target.value = ''; }} className="absolute inset-0 h-full w-full cursor-pointer opacity-0" />
              <span className={`flex h-12 w-12 items-center justify-center rounded-2xl border border-white/10 bg-white/[0.04] text-indigo-300 transition-transform duration-300 ${dragging ? '-translate-y-1 scale-110' : 'group-hover:-translate-y-0.5'}`}><Upload size={20} /></span>
              <span>
                <span className="block text-[14px] font-medium text-slate-100">{dragging ? 'Drop your resume here' : 'Drag your resume here, or click to browse'}</span>
                <span className="mt-1 block text-[12.5px] text-slate-500">PDF only. Used to tailor your questions.</span>
              </span>
            </div>
          ) : (
            <div className="sf-pop flex items-center gap-3.5 rounded-2xl border border-emerald-400/25 bg-emerald-400/[0.05] px-4 py-3.5">
              <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-emerald-300/20 bg-emerald-300/10 text-emerald-200"><FileText size={18} /></span>
              <span className="min-w-0 flex-1">
                <span className="block truncate text-[13.5px] font-medium text-slate-100">{p.file.name}</span>
                <span className="block text-[12px] text-slate-500">{formatFileSize(p.file.size)} · Ready</span>
              </span>
              <button type="button" onClick={() => { p.setFile(null); setFileError(''); }} aria-label="Remove resume" className="sf-focus rounded-lg p-2 text-slate-400 transition-colors hover:bg-white/[0.06] hover:text-slate-100"><X size={16} /></button>
            </div>
          )}
          {fileError && (
            <p role="alert" className="sf-enter-soft mt-2.5 flex items-center gap-2 text-[12.5px] text-rose-300"><AlertCircle size={14} /> {fileError}</p>
          )}
        </div>

        {/* 3. begin */}
        <div className="mt-7">
          <button type="button" onClick={p.onStart} disabled={!ready} className="sf-btn sf-btn-primary group w-full py-3.5 text-[14.5px]">
            Begin interview <ArrowRight size={16} className="transition-transform duration-200 group-hover:translate-x-0.5" />
          </button>
          <p className="mt-3 text-center text-[12px] text-slate-500" aria-live="polite">
            {ready ? `Up to ${MAX_QUESTIONS} questions. You can abort at any time.` : !roleDone ? 'Enter a role title to continue.' : 'Add your resume to continue.'}
          </p>
        </div>
      </div>
    </div>
  );
}
