"use client";

// Evaluation result. Presentation only: the report comes from the existing summary endpoint (schema v2 fields are used when present).

import React, { useState } from 'react';
import { ArrowRight, Check, ChevronDown, FileBadge, MessageSquareQuote, RefreshCw, ShieldCheck, TrendingUp, TriangleAlert } from 'lucide-react';
import { useCountUp } from './useCountUp';
import { ANSWER_TYPE_ORDER, answerTypeLabel, clampScore, ringGeometry, scoreTone, sentenceCase, type ScoreTone } from '../lib/report';

export interface QAAnalysis {
  question: string;
  answer: string;
  score: number;
  feedback: string;
  answerType?: string;
  topic?: string | null;
}
export interface TopicPerf { topic: string; questions: number; averageScore: number }
export interface BehaviorReport { note?: string; dismissive?: number; unprofessional?: number; questions?: number[] }
export interface AnalysisReport {
  overallScore: number;
  summary: string;
  insights: string;
  resume_url?: string;
  breakdown: QAAnalysis[];
  strengths?: string[];
  weaknesses?: string[];
  topicPerformance?: TopicPerf[];
  answerTypeCounts?: Record<string, number>;
  behaviorNote?: string;
  behavior?: BehaviorReport;
}

interface Props {
  report: AnalysisReport;
  isEvaluating: boolean;
  isAdmin: boolean;
  showResume: boolean;
  onToggleResume: () => void;
  onBackToDashboard: () => void;
  onNewSession: () => void;
}

const TONE: Record<ScoreTone, { text: string; bar: string; stroke: string; chip: string }> = {
  high: { text: 'text-emerald-300', bar: 'bg-emerald-400', stroke: '#34d399', chip: 'border-emerald-400/25 bg-emerald-400/10 text-emerald-200' },
  mid: { text: 'text-amber-300', bar: 'bg-amber-400', stroke: '#fbbf24', chip: 'border-amber-400/25 bg-amber-400/10 text-amber-200' },
  low: { text: 'text-rose-300', bar: 'bg-rose-400', stroke: '#fb7185', chip: 'border-rose-400/25 bg-rose-400/10 text-rose-200' },
};

const list = (v: unknown): string[] => (Array.isArray(v) ? v.filter((x): x is string => typeof x === 'string' && x.trim().length > 0) : []);

function ScoreRing({ score, busy }: { score: number; busy: boolean }) {
  const shown = useCountUp(score, !busy);
  const tone = TONE[scoreTone(score)];
  const { circumference, offset } = ringGeometry(score);
  return (
    <div className="relative h-36 w-36 shrink-0" role="img" aria-label={busy ? 'Score is being calculated' : `Overall score ${score} out of 100`}>
      <div aria-hidden className="absolute inset-3 rounded-full bg-indigo-500/15 blur-2xl sf-glow-pulse" />
      <svg viewBox="0 0 132 132" className="relative h-full w-full -rotate-90">
        <circle cx="66" cy="66" r="54" fill="none" stroke="rgba(255,255,255,0.07)" strokeWidth="9" />
        {!busy && (
          <circle key={score} cx="66" cy="66" r="54" fill="none" stroke={tone.stroke} strokeWidth="9" strokeLinecap="round"
            strokeDasharray={circumference} strokeDashoffset={offset} className="sf-ring-fill"
            style={{ ['--sf-circ' as string]: circumference, ['--sf-off' as string]: offset, filter: `drop-shadow(0 0 6px ${tone.stroke}66)` }} />
        )}
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        {busy ? <RefreshCw size={22} className="sf-spin text-indigo-300" /> : (
          <>
            <span className="text-[2.6rem] font-semibold leading-none tabular-nums text-white">{shown}</span>
            <span className="mt-1 text-[11px] uppercase tracking-[0.14em] text-slate-500">out of 100</span>
          </>
        )}
      </div>
    </div>
  );
}

export default function ResultScreen(p: Props) {
  const { report, isEvaluating } = p;
  const [open, setOpen] = useState<Set<number>>(() => new Set());
  const score = clampScore(report.overallScore);
  const strengths = list(report.strengths);
  const weaknesses = list(report.weaknesses);
  const hasLists = Array.isArray(report.strengths) || Array.isArray(report.weaknesses);
  const topics = Array.isArray(report.topicPerformance) ? report.topicPerformance.filter((t) => t && typeof t.topic === 'string') : [];
  const counts = report.answerTypeCounts && typeof report.answerTypeCounts === 'object' ? report.answerTypeCounts : {};
  const countChips = ANSWER_TYPE_ORDER.filter((k) => (counts[k] ?? 0) > 0);
  const behavior = report.behavior;
  const behaviorFlagged = (behavior?.dismissive ?? 0) + (behavior?.unprofessional ?? 0);
  const breakdown = Array.isArray(report.breakdown) ? report.breakdown : [];
  const allOpen = breakdown.length > 0 && open.size === breakdown.length;

  const toggle = (i: number) => setOpen((prev) => { const n = new Set(prev); if (n.has(i)) n.delete(i); else n.add(i); return n; });
  const toggleAll = () => setOpen(allOpen ? new Set() : new Set(breakdown.map((_, i) => i)));

  return (
    <div className="mx-auto w-full max-w-4xl py-4">
      <div className="sf-stagger space-y-5">
        {/* ------------------------------------------------------------ overall */}
        <section aria-label="Overall result" className="sf-card overflow-hidden p-6 sm:p-8">
          <div aria-hidden className="pointer-events-none absolute -right-24 -top-24 h-64 w-64 rounded-full bg-indigo-500/10 blur-3xl" />
          <div className="relative flex flex-col items-center gap-7 text-center sm:flex-row sm:items-center sm:text-left">
            <ScoreRing score={score} busy={isEvaluating} />
            <div className="min-w-0 flex-1">
              <p className="sf-eyebrow">Interview evaluation</p>
              <h1 className="mt-1.5 text-[1.7rem] font-semibold tracking-tight text-white">{isEvaluating ? 'Preparing your evaluation…' : 'Evaluation complete'}</h1>
              {isEvaluating ? (
                <div className="mt-4 space-y-2.5" aria-hidden>
                  <div className="sf-shimmer h-3 w-full rounded-full" /><div className="sf-shimmer h-3 w-[88%] rounded-full" /><div className="sf-shimmer h-3 w-[62%] rounded-full" />
                </div>
              ) : (
                <>
                  <p className="mt-2.5 text-[14.5px] leading-7 text-slate-300">{report.summary}</p>
                  {countChips.length > 0 && (
                    <ul className="mt-4 flex flex-wrap justify-center gap-2 sm:justify-start" aria-label="Answers by type">
                      {countChips.map((k) => (
                        <li key={k} className="sf-chip"><span className="tabular-nums text-slate-100">{counts[k]}</span> {answerTypeLabel(k)}</li>
                      ))}
                    </ul>
                  )}
                </>
              )}
            </div>
          </div>
        </section>

        {!isEvaluating && (
          <>
            {/* ------------------------------------------------------------ strengths / weaknesses */}
            {hasLists ? (
              <div className="grid gap-5 md:grid-cols-2">
                <section aria-label="Strengths" className="sf-card p-5 sm:p-6">
                  <h2 className="flex items-center gap-2.5 text-[14px] font-semibold text-emerald-200"><span className="flex h-7 w-7 items-center justify-center rounded-lg bg-emerald-400/10 text-emerald-300"><TrendingUp size={15} /></span> Strengths</h2>
                  {strengths.length ? (
                    <ul className="mt-4 space-y-3">{strengths.map((s, i) => (
                      <li key={i} className="sf-enter-soft flex items-start gap-2.5 text-[13.5px] leading-6 text-slate-300" style={{ animationDelay: `${300 + i * 70}ms` }}><Check size={15} className="mt-1 shrink-0 text-emerald-300" />{s}</li>
                    ))}</ul>
                  ) : <p className="mt-4 text-[13px] text-slate-500">No clear strengths were identified from these answers.</p>}
                </section>
                <section aria-label="Areas to improve" className="sf-card p-5 sm:p-6">
                  <h2 className="flex items-center gap-2.5 text-[14px] font-semibold text-amber-200"><span className="flex h-7 w-7 items-center justify-center rounded-lg bg-amber-400/10 text-amber-300"><TriangleAlert size={15} /></span> Areas to improve</h2>
                  {weaknesses.length ? (
                    <ul className="mt-4 space-y-3">{weaknesses.map((s, i) => (
                      <li key={i} className="sf-enter-soft flex items-start gap-2.5 text-[13.5px] leading-6 text-slate-300" style={{ animationDelay: `${300 + i * 70}ms` }}><span aria-hidden className="mt-2.5 h-1.5 w-1.5 shrink-0 rounded-full bg-amber-300/80" />{s}</li>
                    ))}</ul>
                  ) : <p className="mt-4 text-[13px] text-slate-500">No weaknesses were identified from these answers.</p>}
                </section>
              </div>
            ) : report.insights ? (
              <section aria-label="Insights" className="sf-card p-5 sm:p-6">
                <h2 className="text-[14px] font-semibold text-indigo-200">Insights</h2>
                <p className="mt-3 text-[13.5px] leading-7 text-slate-300">{report.insights}</p>
              </section>
            ) : null}

            {/* ------------------------------------------------------------ topics */}
            {topics.length > 0 && (
              <section aria-label="Topic performance" className="sf-card p-5 sm:p-6">
                <h2 className="text-[14px] font-semibold text-slate-100">Topic performance</h2>
                <ul className="mt-5 space-y-4">
                  {topics.map((t, i) => {
                    const v = clampScore(t.averageScore);
                    const tone = TONE[scoreTone(v)];
                    return (
                      <li key={t.topic + i}>
                        <div className="mb-1.5 flex items-baseline justify-between gap-3">
                          <span className="min-w-0 truncate text-[13.5px] text-slate-200">{sentenceCase(t.topic)}</span>
                          <span className="shrink-0 text-[12px] text-slate-500">{t.questions} {t.questions === 1 ? 'question' : 'questions'} · <span className={`font-medium tabular-nums ${tone.text}`}>{v}</span></span>
                        </div>
                        <div className="h-2 overflow-hidden rounded-full bg-white/[0.06]" role="meter" aria-valuemin={0} aria-valuemax={100} aria-valuenow={v} aria-label={`${t.topic} average score`}>
                          <div className={`sf-grow-x h-full rounded-full ${tone.bar}`} style={{ width: `${v}%`, animationDelay: `${250 + i * 90}ms` }} />
                        </div>
                      </li>
                    );
                  })}
                </ul>
              </section>
            )}

            {/* ------------------------------------------------------------ behaviour (separate from the technical score) */}
            {behavior && (report.behaviorNote || behavior.note) && (
              <section aria-label="Professionalism" className="sf-card-quiet relative overflow-hidden p-5 sm:p-6">
                <div aria-hidden className="absolute inset-y-0 left-0 w-[3px] bg-gradient-to-b from-violet-400/70 to-indigo-400/20" />
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <h2 className="flex items-center gap-2.5 text-[14px] font-semibold text-violet-200"><span className="flex h-7 w-7 items-center justify-center rounded-lg bg-violet-400/10 text-violet-300"><ShieldCheck size={15} /></span> Professionalism</h2>
                  <span className="sf-chip">Reported separately · not part of the technical score</span>
                </div>
                <p className="mt-3 text-[13.5px] leading-7 text-slate-300">{report.behaviorNote || behavior.note}</p>
                {behaviorFlagged > 0 && (
                  <ul className="mt-3 flex flex-wrap gap-2 text-[12px]">
                    {(behavior.dismissive ?? 0) > 0 && <li className="sf-chip">{behavior.dismissive} dismissive</li>}
                    {(behavior.unprofessional ?? 0) > 0 && <li className="sf-chip">{behavior.unprofessional} unprofessional</li>}
                    {Array.isArray(behavior.questions) && behavior.questions.length > 0 && <li className="sf-chip">Questions {behavior.questions.join(', ')}</li>}
                  </ul>
                )}
              </section>
            )}

            {/* ------------------------------------------------------------ admin: original resume */}
            {p.isAdmin && report.resume_url && (
              <section aria-label="Candidate resume" className="sf-card p-5">
                <button type="button" onClick={p.onToggleResume} aria-expanded={p.showResume} className="sf-focus flex w-full items-center justify-between rounded-lg text-left">
                  <span className="flex items-center gap-2 text-[14px] font-semibold text-indigo-200"><FileBadge size={16} /> Candidate resume (original PDF)</span>
                  <span className="flex items-center gap-1.5 text-[12px] text-slate-500">{p.showResume ? 'Hide' : 'View'} <ChevronDown size={14} className={`transition-transform duration-300 ${p.showResume ? 'rotate-180' : ''}`} /></span>
                </button>
                {p.showResume && (
                  <div className="sf-enter-soft mt-4 h-[600px] overflow-hidden rounded-xl border border-white/10">
                    <iframe src={`${report.resume_url}#toolbar=0`} width="100%" height="100%" title="Candidate Resume" className="border-none bg-slate-950" />
                  </div>
                )}
              </section>
            )}

            {/* ------------------------------------------------------------ per-question breakdown */}
            {breakdown.length > 0 && (
              <section aria-label="Question breakdown">
                <div className="mb-3 flex items-center justify-between px-1">
                  <h2 className="text-[14px] font-semibold text-slate-100">Question breakdown</h2>
                  <button type="button" onClick={toggleAll} className="sf-link sf-focus text-[12.5px] font-medium">{allOpen ? 'Collapse all' : 'Expand all'}</button>
                </div>
                <ol className="space-y-3">
                  {breakdown.map((item, i) => {
                    const v = clampScore(item.score);
                    const tone = TONE[scoreTone(v)];
                    const isOpen = open.has(i);
                    return (
                      <li key={i} className="sf-card-quiet sf-enter-soft overflow-hidden transition-colors duration-200 hover:border-white/[0.12]" style={{ animationDelay: `${i * 45}ms` }}>
                        <button type="button" onClick={() => toggle(i)} aria-expanded={isOpen} aria-controls={`qa-${i}`} className="sf-focus flex w-full items-start gap-3.5 px-4 py-3.5 text-left sm:px-5">
                          <span className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full border border-white/10 text-[11px] tabular-nums text-slate-400">{i + 1}</span>
                          <span className="min-w-0 flex-1">
                            <span className={`block text-[13.5px] leading-6 text-slate-200 ${isOpen ? '' : 'line-clamp-2'}`}>{item.question}</span>
                            <span className="mt-1.5 flex flex-wrap items-center gap-2">
                              {item.answerType && <span className="sf-chip">{answerTypeLabel(item.answerType)}</span>}
                              {item.topic && <span className="hidden text-[11.5px] text-slate-500 sm:inline">{sentenceCase(item.topic)}</span>}
                            </span>
                          </span>
                          <span className={`shrink-0 rounded-lg border px-2.5 py-1 text-[12px] font-medium tabular-nums ${tone.chip}`}>{v}</span>
                          <ChevronDown size={16} className={`mt-1 shrink-0 text-slate-500 transition-transform duration-300 ${isOpen ? 'rotate-180' : ''}`} aria-hidden />
                        </button>
                        <div id={`qa-${i}`} data-open={isOpen} className="sf-collapse" role="region" aria-label={`Details for question ${i + 1}`}>
                          <div>
                            <div className="space-y-3 border-t border-white/[0.06] px-4 py-4 sm:px-5">
                              <div>
                                <p className="mb-1.5 text-[11px] font-medium uppercase tracking-[0.1em] text-slate-500">Your answer</p>
                                <p className="whitespace-pre-wrap break-words rounded-lg border border-white/[0.05] bg-black/20 px-3.5 py-3 text-[13px] leading-6 text-slate-300">{item.answer || '[No response provided]'}</p>
                              </div>
                              <div className="flex items-start gap-2.5 rounded-lg border border-indigo-300/10 bg-indigo-400/[0.05] px-3.5 py-3">
                                <MessageSquareQuote size={15} className="mt-0.5 shrink-0 text-indigo-300" aria-hidden />
                                <p className="text-[13px] leading-6 text-slate-300">{item.feedback}</p>
                              </div>
                            </div>
                          </div>
                        </div>
                      </li>
                    );
                  })}
                </ol>
              </section>
            )}

            <div className="flex justify-end pb-8 pt-2">
              {p.isAdmin ? (
                <button type="button" onClick={p.onBackToDashboard} className="sf-btn sf-btn-primary group">Back to dashboard <ArrowRight size={15} className="transition-transform duration-200 group-hover:translate-x-0.5" /></button>
              ) : (
                <button type="button" onClick={p.onNewSession} className="sf-btn sf-btn-ghost"><RefreshCw size={14} /> Start a new interview</button>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
