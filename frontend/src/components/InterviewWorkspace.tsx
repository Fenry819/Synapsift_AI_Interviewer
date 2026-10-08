"use client";

// Presentation of the candidate interview: header/status, chat history, composer, end state.
// All behaviour (sending, progressive reveal, double-submit guard, abort, evaluation) stays in page.tsx and arrives via props.

import React, { useEffect, useRef } from 'react';
import { Send, AlertCircle, ArrowRight, Network, RefreshCw, X } from 'lucide-react';
import { interviewProgress, workspaceStatus, STATUS_LABEL, MAX_QUESTIONS } from '../lib/interviewProgress';

export interface ChatMessage {
  id: string;
  sender: 'ai' | 'candidate';
  text: string;
  timestamp: string;
}

export interface WorkspaceError { title: string; message: string }

interface Props {
  role: string;
  messages: ChatMessage[];
  revealing: { id: string; shown: string } | null;   // the interviewer message currently being revealed (presentation only)
  isAiThinking: boolean;
  inputLocked: boolean;                                // waiting for the interviewer or revealing: composer disabled
  isComplete: boolean;
  policyClosed: boolean;                               // session closed by domain / policy / resignation (no evaluation to view)
  inputAnswer: string;
  error: WorkspaceError | null;
  onInputChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void;
  onSend: () => void;
  onAbort: () => void;
  onViewEvaluation: () => void;
  onReturnToSetup: () => void;
  onDismissError: () => void;
}

const focusRing = 'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400/70 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950';

export default function InterviewWorkspace(p: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const stickRef = useRef(true);                       // follow the newest message unless the reader scrolled up

  const progress = interviewProgress(p.messages, p.isComplete, MAX_QUESTIONS);
  const status = workspaceStatus({ messageCount: p.messages.length, isAiThinking: p.isAiThinking, isComplete: p.isComplete, policyClosed: p.policyClosed });
  const showProgress = !p.policyClosed && (p.messages.length > 0 || p.isAiThinking);
  const generating = status === 'generating' || status === 'starting';
  const revealingText = p.revealing?.shown.length ?? 0;

  useEffect(() => {
    const el = scrollRef.current;
    if (!el || !stickRef.current) return;
    const reduced = typeof window !== 'undefined' && !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    // While text is being revealed follow it instantly (smooth scrolling would lag behind 30 ms steps).
    el.scrollTo({ top: el.scrollHeight, behavior: reduced || revealingText > 0 ? 'auto' : 'smooth' });
  }, [p.messages.length, p.isAiThinking, revealingText, p.isComplete, p.error]);

  // The textarea grows with its content (and shrinks back when the answer is sent), and takes focus whenever it is usable.
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = 'auto';
    ta.style.height = `${Math.min(ta.scrollHeight, 160)}px`;
  }, [p.inputAnswer, p.isComplete]);

  useEffect(() => {
    if (!p.inputLocked && !p.isComplete) textareaRef.current?.focus();
  }, [p.inputLocked, p.isComplete]);

  const onScroll = () => {
    const el = scrollRef.current;
    if (el) stickRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
  };

  const statusTone = status === 'concluded' ? 'bg-emerald-400' : status === 'closed' ? 'bg-slate-500' : 'bg-indigo-400';

  return (
    <section aria-label="Interview" className="sf-card iw-rise mx-auto flex h-[calc(100dvh-9rem)] min-h-[30rem] w-full max-w-3xl flex-col overflow-hidden">
      {/* ---------------------------------------------------------------- header / status */}
      <header className="relative border-b border-white/[0.06] bg-black/20 px-4 pb-3.5 pt-3 sm:px-5">
        <div className="flex items-center justify-between gap-3">
          <div className="min-w-0">
            <p className="truncate text-[13px] font-medium text-slate-100">{p.role || 'Technical interview'}</p>
            <div className="mt-1.5 flex items-center gap-2 text-[11px] text-slate-400">
              <span key={status} className="iw-fade inline-flex items-center gap-1.5 rounded-full border border-white/[0.08] bg-white/[0.04] px-2 py-0.5">
                <span aria-hidden className={`h-1.5 w-1.5 rounded-full ${statusTone} ${generating ? 'iw-live-dot' : ''}`} />
                {STATUS_LABEL[status]}
              </span>
              {showProgress && (
                <>
                  <span aria-hidden className="text-slate-600">·</span>
                  <span className="tabular-nums" aria-label={`Question ${progress.current} of ${progress.total}`}>
                    {p.isComplete ? 'Completed' : <>Question {progress.current}<span className="hidden sm:inline"> of {progress.total}</span><span className="sm:hidden">/{progress.total}</span></>}
                  </span>
                </>
              )}
            </div>
          </div>
          <button
            type="button"
            onClick={p.onAbort}
            aria-label="Abort interview"
            className={`iw-motion-safe shrink-0 rounded-lg border border-white/[0.08] px-3 py-1.5 text-[12px] text-slate-400 transition duration-200 hover:border-rose-400/30 hover:bg-rose-500/[0.08] hover:text-rose-200 active:scale-[0.98] ${focusRing}`}
          >
            Abort<span className="hidden sm:inline"> interview</span>
          </button>
        </div>
        {showProgress && (
          <div className="mt-3 flex gap-1" role="progressbar" aria-valuemin={0} aria-valuemax={progress.total} aria-valuenow={p.isComplete ? progress.total : progress.answered} aria-label="Interview progress">
            {Array.from({ length: progress.total }).map((_, i) => {
              const done = p.isComplete || i < progress.answered;
              const current = !p.isComplete && i === progress.answered;
              return <span key={i} className={`h-1 flex-1 rounded-full transition-all duration-500 ${done ? 'bg-indigo-400/80 shadow-[0_0_8px_rgba(129,140,248,0.5)]' : current ? 'bg-indigo-400/35 iw-live-dot' : 'bg-white/[0.07]'}`} />;
            })}
          </div>
        )}
      </header>

      {/* ---------------------------------------------------------------- chat history */}
      <div ref={scrollRef} onScroll={onScroll} className="iw-scroll flex-1 overflow-y-auto px-4 py-6 sm:px-6" aria-live="off">
        <ol className="mx-auto flex max-w-[44rem] flex-col gap-5" aria-label="Conversation">
          {p.messages.length === 0 && !p.isAiThinking && (
            <li className="iw-fade py-16 text-center text-[13px] text-slate-500">Your interview will appear here.</li>
          )}
          {p.messages.map((m) => {
            const isAi = m.sender === 'ai';
            const revealingThis = p.revealing?.id === m.id;
            return (
              <li key={m.id} className={`flex flex-col ${isAi ? 'items-start iw-msg-ai' : 'items-end iw-msg-user'}`}>
                <span className={`mb-1.5 inline-flex items-center gap-1.5 px-1 text-[10.5px] font-medium uppercase tracking-[0.08em] ${isAi ? 'text-slate-500' : 'text-indigo-300/60'}`}>
                  {isAi && <span aria-hidden className="flex h-4 w-4 items-center justify-center rounded-md bg-indigo-400/20 text-indigo-200"><Network size={10} /></span>}
                  {isAi ? 'Interviewer' : 'You'}
                  {m.timestamp && <span className="ml-2 font-normal normal-case tracking-normal text-slate-600">{m.timestamp}</span>}
                </span>
                <div
                  className={`whitespace-pre-wrap break-words text-[15px] leading-[1.7] ${isAi
                    ? 'max-w-[92%] rounded-2xl rounded-tl-md border border-white/[0.06] bg-white/[0.03] px-4 py-3 text-slate-100 sm:max-w-[88%]'
                    : 'max-w-[88%] rounded-2xl rounded-tr-md border border-indigo-300/15 bg-indigo-400/[0.12] px-4 py-3 text-indigo-50 sm:max-w-[78%]'}`}
                >
                  {revealingThis ? p.revealing!.shown : m.text}
                  {revealingThis && <span aria-hidden className="iw-caret" />}
                </div>
              </li>
            );
          })}

          {p.isAiThinking && (
            <li className="iw-thinking flex items-start" role="status" aria-live="polite">
              <div className="relative inline-flex items-center gap-3 overflow-hidden rounded-2xl rounded-tl-md border border-white/[0.06] bg-white/[0.025] px-4 py-3 text-[13px] text-slate-400">
                <span className="iw-sheen absolute inset-0" aria-hidden />
                <span className="relative inline-flex items-center gap-1 text-indigo-300" aria-hidden><i className="iw-dot" /><i className="iw-dot" /><i className="iw-dot" /></span>
                <span className="relative">Interviewer is preparing the next question…</span>
              </div>
            </li>
          )}
        </ol>
      </div>

      {/* ---------------------------------------------------------------- error / composer / end state (anchored) */}
      <div className="border-t border-white/[0.06] bg-black/25 px-4 pb-4 pt-3 sm:px-5">
        <div className="mx-auto max-w-[44rem]">
          {p.error && (
            <div role="alert" className="iw-drop mb-3 flex items-start gap-3 rounded-xl border border-rose-400/20 bg-rose-500/[0.07] px-3.5 py-3">
              <AlertCircle aria-hidden size={16} className="mt-0.5 shrink-0 text-rose-300" />
              <div className="min-w-0 flex-1">
                <p className="text-[13px] font-medium text-rose-100">{p.error.title}</p>
                <p className="mt-0.5 text-[12.5px] leading-5 text-rose-200/80">{p.error.message}</p>
              </div>
              <button type="button" onClick={p.onDismissError} aria-label="Dismiss message" className={`iw-motion-safe -m-1 rounded-md p-1 text-rose-200/70 transition duration-150 hover:bg-rose-400/10 hover:text-rose-100 ${focusRing}`}>
                <X size={14} />
              </button>
            </div>
          )}

          {!p.isComplete ? (
            <div key="composer" className="iw-fade">
              <div
                aria-busy={p.inputLocked}
                className={`flex items-end gap-2 rounded-xl border bg-[rgba(5,7,16,0.72)] px-3 py-2 transition-all duration-300 ${p.inputLocked
                  ? 'border-white/[0.05] opacity-60'
                  : 'border-white/[0.09] hover:border-white/[0.16] focus-within:border-indigo-400/70 focus-within:shadow-[0_0_0_4px_rgba(99,102,241,0.15),0_0_28px_-8px_rgba(99,102,241,0.5)]'}`}
              >
                <label htmlFor="interview-answer" className="sr-only">Your answer</label>
                <textarea
                  id="interview-answer"
                  ref={textareaRef}
                  placeholder={p.inputLocked ? 'Please wait…' : 'Write your answer…'}
                  value={p.inputAnswer}
                  onChange={p.onInputChange}
                  disabled={p.inputLocked}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      p.onSend();
                    }
                  }}
                  className="iw-scroll max-h-40 min-h-[44px] flex-1 resize-none border-none bg-transparent px-1 py-2.5 text-[14.5px] leading-6 text-slate-100 placeholder:text-slate-500 focus:outline-none disabled:cursor-not-allowed"
                  rows={1}
                />
                <button
                  type="button"
                  onClick={p.onSend}
                  disabled={p.inputLocked || !p.inputAnswer.trim()}
                  className="sf-btn sf-btn-primary iw-motion-safe mb-0.5 h-9 shrink-0 px-3.5 py-0 text-[13px]"
                >
                  {p.inputLocked ? <RefreshCw aria-hidden size={14} className="animate-spin" /> : <Send aria-hidden size={14} />}
                  <span>Send</span>
                </button>
              </div>
              <p className="mt-2 hidden px-1 text-[11px] text-slate-500 sm:block">
                <kbd className="font-sans text-slate-400">Enter</kbd> to send · <kbd className="font-sans text-slate-400">Shift + Enter</kbd> for a new line
              </p>
            </div>
          ) : (
            <div key="end" className="iw-rise flex flex-col items-center gap-3 rounded-xl border border-white/[0.07] bg-slate-950/50 px-5 py-5 text-center">
              {p.policyClosed ? (
                <>
                  <div>
                    <p className="text-[14px] font-medium text-slate-100">This session has been closed</p>
                    <p className="mt-1 text-[12.5px] text-slate-400">No evaluation is available for this session.</p>
                  </div>
                  <button type="button" onClick={p.onReturnToSetup} className="sf-btn sf-btn-ghost iw-motion-safe">
                    <RefreshCw aria-hidden size={14} /> Return to setup
                  </button>
                </>
              ) : (
                <>
                  <div>
                    <p className="text-[14px] font-medium text-slate-100">The technical interview has concluded</p>
                    <p className="mt-1 text-[12.5px] text-slate-400">Your responses are ready to be evaluated.</p>
                  </div>
                  <button type="button" onClick={p.onViewEvaluation} className="sf-btn sf-btn-primary iw-motion-safe group">
                    View evaluation <ArrowRight aria-hidden size={15} className="transition-transform duration-200 group-hover:translate-x-0.5" />
                  </button>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
