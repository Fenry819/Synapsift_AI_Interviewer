// Presentation helpers shared by the setup and result screens. Pure and framework-free (tested with `node --test`).

export type ScoreTone = 'high' | 'mid' | 'low';

export function clampScore(n: unknown): number {
  const v = typeof n === 'number' && Number.isFinite(n) ? n : 0;
  return Math.max(0, Math.min(100, Math.round(v)));
}

/** Colour band for a 0-100 score (the same thresholds the old result page used: >75 green, >50 amber, else red). */
export function scoreTone(score: number): ScoreTone {
  return score > 75 ? 'high' : score > 50 ? 'mid' : 'low';
}

// Plain-language names for the evaluator's answer types (backend evaluation schema v2).
const ANSWER_TYPE_LABEL: Record<string, string> = {
  strong: 'Strong',
  partial: 'Partial',
  incorrect: 'Inaccurate',
  vague: 'Vague',
  irrelevant: 'Off-topic',
  non_answer: 'No attempt',
  unspecified: 'Assessed',
};

export function answerTypeLabel(type: string | undefined | null): string {
  return (type && ANSWER_TYPE_LABEL[type]) || 'Assessed';
}

export const ANSWER_TYPE_ORDER = ['strong', 'partial', 'incorrect', 'vague', 'irrelevant', 'non_answer', 'unspecified'];

/** SVG ring geometry for a score: stroke-dasharray length and the dashoffset that leaves `score`% of the ring filled. */
export function ringGeometry(score: number, radius = 54): { circumference: number; offset: number } {
  const circumference = 2 * Math.PI * radius;
  return { circumference, offset: circumference * (1 - clampScore(score) / 100) };
}

export function formatFileSize(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return '';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(bytes < 10 * 1024 ? 1 : 0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function isPdfFile(file: { name: string; type?: string }): boolean {
  return file.type === 'application/pdf' || /\.pdf$/i.test(file.name);
}

export function isTextFile(file: { name: string; type?: string }): boolean {
  return /\.txt$/i.test(file.name) || file.type === 'text/plain';
}

/** Resumes may be a PDF or a plain-text (.txt) file. DOCX and everything else is refused. */
export function isResumeFile(file: { name: string; type?: string }): boolean {
  return isPdfFile(file) || isTextFile(file);
}

export const RESUME_ACCEPT = '.pdf,.txt,application/pdf,text/plain';

export function isValidEmail(value: string): boolean {
  return /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test(value.trim());
}

export function initials(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return '?';
  return (parts[0][0] + (parts.length > 1 ? parts[parts.length - 1][0] : '')).toUpperCase();
}

/** "decision trees and ensemble methods" -> "Decision trees and ensemble methods" (first letter only). */
export function sentenceCase(text: string | undefined | null): string {
  const t = (text ?? '').trim();
  return t ? t[0].toUpperCase() + t.slice(1) : '';
}
