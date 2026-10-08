// Progressive reveal of an interviewer message that the backend has ALREADY generated and validated.
// This is presentation only: the revealed text always ends up exactly equal to the approved text.
// Framework-free so it can be tested with `node --test` (see frontend/tests).

export const REVEAL_INTERVAL_MS = 30;      // delay between chunks (adjust here)
export const REVEAL_WORDS_PER_CHUNK = 2;   // words added per chunk

export interface RevealOptions {
  intervalMs?: number;
  wordsPerChunk?: number;
  instant?: boolean;                       // e.g. prefers-reduced-motion: show everything at once
  setTimer?: (fn: () => void, ms: number) => unknown;
  clearTimer?: (handle: unknown) => void;
}

/** Cumulative prefixes of `text`, one per chunk of words. The last entry is `text` itself, character for character. */
export function revealSteps(text: string, wordsPerChunk: number = REVEAL_WORDS_PER_CHUNK): string[] {
  const words = text.match(/\s*\S+/g);       // each word carries the whitespace before it, so no prefix ends in a stray space
  if (!words) return [text];
  const size = Math.max(1, Math.floor(wordsPerChunk));
  const steps: string[] = [];
  let shown = "";
  for (let i = 0; i < words.length; i += size) {
    shown += words.slice(i, i + size).join("");
    steps.push(shown);
  }
  steps[steps.length - 1] = text;            // trailing whitespace (if any) is part of the exact final text
  return steps;
}

/**
 * Reveals `text` chunk by chunk. `onStep(shown)` runs immediately with the first chunk (so the message bubble appears
 * at once), then once per interval; `onDone()` runs after the full text was delivered. Returns `cancel()`: after it is
 * called no further callback runs (no stale state updates after unmount, abort, navigation or a new session).
 */
export function startReveal(text: string, onStep: (shown: string) => void, onDone: () => void, opts: RevealOptions = {}): () => void {
  const setTimer = opts.setTimer ?? ((fn: () => void, ms: number) => setTimeout(fn, ms));
  const clearTimer = opts.clearTimer ?? ((h: unknown) => clearTimeout(h as ReturnType<typeof setTimeout>));
  const steps = opts.instant ? [text] : revealSteps(text, opts.wordsPerChunk);
  const interval = opts.intervalMs ?? REVEAL_INTERVAL_MS;
  let index = 0;
  let handle: unknown = null;
  let cancelled = false;

  const tick = () => {
    if (cancelled) return;
    onStep(steps[index]);
    index += 1;
    if (index >= steps.length) {
      handle = null;
      onDone();
    } else {
      handle = setTimer(tick, interval);
    }
  };
  tick();
  return () => {
    cancelled = true;
    if (handle !== null) clearTimer(handle);
    handle = null;
  };
}
