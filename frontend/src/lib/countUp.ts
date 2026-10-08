// Count-up maths for the score reveal. Pure; the animation loop lives in components/useCountUp.ts.

export function easeOutCubic(t: number): number {
  const x = Math.max(0, Math.min(1, t));
  return 1 - Math.pow(1 - x, 3);
}

/** The integer shown `elapsedMs` into a count-up from 0 to `target` that lasts `durationMs`. Always ends exactly on `target`. */
export function countValue(target: number, elapsedMs: number, durationMs: number): number {
  if (!Number.isFinite(target)) return 0;
  if (durationMs <= 0 || elapsedMs >= durationMs) return Math.round(target);
  return Math.round(target * easeOutCubic(elapsedMs / durationMs));
}
