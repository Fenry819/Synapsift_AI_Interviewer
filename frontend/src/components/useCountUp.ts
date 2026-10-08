"use client";

import { useEffect, useState } from 'react';
import { countValue } from '../lib/countUp';

/** Animates a number from 0 to `target` (ease-out). With reduced motion it jumps straight to the final value. Returns 0 while disabled. */
export function useCountUp(target: number, enabled: boolean, durationMs = 1200): number {
  const [value, setValue] = useState(0);

  useEffect(() => {
    if (!enabled) return;
    const reduced = typeof window !== 'undefined' && !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    let raf = 0;
    const start = performance.now();
    const tick = (now: number) => {
      const next = countValue(target, reduced ? durationMs : now - start, durationMs);
      setValue(next);
      if (next !== Math.round(target)) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, enabled, durationMs]);

  return enabled ? value : 0;
}
