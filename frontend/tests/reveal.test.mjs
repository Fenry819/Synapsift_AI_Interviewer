// Run from frontend/:  npm test   (Node's built-in test runner; Node 22.6+ strips the TypeScript types itself).
// Covers the reveal engine the interview chat uses. The React page wiring (disabled input, double-submit guard) is
// listed in the manual checklist in the Phase 5.1 report because the project has no DOM test framework.
import test from 'node:test';
import assert from 'node:assert/strict';
import { revealSteps, startReveal, REVEAL_INTERVAL_MS, REVEAL_WORDS_PER_CHUNK } from '../src/lib/reveal.ts';

// Deterministic timers: nothing fires until the test advances them.
function fakeTimers() {
  const queue = [];
  let nextId = 1;
  return {
    setTimer: (fn, ms) => { const h = nextId++; queue.push({ h, fn, ms }); return h; },
    clearTimer: (h) => { const i = queue.findIndex((t) => t.h === h); if (i >= 0) queue.splice(i, 1); },
    pending: () => queue.length,
    delays: () => queue.map((t) => t.ms),
    step: () => { const t = queue.shift(); if (t) t.fn(); return !!t; },
    drain: () => { let n = 0; while (queue.length) { queue.shift().fn(); n++; } return n; },
  };
}

const GREETING = "Hi Gautham, welcome to the interview. I'll ask you a few technical questions based on your background and the role you're applying for, and we'll adjust the depth as we go. To begin, can you explain what overfitting is?";
const AWKWARD = "  Leading spaces,\ttabs and\n\nblank lines — “quotes”, ünïcode, emoji 🙂 and trailing spaces.  \n";

test('defaults are word-chunked at a natural pace', () => {
  assert.ok(REVEAL_INTERVAL_MS >= 20 && REVEAL_INTERVAL_MS <= 40);
  assert.ok(REVEAL_WORDS_PER_CHUNK >= 1 && REVEAL_WORDS_PER_CHUNK <= 4);
});

test('revealSteps: progressive prefixes, last step is exactly the approved text', () => {
  for (const text of [GREETING, AWKWARD, 'One.', 'Two words', ' ', '', 'a\n\nb']) {
    const steps = revealSteps(text);
    assert.equal(steps[steps.length - 1], text, JSON.stringify(text));
    steps.forEach((s, i) => assert.ok(text.startsWith(s), `step ${i} is a prefix`));
    steps.forEach((s, i) => i > 0 && assert.ok(s.length > steps[i - 1].length, 'strictly growing'));
  }
  const greetingSteps = revealSteps(GREETING);
  assert.ok(greetingSteps.length > 5, 'a real message is revealed over many chunks');
  assert.equal(revealSteps('one two three four five', 2).length, 3);
});

test('first question reveals progressively: first chunk immediately, the rest on timers, then done once', () => {
  const t = fakeTimers();
  const shown = [];
  let done = 0;
  startReveal(GREETING, (s) => shown.push(s), () => { done++; }, { setTimer: t.setTimer, clearTimer: t.clearTimer });
  assert.equal(shown.length, 1, 'bubble content appears immediately');
  assert.ok(shown[0].length < GREETING.length, 'but not the whole message');
  assert.equal(done, 0);
  assert.deepEqual(t.delays(), [REVEAL_INTERVAL_MS]);
  t.drain();
  assert.equal(done, 1);
  assert.equal(shown[shown.length - 1], GREETING, 'exactly the backend-approved text');
  assert.equal(t.pending(), 0);
});

test('a later reply reveals the same way, whitespace and punctuation untouched', () => {
  const t = fakeTimers();
  const shown = [];
  startReveal(AWKWARD, (s) => shown.push(s), () => {}, { setTimer: t.setTimer, clearTimer: t.clearTimer });
  t.drain();
  assert.ok(shown.length > 3);
  assert.equal(shown[shown.length - 1], AWKWARD);
});

test('cancel (unmount / abort / navigation / new session) stops all further updates', () => {
  const t = fakeTimers();
  const shown = [];
  let done = 0;
  const cancel = startReveal(GREETING, (s) => shown.push(s), () => { done++; }, { setTimer: t.setTimer, clearTimer: t.clearTimer });
  t.step(); t.step();
  const seen = shown.length;
  cancel();
  assert.equal(t.pending(), 0, 'the pending timer was cleared');
  assert.equal(t.drain(), 0);
  assert.equal(shown.length, seen, 'no step after cancel');
  assert.equal(done, 0, 'onDone never runs after cancel');
  cancel(); // idempotent
});

test('cancel inside a callback is safe and onDone is not called afterwards', () => {
  const t = fakeTimers();
  let cancel; let done = 0; let steps = 0;
  cancel = startReveal('a b c d e f g h', () => { steps++; if (steps === 2) cancel(); }, () => { done++; }, { setTimer: t.setTimer, clearTimer: t.clearTimer });
  t.drain();
  assert.equal(steps, 2);
  assert.equal(done, 0);
});

test('instant mode (reduced motion) delivers everything at once', () => {
  const t = fakeTimers();
  const shown = []; let done = 0;
  startReveal(GREETING, (s) => shown.push(s), () => { done++; }, { instant: true, setTimer: t.setTimer, clearTimer: t.clearTimer });
  assert.deepEqual(shown, [GREETING]);
  assert.equal(done, 1);
  assert.equal(t.pending(), 0);
});

test('empty or whitespace-only text completes without hanging', () => {
  for (const text of ['', '   ']) {
    const t = fakeTimers(); const shown = []; let done = 0;
    startReveal(text, (s) => shown.push(s), () => { done++; }, { setTimer: t.setTimer, clearTimer: t.clearTimer });
    t.drain();
    assert.equal(done, 1);
    assert.equal(shown[shown.length - 1], text);
  }
});

test('interval and chunk size are adjustable', () => {
  const t = fakeTimers();
  startReveal('a b c d e f', () => {}, () => {}, { intervalMs: 55, wordsPerChunk: 3, setTimer: t.setTimer, clearTimer: t.clearTimer });
  assert.deepEqual(t.delays(), [55]);
});

test('works with the real timers too (default setTimeout)', async () => {
  const shown = [];
  await new Promise((resolve) => startReveal('one two three four five six', (s) => shown.push(s), resolve, { intervalMs: 1 }));
  assert.equal(shown[shown.length - 1], 'one two three four five six');
  assert.ok(shown.length >= 3);
});
