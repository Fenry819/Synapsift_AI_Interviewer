import test from 'node:test';
import assert from 'node:assert/strict';
import { clampScore, scoreTone, answerTypeLabel, ringGeometry, formatFileSize, isPdfFile, isValidEmail, initials, ANSWER_TYPE_ORDER, sentenceCase, isTextFile, isResumeFile, RESUME_ACCEPT } from '../src/lib/report.ts';
import { easeOutCubic, countValue } from '../src/lib/countUp.ts';

test('scores are clamped and rounded; garbage becomes 0', () => {
  assert.deepEqual([clampScore(88.4), clampScore(-5), clampScore(140), clampScore('x'), clampScore(NaN), clampScore(undefined)], [88, 0, 100, 0, 0, 0]);
});

test('score tones keep the previous thresholds (>75 high, >50 mid)', () => {
  assert.deepEqual([scoreTone(100), scoreTone(76), scoreTone(75), scoreTone(51), scoreTone(50), scoreTone(0)], ['high', 'high', 'mid', 'mid', 'low', 'low']);
});

test('answer types map to plain-language labels, unknown types are neutral', () => {
  assert.equal(answerTypeLabel('non_answer'), 'No attempt');
  assert.equal(answerTypeLabel('incorrect'), 'Inaccurate');
  assert.equal(answerTypeLabel('something_new'), 'Assessed');
  assert.equal(answerTypeLabel(undefined), 'Assessed');
  assert.ok(ANSWER_TYPE_ORDER.includes('unspecified'));
});

test('ring geometry fills the right share of the circle', () => {
  const full = ringGeometry(100), none = ringGeometry(0), half = ringGeometry(50);
  assert.ok(Math.abs(full.offset) < 1e-9);
  assert.ok(Math.abs(none.offset - none.circumference) < 1e-9);
  assert.ok(Math.abs(half.offset - half.circumference / 2) < 1e-9);
  assert.ok(Math.abs(ringGeometry(250).offset) < 1e-9, 'scores above 100 are clamped');
});

test('file helpers', () => {
  assert.equal(formatFileSize(512), '512 B');
  assert.equal(formatFileSize(2048), '2.0 KB');
  assert.equal(formatFileSize(300 * 1024), '300 KB');
  assert.equal(formatFileSize(3.5 * 1024 * 1024), '3.5 MB');
  assert.equal(formatFileSize(-1), '');
  assert.ok(isPdfFile({ name: 'cv.PDF', type: '' }) && isPdfFile({ name: 'x', type: 'application/pdf' }));
  assert.ok(!isPdfFile({ name: 'cv.docx', type: 'application/msword' }));
});

test('auth helpers', () => {
  assert.ok(isValidEmail('a@b.co') && isValidEmail(' gautham@example.com '));
  assert.ok(!isValidEmail('a@b') && !isValidEmail('nope') && !isValidEmail('a b@c.com'));
  assert.deepEqual([initials('Gautham Nair'), initials('  madonna '), initials(''), initials('Jean Luc Picard')], ['GN', 'M', '?', 'JP']);
});

test('count-up: eased, monotonic, ends exactly on the target', () => {
  assert.equal(countValue(88, 0, 1000), 0);
  assert.equal(countValue(88, 1000, 1000), 88);
  assert.equal(countValue(88, 5000, 1000), 88);
  assert.equal(countValue(88, 100, 0), 88, 'a zero duration jumps to the target (reduced motion)');
  let prev = -1;
  for (let t = 0; t <= 1000; t += 50) { const v = countValue(73, t, 1000); assert.ok(v >= prev); prev = v; }
  assert.ok(countValue(100, 250, 1000) > 25, 'ease-out starts fast');
  assert.equal(easeOutCubic(-1), 0);
  assert.equal(easeOutCubic(2), 1);
  assert.equal(countValue(NaN, 10, 100), 0);
});

test('sentence case touches only the first letter', () => {
  assert.equal(sentenceCase('decision trees and ensemble methods'), 'Decision trees and ensemble methods');
  assert.equal(sentenceCase('  ML basics'), 'ML basics');
  assert.equal(sentenceCase(''), '');
  assert.equal(sentenceCase(null), '');
});

test('resume files: PDF or plain text only (no DOCX)', () => {
  assert.ok(isResumeFile({ name: 'cv.pdf', type: 'application/pdf' }) && isResumeFile({ name: 'cv.PDF', type: '' }));
  assert.ok(isResumeFile({ name: 'cv.txt', type: 'text/plain' }) && isResumeFile({ name: 'My CV.TXT', type: '' }) && isTextFile({ name: 'x', type: 'text/plain' }));
  assert.ok(!isResumeFile({ name: 'cv.docx', type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' }));
  assert.ok(!isResumeFile({ name: 'cv.doc', type: 'application/msword' }) && !isResumeFile({ name: 'photo.png', type: 'image/png' }));
  assert.ok(RESUME_ACCEPT.includes('.txt') && RESUME_ACCEPT.includes('.pdf') && !/docx?/.test(RESUME_ACCEPT));
});
