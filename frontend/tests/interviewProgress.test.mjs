import test from 'node:test';
import assert from 'node:assert/strict';
import { interviewProgress, isPolicyClosure, workspaceStatus, STATUS_LABEL, MAX_QUESTIONS } from '../src/lib/interviewProgress.ts';

const ai = { sender: 'ai' };
const you = { sender: 'candidate' };

test('progress is derived from the chat, never invented', () => {
  assert.deepEqual(interviewProgress([], false), { asked: 0, answered: 0, current: 1, total: MAX_QUESTIONS, fraction: 0 });
  assert.equal(interviewProgress([ai], false).current, 1);
  const four = [ai, you, ai, you, ai, you, ai];
  const p = interviewProgress(four, false);
  assert.deepEqual([p.asked, p.answered, p.current], [4, 3, 4]);
  assert.equal(p.fraction, 0.3);
});

test('while the next question is being generated the current number does not jump ahead', () => {
  const p = interviewProgress([ai, you, ai, you], false);
  assert.equal(p.current, 2);
  assert.equal(p.answered, 2);
});

test('never exceeds the total, and a completed interview is full', () => {
  const many = Array.from({ length: 12 }, () => [ai, you]).flat();
  assert.equal(interviewProgress(many, false).current, MAX_QUESTIONS);
  assert.equal(interviewProgress(many, false).fraction, 1);
  assert.equal(interviewProgress([ai, you, ai], true).fraction, 1);
});

test('policy / domain / resignation closures are recognised, ordinary closings are not', () => {
  assert.ok(isPolicyClosure('We are unable to conduct a technical screening for this specific domain.'));
  assert.ok(isPolicyClosure('If you are unwilling to proceed with the technical questions, we will conclude the assessment here.'));
  assert.ok(isPolicyClosure('This session is closed.'));
  assert.ok(!isPolicyClosure('Thanks, that concludes the technical interview. Your responses will now be evaluated and made available for review.'));
  assert.ok(!isPolicyClosure('This interview is being concluded because the conversation has repeatedly moved away from professional participation.'));
  assert.ok(!isPolicyClosure('') && !isPolicyClosure(undefined) && !isPolicyClosure(null));
});

test('status model', () => {
  assert.equal(workspaceStatus({ messageCount: 0, isAiThinking: true, isComplete: false, policyClosed: false }), 'starting');
  assert.equal(workspaceStatus({ messageCount: 3, isAiThinking: true, isComplete: false, policyClosed: false }), 'generating');
  assert.equal(workspaceStatus({ messageCount: 3, isAiThinking: false, isComplete: false, policyClosed: false }), 'active');
  assert.equal(workspaceStatus({ messageCount: 9, isAiThinking: false, isComplete: true, policyClosed: false }), 'concluded');
  assert.equal(workspaceStatus({ messageCount: 1, isAiThinking: false, isComplete: true, policyClosed: true }), 'closed');
  assert.deepEqual(Object.keys(STATUS_LABEL).sort(), ['active', 'closed', 'concluded', 'generating', 'starting']);
});
