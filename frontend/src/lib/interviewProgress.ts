// Progress / status helpers for the interview workspace. Pure and framework-free (tested with `node --test`).
// Progress is derived ONLY from what the chat actually contains: interviewer messages = questions asked,
// candidate messages = answers given. The backend ends an interview after at most MAX_QUESTIONS answers
// (it may end earlier, in which case the workspace shows "Completed").

export const MAX_QUESTIONS = 10;

export interface ProgressInput { sender: 'ai' | 'candidate' }

export interface Progress {
  asked: number;      // interviewer messages shown so far
  answered: number;   // candidate answers given so far
  current: number;    // "Question N": the question being answered, or the one just answered while the next is generated
  total: number;
  fraction: number;   // 0..1 for the progress bar (answers given / total; 1 once the interview is complete)
}

export function interviewProgress(messages: ProgressInput[], isComplete: boolean, total: number = MAX_QUESTIONS): Progress {
  const asked = messages.filter((m) => m.sender === 'ai').length;
  const answered = messages.filter((m) => m.sender === 'candidate').length;
  const current = Math.min(Math.max(asked, 1), total);
  const fraction = isComplete ? 1 : Math.min(answered / total, 1);
  return { asked, answered, current, total, fraction };
}

// Backend messages that mean the session was closed by policy / domain / resignation (no evaluation to view).
const POLICY_CLOSURE_MARKERS = ['unable to conduct', 'session is closed', 'unwilling to proceed'];

export function isPolicyClosure(text: string | undefined | null): boolean {
  return !!text && POLICY_CLOSURE_MARKERS.some((m) => text.includes(m));
}

export type WorkspaceStatus = 'starting' | 'active' | 'generating' | 'concluded' | 'closed';

export function workspaceStatus(opts: { messageCount: number; isAiThinking: boolean; isComplete: boolean; policyClosed: boolean }): WorkspaceStatus {
  if (opts.isComplete) return opts.policyClosed ? 'closed' : 'concluded';
  if (opts.isAiThinking) return opts.messageCount === 0 ? 'starting' : 'generating';
  return 'active';
}

export const STATUS_LABEL: Record<WorkspaceStatus, string> = {
  starting: 'Starting',
  active: 'In progress',
  generating: 'In progress',
  concluded: 'Concluded',
  closed: 'Closed',
};
