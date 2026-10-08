"""Final-evaluation helpers: transcript pairing, the evaluator prompt, report validation/normalisation and cache rules.

Standard library only (plus the stdlib-only answer_signals / interview_style modules), so it can be tested without the
app's heavy imports. The provider chain (Gemini -> backup key -> local model) stays in main.generate_eval_response; this
module only receives a `generate_fn(prompt) -> str` and never talks to a provider itself.

Principles:
  * only real interview question -> candidate answer pairs are scored (never greetings, closings, rejections, resignations);
  * the question that is scored is the exact stored core question (`gen_meta.core_question`), not the display text that
    also carries the backend's greeting/reaction; legacy rows fall back to the display text;
  * the evaluator judges technical content; behaviour/professionalism is reported separately and never changes a score;
  * wording is evidence-based: broad or insulting claims are replaced by neutral, answer-grounded text;
  * overallScore is computed here from the per-question scores, and a result is only ever cached if it validated.
"""
import json
import re
from dataclasses import dataclass, field

from answer_signals import detect_behavior, strip_markers
from interview_style import CLOSING_MESSAGE, CONDUCT_TERMINATION_MESSAGE

# Bump when the report shape or scoring rules change: cached reports carrying another version are recomputed.
EVAL_SCHEMA_VERSION = 2

ANSWER_TYPES = ("strong", "partial", "incorrect", "vague", "irrelevant", "non_answer")
# Expected score range per answer type. A score further than BAND_MARGIN outside it contradicts its label: the evaluator is
# asked ONCE to correct itself; scores are never silently moved to fit a label.
SCORE_BANDS = {"strong": (75, 100), "partial": (35, 74), "incorrect": (0, 30), "vague": (0, 35),
               "irrelevant": (0, 15), "non_answer": (0, 0)}
BAND_MARGIN = 10
MAX_LIST_ITEMS = 4
MAX_PROMPT_ANSWER_CHARS = 800      # prompt only; the stored/reported answer is always complete
MAX_TEXT_CHARS = 400


class EvaluationUnavailableError(Exception):
    """No evaluation could be produced (provider outage or unusable output). Never cached."""


class EvaluationInconsistentError(EvaluationUnavailableError):
    """Structurally valid, but an answerType and its score materially contradict each other."""
    def __init__(self, problems):
        super().__init__("answerType/score conflict: " + "; ".join(problems))
        self.problems = problems


# --------------------------------------------------------------------------------------------------------------
# Transcript -> scored pairs
# --------------------------------------------------------------------------------------------------------------
_CANNED_MARKERS = ("unwilling to proceed", "session is closed", "unable to conduct", "wish you the best in your job search",
                   "already been concluded")
_TERMINATION_MARKERS = ("session is closed", "unwilling to proceed", "unable to conduct")


@dataclass
class QAPair:
    index: int                      # 1-based position among the scored questions
    question: str                   # exactly what is scored (core question, or display text for legacy rows)
    question_source: str            # core_question | display_text
    answer: str
    topic: str | None = None
    difficulty: str | None = None
    flow_quality: str | None = None         # interview-flow label of THIS answer (supporting trace only)
    behavior: str = "normal"                # normal | dismissive | unprofessional
    behavior_signals: list = field(default_factory=list)
    behavior_source: str = "detected"       # stored (from the interview) | detected (re-derived from the answer text)


@dataclass
class Transcript:
    pairs: list
    terminated: bool = False

    @property
    def rejected(self) -> bool:
        return not self.pairs


def _meta(raw) -> dict | None:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def is_canned_text(text: str) -> bool:
    t = text or ""
    return t.strip() in (CLOSING_MESSAGE.strip(), CONDUCT_TERMINATION_MESSAGE.strip()) or any(m in t for m in _CANNED_MARKERS)


def _is_question_row(text: str, provider, meta: dict | None) -> bool:
    """A real interviewer question: not backend text (provider 'system'), not a conclusion, not a canned message."""
    if not (text or "").strip() or provider == "system" or is_canned_text(text):
        return False
    return not (meta and meta.get("action") == "conclude")


def _scored_question(text: str, meta: dict | None) -> tuple:
    """(exact question string to score, source). Prefers gen_meta.core_question; legacy rows use the display text, minus
    the stored preamble when one was recorded."""
    core = (meta or {}).get("core_question")
    if isinstance(core, str) and core.strip():
        return core.strip(), "core_question"
    display = (text or "").strip()
    preamble = (meta or {}).get("preamble")
    if isinstance(preamble, str) and preamble.strip() and display.startswith(preamble.strip()):
        display = display[len(preamble.strip()):].strip() or display
    return display, "display_text"


def build_transcript(rows) -> Transcript:
    """rows: (sender, text, llm_provider, gen_meta) in message order.

    The reaction row that follows an answer carries that answer's flow labels (quality/behaviour), so they are attached to
    the pair just before it. Those labels are context only; the evaluator's own judgement decides the score."""
    pairs, pending, last_pair, last_ai_text = [], None, None, ""
    for sender, text, provider, raw_meta in rows:
        if sender == "ai":
            meta = _meta(raw_meta)
            last_ai_text = text or ""
            if last_pair is not None and meta and meta.get("answer_behavior"):
                last_pair.flow_quality = meta.get("answer_quality")
                last_pair.behavior = meta["answer_behavior"] if meta["answer_behavior"] in ("normal", "dismissive", "unprofessional") else "normal"
                last_pair.behavior_signals = list(meta.get("behavior_signals") or [])
                last_pair.behavior_source = "stored"
            elif last_pair is not None and meta and meta.get("severe_abuse"):     # conduct termination row: the answer before it was severe abuse
                last_pair.behavior, last_pair.behavior_signals, last_pair.behavior_source = "unprofessional", ["conduct:severe_abuse"], "stored"
            last_pair = None
            pending = None
            if _is_question_row(text, provider, meta):
                question, source = _scored_question(text, meta)
                pending = (question, source, meta or {})
        elif sender == "candidate":
            if pending is not None:
                question, source, meta = pending
                pair = QAPair(index=len(pairs) + 1, question=question, question_source=source, answer=(text or "").strip(),
                              topic=meta.get("topic") if isinstance(meta.get("topic"), str) else None,
                              difficulty=meta.get("difficulty") if isinstance(meta.get("difficulty"), str) else None)
                pairs.append(pair)
                last_pair, pending = pair, None
            elif last_pair is not None:                      # an extra message before the next question: keep the evidence
                last_pair.answer = (last_pair.answer + " " + (text or "").strip()).strip()
    for p in pairs:
        if p.behavior_source != "stored":                     # last answer has no reaction row: derive from the text itself
            b = detect_behavior(p.answer)
            p.behavior, p.behavior_signals, p.behavior_source = b.label, b.signals, "detected"
    terminated = any(m in last_ai_text for m in _TERMINATION_MARKERS)
    return Transcript(pairs=pairs, terminated=terminated)


# --------------------------------------------------------------------------------------------------------------
# Evaluator prompt
# --------------------------------------------------------------------------------------------------------------
_PROMPT_HEAD = """Evaluate the technical answers of a completed interview. Judge ONLY what the candidate wrote.
Rules:
1. Output exactly {n} items in "breakdown", in order, each with its "index".
2. Score the TECHNICAL content of each answer 0-100. Tone, slang, typos and casual wording are never evidence of missing knowledge and never lower a score.
3. "answerType" with the matching score range: strong 75-100 (accurate, specific); partial 35-74 (incomplete or imprecise); incorrect 0-30 (an attempt that is technically wrong, NOT a refusal); vague 0-35 (on topic but too general or circular); irrelevant 0-15 (does not address the question); non_answer 0 (no attempt: "I don't know", refusal, gibberish).
4. Neutral wording based only on the answers shown; no insults and no broad claims about the candidate's wider ability (e.g. "complete lack of knowledge", "unprofessional throughout"). Say e.g. "did not demonstrate understanding of the concepts assessed".
5. "feedback": one sentence about this answer. "summary": two neutral sentences. "strengths"/"weaknesses": at most {max_items} short items each, [] if none. No overall score.
JSON only: {{"summary":"","strengths":[],"weaknesses":[],"breakdown":[{{"index":1,"answerType":"partial","score":50,"feedback":""}}]}}
Interview (question = exactly what was asked; topic/difficulty are context):
"""


def build_prompt(pairs) -> str:
    return _PROMPT_HEAD.format(n=len(pairs), max_items=MAX_LIST_ITEMS) + _items_json(pairs)


def _items_json(pairs) -> str:
    items = [{"index": p.index, "question": p.question, "answer": p.answer[:MAX_PROMPT_ANSWER_CHARS], "topic": p.topic,
              "difficulty": p.difficulty} for p in pairs]
    return json.dumps(items, ensure_ascii=False, separators=(",", ":"))


def build_repair_prompt(pairs, previous: dict, problems) -> str:
    """The single correction request: same interview, the evaluator's own earlier JSON and exactly what contradicts."""
    return (_PROMPT_HEAD.format(n=len(pairs), max_items=MAX_LIST_ITEMS) + _items_json(pairs)
            + "\nYour previous result had inconsistencies:\n- " + "\n- ".join(problems)
            + "\nRe-judge those answers and return the corrected COMPLETE JSON in the same structure. "
              "Change the answerType or the score, whichever was wrong.\nPrevious result:\n"
            + json.dumps(previous, ensure_ascii=False, separators=(",", ":")))


# --------------------------------------------------------------------------------------------------------------
# Wording guard: broad / insulting / hiring-style claims are replaced by neutral, evidence-based text
# --------------------------------------------------------------------------------------------------------------
_UNSUPPORTED = re.compile("|".join([
    r"\b(?:complete|total|utter|absolute|entire|zero)\s+(?:lack|absence)\b",
    r"\b(?:no|zero|without any|lacks? (?:all|any))\s+(?:technical\s+)?(?:knowledge|ability|skills?|competen\w+)\b",
    r"\b(?:highly|extremely|very|completely|totally|utterly)\s+unprofessional\b",
    r"\bunprofessional\s+(?:throughout|in every)\b",
    r"\bthroughout\s+(?:the\s+)?(?:entire\s+)?(?:interview|session|assessment)\b",
    r"\b(?:incompeten\w*|clueless|hopeless|pathetic|useless|stupid|idiot\w*|lazy|terrible|awful|dreadful|disgrace\w*|ruthless\w*)\b",
    r"\b(?:total|complete|utter) (?:failure|disaster)\b",
    r"\b(?:completely|totally|utterly) (?:unprepared|ignorant|wrong)\b",
    r"\bshould not be hired\b|\bdo not hire\b|\bnot (?:suitable|fit|recommended) for\b|\bunqualified\b|\bunfit\b",
]), re.I)

_FEEDBACK_BY_TYPE = {
    "strong": "The answer addressed the question accurately and with specific detail.",
    "partial": "The answer covered part of the concept but was incomplete.",
    "incorrect": "The answer contained technical inaccuracies about the concept assessed.",
    "vague": "The answer was too general to show understanding of the concept assessed.",
    "irrelevant": "The answer did not address the question that was asked.",
    "non_answer": "No attempt was made at this technical question.",
    "unspecified": "The answer was assessed on its technical content only.",
}
_WEAKNESS_BY_TYPE = {
    "partial": "{t}: the answer was incomplete.",
    "incorrect": "{t}: the answer contained technical inaccuracies.",
    "vague": "{t}: the answer was too general to show understanding.",
    "irrelevant": "{t}: the answer did not address the question.",
    "non_answer": "{t}: no attempt was made.",
}


def _clean_text(value, limit=MAX_TEXT_CHARS) -> str | None:
    """A usable model string, or None (missing, wrong type, empty, or containing an unsupported claim)."""
    if not isinstance(value, str):
        return None
    text = re.sub(r"\s+", " ", value).strip()
    if not text or _UNSUPPORTED.search(text):
        return None
    return text[:limit].rstrip()


def _clean_list(value) -> list:
    out = []
    for item in value if isinstance(value, list) else []:
        text = _clean_text(item, 200)
        if text:
            out.append(text.rstrip("."))
        if len(out) >= MAX_LIST_ITEMS:
            break
    return out


# --------------------------------------------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------------------------------------------
def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _label(pair) -> str:
    return pair.topic or f"Question {pair.index}"


def _order_items(breakdown, n):
    """Items in question order (by their own index when it is a clean 1..n permutation, otherwise as delivered)."""
    idx = [b.get("index") for b in breakdown]
    if all(isinstance(i, int) and not isinstance(i, bool) for i in idx) and sorted(idx) == list(range(1, n + 1)):
        return [b for _, b in sorted(zip(idx, breakdown), key=lambda x: x[0])]
    return breakdown


def behavior_report(pairs) -> dict:
    flagged = [p for p in pairs if p.behavior in ("dismissive", "unprofessional")]
    d = sum(1 for p in flagged if p.behavior == "dismissive")
    u = len(flagged) - d
    if not flagged:
        note = "No dismissive or unprofessional language was detected in the answers."
    else:
        qs = ", ".join(str(p.index) for p in flagged)
        note = (f"{len(flagged)} of {len(pairs)} responses (questions {qs}) included dismissive or unprofessional language "
                f"({d} dismissive, {u} unprofessional). This is reported separately and is not part of the technical assessment.")
    return {"note": note, "dismissive": d, "unprofessional": u, "questions": [p.index for p in flagged]}


def topic_performance(pairs, scores) -> list:
    groups = {}
    for p, s in zip(pairs, scores):
        if p.topic:
            groups.setdefault(p.topic.strip(), []).append(s)
    return [{"topic": t, "questions": len(v), "averageScore": round(sum(v) / len(v))} for t, v in groups.items()]


def _deterministic_summary(types, n) -> str:
    counts = [(t, types.count(t)) for t in ANSWER_TYPES if types.count(t)]
    names = {"strong": "strong", "partial": "partial", "incorrect": "technically inaccurate", "vague": "vague",
             "irrelevant": "off-topic", "non_answer": "non-answers"}
    breakdown = ", ".join(f"{c} {names[t]}" for t, c in counts)
    first = f"Of {n} scored answers: {breakdown}." if breakdown else f"{n} answers were scored."
    weak = sum(types.count(t) for t in ("incorrect", "vague", "irrelevant", "non_answer"))
    if weak * 2 > n:
        second = "The candidate did not demonstrate understanding of most of the concepts assessed in this interview."
    elif types.count("strong") * 2 >= n:
        second = "Most answers demonstrated sound understanding of the concepts assessed."
    else:
        second = "Understanding of the concepts assessed was demonstrated in part."
    return f"{first} {second} This reflects only the answers given in this interview."


_NON_ANSWER = re.compile(r"\b(?:don'?t know|dunno|no idea|idk|not sure|no clue|pass|skip)\b", re.I)


def obvious_non_answer(answer: str) -> bool:
    """Clearly no attempt: empty, a couple of words of behaviour chatter, or a short 'I don't know'."""
    words = strip_markers(answer or "").split()
    return len(words) <= 2 or (len(words) <= 6 and bool(_NON_ANSWER.search(answer or "")))


def _structured_items(pairs, model_out) -> list:
    breakdown = model_out.get("breakdown") if isinstance(model_out, dict) else None
    if not (isinstance(breakdown, list) and len(breakdown) == len(pairs)
            and all(isinstance(b, dict) and _number(b.get("score")) for b in breakdown)):
        raise EvaluationUnavailableError("Evaluation JSON does not match the expected structure.")
    return _order_items(breakdown, len(pairs))


def _score_of(b) -> int:
    return max(0, min(100, int(round(b["score"]))))     # keeps the number a valid 0-100 score; never fits it to a label


def consistency_problems(pairs, items) -> list:
    """answerType vs score conflicts that are material (more than BAND_MARGIN outside the type's range). An obvious
    non-answer is handled deterministically (score 0) and is not a conflict."""
    problems = []
    for p, b in zip(pairs, items):
        atype, score = b.get("answerType"), _score_of(b)
        if atype not in SCORE_BANDS or (atype == "non_answer" and obvious_non_answer(p.answer)):
            continue
        lo, hi = SCORE_BANDS[atype]
        if score < lo - BAND_MARGIN or score > hi + BAND_MARGIN:
            problems.append(f'item {p.index}: answerType "{atype}" expects a score of {lo}-{hi} but the score is {score}')
    return problems


def assemble_report(transcript: Transcript, model_out) -> dict:
    """Validated, normalised report from the evaluator's parsed JSON. Raises EvaluationUnavailableError when it is
    unusable and EvaluationInconsistentError when a label and its score contradict (the caller may repair once)."""
    pairs = transcript.pairs
    items = _structured_items(pairs, model_out)
    problems = consistency_problems(pairs, items)
    if problems:
        raise EvaluationInconsistentError(problems)

    rows, scores, types = [], [], []
    for p, b in zip(pairs, items):
        score = _score_of(b)
        atype = b.get("answerType") if b.get("answerType") in ANSWER_TYPES else "unspecified"
        adjusted = None
        if atype == "non_answer" and score != 0 and obvious_non_answer(p.answer):
            adjusted, score = score, 0                   # no attempt was made: the score is 0 by definition
        feedback = _clean_text(b.get("feedback"))
        sanitized = feedback is None
        row = {"question": p.question, "answer": p.answer, "score": score,
               "feedback": feedback or _FEEDBACK_BY_TYPE[atype], "answerType": atype,
               "topic": p.topic, "difficulty": p.difficulty, "questionSource": p.question_source,
               "flowQuality": p.flow_quality, "behavior": p.behavior}
        if adjusted is not None:
            row["modelScore"] = adjusted
        if sanitized:
            row["feedbackReplaced"] = True
        rows.append(row)
        scores.append(score)
        types.append(atype)

    n = len(pairs)
    topics = topic_performance(pairs, scores)
    strengths = _clean_list(model_out.get("strengths"))
    weaknesses = _clean_list(model_out.get("weaknesses"))
    if not strengths:
        strengths = [f"{_label(p)}: the answer was accurate and specific" for p, t in zip(pairs, types) if t == "strong"][:MAX_LIST_ITEMS]
    if not weaknesses:
        weaknesses = [_WEAKNESS_BY_TYPE[t].format(t=_label(p)).rstrip(".") for p, t in zip(pairs, types)
                      if t in _WEAKNESS_BY_TYPE][:MAX_LIST_ITEMS]
    summary = _clean_text(model_out.get("summary"), 600) or _deterministic_summary(types, n)
    behavior = behavior_report(pairs)
    return {
        "schemaVersion": EVAL_SCHEMA_VERSION,
        "overallScore": round(sum(scores) / n),
        "summary": summary,
        "insights": f"Strengths: {'; '.join(strengths) or 'none identified'}. Weaknesses: {'; '.join(weaknesses) or 'none identified'}.",
        "strengths": strengths,
        "weaknesses": weaknesses,
        "topicPerformance": topics,
        "answerTypeCounts": {t: types.count(t) for t in ANSWER_TYPES + ("unspecified",) if types.count(t)},
        "behaviorNote": behavior["note"],
        "behavior": behavior,
        "breakdown": rows,
    }


def terminated_report(transcript: Transcript) -> dict:
    """Deterministic report for domain rejections and resignations: no model call, scores voided as before."""
    return {
        "schemaVersion": EVAL_SCHEMA_VERSION,
        "overallScore": 0,
        "summary": "Interview terminated early due to domain rejection, resignation, or policy violation.",
        "insights": "The interview ended before a technical assessment could be completed, so no scores were assigned.",
        "strengths": [], "weaknesses": [], "topicPerformance": [], "answerTypeCounts": {},
        "behaviorNote": behavior_report(transcript.pairs)["note"] if transcript.pairs else "",
        "behavior": behavior_report(transcript.pairs) if transcript.pairs else {"note": "", "dismissive": 0, "unprofessional": 0, "questions": []},
        "breakdown": [{"question": p.question, "answer": p.answer, "score": 0, "feedback": "Score voided due to early termination.",
                       "answerType": "unspecified", "topic": p.topic, "difficulty": p.difficulty,
                       "questionSource": p.question_source, "flowQuality": p.flow_quality, "behavior": p.behavior}
                      for p in transcript.pairs],
    }


def _ask(generate_fn, prompt: str) -> dict:
    """One evaluator call -> parsed JSON object. Every failure is EvaluationUnavailableError."""
    try:
        text = generate_fn(prompt)
    except EvaluationUnavailableError:
        raise
    except Exception as e:
        raise EvaluationUnavailableError(f"provider error: {e}") from e
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        raise EvaluationUnavailableError("No JSON boundaries found.")
    try:
        parsed = json.loads(match.group(0))
    except ValueError as e:
        raise EvaluationUnavailableError(f"Evaluation output is not valid JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise EvaluationUnavailableError("Evaluation output is not a JSON object.")
    return parsed


def evaluate_transcript(transcript: Transcript, generate_fn) -> dict:
    """Run the evaluator and return a validated report. If an answerType and its score materially conflict, ONE repair
    request asks the evaluator for a corrected result; a second invalid result is not patched up. Every failure raises
    EvaluationUnavailableError, so the caller refuses to cache and answers 503."""
    parsed = _ask(generate_fn, build_prompt(transcript.pairs))
    try:
        return assemble_report(transcript, parsed)
    except EvaluationInconsistentError as first:
        repaired = _ask(generate_fn, build_repair_prompt(transcript.pairs, parsed, first.problems))
        return assemble_report(transcript, repaired)


def cached_report(raw) -> dict | None:
    """The stored evaluation if it is a valid report of the CURRENT schema version, else None (recompute).
    Reports written before versioning, by another version, or by an old failed parse are all recomputed."""
    try:
        data = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    except ValueError:
        return None
    if not (isinstance(data, dict) and data.get("schemaVersion") == EVAL_SCHEMA_VERSION
            and _number(data.get("overallScore")) and isinstance(data.get("summary"), str) and data["summary"].strip()
            and data["summary"] != "Evaluation failed to parse." and isinstance(data.get("breakdown"), list)):
        return None
    return data
