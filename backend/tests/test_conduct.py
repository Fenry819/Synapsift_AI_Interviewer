"""Conduct strikes: terminate only after repeated SEVERE abuse aimed at the interviewer; everything else continues.
Separate from technical quality. Embeddings are REAL, providers are scripted, the profanity check is the app's own (better_profanity)."""
import ast
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import BACKEND, check, finish, load_embedder  # noqa: E402

from better_profanity import profanity  # noqa: E402
import interview_config as cfg  # noqa: E402
import interview_engine as ie  # noqa: E402
import interview_style as style  # noqa: E402
import answer_signals as sig  # noqa: E402
import evaluation as ev  # noqa: E402
from llm_providers import LLMResult  # noqa: E402
from rag import RetrievedChunk  # noqa: E402
from resume_profile import build_candidate_profile  # noqa: E402

embed = load_embedder()
PROFILE = build_candidate_profile("SKILLS\nPython, PyTorch, scikit-learn, Machine Learning, NLP\nEXPERIENCE\nML Intern - Acme\n- a\n- b\n")
Q = "What is overfitting?"
T = "overfitting"
severe = lambda t: sig.is_severe_abuse(t, profanity.contains_profanity)
OUTCOME_WORDS = re.compile(r"\bscor\w*|\bpass(?:ed)?\b|\bfail\w*|\bhir\w+|\breject\w*|\bresult\w*|\bgrade\b|\bselected\b|\boffer\b|\bjob\b", re.I)


def turn_state(state, answer, *, profane=None, number=3, severe_flag=None):
    """Classify one answer exactly like main.py does, then plan. -> (state, plan, quality)"""
    p = profanity.contains_profanity(answer) if profane is None else profane
    sv = severe(answer) if severe_flag is None else severe_flag
    q = ie.classify_answer(answer, Q, T, embed, profane=p, severe_abuse=sv)
    st, plan = ie.plan_next_step(state, q, "ai_ml", answer_number=number)
    return st, plan, q


def fresh(**kw): return ie.InterviewState(difficulty="beginner", current_topic=T, asked_topics=[T], deep_probed_topics=[T], **kw)
def advance(st, plan): return ie.record_question(st, plan, T, None, preamble=None)


print("== severity: only profanity AIMED at the interviewer ==")
SEVERE = ["Get the fuck out", "Fuck you man", "just fuck off", "fuck you", "you are a fucking idiot", "shut the fuck up", "you're such a bitch", "go fuck yourself"]
NOT_SEVERE = ["damn", "this is fucking hard", "when you fuck up the train test split the model overfits", "what the hell is this", "bullshit question lol",
              "wow damn bro this is easy", "overfitting is a fitting in toilet", "please give me the job my mom is sick i need money", "whatever", "i already told you in simple terms",
              "I don't know", "ask something else", "lol", "I quit", "the shit hits the fan when variance is high", "you can use dropout", "fuck, I forgot the answer"]
for a in SEVERE: check(f"severe abuse: {a!r}", severe(a))
for a in NOT_SEVERE: check(f"NOT severe: {a!r}", not severe(a))

print("== a single profane / mild / odd answer never terminates and never earns a strike ==")
for a in ["damn this is hard", "this is fucking hard", "overfitting is a fitting in toilet", "please give me the job, my mom is sick and I need money",
          "whatever", "i already told you in simple terms", "I don't know", "lol this is too easy", "wow damn bro"]:
    st, plan, q = turn_state(fresh(), a)
    check(f"{a!r}: no strike, no warning, no termination, interview continues", st.conduct_strikes == 0 and st.conduct_warnings == 0 and plan.conduct_level == 0 and not plan.conduct_terminate and not q.severe_abuse, (st.conduct_strikes, plan.conduct_level))

print("== escalation ==")
a1, a2, a3 = "Fuck you man", "just fuck off", "Get the fuck out"
st, p1, q1 = turn_state(fresh(), a1)
check("1st severe incident: strike 1, professional warning (level 1), interview CONTINUES", st.conduct_strikes == 1 and st.conduct_warnings == 1 and p1.conduct_level == 1 and not p1.conduct_terminate, (st.conduct_strikes, p1.conduct_level))
check("...the severe flag is separate from technical quality (the answer is simply a poor/empty technical attempt)", q1.severe_abuse and q1.label in ("weak", "irrelevant"))
st = advance(st, p1)
st, p2, q2 = turn_state(st, a2, number=4)
check("2nd severe incident: strike 2, FIRMER warning (level 2), interview continues", st.conduct_strikes == 2 and p2.conduct_level == 2 and not p2.conduct_terminate, (st.conduct_strikes, p2.conduct_level))
st = advance(st, p2)
st, p3, q3 = turn_state(st, a3, number=5)
check("3rd severe incident: interview TERMINATES", st.conduct_strikes == 3 and p3.conduct_terminate and p3.conduct_level == 0, (st.conduct_strikes, p3))
# not consecutive: strikes still accumulate, a normal answer in between resets only the 'consecutive' counter
st = fresh()
st, p, _ = turn_state(st, a1); st = advance(st, p)
st, p, _ = turn_state(st, "Overfitting is when a model memorises the training noise and fails on unseen data.", number=4); st = advance(st, p)
check("a normal answer between incidents resets the consecutive counter, strikes remain", st.conduct_strikes == 1 and st.conduct_consecutive == 0)
st, p, _ = turn_state(st, a2, number=5); st = advance(st, p)
check("spaced 2nd incident: still only a firmer warning", st.conduct_strikes == 2 and p.conduct_level == 2 and not p.conduct_terminate)
st, p, _ = turn_state(st, a3, number=6)
check("spaced 3rd incident terminates", p.conduct_terminate and st.conduct_strikes == 3)

print("== two consecutive severe turns AFTER a warning / professionalism reminder end the interview earlier ==")
reminded = fresh(recent_preambles=[style.TRANSITIONS["reminder_follow"][0]])
st, p, _ = turn_state(reminded, a1)
check("after a professionalism reminder, the first severe turn is only warned (consecutive 1)", not p.conduct_terminate and p.conduct_level == 1 and st.conduct_consecutive == 1, (p, st.conduct_consecutive))
st = advance(st, p)
st, p, _ = turn_state(st, a2, number=4)
check("the SECOND consecutive severe turn after that warning terminates (strike 2, earlier than the 3rd strike)", p.conduct_terminate and st.conduct_strikes == 2, (st.conduct_strikes, p))
st, p, _ = turn_state(fresh(), a1); st = advance(st, p)
st, p, _ = turn_state(st, a2, number=4)
check("with no earlier reminder the 2nd severe turn is NOT early-terminated (escalation policy holds)", not p.conduct_terminate and p.conduct_level == 2)
old = (cfg.CONDUCT_TERMINATE_STRIKES, cfg.CONDUCT_EARLY_CONSECUTIVE)
cfg.CONDUCT_TERMINATE_STRIKES = 2
st, p, _ = turn_state(fresh(), a1); st = advance(st, p)
st, p, _ = turn_state(st, a2, number=4)
check("thresholds are backend-controlled and configurable (strikes=2 -> terminates on the 2nd)", p.conduct_terminate)
cfg.CONDUCT_TERMINATE_STRIKES, cfg.CONDUCT_EARLY_CONSECUTIVE = old

print("== warning wording and the composed message ==")
class Script:
    def __init__(self, *r): self.r, self.calls = list(r), []
    def __call__(self, messages, schema):
        self.calls.append(messages); return LLMResult(self.r[min(len(self.calls) - 1, len(self.r) - 1)], "openrouter", "m")
ref = RetrievedChunk(text="Overfitting is when a model memorises noise.", score=0.6, chunk_id="c1")
def ctx_for(state, plan, q, number=3):
    return ie.TurnContext(target_role="AI / Machine Learning Role", domain="ai_ml", profile=PROFILE, answer_number=number, max_answers=10, state=state, plan=plan, quality=q,
                          turns=[(Q, "x")], previous_questions=[Q], reference=ref, latest_answer="x", candidate_name="Gautham")
def J(question, ctx):
    stay = ctx.plan.step in ("follow_up", "redirect")
    return json.dumps({"action": "ask_question", "topic": ctx.plan.topic if stay else "gradient descent", "difficulty": ctx.plan.difficulty, "follow_up": stay, "question": question})
def one_q(text): return sum(1 for s in ie.split_sentences(text) if ie.is_interrogative(s)) == 1 and text.count("?") == 1

st1, pl1, qq1 = turn_state(fresh(), a1)
c1 = ctx_for(st1, pl1, qq1)
o1 = ie.generate_interviewer_turn(c1, Script(J("Could you describe in simple terms what overfitting is?", c1)), embed)
print("   warning 1 ->", o1.question)
check("warning 1: professional, asks to avoid abusive language, then ONE technical question", o1.preamble_kind == "conduct_warning_1" and "respectful and professional" in o1.question.lower() + " respectful and professional"[:0] or "respectful" in o1.question, o1.question)
check("...no reassurance, no hostility, no score / pass / hiring wording", not OUTCOME_WORDS.search(o1.preamble) and one_q(o1.question), o1.question)
stw = advance(st1, pl1)
st2, pl2, qq2 = turn_state(stw, a2, number=4)
c2 = ctx_for(st2, pl2, qq2, number=4)
o2 = ie.generate_interviewer_turn(c2, Script(J("Could you describe in simple terms what overfitting is?", c2)), embed)
print("   warning 2 ->", o2.question)
check("warning 2 is firmer and says the interview will be concluded if it continues", o2.preamble_kind == "conduct_warning_2" and "concluded" in o2.preamble and o2.preamble != o1.preamble and one_q(o2.question), o2.question)
check("every warning wording: no '?', no score/pass/fail/hiring words", all("?" not in t and not OUTCOME_WORDS.search(t) for lvl in style.CONDUCT_WARNINGS.values() for t in lvl))
check("metadata: severe incident, strike count, warning level are stored", (lambda m: m["severe_abuse"] is True and m["conduct_strikes"] == 2 and m["conduct_warning_level"] == 2 and m["answer_behavior"] in ("normal", "unprofessional", "dismissive"))(o2.meta(c2)), o2.meta(c2))
check("the conduct counters survive the stored-state round trip", (lambda r: (r.conduct_strikes, r.conduct_warnings) == (2, 2))(ie.load_state(ie.dump_state(advance(st2, pl2)), PROFILE)))
check("older stored states without conduct fields still load with zeros", (lambda r: (r.conduct_strikes, r.conduct_warnings, r.conduct_consecutive) == (0, 0, 0))(ie.load_state(json.dumps({"version": ie.STATE_VERSION}), PROFILE)))

print("== termination wording ==")
M = style.CONDUCT_TERMINATION_MESSAGE
check("neutral wording exactly as specified", M == "This interview is being concluded because the conversation has repeatedly moved away from professional participation despite prior reminders. Your responses up to this point will remain available for review.")
check("no failure / rejection / score / pass / fail / hiring-outcome wording", not OUTCOME_WORDS.search(M.replace("Your responses up to this point will remain available for review", "")) and not re.search(r"fail|reject|pass|hire|hiring|score|job", M, re.I))
check("it is NOT one of the resignation / policy-violation strings (so the UI offers 'End Interview & Get Result')", not any(x in M for x in ("unwilling to proceed", "session is closed", "unable to conduct")))

print("== resignation path is unchanged and separate ==")
src = open(os.path.join(BACKEND, "main.py"), encoding="utf-8").read()
chat = src[src.index("async def chat_round"):src.index('@app.get("/api/interview/summary')]
check("explicit resignation is handled FIRST by its own branch (conduct is only evaluated in the else-branch)", chat.index("if is_resigning:") < chat.index("is_severe_abuse(") < chat.index("plan.conduct_terminate"))
check("the resignation message and reason are untouched", "RESIGNATION_MESSAGE, True" in chat and "resignation" in chat)
check("conduct termination uses its own message and distinct reason, and does not call a provider", "CONDUCT_TERMINATION_MESSAGE" in chat and "termination_reason\": CONDUCT_TERMINATION_REASON" in chat
      and chat.index("if plan.conduct_terminate:") < chat.index("generate_interviewer_turn("))
check("the termination reason is stored in the AI row's gen_meta", "system_meta" in chat and style.CONDUCT_TERMINATION_REASON == "conduct_termination")
phrases = next(ast.literal_eval(n.value) for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "_RESIGNATION_PHRASES")
rx = re.compile(r"(?<![a-z0-9])(?:" + "|".join(re.escape(p) for p in phrases) + r")(?![a-z0-9])")
norm = lambda t: re.sub(r"[^a-z0-9]+", " ", t.lower().replace("’", "'").replace("'", "")).strip()
check("abuse alone is not a resignation (no resignation phrase in the abusive examples), and 'I quit' is not abuse", all(not rx.search(norm(a)) for a in SEVERE) and rx.search(norm("I quit")) and not severe("I quit"))

print("== earlier answers stay available and evaluable after a conduct termination ==")
def ai(core, behavior=None, quality=None):
    return ("ai", core, "local", json.dumps({"action": "ask_question", "core_question": core, "topic": "overfitting", "difficulty": "beginner", "answer_behavior": behavior, "answer_quality": quality, "behavior_signals": []}))
def cand(t): return ("candidate", t, None, None)
term_meta = json.dumps({"action": "terminate", "termination_reason": "conduct_termination", "severe_abuse": True, "conduct_strikes": 3})
rows = [ai("What is overfitting?"), cand("It is when a model memorises noise and fails on new data."),
        ai("Why does regularization help?", "normal", "strong"), cand("Fuck you man"),
        ai("Could you explain bias and variance?", "unprofessional", "weak"), cand("just fuck off"),
        ai("Please keep to the technical questions. What is a decision tree?", "unprofessional", "weak"), cand("Get the fuck out"),
        ("ai", style.CONDUCT_TERMINATION_MESSAGE, "system", term_meta)]
t = ev.build_transcript(rows)
check("all 4 answers (including the ones before termination) are paired; the termination row is not a question", len(t.pairs) == 4 and all("concluded" not in p.question for p in t.pairs))
check("a conduct termination is NOT treated as a voided termination: the interview is still evaluated", t.terminated is False and not t.rejected)
check("legacy rows (no provider) carrying the conduct message are also recognised as canned text", ev.is_canned_text(style.CONDUCT_TERMINATION_MESSAGE))
prompts = []
def fake_eval(prompt):
    prompts.append(prompt)
    return json.dumps({"summary": "The candidate answered the first question accurately and did not answer the others.", "strengths": [], "weaknesses": [],
                       "breakdown": [{"index": 1, "answerType": "strong", "score": 85, "feedback": "Accurate."}] + [{"index": i, "answerType": "non_answer", "score": 0, "feedback": "No attempt."} for i in (2, 3, 4)]})
rep = ev.evaluate_transcript(t, fake_eval)
check("the evaluator runs on the collected answers and the first answer keeps its technical score", len(prompts) == 1 and rep["breakdown"][0]["score"] == 85 and len(rep["breakdown"]) == 4 and rep["overallScore"] == round(85 / 4))
check("conduct is reported separately (behaviour note), not mixed into technical scores", "not part of the technical assessment" in rep["behaviorNote"] and rep["behavior"]["questions"] == [2, 3, 4], rep["behavior"])

finish()
