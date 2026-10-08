"""Candidate requests to the interviewer (teach me / hint / answer, score / points, "did I pass", "am I getting the job") and
obvious nonsense or mockery in an answer. Embeddings are REAL; providers are scripted. Nothing here scores anything.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import check, finish, load_embedder  # noqa: E402

import interview_engine as ie  # noqa: E402
import interview_style as style  # noqa: E402
import answer_signals as sig  # noqa: E402
from llm_providers import LLMResult  # noqa: E402
from rag import RetrievedChunk  # noqa: E402
from resume_profile import build_candidate_profile  # noqa: E402

embed = load_embedder()
PROFILE = build_candidate_profile("SKILLS\nPython, PyTorch, scikit-learn, Machine Learning, NLP\nEXPERIENCE\nML Intern - Acme\n- a\n- b\n")
Q = "What is overfitting?"
T = "overfitting"
REASSURING = re.compile(r"no problem|that'?s fine|no worries|don'?t worry", re.I)
STALE = re.compile(r"let'?s set that aside|alright, let'?s move on", re.I)
OUTCOME_CLAIM = re.compile(r"you(?:'ll| will| won'?t| will not| might| may)\s+(?:get|be|pass|fail|receive|hear)|i(?:'ll| will| can| could)\s+(?:give|hire|offer|select|recommend|promise)\s+you|"
                           r"\brejected?\b|\bhired\b|\bselected\b|\bpassed\b|\bfailed\b|\bguarantee|\bpromise\b|\bunfortunately\b|\bsorry\b", re.I)
HOSTILE = re.compile(r"unacceptable|ridiculous|pathetic|stupid|lazy|shame|embarrass|inappropriate|manipulat|nonsense|silly", re.I)
# any piece of the subject matter itself: the backend's wording must never carry technical content
TECHNICAL = re.compile(r"noise|training|validation|regulari|variance|bias|generali[sz]|parameters|model|memori[sz]", re.I)


def cls(a, q=Q, t=T): return ie.classify_answer(a, q, t, embed)


print("== requests are detected, separately from answers, appeals and behaviour ==")
HELP = ["can you teach me that?", "can you explain it?", "give me a hint", "what is the answer?", "can you explain the key components?",
        "please explain this to me", "can I get a hint", "tell me the answer", "can you rephrase the question", "I don't understand the question", "what do you mean"]
SCORE = ["give me the points", "what is my score?", "did I pass?", "how many points do I have", "what's the result", "tell me my marks", "how am I doing", "am I passing"]
OUTCOME = ["am I getting the job?", "will I be hired", "did I get the job?", "will you hire me", "can I get selected"]
for a in HELP: check(f"help request: {a!r}", sig.detect_behavior(a).request == "help", sig.detect_behavior(a).signals)
for a in SCORE: check(f"score request: {a!r}", sig.detect_behavior(a).request == "score", sig.detect_behavior(a).signals)
for a in OUTCOME: check(f"hiring-outcome request: {a!r}", sig.detect_behavior(a).request == "outcome", sig.detect_behavior(a).signals)
NOT_REQUESTS = ["Overfitting happens when a model memorises noise in the training data, so validation error stays high.", "The gradient points in the direction of steepest ascent.",
                "I would explain the result by plotting the loss curve", "we show the points on a scatter plot", "the model can get a better score with regularization",
                "I don't know", "whatever", "please give me the job", "my mom is sick and I need money", "it uses the answer from the previous layer",
                "pass the data through the network", "hire me", "early stopping"]
for a in NOT_REQUESTS:
    r = sig.detect_behavior(a).request
    check(f"not a request: {a[:60]!r}", r is None or a == "hire me", (r, sig.detect_behavior(a).signals))
check("'am I getting the job?' is a question, not boasting: behaviour stays normal", sig.detect_behavior("am I getting the job?").label == "normal" and not sig.detect_behavior("am I getting the job?").confident)
check("'im going to get a job wow' is unchanged (boasting, unprofessional, no request)", sig.detect_behavior("im going to get a job wow").label == "unprofessional" and sig.detect_behavior("im going to get a job wow").request is None)

print("== a request is never technical knowledge ==")
for a in HELP + SCORE + OUTCOME:
    q = cls(a)
    check(f"{a!r}: recorded as a request, a weak/empty attempt (not partial/strong), no appeal-as-technical", q.request is not None and q.label in ("weak", "irrelevant"), (q.label, q.request))
check("strip_markers removes the request clause but keeps the technical clause", "memoris" in sig.strip_markers("It memorises the noise. Can you explain it?").lower() and "explain" not in sig.strip_markers("It memorises the noise. Can you explain it?").lower())
mix = cls("Overfitting is when the model memorises the training noise and fails on new data. Can you teach me the rest?")
check("a real answer plus a request: judged on the answer alone (the request adds nothing), request still recorded", mix.request == "help" and mix.label in ("partial", "strong"), (mix.label, mix.request))
mix_wrong = cls("overfitting is when the model is too simple. what is the answer?")
check("a recognised misconception plus a request is still 'incorrect'", mix_wrong.label == "incorrect" and mix_wrong.request == "help", (mix_wrong.label, mix_wrong.request))

print("== wording of the request reactions ==")
ALL = [(k, t) for k, (a, b) in style.REQUEST_REACTIONS.items() for t in a + b]
check("three request families with several variants each", set(style.REQUEST_REACTIONS) == {"request_help", "request_score", "request_outcome"} and all(len(a) >= 3 for a, _b in style.REQUEST_REACTIONS.values()))
check("the brief's own sentences are available", "I can't provide the answer during the assessment, but I can rephrase the question." in style.REQUEST_REACTIONS["request_help"][0]
      and "I can't provide scores during the interview. Let's continue with the assessment." in style.REQUEST_REACTIONS["request_score"][0])
check("no question marks, exclamations, hostility, reassurance or promise/prediction/rejection in any wording", all("?" not in t and "!" not in t and not HOSTILE.search(t) and not REASSURING.search(t) and not OUTCOME_CLAIM.search(t) for _k, t in ALL), [t for _k, t in ALL if OUTCOME_CLAIM.search(t)])
check("NO technical content in any of them (they can never leak an answer)", all(not TECHNICAL.search(t) for _k, t in ALL), [t for _k, t in ALL if TECHNICAL.search(t)])
check("help wording says the answer is not given; score wording says scores are not given; outcome wording keeps 'only assess your interview responses'",
      all(re.search(r"can't|not able|aren't", t) for k, t in ALL if k == "request_help") and all("score" in t.lower() or "result" in t.lower() for k, t in ALL if k == "request_score")
      and all("interview responses" in t for k, t in ALL if k == "request_outcome"))
firsts = {style.request_reaction("help", "weak", True, r, [])[1] for r in range(6)}
check("variants rotate", len(firsts) >= 3, firsts)
used = style.request_reaction("help", "weak", True, 0, [])[1]
check("the variant used last turn is not repeated straight away", style.request_reaction("help", "weak", True, 0, [used])[1] != used)

print("== through the interviewer pipeline ==")
def fresh(**kw): return ie.InterviewState(difficulty="beginner", current_topic=T, asked_topics=[T], deep_probed_topics=[T], **kw)
class Script:
    def __init__(self, *r): self.r, self.calls = list(r), []
    def __call__(self, messages, schema):
        self.calls.append(messages); return LLMResult(self.r[min(len(self.calls) - 1, len(self.r) - 1)], "openrouter", "m")
def ctx_for(answer, state=None, answer_number=3):
    q = cls(answer)
    st, plan = ie.plan_next_step(state or fresh(), q, "ai_ml", answer_number=answer_number)
    ref = RetrievedChunk(text="Overfitting is when a model memorises noise in the training data.", score=0.6, chunk_id="c1")
    return ie.TurnContext(target_role="AI / Machine Learning Role", domain="ai_ml", profile=PROFILE, answer_number=answer_number, max_answers=10, state=st, plan=plan,
                          quality=q, turns=[(Q, "x")], previous_questions=[Q], reference=ref, latest_answer=answer, candidate_name="Gautham")
def J(question, ctx):
    stay = ctx.plan.step in ("follow_up", "redirect")
    return json.dumps({"action": "ask_question", "topic": ctx.plan.topic if stay else "gradient descent", "difficulty": ctx.plan.difficulty, "follow_up": stay, "question": question})
def one_q(text): return sum(1 for s in ie.split_sentences(text) if ie.is_interrogative(s)) == 1 and text.count("?") == 1

c = ctx_for("can you teach me that?")
o = ie.generate_interviewer_turn(c, Script(J("Could you describe in your own words what overfitting means for a model?", c)), embed)
print("   teach me   ->", o.question)
check("TEACH ME: refuses to provide the answer, offers to rephrase, then ONE rephrased question on the same topic", o.preamble_kind == "request_help" and "can't provide the answer" in o.question and "rephrase" in o.question and one_q(o.question) and c.plan.step == "follow_up", (c.plan.step, o.question))
check("...no reassurance / stale 'move on' filler", not REASSURING.search(o.question) and not STALE.search(o.question))
check("...the model is told the candidate asked for help and must not explain or hint", "candidate_asked_for_help: true" in ie.build_user_message(c) and "do NOT explain" in ie.build_user_message(c))
check("...metadata records the request", (lambda m: m["request"] == "help" and "request:help" in m["behavior_signals"] and m["reaction_kind"] == "request_help")(o.meta(c)), o.meta(c))

print("== no answer leakage ==")
c = ctx_for("give me a hint")
leaky = J("Overfitting is when a model memorises noise in the training data. What is overfitting?", c)
clean = J("In your own words, what does overfitting mean?", c)
s = Script(leaky, clean)
o = ie.generate_interviewer_turn(c, s, embed)
check("a model reply that explains the answer before asking is REJECTED (one repair, then the clean one-sentence question)", len(s.calls) == 2 and o.generation == "repaired" and "memorises" not in o.question and one_q(o.question), (o.generation, o.question))
s = Script(leaky, leaky)
o = ie.generate_interviewer_turn(ctx_for("what is the answer?"), s, embed)
check("two leaking replies -> deterministic fallback question; the answer never reaches the candidate", o.generation == "fallback" and "memorises" not in o.question and "noise" not in o.question.lower() and one_q(o.question), o.question)
c = ctx_for("what is the answer?")
o = ie.generate_interviewer_turn(c, Script(J("What is overfitting in machine learning?", c)), embed)
check("the shown message contains only the backend wording plus the model's single question", o.question == f"{o.preamble} {o.core_question}" and o.core_question.endswith("?"))
check("the core question may restate the concept's NAME but offers nothing beyond it", not re.search(r"memori|noise|variance|regulari", o.question, re.I))

c = ctx_for("give me the points")
o = ie.generate_interviewer_turn(c, Script(J("Could you explain what overfitting is in machine learning?", c)), embed)
print("   points     ->", o.question)
check("SCORE request: 'I can't provide scores during the interview. Let's continue with the assessment.' + ONE technical question", o.preamble_kind == "request_score" and "can't" in o.question and "scores" in o.question and "Let's continue with the assessment" in o.question and one_q(o.question), o.question)
c = ctx_for("what is my score?")
o = ie.generate_interviewer_turn(c, Script(J("Could you explain what overfitting is in machine learning?", c)), embed)
check("'what is my score?' -> score reaction, no number, no verdict", o.preamble_kind == "request_score" and not re.search(r"\d|pass|fail|good|bad", o.preamble, re.I), o.question)
c = ctx_for("did I pass?")
o = ie.generate_interviewer_turn(c, Script(J("Could you explain what overfitting is in machine learning?", c)), embed)
print("   did I pass ->", o.question)
check("'did I pass?' -> no verdict either way (neither 'you passed' nor 'you failed')", o.preamble_kind == "request_score" and not OUTCOME_CLAIM.search(o.question), o.question)
c = ctx_for("am I getting the job?")
o = ie.generate_interviewer_turn(c, Script(J("Could you explain what overfitting is in machine learning?", c)), embed)
print("   the job?   ->", o.question)
check("HIRING OUTCOME: no promise, no rejection, no hiring authority; only the interview responses are assessed; then ONE technical question",
      o.preamble_kind == "request_outcome" and "interview responses" in o.question and not OUTCOME_CLAIM.search(o.question) and not HOSTILE.search(o.question) and one_q(o.question), o.question)
check("...it is not treated as a pleading appeal or as boasting", not c.quality.appeal and c.quality.behavior == "normal" and not c.quality.confident, (c.quality.appeal, c.quality.behavior))

c = ctx_for("can you explain it?", answer_number=3)
st1 = ie.record_question(c.state, c.plan, "overfitting", None, preamble=None)
check("the request does not end the interview and is not counted as an appeal", c.plan.step in ("follow_up", "redirect", "new_topic") and st1.appeal_count == 0)
c2 = ctx_for("give me a hint", state=ie.record_question(fresh(), ie.StepPlan("follow_up", "beginner", T, "weak"), T, None, preamble=None), answer_number=4)
check("a second request on the same topic moves on (limited probing) and uses the 'moving on' wording, still no answer", c2.plan.step == "new_topic", c2.plan.step)
o = ie.generate_interviewer_turn(c2, Script(J("Could you explain how gradient descent updates a model's weights?", c2)), embed)
print("   2nd hint   ->", o.question)
check("...'I can't provide the answer during the assessment, so let's move to another area.' style", o.preamble_kind == "request_help" and o.preamble in style.REQUEST_REACTIONS["request_help"][1], o.preamble)

print("== obvious nonsense / mockery gets the existing refocus, not a bland transition ==")
MOCK = ["overfitting is a fitting in toilet", "its fitting a toilet seat lol", "overfitting means the poop is too big", "overfitting is when you wear a fitting suit in the bathroom"]
for a in MOCK:
    q = cls(a)
    check(f"mock answer {a!r} -> irrelevant", q.label == "irrelevant", (q.label, q.signals))
c = ctx_for("overfitting is a fitting in toilet")
o = ie.generate_interviewer_turn(c, Script(J("Could you explain what overfitting means in machine learning?", c)), embed)
print("   toilet     ->", o.question)
check("the real case now gets the professional refocus (redirect family), not 'Alright, let's move on to a different topic.'", c.plan.step == "redirect" and o.preamble_kind == "redirect" and o.preamble in style.TRANSITIONS["redirect"] and not STALE.search(o.question), (c.plan.step, o.question))
check("...still neutral: no insult, no mockery back", not HOSTILE.search(o.question) and "toilet" not in o.question.lower())
mock2 = cls("feature engineering is engineering of a horse", "What is feature engineering?", "feature engineering")
check("another domain/topic, different words: still irrelevant (nothing is hardcoded to the toilet example)", mock2.label == "irrelevant", (mock2.label, mock2.signals))

print("== normal incorrect, weak and short genuine answers are NOT made harsher ==")
for a, want in [("overfitting is when the model is too simple", "incorrect"), ("overfitting means high bias", "incorrect"), ("i dont know", "weak"),
                ("overfitting is a type of fitting", "vague"), ("add a penalty term", "weak"), ("Use dropout", "weak"), ("It memorises the noise", "weak"),
                ("early stopping", "weak"), ("model is too complex", "weak"), ("It fits training data too well", "weak"), ("too many parameters", "weak"),
                ("it is when the model fits well", "weak"), ("yes", "weak")]:
    q = cls(a)
    check(f"{a!r} -> {want} (unchanged), no request", q.label == want and q.request is None, (q.label, q.request))
for a, q_, t_ in [("creating better inputs from raw information", "What is feature engineering?", "feature engineering"), ("turning raw columns into useful signals", "What is feature engineering?", "feature engineering"),
                  ("layers of connected neurons", "What are neural networks?", "neural networks"), ("by asking yes or no questions about the inputs", "How does a decision tree split the data?", "decision trees"),
                  ("choosing the best question at every branch", "How does a decision tree split the data?", "decision trees")]:
    check(f"short genuine answer {a!r} is NOT irrelevant", cls(a, q_, t_).label != "irrelevant", cls(a, q_, t_).signals)
c = ctx_for("overfitting is when the model is too simple")
o = ie.generate_interviewer_turn(c, Script(J("In one sentence, what is overfitting?", c)), embed)
check("a normal incorrect answer still gets the ordinary neutral correction (incorrect_follow)", o.preamble_kind == "incorrect_follow" and o.preamble in style.TRANSITIONS["incorrect_follow"], o.question)
check("older stored states and metadata still load; appeals still detected separately", ie.load_state(json.dumps({"version": ie.STATE_VERSION}), PROFILE).appeal_count == 0 and sig.detect_behavior("please give me the job").appeal and sig.detect_behavior("please give me the job").request is None)

print("== reassuring wording is only for good-faith non-answers ==")
def reassuring(text): return bool(REASSURING.search(text))
check("every wording in the weak_* families is the only place reassurance exists",
      all(not reassuring(t) for k, items in style.TRANSITIONS.items() if not k.startswith("weak_") for t in items) and any(reassuring(t) for t in style.TRANSITIONS["weak_follow"]))
check("the new repeated-nonsense family is firm and neutral: no reassurance, no '?', says it still does not address the technical question",
      len(style.TRANSITIONS["nonsense_new"]) >= 3 and all(not reassuring(t) and "?" not in t and re.search(r"still", t) for t in style.TRANSITIONS["nonsense_new"])
      and "That still doesn't address the technical question. Let's move to a different area." in style.TRANSITIONS["nonsense_new"])
# honest, good-faith non-answers keep the calm reassuring wording
for ans in ["I don't know", "I'm not sure", "i can't remember", "no idea"]:
    q = cls(ans)
    kinds = {style.reaction_kind(step, q.label, q.behavior, confident=q.confident) for step in ("follow_up", "new_topic")}
    check(f"honest non-answer {ans!r} can still get the calm weak wording", q.label == "weak" and q.behavior == "normal" and kinds == {"weak_follow", "weak_new"}, (q.label, kinds))
c = ctx_for("I don't know")
o = ie.generate_interviewer_turn(c, Script(J("Could you describe in simple terms what overfitting is?", c)), embed)
print("   I don't know ->", o.question)
check("pipeline: 'I don't know' -> weak_follow wording (reassurance allowed)", o.preamble_kind == "weak_follow" and o.preamble in style.TRANSITIONS["weak_follow"], o.question)
# bad-faith or non-genuine weak answers never get reassurance, in any step, even when the professionalism reminder is suppressed
bad = [(step, beh, rem, conf) for step in ("follow_up", "new_topic", "redirect") for beh in ("normal", "dismissive", "unprofessional") for rem in (True, False) for conf in (False, True)
       if (beh != "normal" or conf)]
check("weak + (dismissive | unprofessional | confident), any step, reminder allowed or suppressed: never a weak_* (reassuring) family",
      all(not style.reaction_kind(step, "weak", beh, reminder_ok=rem, confident=conf).startswith("weak_") for step, beh, rem, conf in bad))
check("...and the wording picked for them (every variant of the chosen family) contains no reassurance",
      all(not reassuring(t) for step, beh, rem, conf in bad for t in style.TRANSITIONS[style.reaction_kind(step, "weak", beh, reminder_ok=rem, confident=conf)]))
q = cls("lol idk whatever dude")
st_recent = [style.TRANSITIONS["reminder_follow"][0]]     # a professionalism reminder was just shown, so it is suppressed this turn
c = ctx_for("lol idk dude", state=fresh(recent_preambles=st_recent))
o = ie.generate_interviewer_turn(c, Script(J("Could you describe in simple terms what overfitting is?", c)), embed)
print("   unprofessional weak, reminder suppressed ->", o.question)
check("pipeline: unprofessional weak answer with the reminder suppressed does NOT get 'No problem / That's fine'", c.quality.behavior == "unprofessional" and not reassuring(o.question) and not o.preamble_kind.startswith("weak_"), (c.quality.behavior, o.question))

print("== repeated nonsense (the live regression) ==")
Q2 = "Can you describe overfitting in machine learning without using the toilet analogy?"      # the model's own follow-up in the live session
first = ie.classify_answer("overfitting or overshitting when you shit too much on toilet", Q, T, embed, profane=True)
check("1st mock/profane answer: a poor answer with unprofessional behaviour (professionalism reminder, never reassurance)", first.label in ("weak", "irrelevant") and first.behavior == "unprofessional", (first.label, first.behavior))
st_a, plan_a = ie.plan_next_step(fresh(), first, "ai_ml", answer_number=2)
check("...it is remembered as bad-faith for the next turn", st_a.last_bad_faith is True and plan_a.prior_bad_faith is False)
st_a = ie.record_question(st_a, plan_a, T, None, preamble=style.TRANSITIONS["reminder_follow"][1])
second = ie.classify_answer("overfitting wis when yo ugo to toilet", Q2, T, embed)
st_b, plan_b = ie.plan_next_step(st_a, second, "ai_ml", answer_number=3)
check("2nd typo-ridden mock answer: a poor, signal-free answer with no technical term of its own; the planner moves on and knows the previous one was bad-faith",
      second.label in ("weak", "irrelevant") and second.signals.get("own_terms") == 0 and plan_b.step == "new_topic" and plan_b.prior_bad_faith is True, (second.label, second.signals, plan_b.step))
ref = RetrievedChunk(text="Overfitting is when a model memorises noise.", score=0.6, chunk_id="c1")
def tctx(state, plan, quality):
    return ie.TurnContext(target_role="AI / Machine Learning Role", domain="ai_ml", profile=PROFILE, answer_number=3, max_answers=10, state=state, plan=plan, quality=quality,
                          turns=[(Q2, "x")], previous_questions=[Q, Q2], reference=ref, latest_answer="x", candidate_name="Gautham")
cb = tctx(st_b, plan_b, second)
ob = ie.generate_interviewer_turn(cb, Script(J("What is a decision tree in machine learning?", cb)), embed)
print("   repeated nonsense ->", ob.question)
check("REPEATED nonsense: 'That still doesn't address the technical question...' family; NOT 'That's fine. Let's turn to another area.'",
      ob.preamble_kind == "nonsense_new" and ob.preamble in style.TRANSITIONS["nonsense_new"] and not reassuring(ob.question) and one_q(ob.question), ob.question)
for k in range(6):
    check(f"rotation {k}: nonsense wording has no reassurance", not reassuring(style.choose_transition("nonsense_new", k, [])))
check("repeated nonsense that stays on the topic (limited probing) gets the firm refocus, not a simpler-angle reassurance",
      style.reaction_kind("follow_up", "weak", "normal", repeated_nonsense=True) == "redirect" and style.reaction_kind("redirect", "irrelevant", "normal", repeated_nonsense=True) == "redirect")
check("dismissive / unprofessional wording still takes precedence over the nonsense wording",
      style.reaction_kind("new_topic", "weak", "dismissive", repeated_nonsense=True) == "dismissive_new" and style.reaction_kind("new_topic", "weak", "unprofessional", repeated_nonsense=True) == "reminder_new")

honest = cls("i don't know")
st_h, plan_h = ie.plan_next_step(st_a, honest, "ai_ml", answer_number=3)
ch = tctx(st_h, plan_h, honest)
oh = ie.generate_interviewer_turn(ch, Script(J("What is a decision tree in machine learning?", ch)), embed)
check("an honest 'I don't know' right after a nonsense answer is still treated as good faith (calm weak wording, reassurance allowed)", plan_h.prior_bad_faith and oh.preamble_kind in ("weak_follow", "weak_new"), (oh.preamble_kind, oh.question))
real_short = ie.classify_answer("early stopping", Q, T, embed)
st_r, plan_r = ie.plan_next_step(st_a, real_short, "ai_ml", answer_number=3)
cr = tctx(st_r, plan_r, real_short)
orr = ie.generate_interviewer_turn(cr, Script(J("Could you describe in simple terms what overfitting is?", cr)), embed)
check("a genuine short answer ('early stopping') after a joke is NOT treated as nonsense (it has a technical term of its own)", orr.preamble_kind in ("weak_follow", "weak_new"), (orr.preamble_kind, orr.question))
st_n, plan_n = ie.plan_next_step(fresh(), second, "ai_ml", answer_number=3)
check("without a bad-faith answer before it, the same typo reply gets the ordinary handling (no 'still' wording)", plan_n.prior_bad_faith is False and style.reaction_kind(plan_n.step, second.label, second.behavior) in ("weak_follow", "weak_new", "redirect", "irrelevant_new"))

print("== normal weak behaviour is unchanged ==")
for step, want in (("follow_up", "weak_follow"), ("new_topic", "weak_new")):
    check(f"plain weak answer, {step}: {want} (as before)", style.reaction_kind(step, "weak", "normal") == want)
check("plain weak after a vague (not nonsense) answer is unchanged", style.reaction_kind("follow_up", "weak", "normal", repeated_nonsense=False) == "weak_follow")
for ans in ["early stopping", "add a penalty term", "too many parameters", "Use dropout"]:
    c = ctx_for(ans)
    o = ie.generate_interviewer_turn(c, Script(J("Could you describe in simple terms what overfitting is?", c)), embed)
    check(f"short genuine answer {ans!r}: still the ordinary calm weak_follow", o.preamble_kind == "weak_follow" and o.preamble in style.TRANSITIONS["weak_follow"], o.question)
check("strong / partial / vague / incorrect selection is unchanged",
      [style.reaction_kind("new_topic", l, "normal") for l in ("strong", "partial", "vague", "incorrect", "irrelevant")] == ["strong_new", "partial_new", "vague_new", "incorrect_new", "irrelevant_new"]
      and [style.reaction_kind("follow_up", l, "normal") for l in ("partial", "vague", "incorrect")] == ["partial_follow", "vague_follow", "incorrect_follow"])

finish()
