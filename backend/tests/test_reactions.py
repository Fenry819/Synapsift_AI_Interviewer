"""Phase 3.2: answer-aware interviewer reactions (flow only: nothing here scores an answer).

Covers the two-axis classifier (technical quality x behaviour), reaction selection, the planner (depth probes, repeated
poor answers), the strict validator still holding, and traceability metadata. Embeddings are REAL; providers are scripted.
"""
import ast
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import BACKEND, check, finish, load_embedder  # noqa: E402

import interview_config as cfg  # noqa: E402
import interview_engine as ie  # noqa: E402
import interview_style as style  # noqa: E402
import answer_signals as sig  # noqa: E402
from llm_providers import LLMResult  # noqa: E402
from rag import RetrievedChunk  # noqa: E402
from resume_profile import build_candidate_profile  # noqa: E402

embed = load_embedder()
PROFILE = build_candidate_profile("SKILLS\nPython, PyTorch, scikit-learn, Machine Learning, NLP\nEXPERIENCE\nML Intern - Acme\n- a\n- b\n")
DISPLAY = dict(max_sentences=cfg.MAX_DISPLAY_SENTENCES, max_chars=cfg.MAX_DISPLAY_CHARS)

Q_SUP = "What is supervised learning?"
Q_FE = "What is feature engineering?"
Q_ROLE = "What role does labeled data play in supervised learning?"
Q_OVER = "What is overfitting and how do you prevent it?"
STRONG = ("Overfitting happens when a model memorises noise in the training data instead of the underlying pattern, so training error is low "
          "but validation error is high. I prevent it with regularization such as L2 penalties or dropout, early stopping, cross-validation "
          "to detect it, and by collecting more data or simplifying the model.")


def cls(answer, question=Q_OVER, topic="overfitting and regularization", profane=False):
    return ie.classify_answer(answer, question, topic, embed, profane=profane)


REASSURING = re.compile(r"no problem|that'?s fine|no worries|don'?t worry|it'?s okay|that'?s okay", re.I)
HOSTILE = re.compile(r"unacceptable|you(?:'re| are) wrong|clearly (?:you )?don'?t|poor answer|terrible|stupid|ridiculous|pathetic|lazy|"
                     r"embarrass|shame|incompetent|useless|waste of|disappoint|nonsense|idiot|sarcas|you(?:'ll| will) fail", re.I)
OUTCOME_WORDS = re.compile(r"\b(?:scor\w*|pass(?:ed)?|fail(?:ed)?|hire|hired|hiring|reject\w*|offer|job|grade|rating|selected|shortlist\w*)\b", re.I)

# ======================================================================================================
print("== 1. technical quality and behaviour are separate axes (the real examples from a live session) ==")
a = cls("supervised learning is a type of learning", Q_SUP, "supervised learning")
check("circular answer 'supervised learning is a type of learning' -> vague (not partial, not strong)", a.label == "vague" and a.behavior == "normal", (a.label, a.reason))
a = cls("haah easy question..supervised learning is a type of learning", Q_SUP, "supervised learning")
check("same answer + laughter + 'easy question' -> still vague; behaviour unprofessional; confidence recorded",
      a.label == "vague" and a.behavior == "unprofessional" and a.confident and "unprofessional:laughter" in a.behavior_signals and "confident:easy_question" in a.behavior_signals, (a.label, a.behavior, a.behavior_signals))
a = cls("feature engineering gives features to machine learning", Q_FE, "feature engineering")
check("'feature engineering gives features to machine learning' -> vague (adds nothing specific)", a.label == "vague", (a.label, a.reason, a.signals))
a = cls("labeled data is a type of data that has role..damn its an easy question sir", Q_ROLE, "supervised learning")
check("'labeled data is a type of data that has role..damn its an easy question sir' -> vague + unprofessional + confident",
      a.label == "vague" and a.behavior == "unprofessional" and a.confident, (a.label, a.behavior))
a = cls("i already told you in simple terms", Q_FE, "feature engineering")
check("'i already told you in simple terms' -> behaviour dismissive (technical label is poor, not strong)", a.behavior == "dismissive" and a.label in ("weak", "vague", "irrelevant"), (a.label, a.behavior))
a = cls("damn its an easy question sir", Q_FE, "feature engineering")
check("'damn its an easy question sir' (no technical content at all) -> dismissive, nothing counted as an attempt", a.behavior == "dismissive" and a.label == "weak" and a.confident, (a.label, a.behavior))
a = cls("im going to get a job wow", Q_FE, "feature engineering")
check("'im going to get a job wow' -> a poor (off-topic) answer + behaviour UNPROFESSIONAL (boasting/slang), not dismissive", a.label in ("weak", "irrelevant") and a.behavior == "unprofessional", (a.label, a.behavior, a.signals))
a = cls("pink pong shoot a gun")
check("nonsense 'pink pong shoot a gun' -> irrelevant, behaviour normal (it is off-topic, not rude)", a.label == "irrelevant" and a.behavior == "normal", (a.label, a.behavior))
for text in ["whatever", "why are you asking this", "ask something else", "this is easy, next question", "can we move on", "this is boring"]:
    check(f"dismissive wording {text!r} -> behaviour dismissive", cls(text).behavior == "dismissive", cls(text).behavior)

print("-- recognised misconceptions -> incorrect (and only those)")
for text, rule in [("supervised learning uses unlabeled data, obviously", "supervised-needs-no-labels"),
                   ("Overfitting is when the model performs poorly on the training data.", "overfitting-bad-on-training"),
                   ("overfitting happens when the model is too simple to capture the pattern", "overfitting-too-simple"),
                   ("gradient descent works by maximizing the loss function", "gradient-descent-maximizes"),
                   ("regularization increases the model complexity to fit better", "regularization-adds-complexity"),
                   ("cross-validation is used to train the model faster", "cross-validation-speeds-training"),
                   ("k-means is a supervised algorithm that needs labeled data", "kmeans-is-supervised"),
                   ("the test set is used to train the model", "test-set-used-for-training")]:
    a = cls(text, "Explain the concept." if "supervised" not in text else Q_SUP, text.split()[0])
    q = {"supervised": Q_SUP, "Overfitting": Q_OVER, "overfitting": Q_OVER, "gradient": "How does gradient descent work?", "regularization": "What is regularization?",
         "cross-validation": "Why use cross-validation?", "k-means": "How does k-means clustering work?", "the": "What is the role of the test set?"}[text.split()[0]]
    a = ie.classify_answer(text, q, "", embed)
    check(f"misconception -> incorrect ({rule})", a.label == "incorrect" and a.signals.get("misconception") == rule, (a.label, a.signals))
a = ie.classify_answer("Supervised learning uses labeled data, unlike unsupervised learning which works with unlabeled data and finds structure by itself.", Q_SUP, "supervised learning", embed)
check("a correct CONTRAST of supervised vs unsupervised is not mistaken for a misconception", a.label != "incorrect", (a.label, a.signals))
a = ie.classify_answer("Unsupervised learning works on unlabeled data and finds structure such as clusters without target labels.", "What is unsupervised learning?", "unsupervised learning", embed)
check("correct 'unsupervised ... unlabeled' is not flagged", a.label != "incorrect", (a.label, a.signals))

print("-- good answers are not punished for tone")
a = cls(STRONG)
check("strong, plain wording -> strong / normal", a.label == "strong" and a.behavior == "normal", (a.label, a.behavior))
a = cls("lol " + STRONG)
check("technically strong answer with casual wording ('lol') -> still STRONG; behaviour unprofessional is recorded separately", a.label == "strong" and a.behavior == "unprofessional", (a.label, a.behavior))
a = cls("Obviously: " + STRONG)
check("confidence on its own is not punished: strong + 'obviously' -> strong, behaviour normal, confidence flagged", a.label == "strong" and a.behavior == "normal" and a.confident, (a.label, a.behavior, a.confident))
a = cls(STRONG + " Damn, that was easy.", profane=True)
check("casual profanity inside a strong answer -> strong / unprofessional (continue, do not terminate)", a.label == "strong" and a.behavior == "unprofessional", (a.label, a.behavior))
a = cls("It is easy to overfit when the model is too complex for a small noisy dataset, so I would use regularization and cross-validation with early stopping.")
check("the plain word 'easy' inside a real answer is not a confidence marker", not a.confident and a.label in ("partial", "strong"), (a.label, a.confident))
check("partial answer -> partial / normal", cls("Overfitting is when the model does well on training data but badly on new data, so you can use regularization to help with that.").label == "partial")
check("profanity flag from the app marks behaviour unprofessional even without slang words", cls("regularization", profane=True).behavior == "unprofessional")

print("-- unchanged Phase 3 behaviour")
for text, want in [("I don't know", "weak"), ("idk", "weak"), ("pass", "weak"), ("regularization", "weak"), ("It overfits", "weak"), ("add a penalty term", "weak"), ("", "weak")]:
    check(f"{text!r} -> {want}", cls(text).label == want, cls(text).label)
check("'I would pass the data through the pipeline' is not the word 'pass'", "non-answer" not in cls("I would pass the data through the pipeline").reason)
check("no embeddings -> an off-topic reply stays conservative (never 'irrelevant')", ie.classify_answer("pink pong shoot a gun", Q_OVER, "x", None).label != "irrelevant")
check("short genuine answers are not 'vague' or 'irrelevant'", all(cls(t).label not in ("vague", "irrelevant") for t in ["Use dropout", "early stopping", "It memorises the noise", "model is too complex", "add a penalty term"]))
check("eigenvectors-style answer (specific words outside the vocabulary) is not called vague", ie.classify_answer("It projects the data onto the eigenvectors of the covariance matrix with the largest eigenvalues.", "How does PCA work?", "PCA", embed).label != "vague")

# ======================================================================================================
print("== 2. reaction selection (step x quality x behaviour) ==")
RK = style.reaction_kind
cases = [
    (("follow_up", "strong", "normal"), "strong_deeper"), (("new_topic", "strong", "normal"), "strong_new"),
    (("follow_up", "partial", "normal"), "partial_follow"), (("new_topic", "partial", "normal"), "partial_new"),
    (("follow_up", "incorrect", "normal"), "incorrect_follow"), (("new_topic", "incorrect", "normal"), "incorrect_new"),
    (("follow_up", "vague", "normal"), "vague_follow"), (("new_topic", "vague", "normal"), "vague_new"),
    (("redirect", "irrelevant", "normal"), "redirect"), (("new_topic", "irrelevant", "normal"), "irrelevant_new"),
    (("follow_up", "weak", "normal"), "weak_follow"), (("new_topic", "weak", "normal"), "weak_new"),
    (("redirect", "weak", "dismissive"), "dismissive_follow"), (("new_topic", "vague", "dismissive"), "dismissive_new"),
    (("follow_up", "vague", "unprofessional"), "reminder_follow"), (("new_topic", "incorrect", "unprofessional"), "reminder_new"),
    (("redirect", "irrelevant", "unprofessional"), "reminder_follow"),
    (("follow_up", "partial", "unprofessional"), "partial_follow"),        # partial + casual wording: no lecture
    (("new_topic", "strong", "unprofessional"), "strong_new"),             # strong answers are never lectured
    (("follow_up", "strong", "dismissive"), "strong_deeper"),
    (("open", None, "normal"), "open"),
]
for (step, label, behavior), want in cases:
    check(f"{step:9s} {str(label):9s} {behavior:14s} -> {want}", RK(step, label, behavior) == want, RK(step, label, behavior))
check("a reminder already given recently is not repeated: unprofessional + vague falls back to the quality reaction",
      RK("follow_up", "vague", "unprofessional", reminder_ok=False) == "vague_follow" and RK("new_topic", "incorrect", "unprofessional", reminder_ok=False) == "incorrect_new")
check("dismissive is never softened by the reminder logic", RK("redirect", "weak", "dismissive", reminder_ok=False) == "dismissive_follow")
check("confident + 'weak' (claims it is easy, then does not answer) never gets the reassuring family", RK("follow_up", "weak", "normal", confident=True) == "vague_follow" and RK("new_topic", "weak", "normal", confident=True) == "vague_new")
check("an honest 'I don't know' still gets the calm family", RK("follow_up", "weak", "normal") == "weak_follow")

print("== 3. wording rules for every canned reaction ==")
ALL = [(fam, t) for fam, items in style.TRANSITIONS.items() for t in items]
check("no reaction contains a question mark or an exclamation mark", all("?" not in t and "!" not in t for _f, t in ALL))
check("reassuring phrases appear ONLY in the weak_* families", all(not REASSURING.search(t) for f, t in ALL if not f.startswith("weak_")),
      [(f, t) for f, t in ALL if not f.startswith("weak_") and REASSURING.search(t)])
check("never hostile, shaming, sarcastic or threatening", all(not HOSTILE.search(t) for _f, t in ALL), [t for _f, t in ALL if HOSTILE.search(t)])
check("no score / pass-fail / hiring / outcome language", all(not OUTCOME_WORDS.search(t) for _f, t in ALL), [t for _f, t in ALL if OUTCOME_WORDS.search(t)])
check("no 'you are wrong' style phrasing (corrections talk about the explanation, not the person)", all(not re.search(r"\byou(?:'re| are| clearly)?\b.{0,12}\b(?:wrong|incorrect|bad|poor)\b", t, re.I) for _f, t in ALL))
check("no technical content in any reaction (no lexicon term, no definition)", all(not sig_terms for sig_terms in [__import__("resume_profile").find_lexicon_terms(t) for _f, t in ALL]))
check("incorrect-answer reactions never reveal an answer (they only say it was inaccurate)", all(not re.search(r"labeled|input|output|gradient|overfit|regulari|supervised|feature|noise", t, re.I) for f, t in ALL if f.startswith("incorrect")))
check("every reaction family exists and has at least 3 variants", all(len(items) >= 3 for items in style.TRANSITIONS.values()))
check("every reaction + question passes the full validator", all(ie.validate_question(f"{t} How would you explain this concept?", PROFILE, **DISPLAY) == [] for _f, t in ALL))
check("sentence-level phrasing matches the spec for the main cases",
      any("on the right track" in t for t in style.TRANSITIONS["partial_follow"]) and any("isn't technically accurate" in t for t in style.TRANSITIONS["incorrect_follow"])
      and any("too vague" in t for t in style.TRANSITIONS["vague_follow"]) and any("Please answer the technical question directly" in t for t in style.TRANSITIONS["dismissive_follow"])
      and any("professional" in t for t in style.TRANSITIONS["reminder_follow"]) and any("one level deeper" in t for t in style.TRANSITIONS["strong_deeper"]))
check("the closing message still has no score / verdict", not OUTCOME_WORDS.search(style.CLOSING_MESSAGE.replace("evaluated", "")) and "?" not in style.CLOSING_MESSAGE)

print("== 4. varied wording ('No problem' / \"Let's\" every turn is gone) ==")
for kind in ("weak_follow", "vague_follow", "incorrect_follow", "dismissive_follow", "partial_follow", "strong_new", "reminder_follow"):
    recent, seq = [], []
    for n in range(1, 31):
        t = style.choose_transition(kind, n, recent); seq.append(t); recent = (recent + [t])[-cfg.MAX_RECENT_PREAMBLES:]
    check(f"{kind}: 30 consecutive picks never repeat the previous one and use >= 3 variants", all(a != b for a, b in zip(seq, seq[1:])) and len(set(seq)) >= 3)
mix = ["weak_follow", "weak_new", "vague_follow", "partial_follow", "strong_new", "incorrect_follow", "weak_follow", "irrelevant_new", "vague_new", "strong_deeper"] * 3
recent, shown = [], []
for n, k in enumerate(mix, start=1):
    t = style.choose_transition(k, n, recent); shown.append(t); recent = (recent + [t])[-cfg.MAX_RECENT_PREAMBLES:]
firsts = [x.split()[0].lower() for x in shown]
check("across a mixed 30-turn interview no opening phrase repeats within 3 consecutive reactions", all(style._lead(shown[i]) not in [style._lead(x) for x in shown[max(0, i - 3):i]] for i in range(len(shown))))
check("...and the same first word is not used three times in a row", all(not (firsts[i] == firsts[i - 1] == firsts[i - 2]) for i in range(2, len(firsts))), firsts)

# ======================================================================================================
print("== 5. the planner: depth for strong answers, limited probing for poor ones ==")
Aq = lambda label, behavior="normal": ie.AnswerQuality(label, "t", behavior=behavior)
def fresh(topic="overfitting and regularization", **kw):
    return ie.InterviewState(difficulty="intermediate", current_topic=topic, asked_topics=[topic], **kw)
s, p = ie.plan_next_step(fresh(), Aq("strong"), "ai_ml", answer_number=2)
check("first strong topic -> ONE deeper follow-up on the same topic, difficulty +1", p.step == "follow_up" and p.depth_probe and p.difficulty == "advanced" and ie.same_topic(p.topic, "overfitting and regularization"), p)
st2 = ie.record_question(s, p, p.topic, None)
check("recording it marks the topic as deep-probed and counts the probe", st2.probe_count == 1 and st2.deep_probed_topics)
s, p = ie.plan_next_step(st2, Aq("strong"), "ai_ml", answer_number=3)
check("strong again after the deeper probe -> MOVE ON (no endless probing)", p.step == "new_topic" and not p.depth_probe)
st3 = ie.record_question(s, p, "gradient descent", None)
s, p = ie.plan_next_step(st3, Aq("strong"), "ai_ml", answer_number=4)
check("the 2nd strong topic is NOT deep-probed (not every strong answer)", p.step == "new_topic", p)
st4 = ie.record_question(s, p, "decision trees", None)
s, p = ie.plan_next_step(st4, Aq("strong"), "ai_ml", answer_number=5)
check("the 3rd strong topic is deep-probed again", p.step == "follow_up" and p.depth_probe, p)
late = fresh();
s, p = ie.plan_next_step(late, Aq("strong"), "ai_ml", answer_number=8, max_answers=10)
check("no deeper probe when fewer than MIN_ANSWERS_LEFT answers remain", p.step == "new_topic", p)
capped = fresh(deep_probed_topics=["a b", "c d", "e f"])
s, p = ie.plan_next_step(capped, Aq("strong"), "ai_ml", answer_number=2)
check(f"never more than {cfg.MAX_DEEP_PROBES} deeper probes per interview", p.step == "new_topic", p)
s, p = ie.plan_next_step(fresh(deep_probed_topics=["overfitting and regularization"]), Aq("strong"), "ai_ml", answer_number=2)
check("a topic is never deep-probed twice", p.step == "new_topic")

def simulate(labels, behaviors=None, domain="ai_ml"):
    """Run the planner over a scripted sequence of answers; returns [(step, topic, difficulty, depth)] per question."""
    state = ie.InterviewState(difficulty="beginner")
    state, plan = ie.plan_next_step(state, None, domain)
    state = ie.record_question(state, plan, plan.topic, None)
    out, asked = [(plan.step, state.current_topic, state.difficulty, False)], []
    for n, label in enumerate(labels, start=1):
        beh = (behaviors or {}).get(n, "normal")
        state, plan = ie.plan_next_step(state, Aq(label, beh), domain, answer_number=n)
        topic = plan.topic if plan.step in ("follow_up", "redirect") else plan.topic
        state = ie.record_question(state, plan, topic, None)
        out.append((plan.step, state.current_topic, plan.difficulty, plan.depth_probe))
    return out, state
seq, _ = simulate(["strong"] * 9)
steps = [x[0] for x in seq]
check("9 strong answers: some (but not all) topics get a deeper probe: steps=" + ",".join(s[:2] for s in steps),
      0 < sum(1 for x in seq if x[3]) <= cfg.MAX_DEEP_PROBES and steps.count("new_topic") >= 4, steps)
check("...never two consecutive deeper probes, never more than 2 questions in a row on one topic",
      all(not (seq[i][3] and seq[i - 1][3]) for i in range(1, len(seq))) and all(not (seq[i][1] == seq[i - 1][1] == seq[i - 2][1]) for i in range(2, len(seq))), [x[:2] for x in seq])
seq, _ = simulate(["partial"] * 6)
check("partial answers: one focused follow-up per topic, then a new topic", [x[0] for x in seq] == ["open", "follow_up", "new_topic", "follow_up", "new_topic", "follow_up", "new_topic"], [x[0] for x in seq])
for label in ("incorrect", "vague", "weak"):
    seq, _ = simulate([label] * 6)
    check(f"repeated {label} answers: ONE clarification per topic, then the interview moves topic ({','.join(x[0][:2] for x in seq)})",
          [x[0] for x in seq] == ["open", "follow_up", "new_topic", "follow_up", "new_topic", "follow_up", "new_topic"] and len({x[1] for x in seq}) >= 4, [x[:2] for x in seq])
    check(f"...and never more than 2 consecutive questions on one concept", all(not (seq[i][1] == seq[i - 1][1] == seq[i - 2][1]) for i in range(2, len(seq))))
seq, _ = simulate(["weak"] * 4, {1: "dismissive", 2: "dismissive", 3: "dismissive"})
check("dismissive answers: one firm re-ask (redirect) on the same topic, then move on", [x[0] for x in seq][:3] == ["open", "redirect", "new_topic"], [x[0] for x in seq])
seq, _ = simulate(["irrelevant"] * 3)
check("irrelevant answers: one redirect, then move on", [x[0] for x in seq] == ["open", "redirect", "new_topic", "redirect"], [x[0] for x in seq])
seq, _ = simulate(["vague", "partial", "strong", "weak", "incorrect", "strong"])
check("a poor answer followed by a better one is not punished again (bad-attempt counter resets)", seq[2][0] == "new_topic" or seq[2][0] == "follow_up", [x[:2] for x in seq])
diffs = [x[2] for x in simulate(["strong", "strong", "incorrect", "vague", "strong", "weak", "strong", "strong"])[0]]
lv = cfg.DIFFICULTY_LEVELS
check("difficulty never moves more than one level per answer: " + "->".join(d[:3] for d in diffs), all(abs(lv.index(a) - lv.index(b)) <= 1 for a, b in zip(diffs, diffs[1:])))
check("incorrect and vague answers lower the difficulty by one; strong raises it by one",
      ie.adapt_difficulty("intermediate", "incorrect") == "beginner" and ie.adapt_difficulty("intermediate", "vague") == "beginner" and ie.adapt_difficulty("intermediate", "strong") == "advanced")
check("weak_topics gets incorrect/vague topics; strong_topics gets strong ones",
      "overfitting and regularization" in ie.plan_next_step(fresh(), Aq("incorrect"), "ai_ml")[0].weak_topics and "overfitting and regularization" in ie.plan_next_step(fresh(), Aq("vague"), "ai_ml")[0].weak_topics
      and "overfitting and regularization" in ie.plan_next_step(fresh(), Aq("strong"), "ai_ml")[0].strong_topics)
check("state saved before Phase 3.2 (no deep_probed_topics / bad_attempts) still loads", ie.load_state(json.dumps({"version": 1, "difficulty": "beginner", "asked_topics": ["a"], "recent_preambles": ["x"]}), PROFILE).bad_attempts == 0)

# ======================================================================================================
print("== 6. what the candidate actually sees (engine with a scripted model) ==")
def make_ctx(label, behavior="normal", confident=False, step_state=None, answer_number=3, prev=None, reason="t"):
    st = step_state or fresh(deep_probed_topics=["overfitting and regularization"])
    q = ie.AnswerQuality(label, reason, behavior=behavior, behavior_signals=[f"{behavior}:x"] if behavior != "normal" else [], confident=confident)
    st, plan = ie.plan_next_step(st, q, "ai_ml", answer_number=answer_number)
    ref = RetrievedChunk(text="Supervised learning fits a model to labeled input-output pairs.", score=0.6, chunk_id="c1")
    return ie.TurnContext(target_role="AI / Machine Learning Role", domain="ai_ml", profile=PROFILE, answer_number=answer_number, max_answers=10, state=st, plan=plan, quality=q,
                          turns=[(Q_SUP, "x")], previous_questions=prev if prev is not None else [Q_SUP], reference=ref, latest_answer="x", candidate_name="Gautham")
class Script:
    def __init__(self, *r): self.r, self.calls = list(r), []
    def __call__(self, messages, schema):
        self.calls.append(messages); return LLMResult(self.r[min(len(self.calls) - 1, len(self.r) - 1)], "openrouter", "m")
def J(q, ctx, topic=None):
    """A model reply for this turn. A follow-up/redirect must stay on the planned topic; a new topic may name its own."""
    stay = ctx.plan.step in ("follow_up", "redirect")
    return json.dumps({"action": "ask_question", "topic": ctx.plan.topic if stay else (topic or ctx.plan.topic), "difficulty": ctx.plan.difficulty,
                       "follow_up": stay, "question": q})
def turn(ctx, *r): return ie.generate_interviewer_turn(ctx, Script(*r), embed)
def one_question(text): return sum(1 for s in ie.split_sentences(text) if ie.is_interrogative(s)) == 1 and text.count("?") <= 1

c = make_ctx("vague"); o = turn(c, J("What specifically does supervised learning involve, in concrete terms?", c, "supervised learning"))
print("   vague      ->", o.question)
check("VAGUE: neutral 'be more specific' reaction, no reassurance, one question", o.preamble_kind == "vague_follow" and not REASSURING.search(o.question) and one_question(o.question) and o.question.endswith("in concrete terms?"), o.question)
c = make_ctx("incorrect"); o = turn(c, J("What role does labeled data play in supervised learning?", c, "supervised learning"))
print("   incorrect  ->", o.question)
check("INCORRECT: neutral correction + ONE focused question; no reassurance", o.preamble_kind == "incorrect_follow" and not REASSURING.search(o.question) and one_question(o.question), o.question)
leak = "Supervised learning uses labeled input-output pairs. What role does labeled data play in it?"
s = Script(J(leak, c, "supervised learning"), J("What role does labeled data play in supervised learning?", c, "supervised learning"))
o = ie.generate_interviewer_turn(c, s, embed)
check("INCORRECT: a question that states the correct answer is rejected (must be the question alone) and repaired",
      o.generation == "repaired" and "input-output" not in o.question and "ONLY the question" in o.attempts[0]["problems"][0], (o.generation, o.attempts))
s = Script(J(leak, c, "supervised learning"), J(leak, c, "supervised learning")); o = ie.generate_interviewer_turn(c, s, embed)
check("INCORRECT: if the model keeps explaining, the deterministic fallback (a bare focused question) is used instead; answer never leaks",
      o.generation == "fallback" and "input-output" not in o.question and "labeled" not in o.question.lower() and one_question(o.question), o.question)
c = make_ctx("weak", behavior="dismissive", confident=True)
o = turn(c, J("How would you explain supervised learning in simple terms?", c, "supervised learning"))
print("   dismissive ->", o.question)
check("DISMISSIVE: firm, direct, professional; repeats the question; one question", o.preamble_kind == "dismissive_follow" and o.preamble in style.TRANSITIONS["dismissive_follow"] and one_question(o.question) and not REASSURING.search(o.question), o.question)
c = make_ctx("weak", behavior="dismissive")
o = turn(c, J(Q_SUP, c, "supervised learning"))
check("DISMISSIVE re-ask may repeat the question just asked (no duplicate rejection for a redirect)", o.generation == "primary" and o.core_question == Q_SUP, (o.generation, o.attempts))
c2 = make_ctx("partial"); o2 = ie.generate_interviewer_turn(c2, Script(J(Q_SUP, c2, "supervised learning"), J("Which part of supervised learning is least clear to you?", c2, "supervised learning")), embed)
check("...but an ordinary follow-up that near-repeats the question is still rejected", o2.generation == "repaired" and "exact repeat" in o2.attempts[0]["problems"][0], o2.attempts)
c = make_ctx("vague", behavior="unprofessional", confident=True)
o = turn(c, J("What specifically does supervised learning involve, in concrete terms?", c, "supervised learning"))
print("   unprofessional + vague ->", o.question)
check("MILDLY UNPROFESSIONAL + poor answer: ONE brief professionalism reminder, calm, no termination wording", o.preamble_kind == "reminder_follow" and "professional" in o.preamble.lower() + "focused" and one_question(o.question), o.question)
c = make_ctx("vague", behavior="unprofessional", confident=True, step_state=fresh(deep_probed_topics=["overfitting and regularization"], recent_preambles=[style.TRANSITIONS["reminder_follow"][0]]))
o = turn(c, J("What specifically does supervised learning involve, in concrete terms?", c, "supervised learning"))
check("...and the reminder is NOT repeated while it is still in recent memory (falls back to the vague reaction)", o.preamble_kind == "vague_follow", o.question)
st = fresh(deep_probed_topics=["overfitting and regularization"]); shown = []
for n in range(1, 9):
    c = make_ctx("vague", behavior="unprofessional", confident=True, step_state=st, answer_number=n)
    o = turn(c, J("What specifically does supervised learning involve, in concrete terms?", c, c.plan.topic) if c.plan.step != "new_topic" else J(f"How does topic number {n} work in practice?", c, f"topic number {n}"))
    shown.append(o.preamble_kind); st = ie.record_question(c.state, c.plan, o.topic, None, preamble=o.preamble); st.probe_count = 0; st.bad_attempts = 0
reminders = [i for i, k in enumerate(shown) if k.startswith("reminder")]
check(f"8 unprofessional+poor turns in a row: reminders are spaced out, never every turn: {shown}", 0 < len(reminders) <= 3 and all(b - a >= 2 for a, b in zip(reminders, reminders[1:])), shown)
c = make_ctx("partial", behavior="unprofessional"); o = turn(c, J("Which part of supervised learning is unclear?", c, "supervised learning"))
check("casual wording around a PARTIAL answer gets no lecture: the normal partial reaction", o.preamble_kind == "partial_follow", o.question)
c = make_ctx("strong", behavior="unprofessional"); o = turn(c, J("What are the trade-offs of regularization?", c, "overfitting and regularization"))
check("a technically STRONG answer with casual wording gets the normal strong reaction (no reminder)", o.preamble_kind in ("strong_deeper", "strong_new"), o.question)
c = make_ctx("irrelevant", behavior="unprofessional"); o = turn(c, J("How would you explain supervised learning in simple terms?", c, "supervised learning"))
check("nonsense/irrelevant + unprofessional (e.g. 'im going to get a job wow'): a firm professional refocus", o.preamble_kind in ("reminder_follow", "redirect") and one_question(o.question), o.question)
c = make_ctx("irrelevant"); o = turn(c, J("How would you explain supervised learning in simple terms?", c, "supervised learning"))
check("IRRELEVANT: 'doesn't address the question ... refocus'", o.preamble_kind == "redirect" and "refocus" in o.preamble, o.question)
c = make_ctx("strong", step_state=fresh()); o = turn(c, J("What trade-offs come up when choosing the regularization strength?", c, "overfitting and regularization"))
print("   strong (deeper) ->", o.question)
check("STRONG with a deeper follow-up: restrained praise + 'one level deeper'-style wording", c.plan.depth_probe and o.preamble_kind == "strong_deeper" and not re.search(r"excellent|amazing|perfect|!", o.question, re.I), o.question)
c = make_ctx("partial"); o = turn(c, J("Which part of overfitting is least clear to you?", c, "overfitting and regularization"))
check("PARTIAL: 'right track ... incomplete / clarify one part'", o.preamble_kind == "partial_follow", o.question)

print("== 7. nothing rude, leaky or outcome-related is ever shown; strict rules intact ==")
combos = [(l, b, conf) for l in ("strong", "partial", "incorrect", "vague", "irrelevant", "weak") for b in ("normal", "dismissive", "unprofessional") for conf in (False, True)]
bad_msgs, shown_all = [], []
for l, b, conf in combos:
    for state_kw in ({"deep_probed_topics": ["overfitting and regularization"]}, {}):
        c = make_ctx(l, b, conf, step_state=fresh(**state_kw))
        o = ie.generate_interviewer_turn(c, Script("garbage", "garbage"), embed)         # forces the deterministic fallback path
        shown_all.append(o.question)
        problems = ie.validate_question(o.question, PROFILE, **DISPLAY)
        if problems or HOSTILE.search(o.question) or OUTCOME_WORDS.search(o.question) or not one_question(o.question):
            bad_msgs.append((l, b, o.question, problems))
check(f"{len(shown_all)} generated messages (every quality x behaviour x confidence, fallback path): valid, exactly one question, never hostile, no score/hiring words", not bad_msgs, bad_msgs[:2])
check("no generated message contains a reassuring phrase unless the answer was an honest weak one",
      all(not REASSURING.search(m) for m, (l, b, c2) in zip(shown_all[0::2], combos) if l != "weak" or b != "normal" or c2))
for label, bad in {"two questions": "What is supervised learning? How does it differ from unsupervised learning?",
                   "compound": "What is supervised learning, and how does it differ from unsupervised learning?",
                   "bullets": "Discuss:\n- labels\n- features\nWhich matters more?", "multiple choice": "Which is right? (A) labeled (B) unlabeled",
                   "RAG leak": "According to the background knowledge, what is supervised learning?", "provider leak": "As an AI language model, what is supervised learning?",
                   "insult": "That was a ridiculous answer. What is supervised learning?", "fabricated skill": "Given your experience with Kubernetes, what is supervised learning?",
                   "greeting smuggled into the question": "Excellent question about supervised learning, what is it?"}.items():
    c = make_ctx("vague"); s = Script(J(bad, c, "supervised learning"), J("What specifically does supervised learning involve, in concrete terms?", c, "supervised learning"))
    o = ie.generate_interviewer_turn(c, s, embed)
    check(f"still rejected and repaired: {label}", o.generation == "repaired" and len(s.calls) == 2 and one_question(o.question), (o.generation, o.question))
c = make_ctx("vague"); o = turn(c, J("Hello again! Excellent work. What specifically does supervised learning involve, in concrete terms?", c, "supervised learning"))
check("greeting / praise the model writes in front of its question is dropped; only the backend reaction is shown", not re.search(r"hello again|excellent", o.question, re.I) and o.preamble in style.TRANSITIONS["vague_follow"], o.question)

# explicit resignation is still decided by main.py's deterministic list; the new dismissive/unprofessional wording must not match it
src = open(os.path.join(BACKEND, "main.py"), encoding="utf-8").read()
phrases = next(ast.literal_eval(n.value) for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "_RESIGNATION_PHRASES")
rx = re.compile(r"(?<![a-z0-9])(?:" + "|".join(re.escape(p) for p in phrases) + r")(?![a-z0-9])")
norm = lambda t: re.sub(r"[^a-z0-9]+", " ", t.lower().replace("’", "'").replace("'", "")).strip()
SESSION = ["haah easy question..supervised learning is a type of learning", "labeled data is a type of data that has role..damn its an easy question sir",
           "i already told you in simple terms", "im going to get a job wow", "whatever", "why are you asking this", "ask something else", "this is easy",
           "damn its an easy question sir", "lol", "feature engineering gives features to machine learning", "supervised learning uses unlabeled data, obviously",
           "pink pong shoot a gun", "this is boring", "can we move on", "yeah right"]
check("NO accidental termination: none of the live-session / dismissive / unprofessional replies matches the explicit-resignation rules in main.py",
      all(not rx.search(norm(t)) for t in SESSION), [t for t in SESSION if rx.search(norm(t))])
check("explicit resignation still matches ('I quit', 'stop the interview', 'I don't want to continue')", all(rx.search(norm(t)) for t in ["I quit", "stop the interview", "I don't want to continue"]))
check("main.py passes the profanity flag and answer count to the new classifier/planner", "profane=is_profane" in src and "answer_number=answer_number, max_answers=MAX_CANDIDATE_ANSWERS" in src)

print("== 8. traceability (gen_meta) ==")
c = make_ctx("vague", "unprofessional", True); o = turn(c, J("What specifically does supervised learning involve, in concrete terms?", c, "supervised learning"))
m = o.meta(c)
need = {"answer_quality", "answer_behavior", "behavior_signals", "confident", "reaction_kind", "quality_reason", "quality_signals", "step", "core_question", "preamble", "preamble_kind", "depth_probe", "generation", "difficulty", "topic"}
check("gen_meta carries quality, behaviour, behaviour signals, confidence, reaction kind, reason, step, core question, preamble", need <= set(m), need - set(m))
check("...with the right values", m["answer_quality"] == "vague" and m["answer_behavior"] == "unprofessional" and m["confident"] is True and m["reaction_kind"] == m["preamble_kind"]
      and m["core_question"].endswith("concrete terms?") and m["preamble"] and m["step"] == "follow_up", m)
json.dumps(m)
check("gen_meta is JSON-serialisable", True)
check("the SYSTEM prompt no longer claims 'question' is what the candidate reads, and forbids revealing the answer",
      "exactly what the candidate will read" not in ie.SYSTEM_PROMPT and "never reveal the correct answer" in ie.SYSTEM_PROMPT and "CORE technical question" in ie.SYSTEM_PROMPT)

finish()
