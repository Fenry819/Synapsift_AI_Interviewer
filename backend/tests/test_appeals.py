"""Personal / job appeals and over-generous weak reactions (live-interview naturalness patch).

Covers: the separate `appeal` signal, humane + bounded reactions, no hiring promises or rejections, repeated pleading,
appeal mixed with a technical answer, and that normal answers behave exactly as before. Embeddings are REAL; providers are scripted.
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
Q = "What are neural networks?"
T = "neural networks"
REASSURING = re.compile(r"no problem|that'?s fine|no worries|don'?t worry", re.I)
STALE = re.compile(r"set that aside|let'?s move on|move on to another", re.I)
POSITIVE = re.compile(r"part of it|right track|partly there|good\b|solid|reasonable|well done|great", re.I)
# anything that promises / predicts / rejects / claims hiring authority
OUTCOME_CLAIM = re.compile(r"you(?:'ll| will| won'?t| will not| might| may)\s+(?:get|be|pass|fail|receive|hear)|i(?:'ll| will| can| could)\s+(?:give|hire|offer|select|recommend|promise)|"
                           r"\brejected?\b|\bhired\b|\bselected\b|\bpass(?:ed)?\b|\bfail(?:ed)?\b|\bscor\w+|\bgrade\b|\bguarantee|\bpromise\b|\bunfortunately\b|\bsorry\b", re.I)
HOSTILE = re.compile(r"unacceptable|ridiculous|pathetic|stupid|lazy|shame|embarrass|desperate|beg(?:ging)?\b|inappropriate|manipulat", re.I)


def cls(a, q=Q, t=T): return ie.classify_answer(a, q, t, embed)


print("== the appeal signal (separate from technical quality and behaviour) ==")
APPEALS = ["Hey sir please give me the job my mom is sick i need money, i can do work even tho i dont have theoretical knowledge",
           "please give me the job", "I really need this job", "my family needs money", "please hire me", "I need this opportunity",
           "I am very poor and jobless, please select me", "my father is in hospital so I badly need the salary", "just give me a chance sir, I will work hard",
           "Our family is struggling, I desperately need this job to support my kids", "my mother is ill and our family depends on this job"]
for a in APPEALS:
    check(f"appeal detected: {a[:60]!r}", sig.detect_behavior(a).appeal, sig.detect_behavior(a).signals)
NORMAL = ["Neural networks are layers of connected neurons that learn weights through backpropagation.", "Can you give me an example of backpropagation?",
          "let me explain: gradient descent updates weights using the loss gradient", "I need a position encoding for the transformer so the model knows token order",
          "my model needs more data to avoid overfitting", "The role of the activation function is to add non-linearity", "I don't know", "idk what that is",
          "haah easy question..supervised learning is a type of learning", "whatever", "im going to get a job wow", "I worked on a family of models for a job scheduler"]
for a in NORMAL:
    check(f"NOT an appeal: {a[:60]!r}", not sig.detect_behavior(a).appeal, sig.detect_behavior(a).signals)
b = sig.detect_behavior("please hire me, my mom is sick")
check("an appeal is not misconduct: behaviour stays 'normal' (not unprofessional/dismissive), no 'confident' flag", (b.label, b.confident) == ("normal", False), b)
check("'im going to get a job wow' is still boasting/unprofessional (unchanged), and not an appeal", sig.detect_behavior("im going to get a job wow").label == "unprofessional" and not sig.detect_behavior("im going to get a job wow").appeal)
check("appeal clauses are removed before judging technical content, the technical clause stays",
      "neural" in sig.strip_markers("Neural networks are layers of neurons. Please give me the job, my mom is sick i need money").lower()
      and "mom" not in sig.strip_markers("Neural networks are layers of neurons. Please give me the job, my mom is sick i need money").lower())

print("== technical quality is judged on the technical part only ==")
a1 = cls("Hey sir please give me the job my mom is sick i need money, i can do work even tho i dont have theoretical knowledge")
check("the real pleading message: appeal + a poor technical result (no content), nothing 'partial'/'strong'", a1.appeal and a1.label in ("weak", "irrelevant"), (a1.label, a1.signals))
a2 = cls("Neural networks are type of network which can transfer datas\nHey sir please give me the job my mom is sick i need money,")
check("REGRESSION: vague answer + pleading was inflated to 'partial' by the appeal words; now VAGUE", a2.label == "vague" and a2.appeal, (a2.label, a2.signals))
a3 = cls("Neural networks are just a type of network. Please give me the job, I really need money.")
check("the example from the brief: vague + appeal", a3.label == "vague" and a3.appeal, (a3.label, a3.signals))
a3b = cls("Neural networks are type of network which can transfer datas")
check("the same vague sentence without any appeal is vague too", a3b.label == "vague" and not a3b.appeal)
good = "A neural network is made of layers of connected neurons that apply weights and activation functions, and it learns by adjusting the weights with backpropagation."
a4 = cls(good + " Please give me the job.")
check("a good technical answer + an appeal is NOT downgraded (same label as without the appeal)", a4.appeal and a4.label == cls(good).label and a4.label in ("partial", "strong"), (a4.label, cls(good).label))
check("appeal-only message -> weak/irrelevant, never positive", cls("please hire me i really need this opportunity").label in ("weak", "irrelevant"))
check("'I dont have theoretical knowledge' style admissions count as an honest non-answer", cls("sorry i dont have any theoretical knowledge").label == "weak")

print("== reaction wording ==")
ALLTXT = [(k, f"{lead}, {close}" if k == "first" else f"{lead} {close}") for k, items in (("first", style.APPEAL_FIRST), ("repeat", style.APPEAL_REPEAT)) for lead, close in items]
check("several controlled variations exist for the first appeal and for repeats", len(style.APPEAL_FIRST) >= 3 and len(style.APPEAL_REPEAT) >= 3)
check("first-appeal wording acknowledges, then states the boundary (only interview responses are assessed)", all(re.search(r"understand|appreciate|hear", t, re.I) and re.search(r"assess", t, re.I) and re.search(r"interview responses|technical responses", t) for k, t in ALLTXT if k == "first"))
check("the brief's own wording is available", any(t.startswith("I understand this is important to you. However, I can only assess you based on your interview responses") for _k, t in ALLTXT))
check("repeat wording is firmer, still kind, and says to focus on the technical questions", all(re.search(r"focus|keep to", t, re.I) and re.search(r"understand", t, re.I) and re.search(r"personal circumstances|assessment can only", t) for k, t in ALLTXT if k == "repeat"))
check("NO promises, predictions, rejections, scores or hiring authority in any appeal wording", all(not OUTCOME_CLAIM.search(t) for _k, t in ALLTXT), [t for _k, t in ALLTXT if OUTCOME_CLAIM.search(t)])
check("never hostile, shaming or exclamatory; no question mark (the model's question is the only one)", all(not HOSTILE.search(t) and "?" not in t and "!" not in t for _k, t in ALLTXT))
bad = []
for label in ("strong", "partial", "vague", "incorrect", "irrelevant", "weak"):
    for follow in (True, False):
        for count in (1, 2):
            kind, text = style.appeal_reaction(label, follow, count, 3, [])
            ok = (kind == ("appeal_repeat" if count >= 2 else "appeal_first") and "?" not in text and "!" not in text and not OUTCOME_CLAIM.search(text)
                  and not HOSTILE.search(text) and not REASSURING.search(text) and not STALE.search(text) and not POSITIVE.search(text))
            if not ok: bad.append((label, follow, count, kind, text))
check("every label x step x count combination (24) is clean: no reassurance, no 'move on' boilerplate, no praise, no outcome claims, no '?'/'!'", not bad, bad[:2])
check("technical remark follows the boundary: vague -> 'too vague', incorrect -> 'wasn't accurate', partial -> 'incomplete'",
      "too vague" in style.appeal_reaction("vague", True, 1, 0, [])[1] and "wasn't accurate" in style.appeal_reaction("incorrect", True, 1, 0, [])[1]
      and "incomplete" in style.appeal_reaction("partial", True, 1, 0, [])[1])
check("a strong answer with an appeal gets only the boundary (no remark on the answer)", style.appeal_reaction("strong", True, 1, 0, [])[1].endswith("discussion.") or style.appeal_reaction("strong", True, 1, 0, [])[1].endswith("questions."))
firsts = {style.appeal_reaction("vague", True, 1, r, [])[1].split(".")[0] for r in range(8)}
check("wording rotates across the interview (>=3 distinct acknowledgements)", len(firsts) >= 3, firsts)
used = style.appeal_reaction("vague", True, 1, 0, [])[1]
check("a boundary sentence shown in the last turns is not repeated straight away", style.appeal_reaction("vague", True, 1, 0, [used])[1].split(".")[0] != used.split(".")[0])

print("== through the interviewer pipeline (planner + composed message) ==")
def fresh(**kw): return ie.InterviewState(difficulty="beginner", current_topic=T, asked_topics=[T], deep_probed_topics=[T], **kw)
class Script:
    def __init__(self, *r): self.r, self.calls = list(r), []
    def __call__(self, messages, schema):
        self.calls.append(messages); return LLMResult(self.r[min(len(self.calls) - 1, len(self.r) - 1)], "openrouter", "m")
def ctx_for(answer, state=None, answer_number=3, prev=None):
    q = cls(answer)
    st, plan = ie.plan_next_step(state or fresh(), q, "ai_ml", answer_number=answer_number)
    ref = RetrievedChunk(text="Neural networks are built from layers of connected units.", score=0.6, chunk_id="c1")
    return ie.TurnContext(target_role="AI / Machine Learning Role", domain="ai_ml", profile=PROFILE, answer_number=answer_number, max_answers=10, state=st, plan=plan,
                          quality=q, turns=[(Q, "x")], previous_questions=prev or [Q], reference=ref, latest_answer=answer, candidate_name="Gautham")
def J(question, ctx):
    stay = ctx.plan.step in ("follow_up", "redirect")
    return json.dumps({"action": "ask_question", "topic": ctx.plan.topic if stay else "gradient descent", "difficulty": ctx.plan.difficulty, "follow_up": stay, "question": question})
def one_q(text): return sum(1 for s in ie.split_sentences(text) if ie.is_interrogative(s)) == 1 and text.count("?") == 1

c = ctx_for("Neural networks are just a type of network. Please give me the job, I really need money.")
o = ie.generate_interviewer_turn(c, Script(J("What is the basic structure of a neural network?", c)), embed)
print("   appeal + vague   ->", o.question)
check("appeal + vague answer: boundary + 'too vague' + ONE technical question, on the same topic", o.preamble_kind == "appeal_first" and "too vague" in o.question and one_q(o.question) and c.plan.step == "follow_up", (c.plan.step, o.question))
check("...no reassurance, no 'set that aside / move on' boilerplate, no outcome claims, no hostility", not REASSURING.search(o.question) and not STALE.search(o.question) and not OUTCOME_CLAIM.search(o.question) and not HOSTILE.search(o.question))
check("...it does not reveal the correct answer (the preamble is backend text only)", o.preamble in o.question and "layers" not in o.preamble.lower() and o.core_question == "What is the basic structure of a neural network?")
check("...the metadata records the appeal, the kind and the preamble", (lambda m: m["appeal"] is True and m["reaction_kind"] == "appeal_first" and m["preamble"] == o.preamble and "appeal:job_request" in m["behavior_signals"])(o.meta(c)), o.meta(c))

c = ctx_for("Hey sir please give me the job my mom is sick i need money, i can do work even tho i dont have theoretical knowledge")
o = ie.generate_interviewer_turn(c, Script(J("Can you describe, in simple terms, what a neural network is?", c)), embed)
print("   the real message ->", o.question)
check("the real pleading message is no longer 'Let's set that aside and move on to another topic.'", not STALE.search(o.question) and o.preamble_kind == "appeal_first" and one_q(o.question), o.question)
check("...it stays on the topic with a calm remark (the candidate gets one more chance), and the interview continues", c.plan.step in ("follow_up", "redirect") and not REASSURING.search(o.question))

st1 = ie.record_question(c.state, c.plan, "neural networks", None, preamble=o.preamble)
check("the appeal is counted in the persisted state", c.state.appeal_count == 1 and st1.appeal_count == 1)
c2 = ctx_for("please hire me i really need this opportunity, my family needs money", state=st1, answer_number=4)
o2 = ie.generate_interviewer_turn(c2, Script(J("How does gradient descent update the weights of a model?", c2)), embed)
print("   repeated pleading ->", o2.question)
check("REPEATED pleading gets the firmer wording and the interview still continues", o2.preamble_kind == "appeal_repeat" and c2.plan.appeal == 2 and re.search(r"focus|keep to", o2.question) and one_q(o2.question), o2.question)
check("...firmer, but still no hostility / outcome claim / termination", not HOSTILE.search(o2.question) and not OUTCOME_CLAIM.search(o2.question) and o2.action == "ask_question")
check("...and the boundary sentence is not the one just used", o2.preamble.split(".")[0] != o.preamble.split(".")[0])
c3 = ctx_for(good + " Please give me the job.", answer_number=3)
o3 = ie.generate_interviewer_turn(c3, Script(J("How does a neural network learn its weights?", c3)), embed)
print("   appeal + decent  ->", o3.question)
check("appeal + a decent technical answer: the remark follows the answer's own label (same as without the appeal), nothing harsher",
      o3.preamble_kind == "appeal_first" and c3.quality.label == cls(good).label and not re.search(r"vague|accurate|refocus|question itself", o3.preamble)
      and (("incomplete" in o3.preamble) == (c3.quality.label == "partial")) and one_q(o3.question), (c3.quality.label, o3.question))

src = open(os.path.join(BACKEND, "main.py"), encoding="utf-8").read()
phrases = next(ast.literal_eval(n.value) for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "_RESIGNATION_PHRASES")
rx = re.compile(r"(?<![a-z0-9])(?:" + "|".join(re.escape(p) for p in phrases) + r")(?![a-z0-9])")
norm = lambda t: re.sub(r"[^a-z0-9]+", " ", t.lower().replace("’", "'").replace("'", "")).strip()
check("no appeal message is treated as a resignation (the interview is never ended by pleading)", all(not rx.search(norm(a)) for a in APPEALS), [a for a in APPEALS if rx.search(norm(a))])

print("== normal answers are unaffected ==")
cases = [("Neural networks are layers of connected neurons that learn weights through backpropagation.", "partial", "strong"), ("I don't know", "weak", "weak"),
         ("my cat likes the sofa", "irrelevant", "irrelevant"), ("neural networks are a type of network", "vague", "vague")]
for ans, *labels in cases:
    q = cls(ans)
    check(f"{ans[:50]!r}: no appeal, label {q.label}", not q.appeal and q.label in labels, q.label)
st, plan = ie.plan_next_step(fresh(), cls("neural networks are a type of network"), "ai_ml", answer_number=3)
check("a normal vague answer has no appeal count and keeps the ordinary 'vague_follow' reaction", plan.appeal == 0 and st.appeal_count == 0
      and style.reaction_kind(plan.step, "vague", "normal") == "vague_follow")
c = ctx_for("neural networks are a type of network")
o = ie.generate_interviewer_turn(c, Script(J("What do the layers of a neural network do?", c)), embed)
print("   plain vague      ->", o.question)
check("plain vague answers: neutral correction, no positive acknowledgement, unchanged wording family", o.preamble in style.TRANSITIONS["vague_follow"] and not POSITIVE.search(o.preamble) and not REASSURING.search(o.question))
for k in ("vague_follow", "vague_new", "incorrect_follow", "incorrect_new", "irrelevant_new", "redirect"):
    check(f"{k}: none of its wordings is positive or reassuring", all(not POSITIVE.search(t) and not REASSURING.search(t) for t in style.TRANSITIONS[k]))
check("older stored interview states (no appeal_count) still load", ie.load_state(json.dumps({"version": ie.STATE_VERSION, "difficulty": "beginner"}), PROFILE).appeal_count == 0)

finish()
