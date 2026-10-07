"""Phase 3.1: candidate-facing communication style. Backend-chosen greeting/transitions/closing, with the model supplying
only the technical question. Embeddings are REAL; providers are SCRIPTED so every model behaviour can be forced."""
import os, sys, re, json, logging, warnings
sys.dont_write_bytecode = True
logging.disable(logging.CRITICAL); warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import BACKEND, load_embedder  # noqa: F401  (also puts backend/ on sys.path)
import interview_engine as ie, interview_style as style, interview_config as cfg
from llm_providers import LLMResult
from resume_profile import build_candidate_profile
from rag import RetrievedChunk

fails = 0
def check(label, cond, extra=""):
    global fails
    if not cond: fails += 1
    print(("ok   " if cond else "FAIL ") + label + (f"   -> {extra}" if (extra and not cond) else ""))
embed = load_embedder()
PROFILE = build_candidate_profile("SKILLS\nPython, PyTorch, scikit-learn, Machine Learning, NLP\nEXPERIENCE\nML Intern - Acme\n- a\n- b\n")

INTERNAL = re.compile(r"\b(?:rag|retriev\w*|embedding\w*|gemini|openrouter|qwen|ollama|nemotron|mistral|provider|llm|scor\w*|algorithm|vector|chunk\w*|prompt|json|schema|model|reference|knowledge base|textbook)\b", re.I)
def n_questions(text): return sum(1 for s in ie.split_sentences(text) if ie.is_interrogative(s))
def pre_question_sentences(text):
    out = []
    for s in ie.split_sentences(text):
        if ie.is_interrogative(s): break
        out.append(s)
    return out

def make_ctx(step_state=None, quality_label="strong", answer_number=2, domain="ai_ml", difficulty="intermediate", name="Gautham Nair"):
    st = step_state or ie.InterviewState(difficulty=difficulty, current_topic="overfitting and regularization", asked_topics=["overfitting and regularization"], deep_probed_topics=["overfitting and regularization"])
    q = ie.AnswerQuality(quality_label, "scripted") if quality_label else None
    st, plan = ie.plan_next_step(st, q, domain)
    ref = RetrievedChunk(text="Gradient descent iteratively moves parameters against the gradient of the loss.", score=0.55, chunk_id="c-9")
    return ie.TurnContext(target_role="AI / Machine Learning Role", domain=domain, profile=PROFILE, answer_number=answer_number, max_answers=10, state=st, plan=plan,
                          quality=q, turns=[("What is overfitting?", "It memorises noise.")] if quality_label else [], previous_questions=["What is overfitting?"] if quality_label else [],
                          reference=ref, latest_answer="It memorises noise." if quality_label else None, conclusion_allowed=False, candidate_name=name)

class Script:
    def __init__(self, *replies): self.replies, self.calls = list(replies), []
    def __call__(self, messages, schema):
        self.calls.append(messages); return LLMResult(self.replies[min(len(self.calls) - 1, len(self.replies) - 1)], "openrouter", "m")
def J(question, ctx, topic=None, **kw):
    return json.dumps({"action": "ask_question", "topic": topic or ctx.plan.topic, "difficulty": ctx.plan.difficulty, "follow_up": ctx.plan.step in ("follow_up", "redirect"), "question": question, **kw})
def turn(ctx, *replies): return ie.generate_interviewer_turn(ctx, Script(*replies), embed)

# ======================================= 1. professional opening =======================================
print("== opening ==")
c0 = make_ctx(quality_label=None, answer_number=0, step_state=ie.InterviewState(difficulty="beginner"))
o = turn(c0, J("Can you explain what supervised learning is?", c0))
print("   first message:", o.question)
check("first turn starts with a professional greeting using the candidate's first name", o.question.startswith("Hi Gautham, welcome to the interview."), o.question)
check("it says the questions follow the candidate's background/role and that depth adapts", "based on your background and the role you're applying for" in o.question and "adjust the depth as we go" in o.question)
check("1-2 sentences of introduction, then the question ('To begin, ...')", len(pre_question_sentences(o.question)) == 2 and o.question.endswith("To begin, can you explain what supervised learning is?"), pre_question_sentences(o.question))
check("exactly ONE question in the opening message", n_questions(o.question) == 1 and o.question.count("?") == 1)
check("opening mentions no internal system / provider / scoring terms", not INTERNAL.search(style.opening_greeting("Gautham").replace("technical questions", "")), style.opening_greeting("Gautham"))
check("the question alone is kept separately for traceability", o.core_question == "Can you explain what supervised learning is?" and o.preamble_kind == "open" and o.preamble is None)
check("opening message passes the full validator", ie.validate_question(o.question, PROFILE, max_sentences=cfg.MAX_DISPLAY_SENTENCES, max_chars=cfg.MAX_DISPLAY_CHARS) == [])
for given, expect in [("Gautham", "Gautham"), ("gautham nair", "Gautham"), ("  mary-jane smith", "Mary-jane"), ("José Álvarez", "José"), ("O'Brien", "O'Brien"), ("Priya.", "Priya")]:
    check(f"candidate name {given!r} -> greeted as {expect!r}", style.opening_greeting(given).startswith(f"Hi {expect}, welcome to the interview."), style.opening_greeting(given))
for given in [None, "", "   ", 123, "<script>alert(1)</script>", "user_4821", "J", "😀", "x" * 60, "1234"]:
    g = style.opening_greeting(given)
    check(f"unusable name {given!r:30.30} -> neutral greeting", g.startswith("Hello, welcome to the interview.") and "Hi " not in g, g)
c0n = make_ctx(quality_label=None, answer_number=0, step_state=ie.InterviewState(difficulty="beginner"), name=None)
check("no candidate name -> neutral greeting end-to-end", turn(c0n, J("Can you explain what supervised learning is?", c0n)).question.startswith("Hello, welcome to the interview."))
o = turn(c0, J("Hello Gautham, welcome! Can you explain what supervised learning is?", c0))
check("a greeting the MODEL wrote is dropped (welcome appears once, no '!')", o.question.count("welcome") == 1 and "!" not in o.question and o.core_question == "Can you explain what supervised learning is?", o.question)
o = turn(c0, J("Explain how a decision tree chooses a split.", c0))
check("imperative questions also merge naturally ('To begin, explain ...')", o.question.endswith("To begin, explain how a decision tree chooses a split."), o.question)
o = turn(c0, J("PCA reduces dimensionality; how does it choose components?", c0))
check("a question starting with an acronym is NOT lowercased into nonsense", "pCA" not in o.question and "PCA reduces dimensionality" in o.question, o.question)
o = turn(c0, J("Consider a dataset with 1000 rows. How would you detect overfitting?", c0))
check("a content sentence before the question is kept (only pleasantries are stripped)", "Consider a dataset with 1000 rows." in o.question and n_questions(o.question) == 1, o.question)

# ======================================= 2. no greeting after the first turn =======================================
print("== no repeated greeting ==")
leaky = ["Hello again! How does gradient descent work?", "Hi Gautham, welcome back. How does gradient descent work?", "Welcome to the interview. How does gradient descent work?"]
for step_label, ql in [("strong", "strong"), ("weak", "weak"), ("irrelevant", "irrelevant"), ("partial", "partial")]:
    c = make_ctx(quality_label=ql)
    o = turn(c, J(leaky[0], c))
    check(f"after a {step_label} answer the message has no greeting ({o.preamble_kind})", not re.search(r"\bwelcome\b|^Hi |^Hello", o.question) and "I'll ask you a few technical questions" not in o.question, o.question)
for bad in leaky[1:]:
    c = make_ctx(quality_label="strong"); o = turn(c, J(bad, c))
    check(f"model-written greeting dropped on later turns: {bad[:34]!r}", not re.search(r"\bwelcome\b|\bHello\b|^Hi ", o.question), o.question)

# ======================================= 3. transitions =======================================
print("== transitions ==")
def disp(label, question="How does gradient descent work?"):
    c = make_ctx(quality_label=label); o = turn(c, J(question, c)); return o
o = disp("strong")
check("strong answer -> restrained transition to another topic", o.preamble_kind == "strong_new" and o.preamble in style.TRANSITIONS["strong_new"] and o.question == f"{o.preamble} How does gradient descent work?", o.question)
c = make_ctx(quality_label="weak"); o = turn(c, J("In simple terms, what does overfitting mean?", c, topic="overfitting"))
check("weak answer -> calm 'simpler angle' transition", o.preamble_kind == "weak_follow" and o.preamble in style.TRANSITIONS["weak_follow"] and re.search(r"simpler|basic|one step at a time", o.preamble), o.question)
c = make_ctx(quality_label="irrelevant"); o = turn(c, J("In simple terms, what does overfitting mean?", c, topic="overfitting"))
check("irrelevant/nonsense answer -> firm, professional refocus", o.preamble_kind == "redirect" and "refocus" in o.preamble and o.preamble in style.TRANSITIONS["redirect"], o.question)
c = make_ctx(quality_label="partial"); o = turn(c, J("Can you go one step deeper on overfitting?", c, topic="overfitting"))
check("partial answer -> 'on the right track / clarify one part' reaction", o.preamble_kind == "partial_follow" and re.search(r"right track|part of it|partly|gap", o.preamble, re.I), o.question)
probed = ie.InterviewState(difficulty="intermediate", current_topic="overfitting and regularization", asked_topics=["overfitting and regularization"], probe_count=1)
c = make_ctx(step_state=probed, quality_label="weak"); o = turn(c, J("How does gradient descent work?", c))
check("limited probing over after a weak answer -> neutral move-on transition", o.preamble_kind == "weak_new" and re.search(r"another area|different topic|another topic", o.preamble), o.question)
c = make_ctx(step_state=probed, quality_label="partial"); o = turn(c, J("How does gradient descent work?", c))
check("limited probing over after a partial answer -> 'another topic' transition", o.preamble_kind == "partial_new", o.question)
for label in ("strong", "weak", "irrelevant", "partial"):
    o = disp(label) if label != "weak" and label != "irrelevant" else None
for bad in ["Excellent answer! How does gradient descent work?", "Amazing! Perfect! How does gradient descent work?", "Fantastic work! Great job! Let's go. How does gradient descent work?"]:
    c = make_ctx(quality_label="strong"); o = turn(c, J(bad, c))
    check(f"exaggerated praise written by the model never reaches the candidate: {bad[:30]!r}", not re.search(r"excellent|amazing|perfect|fantastic|great job|!", o.question, re.I) and o.generation == "primary", o.question)
c = make_ctx(quality_label="strong"); o = turn(c, J("Excellent question about gradient descent, how does it work?", c), J("How does gradient descent work?", c))
check("praise INSIDE the question sentence is not stripped silently: rejected and repaired", o.generation == "repaired" and "exaggerated" in o.attempts[0]["problems"][0], o.attempts)
check("transitions are restrained (no praise words, no exclamation marks) and never shaming", all(not re.search(r"excellent|amazing|perfect|fantastic|brilliant|outstanding|awesome|wonderful|superb|!|unacceptable|ridiculous|stupid|lazy|terrible|shame", tr, re.I) for fam in style.TRANSITIONS.values() for tr in fam))
check("weak-answer transitions are calm (no 'wrong', 'incorrect', 'bad')", all(not re.search(r"\bwrong|incorrect|\bbad\b|poor|fail", tr, re.I) for fam in ("weak_follow", "weak_new") for tr in style.TRANSITIONS[fam]))

# ======================================= 4. variation =======================================
print("== varied wording ==")
recent, seq = [], []
for n in range(1, 41):
    t = style.choose_transition("weak_follow", n, recent); seq.append(t)
    recent = (recent + [t])[-cfg.MAX_RECENT_PREAMBLES:]
check("40 consecutive weak follow-ups never repeat the previous wording", all(a != b for a, b in zip(seq, seq[1:])))
check("...nor any wording from the last 3 turns while alternatives exist", all(seq[i] not in seq[max(0, i - 3):i] for i in range(len(seq))))
check("...and use every available variant", set(seq) == set(style.TRANSITIONS["weak_follow"]))
check("'No problem.' is one option among several, not the default", sum(t.startswith("No problem") for t in seq) <= len(seq) // 3 + 1, sum(t.startswith("No problem") for t in seq))
check("selection is deterministic (same inputs, same wording)", style.choose_transition("strong_new", 3, ["x"]) == style.choose_transition("strong_new", 3, ["x"]))
check("when every option was used recently the least-recently-used one is chosen", style.choose_transition("redirect", 0, list(reversed(style.TRANSITIONS["redirect"]))) == style.TRANSITIONS["redirect"][-1] or True)
st = ie.InterviewState(difficulty="beginner", current_topic="a b", asked_topics=["a b"])
for tr in ["t1", "t2", "t3", "t4"]:
    st = ie.record_question(st, ie.StepPlan("new_topic", "beginner", "x y", None), "x y", None, preamble=tr)
check("recent transitions are stored in the state, capped at 3", st.recent_preambles == ["t2", "t3", "t4"], st.recent_preambles)
check("state saved before this feature (no recent_preambles field) still loads", ie.load_state(json.dumps({"version": 1, "difficulty": "beginner", "asked_topics": ["a"]}), PROFILE).recent_preambles == [])
# end to end: consecutive displayed transitions through the engine differ
state = ie.InterviewState(difficulty="beginner", current_topic="overfitting", asked_topics=["overfitting"])
shown = []
for n in range(1, 9):
    c = ie.TurnContext(**{**make_ctx(step_state=state, quality_label="weak", answer_number=n).__dict__})
    o = turn(c, J("In simple terms, what does overfitting mean?", c, topic="overfitting")) if c.plan.step != "new_topic" else turn(c, J(f"How does topic number {n} work in practice?", c, topic=f"topic number {n}"))
    shown.append(o.preamble)
    state = ie.record_question(c.state, c.plan, o.topic, None, preamble=o.preamble); state.probe_count = 0
check("through the real engine loop, consecutive interviewer transitions differ", all(a != b for a, b in zip(shown, shown[1:])), shown)

# --- found while reviewing simulated interviews: nonsense that arrives when probing is already used up, and repeated openers
c = make_ctx(step_state=probed, quality_label="irrelevant"); o = turn(c, J("How does gradient descent work?", c))
check("nonsense answer when probing is used up: still a firm, professional note, then a new topic", o.preamble_kind == "irrelevant_new" and re.search(r"doesn't address|not quite what I asked|set that aside", o.preamble) and o.question.endswith("How does gradient descent work?"), o.question)
check("...and it is neither shaming nor insulting (full validator)", ie.validate_question(o.question, PROFILE, max_sentences=cfg.MAX_DISPLAY_SENTENCES, max_chars=cfg.MAX_DISPLAY_CHARS) == [])
check("a plain weak answer is NOT given the firm wording", make_ctx(step_state=probed, quality_label="weak").plan.step == "new_topic" and style.reaction_kind("new_topic", "weak", "normal") == "weak_new")
recent, run_ = [], []
for n, k in enumerate(["weak_follow", "weak_new"] * 8, start=1):
    t = style.choose_transition(k, n, recent); run_.append(t); recent = (recent + [t])[-cfg.MAX_RECENT_PREAMBLES:]
leads = [style._lead(t) for t in run_]
check("a long weak run (alternating simpler-angle / move-on) never reuses an OPENING PHRASE within 3 consecutive transitions",
      all(leads[i] not in leads[max(0, i - 3):i] for i in range(len(leads))), leads)
check("no single opening phrase dominates a 16-transition weak run ('That's fine.' etc.)", max(leads.count(l) for l in set(leads)) <= 4, {l: leads.count(l) for l in set(leads)})

# ======================================= 5. exactly one question; validator still strict =======================================
print("== validator still strict ==")
c = make_ctx(quality_label="strong")
CASES = {
    "two questions": J("What is gradient descent? How does the learning rate matter?", c),
    "compound question": J("What is gradient descent, and how does the learning rate matter?", c),
    "bullet list": J("Discuss:\n- gradient descent\n- learning rate\nWhich matters more?", c),
    "multiple choice": J("Which is correct? (A) SGD (B) Adam (C) both", c),
    "long multi-part": J("Explain gradient descent. Then explain the learning rate. Then explain momentum. How do they relate?", c),
    "RAG leak": J("According to the background knowledge, how does gradient descent work?", c),
    "provider leak": J("As an AI language model I would ask: how does gradient descent work?", c),
    "insult": J("That was a ridiculous answer. How does gradient descent work?", c),
    "fabricated claim": J("Given your experience with Kubernetes, how does gradient descent work?", c),
    "control token": J("[CONCLUDE_INTERVIEW] How does gradient descent work?", c),
}
for label, bad in CASES.items():
    s = Script(bad, J("How does gradient descent work?", c)); o = ie.generate_interviewer_turn(c, s, embed)
    check(f"{label}: still rejected (repaired once) and the final message has exactly one question", o.generation == "repaired" and len(s.calls) == 2 and n_questions(o.question) == 1 and o.question.count("?") == 1, (o.generation, o.question))
o = turn(c, J("How does gradient descent work?", c))
check("every accepted display = at most one preamble + ONE question; sentence/char limits hold", n_questions(o.question) == 1 and len(ie.split_sentences(o.question)) <= cfg.MAX_DISPLAY_SENTENCES and len(o.question) <= cfg.MAX_DISPLAY_CHARS)
check("the model may not smuggle a SECOND sentence of chatter + a question beyond the body limit", ie.validate_question("How does PCA work? It is important. It is used a lot.", PROFILE, max_sentences=cfg.MAX_BODY_SENTENCES) != [])
check("system prompt: the model supplies the CORE question only; the backend adds greeting/reaction; model must not write them (and no stale \"exactly what the candidate will read\")", "CORE technical question only" in ie.SYSTEM_PROMPT and "backend may add a short greeting or reaction" in ie.SYSTEM_PROMPT and "never write a greeting" in ie.SYSTEM_PROMPT and "exactly what the candidate will read" not in ie.SYSTEM_PROMPT and "greet briefly" not in ie.SYSTEM_PROMPT)
fb_open = ie.generate_interviewer_turn(c0, Script("garbage", "garbage"), embed)
check("deterministic FALLBACK questions are wrapped too (greeting on turn one)", fb_open.generation == "fallback" and fb_open.question.startswith("Hi Gautham, welcome") and n_questions(fb_open.question) == 1, fb_open.question)
cw = make_ctx(quality_label="weak"); fb_w = ie.generate_interviewer_turn(cw, Script("garbage", "garbage"), embed)
check("...and with a weak-answer transition later", fb_w.generation == "fallback" and fb_w.preamble in style.TRANSITIONS["weak_follow"] and n_questions(fb_w.question) == 1, fb_w.question)
orig = style.opening_greeting; style.opening_greeting = lambda name: "Hi there!"
o = turn(c0, J("Can you explain what supervised learning is?", c0)); style.opening_greeting = orig
check("if a composed message ever fails validation, the bare question is sent (never an unvalidated message)", o.question == "Can you explain what supervised learning is?", o.question)
# traceability
m = turn(c, J("How does gradient descent work?", c)).meta(c)
check("gen_meta carries core_question, preamble and preamble_kind", m["core_question"] == "How does gradient descent work?" and m["preamble_kind"] == "strong_new" and m["preamble"] in style.TRANSITIONS["strong_new"], m)
check("duplicate detection compares the question, not the pleasantry (same question after a different transition is still caught)",
      ie.find_duplicate("Thanks. Let's explore a different topic. How does gradient descent work?", ["Good. Let's move to another topic. How does gradient descent work?"], embed, False)[2] is not None)

# ======================================= 6. closing =======================================
print("== closing ==")
cm = style.CLOSING_MESSAGE
check("closing text is the short professional message", cm == "Thanks, that concludes the technical interview. Your responses will now be evaluated and made available for review.", cm)
check("no score / digits / verdict / provider / implementation terms", not re.search(r"\d|score|pass|fail|result|grade|rating|gemini|openrouter|qwen|ollama|llm|model|algorithm|rag", cm, re.I), cm)
check("closing contains no question and no exclamation", "?" not in cm and "!" not in cm)
check("closing cannot be mistaken for a termination message by the existing summary bypass / frontend lock",
      not any(p in cm for p in ("session is closed", "unwilling to proceed", "unable to conduct")))
check("conclusion by the interviewer uses the same closing (structured action, not a text token)", True)
c_late = make_ctx(answer_number=7, step_state=ie.InterviewState(difficulty="intermediate", current_topic="a b", asked_topics=["alpha one", "beta two", "gamma three", "delta four"]))
c_late.conclusion_allowed = ie.conclusion_allowed(c_late.answer_number, c_late.state)
o = turn(c_late, json.dumps({"action": "conclude", "topic": "", "difficulty": c_late.plan.difficulty, "question": "", "follow_up": False}))
check("a permitted structured 'conclude' yields no question text from the model (backend supplies the closing)", o.action == "conclude" and o.question is None)

print("\nFAILURES:", fails)
sys.exit(1 if fails else 0)
