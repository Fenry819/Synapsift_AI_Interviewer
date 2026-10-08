"""Phase 4: evaluation pairing, report schema, wording guard, cache/versioning. Standard library only: the evaluator
provider is a scripted function, nothing is sent to Gemini or Ollama and no database is touched."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import check, finish  # noqa: E402
import evaluation as ev  # noqa: E402
from interview_style import CLOSING_MESSAGE  # noqa: E402

RESIGN = "If you are unwilling to proceed with the technical questions, we will conclude the assessment here. Thank you for your time."
REJECT = "Thank you for your interest. We are unable to conduct a technical screening for this specific domain."


def ai(core, display=None, topic="overfitting", provider="local", behavior=None, quality=None, **extra):
    meta = {"action": "ask_question", "core_question": core, "topic": topic, "difficulty": "intermediate", "preamble": None,
            "answer_quality": quality, "answer_behavior": behavior, "behavior_signals": extra.pop("signals", [])}
    meta.update(extra)
    return ("ai", display or core, provider, json.dumps(meta))

def cand(text): return ("candidate", text, None, None)
def system(text): return ("ai", text, "system", None)

def good_model(n, types=None, scores=None, **over):
    types = types or ["partial"] * n
    scores = scores or [50] * n
    out = {"summary": "The candidate gave partial answers on the topics assessed.", "strengths": ["Explained the basic idea"],
           "weaknesses": ["Details were incomplete"],
           "breakdown": [{"index": i + 1, "answerType": types[i], "score": scores[i], "feedback": f"Feedback {i + 1} based on the answer."} for i in range(n)]}
    out.update(over)
    return out

def run(rows, model):
    t = ev.build_transcript(rows)
    return ev.evaluate_transcript(t, lambda prompt: json.dumps(model))

# ---------------------------------------------------------------- pairing
print("== pairing: core_question preferred, legacy fallback, system rows excluded ==")
rows = [ai("What is overfitting?", display="Hi Gautham, welcome to the interview. I'll ask a few questions. To begin, what is overfitting?"),
        cand("It memorises noise."),
        ai("How does regularization help?", display="Thanks. Let's move on. How does regularization help?", topic="regularization"),
        cand("It penalises weights."),
        system(CLOSING_MESSAGE)]
t = ev.build_transcript(rows)
check("two pairs, closing row excluded", len(t.pairs) == 2 and not t.terminated and not t.rejected, [p.question for p in t.pairs])
check("the scored question is gen_meta.core_question, not the greeting + question display text",
      [p.question for p in t.pairs] == ["What is overfitting?", "How does regularization help?"] and all(p.question_source == "core_question" for p in t.pairs))
check("no greeting / transition text reaches the evaluator prompt", "welcome to the interview" not in ev.build_prompt(t.pairs) and "Let's move on" not in ev.build_prompt(t.pairs))
check("exact core question preserved in the final breakdown", [b["question"] for b in run(rows, good_model(2))["breakdown"]] == ["What is overfitting?", "How does regularization help?"])

legacy = [("ai", "What is a decision tree?", "openrouter", None), cand("A tree of splits."),
          ("ai", "Explain bagging.", None, None), cand("Averaging models."),
          ("ai", "Thanks. Now: what is boosting?", "local", json.dumps({"action": "ask_question", "preamble": "Thanks. Now:"})), cand("Sequential learners.")]
t = ev.build_transcript(legacy)
check("legacy rows without gen_meta fall back to the display text", [p.question for p in t.pairs][:2] == ["What is a decision tree?", "Explain bagging."] and t.pairs[0].question_source == "display_text")
check("legacy row with a stored preamble but no core_question: the preamble is stripped", t.pairs[2].question == "what is boosting?", t.pairs[2].question)
check("corrupt gen_meta JSON does not break pairing", len(ev.build_transcript([("ai", "Q?", "local", "{not json"), cand("a")]).pairs) == 1)

print("== pairing: canned/system rows never become questions or answers' questions ==")
t = ev.build_transcript([ai("What is overfitting?"), cand("I quit"), system(RESIGN)])
check("resignation: the only pair is the real question; terminated flag set", len(t.pairs) == 1 and t.terminated and t.pairs[0].question == "What is overfitting?")
check("resignation message is never scored as a question", all("unwilling" not in p.question for p in t.pairs))
t = ev.build_transcript([system(REJECT)])
check("domain rejection: no pairs, flagged as rejected/terminated", t.rejected and t.terminated)
t = ev.build_transcript([ai("Q1?"), cand("a1"), ("ai", "Thanks, wrapping up.", "local", json.dumps({"action": "conclude"})), cand("stray")])
check("model-requested conclusion row is not a question; a stray later message creates no pair", len(t.pairs) == 1)
t = ev.build_transcript([("ai", CLOSING_MESSAGE, None, None), cand("hello")])
check("legacy closing text (no provider, no meta) recognised as canned", t.rejected)
t = ev.build_transcript([ai("Q1?"), cand("first part"), cand("second part"), ai("Q2?"), cand("b")])
check("two consecutive candidate messages stay in ONE answer (no evidence lost, no extra pair)", len(t.pairs) == 2 and t.pairs[0].answer == "first part second part")

print("== pairing: flow labels come from the reaction row after the answer ==")
rows = [ai("Q1?", topic="supervised learning"), cand("haah easy..supervised learning is a type of learning"),
        ai("Q2?", topic="overfitting", quality="vague", behavior="unprofessional", signals=["unprofessional:laughter"]),
        cand("a normal answer"), ai("Q3?", quality="strong", behavior="normal"), cand("lol whatever"), system(CLOSING_MESSAGE)]
t = ev.build_transcript(rows)
check("answer 1 gets the labels stored on the NEXT interviewer row", (t.pairs[0].flow_quality, t.pairs[0].behavior, t.pairs[0].behavior_source) == ("vague", "unprofessional", "stored"))
check("answer 2 -> normal (stored)", t.pairs[1].behavior == "normal" and t.pairs[1].flow_quality == "strong")
check("last answer has no reaction row -> behaviour re-derived from the text (dismissive)", t.pairs[2].behavior == "dismissive" and t.pairs[2].behavior_source == "detected")

# ---------------------------------------------------------------- prompt
print("== prompt rules ==")
p = ev.build_prompt(ev.build_transcript([ai("What is overfitting?"), cand("x")]).pairs)
check("prompt separates wrong answers from refusals and allows casual wording", "NOT a refusal" in p and "never evidence of missing knowledge" in p)
check("the old 'slang => score exactly 0' rule is gone", "slang, the score" not in p and "RUTHLESS" not in p)
check("prompt asks for neutral, evidence-based wording and lists the answer types", all(k in p for k in ("did not demonstrate understanding of the concepts assessed", "non_answer", "vague", "incorrect", "irrelevant")))

# ---------------------------------------------------------------- failure types
print("== wrong answer vs refusal vs vague ==")
rows = [ai("Q1?"), cand("Supervised learning uses unlabeled data."), ai("Q2?"), cand("I don't know"), ai("Q3?"), cand("it is a type of learning"),
        ai("Q4?"), cand("My favourite colour is blue"), ai("Q5?"), cand("backprop computes gradients via the chain rule, lol")]
r = run(rows, good_model(5, ["incorrect", "non_answer", "vague", "irrelevant", "strong"], [25, 0, 20, 5, 90]))
by = [b["answerType"] for b in r["breakdown"]]
check("each failure type is kept distinct in the report", by == ["incorrect", "non_answer", "vague", "irrelevant", "strong"], by)
check("a wrong technical answer keeps its own type and is not a non_answer", r["breakdown"][0]["answerType"] == "incorrect" and r["breakdown"][0]["score"] == 25)
check("answerTypeCounts summarises the types", r["answerTypeCounts"] == {"strong": 1, "incorrect": 1, "vague": 1, "irrelevant": 1, "non_answer": 1})
check("a casual strong answer ('...lol') is scored on its technical content (not zeroed)", r["breakdown"][4]["score"] >= 75)
r = run([ai("Q1?"), cand("I think it reduces variance"), ai("Q2?"), cand("not sure")], good_model(2, ["partial", "partial"], [55, 40], breakdown=None) | {"breakdown": [{"score": 55}, {"score": 40}]})
check("a missing answerType does not fail the evaluation: marked 'unspecified', score untouched", [b["answerType"] for b in r["breakdown"]] == ["unspecified"] * 2 and [b["score"] for b in r["breakdown"]] == [55, 40])

# ---------------------------------------------------------------- vague handling
print("== vague answers ==")
r = run([ai("Explain supervised learning."), cand("supervised learning is a type of learning")], good_model(1, ["vague"], [15], summary="Responses were vague."))
check("vague answer: low score, 'vague' type, and the feedback names it as too general", r["breakdown"][0]["answerType"] == "vague" and r["breakdown"][0]["score"] <= 35)
r = run([ai("Explain supervised learning."), cand("supervised learning is a type of learning")], good_model(1, ["vague"], [15], breakdown=[{"index": 1, "answerType": "vague", "score": 15, "feedback": ""}]))
check("empty model feedback is replaced by neutral text tied to the type", "too general" in r["breakdown"][0]["feedback"] and r["breakdown"][0].get("feedbackReplaced"))

# ---------------------------------------------------------------- professionalism separate
print("== professionalism separated from the technical assessment ==")
rows = [ai("Q1?"), cand("haha easy, overfitting is when a model memorises training noise and fails on new data"),
        ai("Q2?", quality="strong", behavior="unprofessional", signals=["unprofessional:laughter", "confident:easy_question"]),
        cand("whatever"), ai("Q3?", quality="weak", behavior="dismissive", signals=["dismissive:whatever"]), cand("L2 adds a weight penalty to reduce variance."), system(CLOSING_MESSAGE)]
r = run(rows, good_model(3, ["strong", "non_answer", "strong"], [85, 0, 80]))
check("a casual-but-correct answer keeps its technical score; tone does not reduce it", r["breakdown"][0]["score"] == 85 and r["breakdown"][0]["answerType"] == "strong")
check("behaviour is reported in its own fields", r["behavior"]["questions"] == [1, 2] and (r["behavior"]["unprofessional"], r["behavior"]["dismissive"]) == (1, 1) and "not part of the technical assessment" in r["behaviorNote"], r["behavior"])
check("overall score is the mean of the technical scores only", r["overallScore"] == round((85 + 0 + 80) / 3))
check("behaviour note is written by the backend, not the model (a model note cannot inject exaggeration)",
      "unprofessional throughout" not in run(rows, good_model(3, ["strong", "non_answer", "strong"], [85, 0, 80], behaviorNote="highly unprofessional throughout"))["behaviorNote"])
clean = run([ai("Q1?"), cand("a precise correct answer about bias and variance trade-off"), system(CLOSING_MESSAGE)], good_model(1, ["strong"], [90]))
check("no flagged behaviour -> neutral note", "No dismissive or unprofessional language was detected" in clean["behaviorNote"] and clean["behavior"]["questions"] == [])

# ---------------------------------------------------------------- unsupported claims
print("== no exaggerated claims without evidence ==")
BAD = ["The candidate shows a complete lack of technical knowledge.", "Total absence of technical knowledge was evident.",
       "The candidate was highly unprofessional throughout the interview.", "A terrible, incompetent answer.", "Candidate should not be hired.",
       "They have no technical knowledge at all."]
for phrase in BAD:
    r = run([ai("Q1?"), cand("idk")], good_model(1, ["non_answer"], [0], summary=phrase, strengths=[phrase], weaknesses=[phrase],
            breakdown=[{"index": 1, "answerType": "non_answer", "score": 0, "feedback": phrase}]))
    text = json.dumps(r)
    check(f"exaggerated wording removed everywhere: {phrase[:48]!r}", not ev._UNSUPPORTED.search(" ".join([r["summary"], r["insights"], r["behaviorNote"]] + [b["feedback"] for b in r["breakdown"]] + r["strengths"] + r["weaknesses"])), text[:160])
check("...replaced by neutral, answer-grounded wording", "did not demonstrate understanding" in r["summary"] or "Of 1 scored answers" in r["summary"] or "No attempt was made" in r["breakdown"][0]["feedback"])
r = run([ai("Q1?"), cand("a")] * 1, good_model(1, ["partial"], [50], summary="The candidate explained the idea only in part and missed the trade-offs."))
check("neutral, specific wording from the model is kept unchanged", r["summary"] == "The candidate explained the idea only in part and missed the trade-offs." and not r["breakdown"][0].get("feedbackReplaced"))
w = run([ai("Q1?"), cand("a"), ai("Q2?"), cand("b"), ai("Q3?"), cand("c")], {"summary": "", "breakdown": [{"score": 10, "answerType": "vague"}, {"score": 12, "answerType": "vague"}, {"score": 0, "answerType": "non_answer"}]})
check("deterministic fallback summary only states counts/limits ('did not demonstrate understanding', 'only the answers given')",
      "did not demonstrate understanding of most of the concepts assessed" in w["summary"] and "only the answers given" in w["summary"] and not ev._UNSUPPORTED.search(w["summary"]), w["summary"])
check("deterministic weaknesses are per-answer and evidence based", all("no technical knowledge" not in x.lower() for x in w["weaknesses"]) and len(w["weaknesses"]) == 3, w["weaknesses"])

# ---------------------------------------------------------------- report schema
print("== report schema ==")
rows = [ai("Q1?", topic="overfitting"), cand("a"), ai("Q2?", topic="overfitting"), cand("b"), ai("Q3?", topic="decision trees"), cand("c")]
r = run(rows, good_model(3, ["strong", "partial", "vague"], [90, 60, 20]))
need = {"schemaVersion", "overallScore", "summary", "insights", "strengths", "weaknesses", "topicPerformance", "answerTypeCounts", "behaviorNote", "behavior", "breakdown"}
check("report carries every field (old frontend fields kept: overallScore, summary, insights, breakdown[question,answer,score,feedback])", need <= set(r) and all({"question", "answer", "score", "feedback"} <= set(b) for b in r["breakdown"]))
check("each scored question has score, reason, answer type, topic, question source", all({"score", "feedback", "answerType", "topic", "questionSource", "difficulty"} <= set(b) for b in r["breakdown"]))
check("overallScore is computed by the backend (mean of per-question scores), ignoring the model's own figure", run(rows, good_model(3, ["strong", "partial", "vague"], [90, 60, 20], overallScore=99))["overallScore"] == round((90 + 60 + 20) / 3))
check("topic-level performance computed from the per-question scores", r["topicPerformance"] == [{"topic": "overfitting", "questions": 2, "averageScore": 75}, {"topic": "decision trees", "questions": 1, "averageScore": 20}], r["topicPerformance"])
check("insights text assembled from strengths and weaknesses", r["insights"].startswith("Strengths: ") and "Weaknesses: " in r["insights"])
check("schema version stamped", r["schemaVersion"] == ev.EVAL_SCHEMA_VERSION)
r = run(rows, good_model(3, ["strong", "strong", "strong"], [500, 99.6, 80.4]))
check("scores are only rounded and kept inside 0-100 (500 -> 100, 99.6 -> 100, 80.4 -> 80)", [b["score"] for b in r["breakdown"]] == [100, 100, 80], [b["score"] for b in r["breakdown"]])
r = run(rows, good_model(3, ["partial", "partial", "partial"], [70, 38, 45]))
check("scores inside the margin of their label's range are never moved (a legitimate 38 'partial' stays 38)", [b["score"] for b in r["breakdown"]] == [70, 38, 45])
check("...and a label with a score a little outside its range (incorrect 38, margin 10) needs no repair and is not moved",
      run([ai("Q1?"), cand("It memorises the test set during training.")], good_model(1, ["incorrect"], [38]))["breakdown"][0]["score"] == 38)
shuffled = good_model(3, ["strong", "partial", "vague"], [90, 60, 20]); shuffled["breakdown"].reverse()
check("items delivered out of order are put back in question order by their index", [b["score"] for b in run(rows, shuffled)["breakdown"]] == [90, 60, 20])
check("terminated report: voided scores, schema version, traceable questions", (lambda tr: tr["overallScore"] == 0 and tr["schemaVersion"] == ev.EVAL_SCHEMA_VERSION and tr["breakdown"][0]["question"] == "Q1?")(ev.terminated_report(ev.build_transcript([ai("Q1?"), cand("I quit"), system(RESIGN)]))))

# ---------------------------------------------------------------- consistency + single repair
print("== answerType/score consistency: at most ONE repair, never a silent correction ==")
def raises(fn):
    try: fn(); return False
    except ev.EvaluationUnavailableError: return True
class Scripted:
    def __init__(self, *replies): self.replies, self.prompts = list(replies), []
    def __call__(self, prompt):
        self.prompts.append(prompt); return json.dumps(self.replies[min(len(self.prompts), len(self.replies)) - 1])
rows = [ai("Q1?"), cand("Supervised learning uses unlabeled data to predict labels."), ai("Q2?"), cand("It prevents overfitting by penalising large weights in the loss.")]
tr = ev.build_transcript(rows)
bad = good_model(2, ["incorrect", "strong"], [80, 85])          # 'incorrect' labelled but scored 80
fixed = good_model(2, ["partial", "strong"], [80, 85])          # evaluator corrects the LABEL, keeps its technical judgement
s1 = Scripted(good_model(2, ["incorrect", "strong"], [20, 85]))
r = ev.evaluate_transcript(tr, s1)
check("consistent result: no repair request (exactly one evaluator call)", len(s1.prompts) == 1 and [b["score"] for b in r["breakdown"]] == [20, 85])
s2 = Scripted(bad, fixed)
r = ev.evaluate_transcript(tr, s2)
check("conflicting label/score -> exactly ONE repair request", len(s2.prompts) == 2)
check("the repair request names the conflict and carries the previous result", "item 1" in s2.prompts[1] and '"incorrect"' in s2.prompts[1] and "Previous result" in s2.prompts[1])
check("successful repair is accepted as returned: the evaluator's score is kept (80), only its label changed", [(b["answerType"], b["score"]) for b in r["breakdown"]] == [("partial", 80), ("strong", 85)])
check("no band clamping: the corrected result is not post-processed (no modelScore field)", all("modelScore" not in b for b in r["breakdown"]))
s3 = Scripted(bad, bad, fixed)
check("repair that is STILL inconsistent -> EvaluationUnavailableError, no third call, nothing invented", raises(lambda: ev.evaluate_transcript(tr, s3)) and len(s3.prompts) == 2)
s4 = Scripted(bad, {"breakdown": "nope"})
check("repair that is malformed -> EvaluationUnavailableError", raises(lambda: ev.evaluate_transcript(tr, s4)) and len(s4.prompts) == 2)
def second_call_fails(prompt, _n=[0]):
    _n[0] += 1
    if _n[0] == 1: return json.dumps(bad)
    raise RuntimeError("gemini 503")
check("provider failure during the repair -> EvaluationUnavailableError", raises(lambda: ev.evaluate_transcript(tr, second_call_fails)))
check("the inconsistency error is an EvaluationUnavailableError (the endpoint answers 503 and saves nothing)", issubclass(ev.EvaluationInconsistentError, ev.EvaluationUnavailableError))
r = ev.evaluate_transcript(tr, Scripted(good_model(2, ["partial", "strong"], [60, 100])))
check("a legitimate score is never changed because of its label (partial 60, strong 100 untouched)", [b["score"] for b in r["breakdown"]] == [60, 100])

print("== obvious non_answer stays deterministically 0 (no repair needed) ==")
rows = [ai("Q1?"), cand("I don't know"), ai("Q2?"), cand("idk"), ai("Q3?"), cand("lol whatever")]
s5 = Scripted(good_model(3, ["non_answer"] * 3, [40, 10, 5]))
r = ev.evaluate_transcript(ev.build_transcript(rows), s5)
check("non_answer on an obvious 'I don't know' -> score forced to 0, no repair request", len(s5.prompts) == 1 and [b["score"] for b in r["breakdown"]] == [0, 0, 0])
check("...the evaluator's original score is kept for traceability", [b.get("modelScore") for b in r["breakdown"]] == [40, 10, 5])
rows = [ai("Q1?"), cand("Supervised learning uses unlabeled data and clusters the points by similarity."), ai("Q2?"), cand("x")]
s6 = Scripted(good_model(2, ["non_answer", "non_answer"], [50, 0]), good_model(2, ["incorrect", "non_answer"], [20, 0]))
r = ev.evaluate_transcript(ev.build_transcript(rows), s6)
check("a wrong technical answer labelled non_answer is NOT deterministically zeroed: it triggers a repair instead", len(s6.prompts) == 2 and r["breakdown"][0]["answerType"] == "incorrect" and r["breakdown"][0]["score"] == 20)
check("obvious_non_answer: empty / 'I don't know' / 'lol whatever' yes; a real attempt no", all(ev.obvious_non_answer(x) for x in ("", "I don't know", "idk", "lol whatever")) and not ev.obvious_non_answer("Supervised learning uses unlabeled data."))

print("== evaluation prompt is compact and contains only what scoring needs ==")
rows = [ai(f"What is concept {i}?", display=f"Thanks, that's fine. Let's move on. What is concept {i}?", topic="supervised learning", answer_quality="vague",
           answer_behavior="unprofessional", signals=["unprofessional:laughter"], reaction_kind="vague_follow", step="follow_up", quality_reason="circular") if i else
        ai("What is concept 0?") for i in range(10)]
rows = [x for q in rows for x in (q, cand("a typical two sentence answer about the concept " * 4))]
pp = ev.build_prompt(ev.build_transcript(rows).pairs)
items = json.loads(pp.splitlines()[-1])
check("each item has exactly index/question/answer/topic/difficulty", all(set(i) == {"index", "question", "answer", "topic", "difficulty"} for i in items))
check("no greeting/transition text, flow labels, signals, reaction kinds or RAG/state leak into the items sent to the evaluator", not any(k in pp.splitlines()[-1] for k in ("Let's move on", "that's fine", "unprofessional", "laughter", "vague_follow", "reaction", "circular", "follow_up", "signals")))
check("10-answer prompt stays small enough for the local model (< 6000 chars ~ <1500 tokens of a 4096 window)", len(pp) < 6000, len(pp))
check("a very long answer is truncated in the PROMPT only; the reported answer stays complete",
      len(json.loads(ev.build_prompt(ev.build_transcript([ai("Q?"), cand("word " * 1000)]).pairs).splitlines()[-1])[0]["answer"]) == ev.MAX_PROMPT_ANSWER_CHARS
      and len(run([ai("Q?"), cand("word " * 1000)], good_model(1))["breakdown"][0]["answer"]) >= 4999)

# ---------------------------------------------------------------- invalid evaluator output / provider failure
print("== failed evaluations are never turned into reports ==")
t = ev.build_transcript([ai("Q1?"), cand("a"), ai("Q2?"), cand("b")])
def boom(prompt): raise RuntimeError("503 gemini down")
check("provider exception -> EvaluationUnavailableError (nothing to cache)", raises(lambda: ev.evaluate_transcript(t, boom)))
check("provider already raised EvaluationUnavailableError -> propagates", raises(lambda: ev.evaluate_transcript(t, lambda p: (_ for _ in ()).throw(ev.EvaluationUnavailableError("all down")))))
check("no JSON in the reply -> unavailable", raises(lambda: ev.evaluate_transcript(t, lambda p: "sorry, I cannot do that")))
check("malformed JSON -> unavailable", raises(lambda: ev.evaluate_transcript(t, lambda p: '{"breakdown": [oops]}')))
check("wrong number of breakdown items -> unavailable", raises(lambda: ev.evaluate_transcript(t, lambda p: json.dumps(good_model(1)))))
check("non-numeric score -> unavailable", raises(lambda: ev.evaluate_transcript(t, lambda p: json.dumps({"breakdown": [{"score": "high"}, {"score": 5}]}))))
check("boolean score -> unavailable", raises(lambda: ev.evaluate_transcript(t, lambda p: json.dumps({"breakdown": [{"score": True}, {"score": 5}]}))))
check("JSON wrapped in prose / code fences is still accepted", ev.evaluate_transcript(t, lambda p: "Here you go:\n```json\n" + json.dumps(good_model(2)) + "\n```")["schemaVersion"] == ev.EVAL_SCHEMA_VERSION)

# ---------------------------------------------------------------- cache / versioning
print("== cache / schema version ==")
fresh = run([ai("Q1?"), cand("a")], good_model(1))
check("a valid current-version report is served from cache", ev.cached_report(json.dumps(fresh)) == fresh)
old = {k: v for k, v in fresh.items() if k != "schemaVersion"}
check("pre-versioning cached report (no schemaVersion) -> recompute", ev.cached_report(json.dumps(old)) is None)
check("older version number -> recompute", ev.cached_report(json.dumps(dict(fresh, schemaVersion=ev.EVAL_SCHEMA_VERSION - 1))) is None)
check("future/other version number -> recompute", ev.cached_report(json.dumps(dict(fresh, schemaVersion=ev.EVAL_SCHEMA_VERSION + 1))) is None)
check("old failed-parse marker -> recompute", ev.cached_report(json.dumps(dict(fresh, summary="Evaluation failed to parse."))) is None)
check("missing overallScore / breakdown / summary -> recompute", all(ev.cached_report(json.dumps({k: v for k, v in fresh.items() if k != drop})) is None for drop in ("overallScore", "breakdown", "summary")))
check("garbage / empty / non-object cache values -> recompute", all(ev.cached_report(x) is None for x in ("not json", "[]", "null", "", None, "{}")))
check("boolean overallScore -> recompute", ev.cached_report(json.dumps(dict(fresh, overallScore=True))) is None)

# The endpoint wiring (main.py imports the vector store, so it is exercised here by source inspection and, outside
# this repo suite, by the end-to-end run): the endpoint must serve only cached_report() hits and save only validated reports.
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "main.py"), encoding="utf-8").read()
body = src[src.index("async def fetch_session_summary"):src.index('@app.get("/api/admin/candidates")')]
check("endpoint serves cache only through cached_report()", "cached_report(" in body and "json.loads(row[0])" not in body)
check("endpoint saves only after evaluate_transcript/terminated_report; the 503 path raises before any UPDATE",
      body.index("evaluate_transcript(") < body.index("UPDATE interviews") and body.index("raise HTTPException(status_code=503") < body.index("UPDATE interviews"))
check("endpoint keeps the Gemini->backup->local chain (generate_eval_response is the provider function)", "generate_eval_response([HumanMessage(content=prompt)])" in body)

finish()
