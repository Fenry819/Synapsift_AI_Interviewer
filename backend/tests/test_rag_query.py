"""RAG query quality: only a substantive answer may add its text to the knowledge-base query. Embeddings are REAL (the same
classifier the live interview uses); no vector store or provider is called."""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import BACKEND, check, finish, load_embedder  # noqa: E402

import interview_engine as ie  # noqa: E402
import rag  # noqa: E402
import rag_config as rc  # noqa: E402
from resume_profile import build_candidate_profile  # noqa: E402

embed = load_embedder()
PROFILE = build_candidate_profile("SKILLS\nPython, PyTorch, scikit-learn, Machine Learning, NLP, Deep Learning\nEXPERIENCE\nML Intern - Acme\n- a\n- b\n")
ROLE = "AI / Machine Learning Role"
Q = "What is the purpose of feature engineering in machine learning?"
T = "feature engineering"
TOPIC = "feature engineering and data preparation"
GAMING = "I like gaming and laptops a lot, especially comparing hardware and graphics settings."
STRONG = ("Feature engineering turns raw data into inputs a model can learn from: scaling numeric columns, encoding categorical variables, "
          "creating interaction terms and removing noisy features, which usually improves accuracy and generalisation.")


def cls(answer, profane=False, severe=False):
    return ie.classify_answer(answer, Q, T, embed, profane=profane, severe_abuse=severe)


def query(answer, quality, number=3):
    return rag.build_retrieval_query("ai_ml", ROLE, PROFILE, number, ie.answer_for_retrieval(answer, quality), TOPIC)


print("== the gaming / laptop answer must not contaminate the query (the reported case) ==")
q_gaming = cls(GAMING)
check("the gaming answer is classified as a poor answer", q_gaming.label in ("irrelevant", "weak", "vague"), q_gaming.label)
check("answer_for_retrieval returns nothing for it", ie.answer_for_retrieval(GAMING, q_gaming) is None)
built = query(GAMING, q_gaming)
check("the query contains no word of the gaming answer", not re.search(r"gaming|laptops|graphics|hardware", built, re.I), built)
check("CONTROL: appending that raw answer (the old behaviour) WOULD have contaminated the query",
      "gaming" in rag.build_retrieval_query("ai_ml", ROLE, PROFILE, 3, GAMING, TOPIC).lower())
check("role, resume terms and the planned topic are all still in the query",
      TOPIC in built and ROLE in built and "AI and machine learning" in built and "Machine Learning" in built and "Candidate background:" in built, built)
check("no 'Candidate's last answer' fragment at all", "last answer" not in built)

print("== non-substantive answers never reach the query ==")
NON = {
    "I don't know": cls("I don't know"),
    "idk": cls("idk"),
    "whatever": cls("whatever"),
    "ask something else": cls("ask something else"),
    "haah easy question..supervised learning is a type of learning": cls("haah easy question..supervised learning is a type of learning"),
    "feature engineering gives features to machine learning (vague)": cls("feature engineering gives features to machine learning"),
    "supervised learning uses unlabeled data, obviously (recognised misconception)": ie.classify_answer("supervised learning uses unlabeled data, obviously", "What is supervised learning?", "supervised learning", embed),
    "give me a hint": cls("give me a hint"),
    "what is my score?": cls("what is my score?"),
    "please give me the job, my mom is sick and I need money": cls("please give me the job, my mom is sick and I need money"),
    "overfitting is a fitting in toilet (mock)": ie.classify_answer("overfitting is a fitting in toilet", "What is overfitting?", "overfitting", embed),
    "Fuck you man (severe abuse)": cls("Fuck you man", profane=True, severe=True),
    "my cat likes the sofa": cls("my cat likes the sofa"),
}
for label, q in NON.items():
    check(f"{label!r} -> label {q.label}, nothing is added to the query", ie.answer_for_retrieval(label, q) is None, q.label)
check("no quality object (first question) -> nothing added", ie.answer_for_retrieval("anything", None) is None)

print("== substantive answers still contribute ==")
q_strong = cls(STRONG)
check("a substantive answer is judged strong/partial", q_strong.label in ("strong", "partial"), q_strong.label)
check("answer_for_retrieval returns its text", ie.answer_for_retrieval(STRONG, q_strong) and "scaling numeric columns" in ie.answer_for_retrieval(STRONG, q_strong))
sq = query(STRONG, q_strong)
check("the query carries 'Candidate's last answer' with that technical text, after the role/profile/topic parts", "Candidate's last answer: Feature engineering turns raw data" in sq and sq.index(TOPIC) < sq.index("last answer"), sq)
check("it is still length-capped", len(sq.split("Candidate's last answer: ")[1]) <= rc.MAX_ANSWER_CHARS_IN_QUERY)
mixed = STRONG + " Please give me the job, my mom is sick and I need money. lol easy question"
q_mixed = cls(mixed)
text = ie.answer_for_retrieval(mixed, q_mixed)
check("appeals, chatter and requests inside a substantive answer are stripped before use", text is not None and not re.search(r"job|mom|sick|money|lol|easy", text, re.I) and "scaling" in text, text)
partial = "Feature engineering is about creating better input features from the raw data for the model."
q_partial = cls(partial)
check("a partial answer also contributes", q_partial.label in ("strong", "partial") and ie.answer_for_retrieval(partial, q_partial), q_partial.label)

print("== wiring in main.py ==")
src = open(os.path.join(BACKEND, "main.py"), encoding="utf-8").read()
chat = src[src.index("async def chat_round"):src.index('@app.get("/api/interview/summary')]
check("the chat endpoint passes answer_for_retrieval(...) to retrieval, never the raw message", "current_answer=answer_for_retrieval(payload.message, quality)" in chat and "current_answer=payload.message" not in chat)
check("retrieval still receives role, profile and the planned topic", "target_role, profile," in chat and "topic_hint=plan.topic" in chat)

finish()
