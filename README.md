# SynapSift AI

## Overview

An AI technical-interview platform. A candidate signs in, picks a target role and uploads a resume (PDF or plain text). The system then holds a short, adaptive, one-question-at-a-time technical interview, grounded in a vetted knowledge base (RAG), and finishes with an evidence-based evaluation. An administrator can review every candidate's result.

- **Backend:** FastAPI + SQLite + embedded Qdrant vector store, LangChain model clients.
- **Frontend:** Next.js (App Router) + Tailwind.
- **Models:** OpenRouter (when a key is set) with automatic fallback to a **local Ollama model**; evaluation by Gemini with fallback to the same local model. Nothing in the interview or evaluation is faked: if no model answers, the request fails with a clear error instead of inventing a reply or a score.

---

## Contents

0. [Overview](#overview)
1. [How it works (system flow)](#how-it-works-system-flow)
2. [Architecture](#architecture)
3. [Key design decisions](#key-design-decisions)
4. [Prerequisites](#prerequisites)
5. [Backend setup](#backend-setup)
6. [Environment configuration (`.env`)](#environment-configuration-env)
7. [Local model setup (Ollama)](#local-model-setup-ollama)
8. [Knowledge base and `python ingest.py`](#knowledge-base-and-python-ingestpy)
9. [Frontend setup](#frontend-setup)
10. [Running the app](#running-the-app)
11. [Tests](#tests)
12. [Resume and role behavior](#resume-and-role-behavior)
13. [Interview flow in detail](#interview-flow-in-detail)
14. [Evaluation](#evaluation)
15. [Question traceability](#question-traceability)
16. [Known limitations](#known-limitations)
17. [Repository layout](#repository-layout)

---

## How it works (system flow)

```
Candidate                         Frontend (Next.js)                Backend (FastAPI)
---------                         ------------------                -----------------
sign up / sign in  ───────────►   auth screen            ───────►   /api/auth/*          bcrypt + JWT
choose role + resume ─────────►   setup screen           ───────►   /api/interview/start
                                                                     1. validate resume (PDF or .txt), role gate
                                                                     2. build a structured profile (deterministic)
                                                                     3. plan the first topic, retrieve context (RAG)
                                                                     4. model writes ONE question as JSON; backend validates it
◄──────── first question  ◄────   chat (progressive reveal)  ◄──────
answer ───────────────────────►   composer                 ───────►   /api/interview/chat
                                                                     classify answer -> plan next step -> retrieve -> generate
                                                                     (repeat up to 10 answers)
"View evaluation" ────────────►   result screen            ───────►   /api/interview/summary/{id}
                                                                     pair question/answer -> evaluate -> validate -> cache
Administrator ────────────────►   dashboard                ───────►   /api/admin/candidates, summaries, delete
```

1. **Sign-in.** Candidates and administrators have separate roles. Passwords are bcrypt-hashed; sessions are JWTs.
2. **Setup.** The candidate chooses a target role and uploads a resume. The backend validates the file, extracts text, and builds a structured profile (skills, technologies, projects, experience signals) with deterministic rules, so it cannot invent skills.
3. **Interview.** Up to **10** candidate answers. After each answer the backend decides what happens next (follow-up, new topic, deeper probe, firmer re-ask). The model only writes the next technical question.
4. **Evaluation.** When the interview ends, the transcript is evaluated question by question and an overall report is produced and stored.
5. **Review.** Administrators can open any completed candidate's report and original resume.

## Architecture

| Layer | Component | Where |
|---|---|---|
| UI | Next.js app: login/registration, setup + resume upload, interview chat (progressive reveal), results, admin dashboard | `frontend/src` |
| API | FastAPI app, auth, interview + evaluation endpoints | `backend/main.py` |
| Interview engine | State, planning, answer classification, structured output, validation, repair, deterministic fallback | `backend/interview_engine.py`, `interview_style.py`, `answer_signals.py`, `interview_config.py` |
| Profile | Resume text to a structured candidate profile | `backend/resume_profile.py`, `resume_intake.py` |
| RAG | Role-to-domain taxonomy, query building, domain-filtered retrieval with a relevance threshold, per-question trace | `backend/rag.py`, `rag_config.py`, `ingest.py` |
| Providers | OpenRouter, local Ollama, Gemini | `backend/llm_providers.py`, `generate_eval_response` in `main.py` |
| Evaluation | Transcript pairing, evaluator prompt, report validation, cache versioning | `backend/evaluation.py` |
| Data | SQLite (`synapsift.db`): users, interviews, messages; embedded Qdrant (`qdrant_db/`): the knowledge base | `backend/` (both git-ignored runtime data) |

**RAG.** Seven textbooks are chunked (800 characters, 150 overlap), embedded with `all-MiniLM-L6-v2` and stored in Qdrant, each chunk tagged with the knowledge domain(s) it belongs to (`ai_ml`, `data_science`, `advanced_ml`). At question time the backend builds a query from the **planned topic, the target role, the resume terms that belong to the domain**, and (only when the candidate's last answer was substantive) a short excerpt of that answer. The search is filtered to the role's domain and chunks below a cosine relevance of 0.40 are discarded, so an unrelated chunk is never passed to the model.

## Key design decisions

- **The backend owns the interview, not the model.** Difficulty, topic coverage, follow-up vs. new topic, answer limits and when to stop are decided in Python from a versioned interview state stored in SQLite. The model is asked for one question as structured JSON.
- **Everything the model writes is validated before a candidate sees it:** exactly one question, no lists or multiple choice, no leaked reference material, no claims about the candidate that the resume does not support, no insults, no early conclusion, no repeated questions (semantic similarity check). One repair attempt, then a deterministic fallback question. Provider failures never produce a made-up reply.
- **Conversation wording is backend-controlled.** Greetings, transitions and reactions to the previous answer (strong, partial, vague, incorrect, irrelevant, dismissive, appeals, requests for hints or scores) are chosen by deterministic rules, so tone does not depend on model compliance. The interviewer never gives the correct answer, a score or a hiring decision.
- **Role-aware, domain-filtered RAG.** A role maps to a knowledge domain; roles without a corpus run without retrieval rather than being matched to the nearest-looking material.
- **Conduct is separate from technical quality.** Repeated severe abuse ends an interview (warning, firmer warning, then conclusion); a single swear word, slang, nonsense or a personal appeal never does.
- **Evaluation is evidence-based.** The evaluator scores the exact question that was asked, wording that overstates ("complete lack of…") is replaced by neutral text, an answer type and its score must agree (one repair request, otherwise the evaluation is reported unavailable), and professionalism is reported separately from the technical score. A failed evaluation is never cached.
- **Local-first resilience.** Any Ollama model can serve as the interviewer fallback and the evaluation fallback; the model name comes from configuration.

## Prerequisites

- **Python 3.11+** (developed and tested on 3.13 and 3.14).
- **Node.js 20+** and npm.
- **Ollama** (optional but recommended): <https://ollama.com>. Required for the local fallback; without a hosted key it is the only interviewer model. A 14B model needs roughly 10 GB of free RAM/VRAM; smaller models work with lower quality.
- Disk space and a few minutes for the first install: PyTorch is pulled in by `sentence-transformers`, and the embedding model (~90 MB) downloads on first use.
- Internet access for the first install and (optionally) for OpenRouter and Gemini.

## Backend setup

```bash
cd backend
python -m venv venv
# Windows (PowerShell):   venv\Scripts\Activate.ps1
# macOS / Linux:          source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` lists only the top-level runtime dependencies, each with a tested lower bound and a major-version upper bound.

## Environment configuration (`.env`)

```bash
cp backend/.env.example .env        # Windows: copy backend\.env.example .env
```

Place `.env` in the **repository root** (or in `backend/`); both are git-ignored and the backend finds either. Every variable is optional, but set the two security values before real use:

| Variable | Purpose |
|---|---|
| `JWT_SECRET` | Signing key for login tokens. Use a long random string (`python -c "import secrets; print(secrets.token_urlsafe(48))"`). If empty, a temporary random key is used and everyone is signed out whenever the server restarts. |
| `ADMIN_SIGNUP_CODE` | Code required to register an **administrator** account. If empty, administrator sign-up is disabled. |
| `OPENROUTER_API_KEY` | Hosted interviewer model. Without it the interview runs on the local Ollama model. |
| `OPENROUTER_MODEL`, `OPENROUTER_TIMEOUT_SECONDS` | Optional OpenRouter tuning. |
| `GEMINI_API_KEY`, `GEMINI_BACKUP_API_KEY` | Evaluation models (primary, then backup). Without them evaluation uses the local model. |
| `LOCAL_LLM_MODEL` | Ollama model name (default `qwen2.5:14b`). |
| `OLLAMA_BASE_URL` | Default `http://127.0.0.1:11434`. |
| `LOCAL_LLM_NUM_CTX`, `LOCAL_LLM_TIMEOUT_SECONDS` | Context window and timeout for the local model. |

No real keys are stored in the repository.

## Local model setup (Ollama)

```bash
ollama pull qwen2.5:14b        # or any model, then set LOCAL_LLM_MODEL
ollama serve                   # usually already running as a background service
```

**Provider order.** Interviewer: OpenRouter (if a key is set) -> local Ollama. Evaluation: Gemini primary key -> Gemini backup key -> local Ollama. Both providers receive exactly the same context. If every provider fails, the API answers `503` (the answer a candidate just sent is withdrawn so they can resend it) and no fake result is stored.

## Knowledge base and `python ingest.py`

The knowledge base is built from seven textbook PDFs listed in `backend/rag_config.py` (`CORPUS`). Place them in **`For Rag injesting/`** at the repository root (ingestion also looks in `backend/`):

| File (exact name) | Feeds domain(s) |
|---|---|
| `2019BurkovTheHundred-pageMachineLearning.pdf` | AI/ML, Data Science |
| `Artificial Intelligence, Machine Learning, and Deep Learning.pdf` | AI/ML |
| `Machine Learning For Absolute Beginners.pdf` | AI/ML |
| `Master Machine Learning Algorithms - Discover how they work and Implement Them From Scratch by Jason Brownlee (z-lib.org).pdf` | AI/ML, Data Science |
| `Introduction to Machine Learning with Python ( PDFDrive.com )-min.pdf` | Data Science |
| `Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf` | Advanced ML |
| `MachineLearningTomMitchell.pdf` | Advanced ML |

Then build the vector store (a deliberate, one-off step; the API server never does this):

```bash
cd backend
python ingest.py --dry-run     # parse, clean and chunk; report only, writes nothing
python ingest.py               # rebuild the Qdrant collection (a few minutes of CPU)
```

- **Stop the API server first:** embedded Qdrant allows one process per `backend/qdrant_db/`.
- Ingestion **rebuilds** the collection so it always matches the manifest, and writes nothing until every source has been parsed.
- If the vector store is missing or empty the server still starts, but every interview then runs **without** RAG context and prints a warning telling you to run `python ingest.py`.
- The textbooks are third-party copyrighted material; use copies you are entitled to use.

## Frontend setup

```bash
cd frontend
npm install
```

The frontend calls the backend at `http://127.0.0.1:8000` (the CORS settings allow `http://localhost:3000` and `http://127.0.0.1:3000`).

## Running the app

Two terminals:

```bash
# 1. backend  (from backend/, with the venv active)
uvicorn main:app --reload            # http://127.0.0.1:8000  (API docs at /docs)

# 2. frontend (from frontend/)
npm run dev                          # http://localhost:3000
```

Open <http://localhost:3000>, create a candidate account, choose a role, upload a resume and start. To create an administrator, set `ADMIN_SIGNUP_CODE` in `.env`, restart the backend and choose "Administrator" when registering.

## Tests

```bash
# backend (from backend/, venv active): standard-library test scripts, no server or API keys needed
python tests/run_tests.py            # all suites
python tests/run_tests.py reactions  # only suites whose name contains the text

# frontend (from frontend/)
npm test                             # Node's built-in test runner
npx tsc --noEmit                     # type check
npm run lint
```

The backend suites cover the interview engine, answer classification and reactions, conduct handling, resume intake and profiling, RAG query construction, evaluation and caching, provider failover and the setup files. They use the real embedding model (downloaded on first run); the few checks that call a real local Ollama model run only when Ollama is reachable. They never touch `synapsift.db` or `qdrant_db/`.

## Resume and role behavior

**Resume.** Two formats are accepted: **PDF** (text-based, up to 20 pages) and **plain text** (`.txt`, UTF-8). Maximum size is 5 MB and at least 50 characters of readable text are required. Scanned/image-only PDFs, password-protected PDFs, binary files and other formats (including DOCX) are rejected with a clear message. The frontend only offers `.pdf` and `.txt`; the backend enforces the same rules independently.

**Roles.** The role is chosen from a list or typed as a custom role.

| Role | Behavior |
|---|---|
| AI / Machine Learning, Data Science / Applied ML, Advanced / Theoretical ML (and similar titles) | Full RAG from the matching knowledge domain |
| Other technical roles (for example Backend Engineering Intern) | Interview runs with generic engineering topics and **no retrieval**; nothing is matched to ML material |
| Clearly non-technical roles (for example yoga teacher) | Politely rejected up front by a deterministic check; no model is called |

## Interview flow in detail

1. The first topic is chosen by the backend: a topic that matches the resume's skills when there is a clear signal, otherwise one of the foundational topics, picked reproducibly from the interview id.
2. For every question the backend retrieves one reference chunk (role + resume terms + planned topic, domain-filtered, relevance-checked, never reusing a chunk) and asks the model for a single question as structured JSON.
3. The reply is validated; if it fails once it is repaired once; if it still fails a deterministic question for the planned topic is used.
4. After each answer the backend classifies it (strong, partial, vague, incorrect, irrelevant, or a non-answer) and separately classifies behavior, then plans the next step: strong answers sometimes get one deeper follow-up, weak ones one clarification and then a new topic, and so on. Difficulty moves at most one level per answer.
5. Requests for the answer, a hint, a score or the hiring outcome get a calm fixed reply; personal appeals are acknowledged without changing the assessment; repeated severe abuse ends the interview.
6. The interview ends after 10 answers (hard cap), when the interviewer concludes early (only after a minimum number of answers and topics), or when the candidate explicitly resigns or is ended for conduct.

## Evaluation

1. Only real question/answer pairs are scored; greetings, closings and system messages are excluded, and each question is the exact technical question that was asked.
2. The evaluator returns an answer type (strong, partial, incorrect, vague, irrelevant, no attempt), a score and one sentence of feedback per question. The backend computes the overall score as the mean, builds topic-level performance, strengths and weaknesses, and a separate professionalism note.
3. Inconsistent output (for example a score that contradicts the answer type) triggers one repair request, then the evaluation is reported unavailable. Unusable output or a provider outage returns `503` and **nothing is cached**.
4. Reports carry a schema version, so a report produced by an older version is recomputed.
5. Explicit resignation or domain rejection returns a deterministic zero-score report without calling a model. A conduct termination is evaluated normally on the answers collected so far.

## Question traceability

Every interviewer message stores how it was produced, in the `messages` table:

- `gen_meta`: generation path (primary, repaired or fallback), planned step, topic, difficulty, answer quality and behavior of the previous answer, the core question, the reaction text and its kind, similarity to earlier questions.
- `rag_trace`: whether retrieval ran, the **exact retrieval query**, the domain, threshold, how many candidates were considered, and the **selected chunk** (chunk id, source file, title, page, score, excerpt).
- `rag_source_chunk`: the full text of the chunk offered to the model; `llm_provider` / `llm_model`: who produced the message.
- `interviews.interview_state`: topics covered, weak/strong topics, and the ids of every chunk used so far.

Inspect them directly, for example:

```bash
sqlite3 backend/synapsift.db "SELECT id, text_content, rag_trace FROM messages WHERE interview_id='interview_xxxxxxxx' AND sender='ai' ORDER BY id;"
```

"Used" means the chunk was **provided to the model** for that question; the system records it but cannot prove how much the model relied on it. These records are not currently shown in the web UI.

## Known limitations

- **Answer correctness is judged by rules, not understanding.** Vague, circular, off-topic and a set of well-known misconceptions are detected deterministically; a fluent but wrong answer on another point can still be rated partial or strong during the interview. The final evaluation is done by an LLM and is more discriminating, but it is still a model's judgment.
- **English only** for behavior, request and appeal detection.
- **Retrieval relevance** is a cosine threshold; a chunk can pass it and still be only loosely related to the question.
- **Single-machine design:** SQLite and embedded Qdrant (one process at a time); no session restore after a page reload; the frontend API address is fixed to `127.0.0.1:8000`.
- **Local-model quality and speed** depend on your hardware and chosen Ollama model.
- The admin interface is a simple candidate list with report and resume viewing.
- The textbook PDFs are not redistributable and are not required for the app to start; without them (or without running `python ingest.py`) interviews run without RAG.

## Repository layout

```
backend/               FastAPI app, interview engine, RAG, evaluation, tests, requirements.txt, .env.example
  tests/               stdlib test scripts (python tests/run_tests.py)
frontend/              Next.js app (src/app, src/components, src/lib, tests)
For Rag injesting/     source PDFs for the knowledge base (see "Knowledge base")
README.md              this file
```
