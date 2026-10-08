import os
import json
import sqlite3
import random
import secrets
import hmac
import jwt
import datetime
import PyPDF2
import bcrypt
import re
from io import BytesIO
from better_profanity import profanity
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from google.genai.types import AutomaticFunctionCallingConfig
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore 
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_text_splitters import RecursiveCharacterTextSplitter
from resume_profile import (
    build_candidate_profile, empty_candidate_profile, parse_stored_profile,
    profile_to_prompt_text, clean_resume_text,
)
import rag_config as rag_cfg
from rag import retrieve_context, build_rag_trace, knowledge_base_status, RetrievalResult
from llm_providers import generate_chat_response, call_local_llm, ProviderUnavailableError
from resume_intake import KIND_PDF, ResumeFormatError, detect_resume_kind, extract_text_resume
from interview_style import CLOSING_MESSAGE, CONDUCT_TERMINATION_MESSAGE, CONDUCT_TERMINATION_REASON
from answer_signals import is_severe_abuse
from evaluation import (
    EvaluationUnavailableError, build_transcript, cached_report, evaluate_transcript, terminated_report,
)
from interview_engine import (
    TurnContext, answer_for_retrieval, classify_answer, conclusion_allowed, dump_state, fresh_state, generate_interviewer_turn,
    history_from_messages, load_state, plan_next_step, record_question,
)

# 1. Load Configurations & Security Elements
load_dotenv()
PRIMARY_KEY = os.getenv("GEMINI_API_KEY")
BACKUP_KEY = os.getenv("GEMINI_BACKUP_API_KEY")
JWT_ALGORITHM = "HS256"
# Signing key for login tokens. It must come from the environment (.env). If it is missing a random per-process key is
# used so nothing is ever signed with a published default; sessions then end whenever the server restarts.
JWT_SECRET = (os.getenv("JWT_SECRET") or "").strip()
if not JWT_SECRET:
    JWT_SECRET = secrets.token_urlsafe(48)
    print("⚠️ JWT_SECRET is not set: using a temporary random key (logins will not survive a restart). Set JWT_SECRET in .env.")
# Access code required to register an administrator account. Unset = administrator sign-up is disabled.
ADMIN_SIGNUP_CODE = (os.getenv("ADMIN_SIGNUP_CODE") or "").strip()

# 2. Relational Database Initialization
def init_relational_db():
    conn = sqlite3.connect("synapsift.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT CHECK(role IN ('candidate', 'admin')) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interviews (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            target_role TEXT NOT NULL,
            overall_score INTEGER DEFAULT 0,
            evaluation_summary TEXT,
            status TEXT CHECK(status IN ('SETUP', 'ONGOING', 'COMPLETED')) DEFAULT 'SETUP',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            interview_id TEXT NOT NULL,
            sender TEXT CHECK(sender IN ('ai', 'candidate')) NOT NULL,
            text_content TEXT NOT NULL,
            rag_source_chunk TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(interview_id) REFERENCES interviews(id)
        )
    """)

    try:
        cursor.execute("ALTER TABLE interviews ADD COLUMN resume_text TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE interviews ADD COLUMN evaluation_data TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE interviews ADD COLUMN resume_url TEXT")
    except sqlite3.OperationalError:
        pass
    # Structured candidate profile (JSON). NULL for interviews created before this column existed.
    try:
        cursor.execute("ALTER TABLE interviews ADD COLUMN candidate_profile TEXT")
    except sqlite3.OperationalError:
        pass
    # Versioned JSON interview state (difficulty, topics, used RAG chunks). NULL for older interviews.
    try:
        cursor.execute("ALTER TABLE interviews ADD COLUMN interview_state TEXT")
    except sqlite3.OperationalError:
        pass
    # Per-AI-message RAG traceability (JSON): whether RAG was used, source, page/chunk, score, excerpt.
    try:
        cursor.execute("ALTER TABLE messages ADD COLUMN rag_trace TEXT")
    except sqlite3.OperationalError:
        pass
    # Which provider authored each AI message ('openrouter' | 'local' | 'system'), the model name, and a JSON
    # record of how the question was produced (topic, difficulty, answer quality, primary/repaired/fallback ...).
    for column in ("llm_provider", "llm_model", "gen_meta"):
        try:
            cursor.execute(f"ALTER TABLE messages ADD COLUMN {column} TEXT")
        except sqlite3.OperationalError:
            pass

    conn.commit()
    conn.close()

init_relational_db()

# 3. AI engines
# Interviewer chat (OpenRouter -> local Ollama model) lives in llm_providers.generate_chat_response and
# raises ProviderUnavailableError when no provider answers; it never fabricates a question.
# Evaluation: Gemini (primary key, then backup key), then the same local Ollama model.
def generate_eval_response(messages: list) -> str:
    gemini_model = "gemini-3.5-flash"
    # Evaluation uses no tools, so automatic function calling is switched off explicitly (it also silences the SDK's AFC warning).
    no_afc = {"automatic_function_calling": AutomaticFunctionCallingConfig(disable=True)}

    def extract_text(content):
        if isinstance(content, list):
            if len(content) > 0 and isinstance(content[0], dict):
                return content[0].get("text", str(content))
        return str(content)

    try:
        print(f"🧠 Querying {gemini_model} API for Evaluation...")
        llm = ChatGoogleGenerativeAI(model=gemini_model, google_api_key=PRIMARY_KEY)
        response = llm.invoke(messages, **no_afc)
        return extract_text(response.content)
    except Exception as e_primary:
        print(f"⚠️ Primary Gemini failed: {e_primary}. Transitioning to Backup Key...")
        try:
            if BACKUP_KEY:
                llm_backup = ChatGoogleGenerativeAI(model=gemini_model, google_api_key=BACKUP_KEY)
                backup_response = llm_backup.invoke(messages, **no_afc)
                return extract_text(backup_response.content)
        except Exception as e_backup:
            print(f"⚠️ Backup Gemini failed: {e_backup}. Falling back to the local model...")

    # Fallback to the local Ollama model for Evaluation (model chosen by LOCAL_LLM_MODEL)
    try:
        return call_local_llm(messages).text
    except Exception as e_local:
        print(f"🚨 Local model failed for evaluation: {e_local}")

    # Provider failure must never masquerade as a candidate result.
    raise EvaluationUnavailableError("No evaluation provider produced a response.")

# 4. RAG Ingestion Pipeline: Qdrant Setup
# Enforce offline mode to bypass Hugging Face network checks
# Removed offline locks so the new PC can download the model
embeddings = HuggingFaceEmbeddings(
    model_name=rag_cfg.EMBEDDING_MODEL
)

def initialize_vector_db():
    # The server only OPENS the knowledge base. Filling it is the deliberate `python ingest.py` step.
    client = QdrantClient(path=rag_cfg.QDRANT_PATH)

    if not client.collection_exists(rag_cfg.COLLECTION_NAME):
        client.create_collection(
            collection_name=rag_cfg.COLLECTION_NAME,
            vectors_config=VectorParams(size=rag_cfg.EMBEDDING_DIM, distance=Distance.COSINE),
        )
    kb_state, kb_total, kb_tagged = knowledge_base_status(client)
    if kb_state == "empty":
        print("⚠️ Knowledge base is EMPTY: every interview will run WITHOUT RAG context. Stop the server and run `python ingest.py`.")
    elif kb_state == "legacy_schema":
        print(f"⚠️ Knowledge base ({kb_total} chunks) was built by an OLD ingest.py and has no domain metadata: every interview "
              "will run WITHOUT RAG context. Stop the server and run `python ingest.py` to rebuild it.")
    else:
        print(f"📚 Knowledge base ready: {kb_tagged} chunks across domains {list(rag_cfg.DOMAIN_INFO)}.")
    return QdrantVectorStore(client=client, collection_name=rag_cfg.COLLECTION_NAME, embedding=embeddings)

vector_db = initialize_vector_db()

# 5. API Setup & Authentication Guardrails
app = FastAPI(title="SynapSift AI Advanced Backend")

os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SignUpRequest(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: str
    admin_code: str | None = None  

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class ChatPayload(BaseModel):
    interview_id: str
    message: str
    role: str | None = None  # legacy field sent by the frontend; IGNORED - the stored target_role is authoritative

def get_current_user(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authentication credentials")
    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Authentication session expired or malformed")

# --- 🧹 Deterministic Text Guardrails (whole-word / whole-phrase matching) ---
MAX_CANDIDATE_ANSWERS = 10

def normalize_text(text: str) -> str:
    # Lowercase, drop apostrophes ("don't" -> "dont"), turn every other run of
    # non-alphanumerics into a single space, so matching is punctuation-insensitive.
    text = text.lower().replace("’", "'").replace("'", "")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()

# Explicit intent to STOP or REFUSE the interview ONLY. Profanity and generic hostility /
# disengagement ("shut up", "go away", "waste of time") are deliberately not in this list: they
# must not hard-terminate an interview (profanity is detected separately by better_profanity).
# Phrases are written in normalized form, and ambiguous single words that occur in normal
# technical answers ("pass", "skip", "refuse", "i won't") are deliberately NOT listed.
_RESIGNATION_PHRASES = [
    "i quit", "i give up", "i want to give up", "i am giving up", "im giving up",
    "i refuse to answer", "i wont answer", "i will not answer",
    "i wont continue", "i will not continue",
    "i dont want to continue", "i do not want to continue", "i dont want to answer",
    "stop asking", "stop the interview", "end the interview", "end this interview",
    "im done with this interview", "i am done with this interview",
]
_RESIGNATION_RE = re.compile(
    r"(?<![a-z0-9])(?:" + "|".join(re.escape(p) for p in _RESIGNATION_PHRASES) + r")(?![a-z0-9])"
)

def is_explicit_resignation(message: str) -> bool:
    return _RESIGNATION_RE.search(normalize_text(message)) is not None

# A role is rejected only if it contains a non-technical WORD (not a substring of a longer
# word) and nothing in it signals a technical role ("Device Driver Engineer" stays valid).
_NON_TECH_ROLE_TERMS = {"yoga", "gym", "fitness", "chef", "plumber", "sales", "hr", "driver", "retail", "medical"}
_TECH_ROLE_TERMS = {
    "engineer", "engineers", "engineering", "developer", "developers", "programmer", "software",
    "ml", "ai", "devops", "sre", "backend", "frontend", "fullstack", "scientist", "architect",
    "machine", "learning", "cloud", "security", "qa", "web", "mobile", "android", "ios",
    "python", "java", "javascript", "js", "computer", "systems", "firmware", "embedded",
}

def is_unsupported_role(role: str) -> bool:
    tokens = set(normalize_text(role).split())
    return bool(tokens & _NON_TECH_ROLE_TERMS) and not (tokens & _TECH_ROLE_TERMS)

def get_interview_or_404(cursor, interview_id: str):
    """Returns (user_id, status, target_role, candidate_profile_json, interview_state_json)."""
    cursor.execute("SELECT user_id, status, target_role, candidate_profile, interview_state FROM interviews WHERE id = ?", (interview_id,))
    row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Interview not found")
    return row

# --- 🤖 Message authorship / provider outage ---
PROVIDER_SYSTEM = "system"   # canned backend text (rejections, closing messages) rather than an LLM reply
PROVIDER_UNAVAILABLE_DETAIL = "The AI interviewer is temporarily unavailable. Please try again in a moment."

# --- 📄 Resume intake & interview context ---
MAX_RESUME_BYTES = 5 * 1024 * 1024
MAX_RESUME_PAGES = 20
MIN_RESUME_TEXT_CHARS = 50
MAX_ROLE_CHARS = 100

def normalize_role(role: str) -> str:
    role = re.sub(r"\s+", " ", role or "").strip()
    if not role:
        raise HTTPException(status_code=400, detail="Target role is required")
    if len(role) > MAX_ROLE_CHARS:
        raise HTTPException(status_code=400, detail=f"Target role must be at most {MAX_ROLE_CHARS} characters")
    return role

def extract_resume_text(pdf_bytes: bytes) -> str:
    """PDF bytes -> cleaned text, or a clear HTTP error if the file is unusable."""
    try:
        reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("password-protected PDF")
        pages = []
        for index, page in enumerate(reader.pages):
            if index >= MAX_RESUME_PAGES:
                break
            pages.append(page.extract_text() or "")
        text = clean_resume_text("\n".join(pages))
    except Exception as e:
        print(f"Resume PDF could not be parsed: {e}")
        raise HTTPException(status_code=422, detail="The resume could not be read. Please upload a valid, unprotected PDF.")
    if len(text) < MIN_RESUME_TEXT_CHARS:
        raise HTTPException(status_code=422, detail="No readable text was found in the resume. Please upload a text-based PDF (scanned images are not supported).")
    return text

def retrieve_for_interview(target_role: str, candidate_profile: dict, answer_number: int,
                           current_answer: str | None = None, topic_hint: str | None = None,
                           exclude_chunk_ids=None) -> RetrievalResult:
    """The one place endpoints call retrieval (domain filter, scoring, relevance and chunk-dedup live in rag.py)."""
    return retrieve_context(vector_db, target_role=target_role, candidate_profile=candidate_profile,
                            answer_number=answer_number, current_answer=current_answer,
                            topic_hint=topic_hint, exclude_chunk_ids=exclude_chunk_ids)

def embed_texts(texts: list) -> list:
    """Embeddings for the interview engine (answer relevance, duplicate-question detection): the same
    sentence-transformer model already loaded for RAG."""
    return embeddings.embed_documents(texts)

def load_turn_history(cursor, interview_id: str):
    """Transcript rows -> (Q/A pairs, earlier interviewer questions). The backend rebuilds memory from the
    database every turn, so nothing depends on which provider answered before."""
    cursor.execute("SELECT sender, text_content, llm_provider FROM messages WHERE interview_id = ? ORDER BY id ASC", (interview_id,))
    return history_from_messages(cursor.fetchall())

# Canned texts authored by the backend itself (never by a model)
RESIGNATION_MESSAGE = "If you are unwilling to proceed with the technical questions, we will conclude the assessment here. Thank you for your time."
# Normal ending (model-requested conclusion or the answer limit): no score, no verdict, no mention of how it is evaluated.
CONCLUSION_MESSAGE = CLOSING_MESSAGE

def build_interview_context(target_role: str, candidate_profile: dict, answer_number: int) -> dict:
    """Compact per-turn context built from stored data only (never from client input)."""
    return {
        "target_role": target_role,
        "candidate_profile": candidate_profile,
        "progress": {
            "answer_number": answer_number,
            "max_answers": MAX_CANDIDATE_ANSWERS,
            "answers_remaining": max(MAX_CANDIDATE_ANSWERS - answer_number, 0),
        },
    }

# --- 🔑 User & Access Management Endpoints ---

@app.post("/api/auth/signup")
async def signup(user: SignUpRequest):
    if user.role == "admin":
        if not ADMIN_SIGNUP_CODE:
            raise HTTPException(status_code=403, detail="Administrator sign-up is not enabled on this server")
        if not hmac.compare_digest((user.admin_code or "").encode("utf-8"), ADMIN_SIGNUP_CODE.encode("utf-8")):
            raise HTTPException(status_code=403, detail="Invalid or missing Admin Access Code")
    conn = sqlite3.connect("synapsift.db")
    cursor = conn.cursor()
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), salt).decode('utf-8')
    user_id = "user_" + os.urandom(4).hex()
    try:
        cursor.execute("INSERT INTO users (id, name, email, password_hash, role) VALUES (?, ?, ?, ?, ?)",
                      (user_id, user.name, user.email, hashed_password, user.role))
        conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Account registration failed: Email already registered")
    finally:
        conn.close()
    return {"message": "User registration completed successfully"}

@app.post("/api/auth/login")
async def login(credentials: LoginRequest):
    conn = sqlite3.connect("synapsift.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, password_hash, role FROM users WHERE email = ?", (credentials.email,))
    row = cursor.fetchone()
    conn.close()
    
    if not row or not bcrypt.checkpw(credentials.password.encode('utf-8'), row[2].encode('utf-8')):
        raise HTTPException(status_code=400, detail="Invalid credential pairing provided")
        
    token_payload = {
        "user_id": row[0],
        "name": row[1],
        "role": row[3],
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=2)
    }
    token = jwt.encode(token_payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return {"token": token, "role": row[3], "name": row[1]}

# --- 🤖 Dynamic Relational Interview Pipeline Endpoints ---

@app.post("/api/interview/start")
async def start_interview(role: str = Form(...), resume: UploadFile = File(...), user_meta: dict = Depends(get_current_user)):
    # --- Validate everything BEFORE anything is written to disk or the database ---
    role = normalize_role(role)

    resume_bytes = await resume.read(MAX_RESUME_BYTES + 1)
    if len(resume_bytes) > MAX_RESUME_BYTES:
        raise HTTPException(status_code=413, detail=f"Resume is too large (limit {MAX_RESUME_BYTES // (1024 * 1024)} MB)")
    # PDF (PyPDF2) or plain UTF-8 text; anything else (DOCX included) is refused with a clear error.
    try:
        resume_kind = detect_resume_kind(resume.filename, resume.content_type, resume_bytes[:1024])
        resume_text = extract_resume_text(resume_bytes) if resume_kind == KIND_PDF else extract_text_resume(resume_bytes, MIN_RESUME_TEXT_CHARS)
    except ResumeFormatError as e:
        raise HTTPException(status_code=e.status, detail=str(e))

    # Resume -> structured profile, once. Deterministic, so it cannot invent skills; if it ever
    # fails the interview still proceeds with an empty profile instead of crashing.
    try:
        candidate_profile = build_candidate_profile(resume_text)
    except Exception as e:
        print(f"Candidate profile extraction failed, continuing with empty profile: {e}")
        candidate_profile = empty_candidate_profile("failed")
    profile_text = profile_to_prompt_text(candidate_profile)

    interview_id = "interview_" + os.urandom(4).hex()
    file_path = f"uploads/{interview_id}.{'pdf' if resume_kind == KIND_PDF else 'txt'}"     # written only after the first question exists (see below)
    resume_url = f"http://127.0.0.1:8000/{file_path}"

    # === 🛡️ PYTHON DOMAIN GATEKEEPER (Fail-Fast) ===
    # Catch non-tech roles instantly in Python so the LLM doesn't get confused (and skip retrieval entirely)
    role_rejected = is_unsupported_role(role)

    retrieval = None     # RAG result behind the first question (None when nothing was retrieved)
    outcome = None       # the engine's decision for the first question (None for a rejected role)
    turn_ctx = None
    state = fresh_state(candidate_profile)   # starting difficulty comes from VERIFIED resume signals only

    if role_rejected:
        first_question = f"Thank you for your interest in the {role} position. However, this platform is explicitly designed to evaluate candidates for Software Engineering, Data Science, and AI roles. We are unable to conduct a technical screening for this specific domain. We appreciate your time and wish you the best in your job search!"
        initial_status = "COMPLETED"
    else:
        initial_status = "ONGOING"
        domain = rag_cfg.resolve_domain(role)
        state, plan = plan_next_step(state, None, domain, profile=candidate_profile, seed=interview_id)
        # Role-aware retrieval for the FIRST question: query = topic to cover + role + the profile terms that
        # belong to the domain. Roles without a corpus get no retrieval and no substitute material.
        retrieval = retrieve_for_interview(role, candidate_profile, answer_number=0, topic_hint=plan.topic)
        turn_ctx = TurnContext(
            target_role=role, domain=domain, profile=candidate_profile, answer_number=0, max_answers=MAX_CANDIDATE_ANSWERS,
            state=state, plan=plan, quality=None, turns=[], previous_questions=[], reference=retrieval.selected,
            latest_answer=None, conclusion_allowed=False, candidate_name=user_meta.get("name"))
        try:
            outcome = generate_interviewer_turn(turn_ctx, generate_chat_response, embed_texts)
        except ProviderUnavailableError as e:
            # No interview row, no message and no resume file exist yet: nothing to clean up.
            print(f"Interview start aborted, no LLM provider available: {e}")
            raise HTTPException(status_code=503, detail=PROVIDER_UNAVAILABLE_DETAIL)
        first_question = outcome.question

    with open(file_path, "wb") as f:
        f.write(resume_bytes)

    # Provenance. Canned rejection text belongs to the system; a question to whoever produced it (a provider,
    # or the backend's deterministic fallback when a provider answered but nothing acceptable came back).
    author = (PROVIDER_SYSTEM, None) if outcome is None else (outcome.provider, outcome.model)
    # The retrieved chunk only counts as "used" if a model actually built the question on it.
    used_retrieval = retrieval if (outcome is not None and outcome.generation != "fallback") else None
    used_chunk = used_retrieval.selected if used_retrieval else None
    state_json = None
    if outcome is not None:
        state_json = dump_state(record_question(state, plan, outcome.topic, used_chunk.chunk_id if used_chunk else None,
                                                preamble=outcome.preamble))

    conn = sqlite3.connect("synapsift.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO interviews (id, user_id, target_role, resume_text, resume_url, candidate_profile, interview_state, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                  (interview_id, user_meta["user_id"], role, resume_text, resume_url, json.dumps(candidate_profile), state_json, initial_status))
    if outcome is not None and outcome.generation == "fallback":
        trace = build_rag_trace(None, role, reason_override="deterministic fallback question; retrieved material was not used")
    else:
        trace = build_rag_trace(retrieval, role, reason_override="role rejected before retrieval")
    cursor.execute("INSERT INTO messages (interview_id, sender, text_content, rag_source_chunk, rag_trace, llm_provider, llm_model, gen_meta) VALUES (?, 'ai', ?, ?, ?, ?, ?, ?)",
                  (interview_id, first_question,
                   "System Guardrail" if initial_status == "COMPLETED" else (used_chunk.text if used_chunk else None),
                   json.dumps(trace), author[0], author[1],
                   json.dumps(outcome.meta(turn_ctx)) if outcome is not None else None))
    conn.commit()
    conn.close()

    return {"interview_id": interview_id, "first_question": first_question, "status": initial_status}

@app.post("/api/interview/chat")
async def chat_round(payload: ChatPayload, user_meta: dict = Depends(get_current_user)):
    conn = sqlite3.connect("synapsift.db")
    cursor = conn.cursor()

    try:
        # BEGIN IMMEDIATE makes ownership/status/turn-count/insert one atomic step, so
        # concurrent submissions cannot both slip under the answer cap.
        cursor.execute("BEGIN IMMEDIATE")
        owner_id, interview_status, target_role, stored_profile, stored_state = get_interview_or_404(cursor, payload.interview_id)

        # === 🔐 OWNERSHIP: only the candidate who owns the interview may take part in it ===
        if owner_id != user_meta["user_id"]:
            raise HTTPException(status_code=403, detail="You do not have access to this interview")

        # === 🛑 ZOMBIE CHAT GUARDRAIL ===
        # Check if the interview was already rejected or completed
        already_concluded = interview_status == 'COMPLETED'
        answers_before = 0
        candidate_message_id = None   # row id of THIS turn's answer, so it can be withdrawn if no provider answers

        if not already_concluded:
            cursor.execute("SELECT COUNT(*) FROM messages WHERE interview_id = ? AND sender = 'candidate'", (payload.interview_id,))
            answers_before = cursor.fetchone()[0]

            if answers_before >= MAX_CANDIDATE_ANSWERS:
                # Defensive: an interview already holding the maximum number of answers never accepts another.
                cursor.execute("UPDATE interviews SET status = 'COMPLETED' WHERE id = ?", (payload.interview_id,))
                already_concluded = True
            else:
                cursor.execute("INSERT INTO messages (interview_id, sender, text_content) VALUES (?, 'candidate', ?)",
                              (payload.interview_id, payload.message))
                candidate_message_id = cursor.lastrowid
        conn.commit()
    except Exception:
        conn.close()  # discards any open transaction
        raise

    if already_concluded:
        conn.close()
        return {"reply": "This interview session has already been concluded.", "status": "COMPLETED"}

    # This answer is the Nth accepted one (1-based). N == MAX means it is the last legitimate answer.
    answer_number = answers_before + 1

    # Authoritative context: role and profile come from the database, never from the request.
    interview_context = build_interview_context(target_role, parse_stored_profile(stored_profile), answer_number)

    # === 🛡️ FAIL-FAST TRAP (runs on EVERY answer, including the last one) ===
    # Only explicit resignation ends the interview. Profanity is detected but, on its own,
    # is not grounds for termination (no conduct-warning system yet).
    is_profane = profanity.contains_profanity(payload.message)
    if is_profane:
        print(f"Profanity detected in {payload.interview_id} (answer {answer_number}); not terminating on profanity alone.")
    is_resigning = is_explicit_resignation(payload.message)

    if answer_number >= MAX_CANDIDATE_ANSWERS and not is_resigning:
        closing_msg = CLOSING_MESSAGE
        cursor.execute("INSERT INTO messages (interview_id, sender, text_content, rag_source_chunk, rag_trace, llm_provider, llm_model) VALUES (?, 'ai', ?, ?, ?, ?, ?)",
                      (payload.interview_id, closing_msg, "System Hard Stop",
                       json.dumps(build_rag_trace(None, target_role, reason_override="interview answer limit reached")),
                       PROVIDER_SYSTEM, None))
        cursor.execute("UPDATE interviews SET status = 'COMPLETED' WHERE id = ?", (payload.interview_id,))
        conn.commit()
        conn.close()
        return {"reply": closing_msg, "status": "COMPLETED"}

    # === 🛡️ PRODUCTION GATEWAY ===
    # (resignation flag was already computed above)
    next_question, is_completed = None, False
    outcome = turn_ctx = None
    trace_retrieval, trace_reason, new_state_json = None, "interview ended", None
    system_meta = None

    if is_resigning:
        # Explicit resignation is decided by Python alone: no retrieval, no provider, canned text.
        next_question, is_completed = RESIGNATION_MESSAGE, True
        trace_reason = "interview ended by the candidate's resignation"
    else:
        profile = interview_context["candidate_profile"]
        domain = rag_cfg.resolve_domain(target_role)

        # --- Interview memory: stored state + the transcript. Built here, BEFORE any provider is called. ---
        state = load_state(stored_state, profile)
        pairs, previous_questions = load_turn_history(cursor, payload.interview_id)   # includes this answer
        last_question = pairs[-1][0] if pairs else None
        severe_abuse = is_severe_abuse(payload.message, profanity.contains_profanity)
        quality = classify_answer(payload.message, last_question, state.current_topic, embed_texts, profane=is_profane, severe_abuse=severe_abuse)
        state, plan = plan_next_step(state, quality, domain, previous_questions,
                                     answer_number=answer_number, max_answers=MAX_CANDIDATE_ANSWERS)

        if plan.conduct_terminate:
            # Repeated severe abuse aimed at the interviewer: decided by Python alone (no retrieval, no provider). The answers given
            # so far are untouched and remain evaluable; this is NOT a resignation and carries its own termination reason.
            next_question, is_completed = CONDUCT_TERMINATION_MESSAGE, True
            new_state_json = dump_state(state)
            trace_reason = "interview concluded after repeated severe conduct violations"
            system_meta = {"action": "terminate", "termination_reason": CONDUCT_TERMINATION_REASON, "severe_abuse": True,
                           "conduct_strikes": state.conduct_strikes, "conduct_warnings": state.conduct_warnings,
                           "conduct_consecutive_after_warning": state.conduct_consecutive, "answer_number": answer_number}
        else:
            # Role-aware retrieval from trusted context. Chunks already used in this interview are skipped; the
            # candidate's answer only nudges the query, and only when it was substantive (see answer_for_retrieval).
            retrieval = retrieve_for_interview(
                target_role, profile, interview_context["progress"]["answer_number"], current_answer=answer_for_retrieval(payload.message, quality),
                topic_hint=plan.topic, exclude_chunk_ids=set(state.used_chunk_ids))

            turn_ctx = TurnContext(
                target_role=target_role, domain=domain, profile=profile, answer_number=answer_number,
                max_answers=MAX_CANDIDATE_ANSWERS, state=state, plan=plan, quality=quality, turns=pairs,
                previous_questions=previous_questions, reference=retrieval.selected, latest_answer=payload.message,
                conclusion_allowed=conclusion_allowed(answer_number, state), candidate_name=user_meta.get("name"))
            try:
                outcome = generate_interviewer_turn(turn_ctx, generate_chat_response, embed_texts)
            except ProviderUnavailableError as e:
                # Nothing is invented. Withdraw this turn's answer so the transcript and the answer count are
                # exactly as before the request, and let the candidate resend once a provider is back. The stored
                # interview state was never touched.
                print(f"Chat turn aborted for {payload.interview_id}, no LLM provider available: {e}")
                cursor.execute("DELETE FROM messages WHERE id = ?", (candidate_message_id,))
                conn.commit()
                conn.close()
                raise HTTPException(status_code=503, detail=PROVIDER_UNAVAILABLE_DETAIL)

            if outcome.action == "conclude":
                # The model asked to end early (allowed only after a minimum number of answers/topics).
                next_question, is_completed = CONCLUSION_MESSAGE, True
                new_state_json = dump_state(state)
                trace_reason = "interview concluded by the interviewer; closing message"
            else:
                next_question = outcome.question
                if outcome.generation == "fallback":
                    trace_reason = "deterministic fallback question; retrieved material was not used"
                else:
                    trace_retrieval = retrieval
                used_chunk = trace_retrieval.selected if trace_retrieval else None
                new_state_json = dump_state(record_question(state, plan, outcome.topic, used_chunk.chunk_id if used_chunk else None,
                                                            preamble=outcome.preamble))

    selected_chunk = trace_retrieval.selected if trace_retrieval else None
    # Canned text (resignation) is the system's; a question is attributed to whoever produced it.
    author = (outcome.provider, outcome.model) if outcome else (PROVIDER_SYSTEM, None)
    cursor.execute("INSERT INTO messages (interview_id, sender, text_content, rag_source_chunk, rag_trace, llm_provider, llm_model, gen_meta) VALUES (?, 'ai', ?, ?, ?, ?, ?, ?)",
                  (payload.interview_id, next_question, selected_chunk.text if selected_chunk else None,
                   json.dumps(build_rag_trace(trace_retrieval, target_role, reason_override=trace_reason)),
                   author[0], author[1], json.dumps(outcome.meta(turn_ctx)) if outcome else (json.dumps(system_meta) if system_meta else None)))

    if new_state_json is not None:
        cursor.execute("UPDATE interviews SET interview_state = ? WHERE id = ?", (new_state_json, payload.interview_id))
    if is_completed:
        cursor.execute("UPDATE interviews SET status = 'COMPLETED' WHERE id = ?", (payload.interview_id,))

    conn.commit()
    conn.close()

    return {"reply": next_question, "status": "COMPLETED" if is_completed else "ONGOING"}

@app.get("/api/interview/summary/{interview_id}")
async def fetch_session_summary(interview_id: str, user_meta: dict = Depends(get_current_user)):
    conn = sqlite3.connect("synapsift.db")
    cursor = conn.cursor()
    
    # 0. OWNERSHIP: the owning candidate or an admin may read a summary
    cursor.execute("SELECT user_id, evaluation_data, resume_url FROM interviews WHERE id = ?", (interview_id,))
    interview_row = cursor.fetchone()
    if not interview_row:
        conn.close()
        raise HTTPException(status_code=404, detail="Interview not found")
    if interview_row[0] != user_meta["user_id"] and user_meta.get("role") != "admin":
        conn.close()
        raise HTTPException(status_code=403, detail="You do not have access to this interview")
    row = (interview_row[1], interview_row[2])

    # 1. CHECK CACHE FIRST. Only a valid report of the CURRENT schema version is served; anything older (or
    # malformed) is recomputed, and a failed recomputation leaves the stored row untouched.
    cached_json = cached_report(row[0]) if row[0] else None
    if cached_json is not None:
        cached_json["resume_url"] = row[1]
        conn.close()
        return cached_json

    # 2. BUILD THE TRANSCRIPT: only real question -> answer pairs (never greetings, closings, rejections, resignations)
    cursor.execute("SELECT sender, text_content, llm_provider, gen_meta FROM messages WHERE interview_id = ? ORDER BY id ASC", (interview_id,))
    rows = cursor.fetchall()

    if not rows:
        conn.close()
        raise HTTPException(status_code=404, detail="Requested screening logs could not be located")

    transcript = build_transcript(rows)

    # === 🛑 TOKEN SAVER: BYPASS LLM FOR TERMINATIONS & REJECTIONS ===
    if transcript.rejected or transcript.terminated:
        report = terminated_report(transcript)
    else:
        # 3. EVALUATE. Provider outages and unusable output are NOT candidate results: nothing is saved, the
        # client gets a 503, and the evaluation can simply be requested again.
        try:
            report = evaluate_transcript(transcript, lambda prompt: generate_eval_response([HumanMessage(content=prompt)]))
        except EvaluationUnavailableError as e:
            print(f"Evaluation unavailable, nothing saved: {e}")
            conn.close()
            raise HTTPException(status_code=503, detail="Evaluation is temporarily unavailable. Please try again in a moment.")

    cursor.execute("UPDATE interviews SET overall_score = ?, evaluation_summary = ?, evaluation_data = ?, status = 'COMPLETED' WHERE id = ?",
                  (report["overallScore"], report["summary"], json.dumps(report), interview_id))
    conn.commit()

    report["resume_url"] = row[1]
    conn.close()
    return report

@app.get("/api/admin/candidates")
async def get_all_candidates(user_meta: dict = Depends(get_current_user)):
    if user_meta.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Access denied: Admin credentials required")
        
    conn = sqlite3.connect("synapsift.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT u.name, u.email, i.id, i.target_role, i.status, i.created_at 
        FROM interviews i 
        JOIN users u ON i.user_id = u.id 
        ORDER BY i.created_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {"name": r[0], "email": r[1], "interview_id": r[2], "role": r[3], "status": r[4], "date": r[5]} 
        for r in rows
    ]

@app.delete("/api/user/account")
async def delete_account(user_meta: dict = Depends(get_current_user)):
    conn = sqlite3.connect("synapsift.db")
    cursor = conn.cursor()
    user_id = user_meta["user_id"]
    
    cursor.execute("DELETE FROM messages WHERE interview_id IN (SELECT id FROM interviews WHERE user_id = ?)", (user_id,))
    cursor.execute("DELETE FROM interviews WHERE user_id = ?", (user_id,))
    cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
    
    conn.commit()
    conn.close()
    return {"message": "Account permanently deleted"}

@app.delete("/api/admin/interview/{interview_id}")
async def delete_interview(interview_id: str, user_meta: dict = Depends(get_current_user)):
    if user_meta.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Access denied: Admin credentials required")
        
    conn = sqlite3.connect("synapsift.db")
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM messages WHERE interview_id = ?", (interview_id,))
    cursor.execute("DELETE FROM interviews WHERE id = ?", (interview_id,))
    
    conn.commit()
    conn.close()
    return {"message": "Candidate record wiped"}

@app.delete("/api/interview/abort/{interview_id}")
async def abort_interview(interview_id: str, user_meta: dict = Depends(get_current_user)):
    conn = sqlite3.connect("synapsift.db")
    cursor = conn.cursor()
    
    try:
        owner_id = get_interview_or_404(cursor, interview_id)[0]
        if owner_id != user_meta["user_id"]:
            raise HTTPException(status_code=403, detail="You do not have access to this interview")
    except HTTPException:
        conn.close()
        raise

    cursor.execute("DELETE FROM messages WHERE interview_id = ?", (interview_id,))
    cursor.execute("DELETE FROM interviews WHERE id = ? AND user_id = ?", (interview_id, user_meta["user_id"]))

    conn.commit()
    conn.close()
    return {"message": "Interview aborted. Data wiped from system."}