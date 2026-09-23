import os
import json
import sqlite3
import random
import jwt
import datetime
import PyPDF2
import httpx
import bcrypt
import re
from io import BytesIO
from better_profanity import profanity
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore 
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load Configurations & Security Elements
load_dotenv()
PRIMARY_KEY = os.getenv("GEMINI_API_KEY")
BACKUP_KEY = os.getenv("GEMINI_BACKUP_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
JWT_SECRET = os.getenv("JWT_SECRET", "super_secret_synapsift_key_2026")
JWT_ALGORITHM = "HS256"

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

    conn.commit()
    conn.close()

init_relational_db()

# 3. Hybrid AI Engine: OpenRouter (Chat) + Gemini 3.5 (Eval) + Local Ollama
def generate_ai_response(messages: list, task_type: str = "chat") -> str:
    # Look, I remembered my own version this time!
    gemini_model = "gemini-3.5-flash"
    
    def extract_text(content):
        if isinstance(content, list):
            if len(content) > 0 and isinstance(content[0], dict):
                return content[0].get("text", str(content))
        return str(content)

    raw_prompt_string = "\n".join([str(m.content) for m in messages])

    # --- TASK: INTERVIEW CHAT (Strictly Free APIs + Local) ---
    if task_type == "chat":
        if OPENROUTER_API_KEY:
            try:
                print("🌐 Querying OpenRouter (NVIDIA Nemotron-3-ultra-550b) for Chat...")
                llm_openrouter = ChatOpenAI(
                    model="nvidia/nemotron-3-ultra-550b-a55b:free", 
                    api_key=OPENROUTER_API_KEY,
                    base_url="https://openrouter.ai/api/v1"
                )
                response = llm_openrouter.invoke(messages)
                res_text = extract_text(response.content).strip()
                
                # --- THE BLANK TEXT GUARDRAIL ---
                # If the AI returns nothing, force a failure to trigger Mistral
                if not res_text:
                    raise ValueError("OpenRouter API succeeded, but the model returned an empty string.")
                    
                return res_text
            except Exception as e_or:
                print(f"⚠️ OpenRouter failed: {e_or}. Activating Local Mistral...")

        # Fallback directly to Local Mistral. NO Gemini used here to save API limits!
        try:
            print("🤖 Querying Local Mistral Node via Ollama...")
            with httpx.Client(timeout=300.0) as client:
                ollama_response = client.post(
                    "http://127.0.0.1:11434/api/chat",
                    json={
                        "model": "mistral",
                        "messages": [{"role": "user", "content": raw_prompt_string}],
                        "stream": False
                    }
                )
                if ollama_response.status_code == 200:
                    return ollama_response.json().get("message", {}).get("content", "Local Mistral failed to generate content.")
        except Exception as e_ollama:
            print(f"🚨 Critical Failure: Local Ollama instance unreachable. Details: {e_ollama}")
            
        return "Interviewer service temporarily degraded. Please submit your answer again."

    # --- TASK: EVALUATION (Strictly Gemini 3.5 + Local) ---
    if task_type == "eval":
        try:
            print(f"🧠 Querying {gemini_model} API for Evaluation...")
            llm = ChatGoogleGenerativeAI(model=gemini_model, google_api_key=PRIMARY_KEY)
            response = llm.invoke(messages)
            return extract_text(response.content)
        except Exception as e_primary:
            print(f"⚠️ Primary Gemini failed: {e_primary}. Transitioning to Backup Key...")
            try:
                if BACKUP_KEY:
                    llm_backup = ChatGoogleGenerativeAI(model=gemini_model, google_api_key=BACKUP_KEY)
                    backup_response = llm_backup.invoke(messages)
                    return extract_text(backup_response.content)
            except Exception as e_backup:
                print(f"⚠️ Backup Gemini failed: {e_backup}. Activating Local Mistral...")

        # Fallback to local Mistral for Evaluation
        try:
            print("🤖 Querying Local Mistral Node via Ollama for Evaluation...")
            with httpx.Client(timeout=300.0) as client:
                ollama_response = client.post(
                    "http://127.0.0.1:11434/api/chat",
                    json={
                        "model": "mistral",
                        "messages": [{"role": "user", "content": raw_prompt_string}],
                        "stream": False
                    }
                )
                if ollama_response.status_code == 200:
                    return ollama_response.json().get("message", {}).get("content", "Local Mistral failed to generate content.")
        except Exception as e_ollama:
            print(f"🚨 Critical Failure: Local Ollama instance unreachable. Details: {e_ollama}")
            
        return "{}"
        
    return "Interviewer service temporarily degraded. Please submit your answer again."

# 4. RAG Ingestion Pipeline: Qdrant Setup
# Enforce offline mode to bypass Hugging Face network checks
os.environ["HF_HUB_OFFLINE"] = "1"
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'local_files_only': True}
)

def initialize_vector_db():
    local_qdrant_path = "./qdrant_db"
    collection_name = "textbook_knowledge"
    client = QdrantClient(path=local_qdrant_path)

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
    return QdrantVectorStore(client=client, collection_name=collection_name, embedding=embeddings)

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
    role: str

def get_current_user(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authentication credentials")
    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Authentication session expired or malformed")

# --- 🔑 User & Access Management Endpoints ---

@app.post("/api/auth/signup")
async def signup(user: SignUpRequest):
    if user.role == "admin" and user.admin_code != "PGAGI-RECRUITER-2026":
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
    interview_id = "interview_" + os.urandom(4).hex()
    
    pdf_bytes = await resume.read()
    
    file_path = f"uploads/{interview_id}.pdf"
    with open(file_path, "wb") as f:
        f.write(pdf_bytes)
    resume_url = f"http://127.0.0.1:8000/{file_path}"

    reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    resume_text = "".join([page.extract_text() + "\n" for page in reader.pages])[:1500]

    search_query = f"Target Role: {role}. Candidate Background: {resume_text[:300]}"
    
    try:
        docs = vector_db.similarity_search(search_query, k=1, filter={"target_role": role})
    except Exception:
        docs = []

    if not docs:
        print(f"🔎 Broadening RAG search to global knowledge base for '{role}'...")
        docs = vector_db.similarity_search(search_query, k=1)
        
    rag_context = docs[0].page_content if docs else "General software engineering and machine learning principles."

    # === 🛡️ PYTHON DOMAIN GATEKEEPER (Fail-Fast) ===
    # Catch non-tech roles instantly in Python so the LLM doesn't get confused
    invalid_domains = ["yoga", "gym", "fitness", "chef", "plumber", "sales", "hr", "driver", "retail", "medical"]
    
    if any(word in role.lower() for word in invalid_domains):
        first_question = "[DOMAIN_REJECT]"
    else:
        # --- AI GENERATION FOR VALID ROLES ---
        prompt = HumanMessage(
            content=f"""
            [SYSTEM: SENIOR STAFF ENGINEER]
            Candidate Role: {role}
            Resume Extract: {resume_text[:300]}
            Technical Topic: {rag_context}
            
            STRICT RULES:
            1. NO MULTIPLE CHOICE: NEVER ask multiple-choice questions. NEVER give options (A, B, C, D). 
            2. OPEN-ENDED ONLY: Ask an open-ended, conversational question that forces the candidate to explain the concept.
            3. NO LISTS: NEVER use bullet points or numbered lists.
            4. Greet the candidate and mention ONE tech skill from their resume.
            5. Ask EXACTLY ONE technical question based on the topic. Keep it under 3 sentences.
            6. BLIND RULE: NEVER use the words "textbook", "context", "Figure", "Table", or "quote".
            """
        )
        first_question = generate_ai_response([prompt], task_type="chat")

    # --- DOMAIN REJECTION HANDLER ---
    if "[DOMAIN_REJECT]" in first_question:
        first_question = f"Thank you for your interest in the {role} position. However, this platform is explicitly designed to evaluate candidates for Software Engineering, Data Science, and AI roles. We are unable to conduct a technical screening for this specific domain. We appreciate your time and wish you the best in your job search!"
        initial_status = "COMPLETED"
    else:
        initial_status = "ONGOING"

    conn = sqlite3.connect("synapsift.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO interviews (id, user_id, target_role, resume_text, resume_url, status) VALUES (?, ?, ?, ?, ?, ?)", 
                  (interview_id, user_meta["user_id"], role, resume_text, resume_url, initial_status))
    cursor.execute("INSERT INTO messages (interview_id, sender, text_content, rag_source_chunk) VALUES (?, 'ai', ?, ?)", 
                  (interview_id, first_question, "System Guardrail" if initial_status == "COMPLETED" else rag_context))
    conn.commit()
    conn.close()

    return {"interview_id": interview_id, "first_question": first_question, "status": initial_status}

@app.post("/api/interview/chat")
async def chat_round(payload: ChatPayload, user_meta: dict = Depends(get_current_user)):
    conn = sqlite3.connect("synapsift.db")
    cursor = conn.cursor()
    
    # === 🛑 ZOMBIE CHAT GUARDRAIL ===
    # Check if the interview was already rejected or completed
    cursor.execute("SELECT status FROM interviews WHERE id = ?", (payload.interview_id,))
    status_row = cursor.fetchone()
    if not status_row or status_row[0] == 'COMPLETED':
        conn.close()
        return {"reply": "This interview session has already been concluded.", "status": "COMPLETED"}
    
    cursor.execute("SELECT COUNT(*) FROM messages WHERE interview_id = ? AND sender = 'candidate'", (payload.interview_id,))
    turns_taken = cursor.fetchone()[0]
    
    cursor.execute("INSERT INTO messages (interview_id, sender, text_content) VALUES (?, 'candidate', ?)", 
                  (payload.interview_id, payload.message))
    conn.commit()

    if turns_taken >= 10:
        closing_msg = "Thank you so much for your time today. We've covered a lot of ground, and I have all the information I need. I wish you the best of luck, and you can proceed to generate your evaluation results now!"
        cursor.execute("INSERT INTO messages (interview_id, sender, text_content, rag_source_chunk) VALUES (?, 'ai', ?, ?)", 
                      (payload.interview_id, closing_msg, "System Hard Stop"))
        cursor.execute("UPDATE interviews SET status = 'COMPLETED' WHERE id = ?", (payload.interview_id,))
        conn.commit()
        conn.close()
        return {"reply": closing_msg, "status": "COMPLETED"}

    search_query = payload.message if len(payload.message) > 15 else f"Advanced interview concepts for {payload.role}"
    
    try:
        docs = vector_db.similarity_search(search_query, k=1, filter={"target_role": payload.role})
    except Exception:
        docs = []

    if not docs:
        docs = vector_db.similarity_search(search_query, k=1)
        
    rag_context = docs[0].page_content if docs else "Core computer systems engineering."

    # === 🛡️ PRODUCTION GATEWAY: FAIL-FAST TRAP ===
    
    # 1. Expanded Hardcoded Fail-Fast: Profanity, Hostility & Resignations
    is_profane = profanity.contains_profanity(payload.message)
    hostile_keywords = [
        "give up", "i won't", "refuse", "stop asking", "i quit", "pass", "skip",
        "shut up", "go away", "fck", "stfu", "bullshit", "waste of time", "fuck off"
    ]
    is_resigning = any(word in payload.message.lower() for word in hostile_keywords)

    if is_profane or is_resigning:
        next_question = "[TERMINATE_PROFANITY]" if is_profane else "[TERMINATE_RESIGNATION]"
    else:
        # 2. AI Reasoning: Dynamic Follow-Ups and Escapes
        candidate_name = user_meta.get("name", "the candidate")
        
        prompt = HumanMessage(
            content=f"""
            [SYSTEM ROLE: SENIOR STAFF ENGINEER]
            Candidate Name: {candidate_name}
            Candidate's Answer: "{payload.message}"
            Available New Topic: {rag_context}
            
            [INSTRUCTIONS]
            1. JUDGE & REACT: 
               - If their answer is lazy, vague, or short (e.g., "I don't know" or "just test it"): DO NOT conclude. You MUST push back aggressively. Ask a tough follow-up question demanding technical specifics.
               - If their answer is strong and detailed: Ask a NEW question based on the Available New Topic.
            2. DYNAMIC ENDING: Output EXACTLY [CONCLUDE_INTERVIEW] ONLY if the candidate has given highly detailed, masterful answers and you are completely satisfied. Never conclude after a weak or lazy answer!
            
            [STRICT BAN LIST]
            1. NO LISTS: Never output numbered lists or bullet points.
            2. NO THIRD PERSON: Speak directly to {candidate_name} using "you".
            3. BLIND RULE: Never mention "Figure", "Table", "Section", or "textbook".
            4. Keep your response strictly under 3 sentences.
            """
        )
        next_question = generate_ai_response([prompt], task_type="chat")

    # --- TERMINATION HANDLER ---
    is_completed = False
    
    if "[TERMINATE_PROFANITY]" in next_question:
        next_question = "I'm ending the interview right here. We maintain strict professional standards for our engineers, and that kind of language is completely unacceptable. This session is closed."
        is_completed = True
        
    elif "[TERMINATE_RESIGNATION]" in next_question:
        next_question = "If you are unwilling to proceed with the technical questions, we will conclude the assessment here. Thank you for your time."
        is_completed = True
        
    elif "[CONCLUDE_INTERVIEW]" in next_question:
        next_question = "Thank you, that gives me a solid understanding of your technical depth. We've covered everything I need today. I wish you the best, and you can proceed to generate your results now!"
        is_completed = True

    cursor.execute("INSERT INTO messages (interview_id, sender, text_content, rag_source_chunk) VALUES (?, 'ai', ?, ?)", 
                  (payload.interview_id, next_question, rag_context))
    
    if is_completed:
        cursor.execute("UPDATE interviews SET status = 'COMPLETED' WHERE id = ?", (payload.interview_id,))
        
    conn.commit()
    conn.close()

    return {"reply": next_question, "status": "COMPLETED" if is_completed else "ONGOING"}

@app.get("/api/interview/summary/{interview_id}")
async def fetch_session_summary(interview_id: str, user_meta: dict = Depends(get_current_user)):
    conn = sqlite3.connect("synapsift.db")
    cursor = conn.cursor()
    
    # 1. CHECK CACHE FIRST
    cursor.execute("SELECT evaluation_data, resume_url FROM interviews WHERE id = ?", (interview_id,))
    row = cursor.fetchone()
    
    if row and row[0]: 
        parsed_json = json.loads(row[0])
        parsed_json["resume_url"] = row[1]
        conn.close()
        return parsed_json
        
    # 2. GENERATE WITH GEMINI
    cursor.execute("SELECT sender, text_content FROM messages WHERE interview_id = ? ORDER BY id ASC", (interview_id,))
    rows = cursor.fetchall()
    
    if not rows:
        conn.close()
        raise HTTPException(status_code=404, detail="Requested screening logs could not be located")
        
    qa_pairs = []
    current_question = None
    
    for sender, text in rows:
        if sender == 'ai':
            current_question = text
        elif sender == 'candidate' and current_question:
            qa_pairs.append({"question": current_question, "answer": text})
            current_question = None

    # === 🛑 TOKEN SAVER: BYPASS LLM FOR TERMINATIONS & REJECTIONS ===
    # Look at the final message the AI sent
    last_ai_msg = next((text for sender, text in reversed(rows) if sender == 'ai'), "")
    
    # If the transcript is empty (Domain Reject) or contains our termination strings
    is_rejected = len(qa_pairs) == 0
    is_terminated = "session is closed" in last_ai_msg or "unwilling to proceed" in last_ai_msg or "unable to conduct" in last_ai_msg

    if is_rejected or is_terminated:
        parsed_json = {
            "overallScore": 0,
            "summary": "Interview terminated early due to domain rejection, resignation, or policy violation.",
            "insights": "Automated failure. The LLM evaluation phase was bypassed to conserve system resources.",
            "breakdown": [
                {
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "score": 0,
                    "feedback": "Score voided due to early termination."
                } for qa in qa_pairs
            ]
        }
        
        # Save bypass result directly to database
        json_str = json.dumps(parsed_json)
        cursor.execute("UPDATE interviews SET overall_score = 0, evaluation_summary = ?, evaluation_data = ?, status = 'COMPLETED' WHERE id = ?", 
                      (parsed_json["summary"], json_str, interview_id))
        conn.commit()
        
        parsed_json["resume_url"] = row[1] if row else None
        conn.close()
        return parsed_json

    # --- COMPLETE TRANSCRIPT EVALUATION ---
    verification_payload = json.dumps(qa_pairs, indent=2)
    
    prompt = HumanMessage(
        content=f"""Analyze this COMPLETE technical interview transcript:
        {verification_payload}
        
        Evaluate the candidate and output ONLY a valid JSON object matching this exact structure. 
        CRITICAL RULES:
        1. You must evaluate EVERY question. Output EXACTLY {len(qa_pairs)} items in the "breakdown" array. Do not skip any.
        2. RUTHLESS SCORING: If the candidate answers "I don't know", "no", "uhh", gives gibberish, or uses slang, the score for that question MUST BE EXACTLY 0.
        3. The "overallScore" must be the true mathematical average of all {len(qa_pairs)} individual question scores.
        
        {{
            "overallScore": 85,
            "summary": "2 sentence overall summary.",
            "insights": "Strengths: X. Weaknesses: Y.",
            "breakdown": [
                {{
                    "question": "The question asked",
                    "answer": "The candidate's answer",
                    "score": 0,
                    "feedback": "1 sentence strict feedback explaining the score."
                }}
            ]
        }}"""
    )
    response_text = generate_ai_response([prompt], task_type="eval")
    
    try:
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            parsed_json = json.loads(json_str)
        else:
            raise ValueError("No JSON boundaries found.")
            
    except Exception as e:
        print(f"JSON Parsing Failed: {e}. Raw Response: {response_text}")
        parsed_json = {
            "overallScore": 0, 
            "summary": "Evaluation failed to parse.",
            "insights": "Please review logs manually.",
            "breakdown": []
        }

    # === 🛡️ BACKEND STRUCTURAL INTEGRITY OVERRIDE LOOP ===
    if "breakdown" in parsed_json and isinstance(parsed_json["breakdown"], list):
        # We ONLY override the text of the questions the AI actually evaluated. 
        # No unfair padding with 0s!
        for i in range(min(len(parsed_json["breakdown"]), len(qa_pairs))):
            parsed_json["breakdown"][i]["question"] = qa_pairs[i]["question"]
            parsed_json["breakdown"][i]["answer"] = qa_pairs[i]["answer"]

    json_str = json.dumps(parsed_json)

    cursor.execute("UPDATE interviews SET overall_score = ?, evaluation_summary = ?, evaluation_data = ?, status = 'COMPLETED' WHERE id = ?", 
                  (parsed_json.get("overallScore", 0), parsed_json.get("summary", ""), json_str, interview_id))
    conn.commit()
    
    parsed_json["resume_url"] = row[1] if row else None
    conn.close()
    return parsed_json

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
    
    cursor.execute("DELETE FROM messages WHERE interview_id = ?", (interview_id,))
    cursor.execute("DELETE FROM interviews WHERE id = ? AND user_id = ?", (interview_id, user_meta["user_id"]))
    
    conn.commit()
    conn.close()
    return {"message": "Interview aborted. Data wiped from system."}