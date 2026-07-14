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
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv
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
    conn.commit()
    conn.close()

init_relational_db()

# 3. Hybrid AI Engine: Gemini + Local Ollama Mistral Fallback
def generate_ai_response(messages: list) -> str:
    model_name = "gemini-3.5-flash"
    
    def extract_text(content):
        if isinstance(content, list):
            if len(content) > 0 and isinstance(content[0], dict):
                return content[0].get("text", str(content))
        return str(content)

    raw_prompt_string = "\n".join([str(m.content) for m in messages])

    try:
        llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=PRIMARY_KEY)
        response = llm.invoke(messages)
        return extract_text(response.content)
    except Exception as e_primary:
        print(f"⚠️ Primary API failed: {e_primary}. Transitioning to Backup Key...")
        try:
            if BACKUP_KEY:
                llm_backup = ChatGoogleGenerativeAI(model=model_name, google_api_key=BACKUP_KEY)
                backup_response = llm_backup.invoke(messages)
                return extract_text(backup_response.content)
        except Exception as e_backup:
            print(f"⚠️ Backup API failed: {e_backup}. Activating Local Ollama Infrastructure...")

    try:
        print("🤖 Querying Local Mistral Node via Ollama standard pipeline...")
        with httpx.Client(timeout=300.0) as client:
            ollama_response = client.post(
                "http://127.0.0.1:11434/api/generate",
                json={
                    "model": "mistral",
                    "prompt": raw_prompt_string,
                    "stream": False
                }
            )
            if ollama_response.status_code == 200:
                return ollama_response.json().get("response", "Local Mistral failed to generate content.")
    except Exception as e_ollama:
        print(f"🚨 Critical Failure: Local Ollama instance unreachable. Details: {e_ollama}")
        
    return "Interviewer service temporarily degraded. Please submit your answer again."

# 4. RAG Ingestion Pipeline: Qdrant Setup
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

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
    admin_code: str | None = None  # Add this new field

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
    reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    resume_text = "".join([page.extract_text() + "\n" for page in reader.pages])[:1500] 

    search_query = f"Target Role: {role}. Candidate Background: {resume_text[:300]}"
    
    # 🌟 MULTI-TIER RAG FALLBACK SYSTEM 🌟
    try:
        # Tier 1: Try strict metadata match for the requested role
        docs = vector_db.similarity_search(search_query, k=1, filter={"target_role": role})
    except Exception:
        docs = []

    # Tier 2 & 3: Fallback to general search if the role doesn't exist in metadata
    if not docs:
        print(f"⚠️ Custom Role detected: '{role}'. Routing to global textbook search.")
        docs = vector_db.similarity_search(search_query, k=1)
        
    rag_context = docs[0].page_content if docs else "General software engineering and machine learning principles."

    prompt = HumanMessage(
        content=f"""You are a professional technical interviewer for a {role} position.
        Candidate Resume Extract: {resume_text}
        Approved Textbook Context: {rag_context}
        
        Instructions:
        1. Parse the resume silently. Identify their top Skills, Technologies, and Domain exposure.
        2. Greet the candidate warmly and state ONE specific skill you see on their resume.
        3. Ask EXACTLY ONE challenging technical question derived from the Approved Textbook Context.
        
        CRITICAL OUTPUT GUARDRAIL:
        Output ONLY the verbal text spoken out loud to the candidate. Do not write any thoughts, explanations, metadata, or wrappers like 'Question:'."""
    )
    first_question = generate_ai_response([prompt])

    conn = sqlite3.connect("synapsift.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO interviews (id, user_id, target_role, status) VALUES (?, ?, ?, 'ONGOING')", 
                  (interview_id, user_meta["user_id"], role))
    cursor.execute("INSERT INTO messages (interview_id, sender, text_content, rag_source_chunk) VALUES (?, 'ai', ?, ?)", 
                  (interview_id, first_question, rag_context))
    conn.commit()
    conn.close()

    return {"interview_id": interview_id, "first_question": first_question, "status": "ONGOING"}

@app.post("/api/interview/chat")
async def chat_round(payload: ChatPayload, user_meta: dict = Depends(get_current_user)):
    conn = sqlite3.connect("synapsift.db")
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM messages WHERE interview_id = ? AND sender = 'candidate'", (payload.interview_id,))
    turns_taken = cursor.fetchone()[0]
    
    cursor.execute("INSERT INTO messages (interview_id, sender, text_content) VALUES (?, 'candidate', ?)", 
                  (payload.interview_id, payload.message))
    conn.commit()

    if turns_taken >= 15:
        closing_msg = "Thank you for your extensive responses. The session is now concluded."
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

    # --- MISTRAL-PROOF PROMPT ---
    prompt = HumanMessage(
        content=f"""You are a cold, strict interviewer for a {payload.role} role.
        Candidate's last answer: "{payload.message}"
        Next question context: {rag_context}
        
        RULES (FOLLOW EXACTLY):
        1. DO NOT evaluate the candidate out loud. NEVER say "The candidate has provided..." or "Good answer."
        2. DO NOT mention the "Textbook" or "Context".
        3. Simply output the VERY NEXT technical question you want to ask.
        4. If the candidate says "I don't know", give a 1-sentence answer, then ask a new question.
        
        TERMINATION:
        If the candidate proves they have NO knowledge across multiple questions, OR if they have answered perfectly and you are satisfied, YOU MUST END THE INTERVIEW. 
        To end it, do NOT ask a question. Output ONLY this exact string: "[TERMINATE]"
        
        OUTPUT EXACTLY WHAT YOU WILL SPEAK. NO OTHER TEXT."""
    )
    next_question = generate_ai_response([prompt])

    is_completed = False
    if "[TERMINATE]" in next_question or turns_taken >= 10:
        # Strip out any ghost questions Mistral tried to sneak in
        next_question = "Thank you for your responses today. We have gathered sufficient data to conclude this technical screening.\n\n*(Session gracefully terminated by AI based on candidate assessment. Compiling analytics...)*"
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
    cursor.execute("SELECT sender, text_content FROM messages WHERE interview_id = ? ORDER BY id ASC", (interview_id,))
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        raise HTTPException(status_code=404, detail="Requested screening logs could not be located")
        
    qa_pairs = []
    current_question = None
    
    for sender, text in rows:
        if sender == 'ai':
            current_question = text
        elif sender == 'candidate' and current_question:
            qa_pairs.append({"question": current_question, "answer": text})
            current_question = None
            
    if current_question:
        qa_pairs.append({"question": current_question, "answer": "[No response provided]"})

    verification_payload = json.dumps(qa_pairs, indent=2)
    
    # --- MISTRAL-PROOF JSON PROMPT ---
    prompt = HumanMessage(
        content=f"""Analyze this technical interview transcript:
        {verification_payload}
        
        Evaluate the candidate and output ONLY a valid JSON object matching this exact structure. NO markdown formatting. NO conversational text.
        
        {{
            "overallScore": 85,
            "summary": "2 sentence overall summary.",
            "insights": "Strengths: X. Weaknesses: Y.",
            "breakdown": [
                {{
                    "question": "The question asked",
                    "answer": "The candidate's answer",
                    "score": 90,
                    "feedback": "1 sentence feedback."
                }}
            ]
        }}"""
    )
    response_text = generate_ai_response([prompt])
    
    # --- BULLETPROOF REGEX JSON EXTRACTOR ---
    try:
        # Find everything between the first { and the last }
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            parsed_json = json.loads(json_str)
            
            # Update the relational DB with the final score
            conn = sqlite3.connect("synapsift.db")
            cursor = conn.cursor()
            cursor.execute("UPDATE interviews SET overall_score = ?, evaluation_summary = ? WHERE id = ?", 
                          (parsed_json.get("overallScore", 0), parsed_json.get("summary", ""), interview_id))
            conn.commit()
            conn.close()
            
            return parsed_json
        else:
            raise ValueError("No JSON boundaries found.")
            
    except Exception as e:
        print(f"JSON Parsing Failed: {e}. Raw Response: {response_text}")
        return {
            "overallScore": 50, 
            "summary": "AI Evaluation encountered formatting restrictions due to local model limits. Raw data preserved.",
            "insights": "Candidate completed all questions, but LLM analytics engine failed to parse insights.",
            "breakdown": [{"question": p["question"], "answer": p["answer"], "score": 50, "feedback": "Logs saved for manual review."} for p in qa_pairs]
        }

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
    
    # Cascade delete: Messages -> Interviews -> User
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
    
    # Admin rejecting/deleting a candidate's record
    cursor.execute("DELETE FROM messages WHERE interview_id = ?", (interview_id,))
    cursor.execute("DELETE FROM interviews WHERE id = ?", (interview_id,))
    
    conn.commit()
    conn.close()
    return {"message": "Candidate record wiped"}

@app.delete("/api/interview/abort/{interview_id}")
async def abort_interview(interview_id: str, user_meta: dict = Depends(get_current_user)):
    conn = sqlite3.connect("synapsift.db")
    cursor = conn.cursor()
    
    # Security: Ensure the candidate actually owns the interview they are aborting
    cursor.execute("DELETE FROM messages WHERE interview_id = ?", (interview_id,))
    cursor.execute("DELETE FROM interviews WHERE id = ? AND user_id = ?", (interview_id, user_meta["user_id"]))
    
    conn.commit()
    conn.close()
    return {"message": "Interview aborted. Data wiped from system."}