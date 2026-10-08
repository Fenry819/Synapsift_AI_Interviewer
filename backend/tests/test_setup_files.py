"""Submission readiness: requirements.txt covers every import, no hardcoded secrets, .env.example lists the settings, the README
exists with the sections a reviewer needs. Reads files only; imports nothing from the app."""
import ast
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import BACKEND, check, finish  # noqa: E402

ROOT = os.path.abspath(os.path.join(BACKEND, ".."))
read = lambda *p: open(os.path.join(*p), encoding="utf-8").read()

print("== backend/requirements.txt ==")
lines = [l.strip() for l in read(BACKEND, "requirements.txt").splitlines()]
reqs = [re.split(r"[<>=!~;\[ #]", l, maxsplit=1)[0] for l in lines if l and not l.startswith("#")]
norm = lambda n: re.sub(r"[-_.]+", "-", n).lower()
declared = {norm(r) for r in reqs}
check("the file is populated", len(declared) >= 15, sorted(declared))
specs = [l for l in lines if l and not l.startswith("#")]
check("every requirement carries a version range with an upper bound", all(">=" in l and "<" in l for l in specs), [l for l in specs if not (">=" in l and "<" in l)])

local = {f[:-3] for f in os.listdir(BACKEND) if f.endswith(".py")}
imported = set()
for f in os.listdir(BACKEND):
    if not f.endswith(".py"):
        continue
    for node in ast.walk(ast.parse(read(BACKEND, f))):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
third_party = {m for m in imported if m not in sys.stdlib_module_names and m not in local}
DIST = {"dotenv": "python-dotenv", "jwt": "pyjwt", "google": "google-genai", "better_profanity": "better-profanity", "qdrant_client": "qdrant-client"}
dist_of = lambda m: norm(DIST.get(m, m.replace("_", "-")))
missing = sorted(m for m in third_party if dist_of(m) not in declared)
check("every third-party module the backend imports is covered by requirements.txt", not missing, missing)
print("   third-party imports:", sorted(third_party))
for needed, why in [("python-multipart", "multipart uploads"), ("email-validator", "EmailStr"), ("uvicorn", "the server"), ("pypdf", "PyPDFLoader in ingest.py"), ("sentence-transformers", "the embedding model")]:
    check(f"{needed} is listed ({why}); it is not imported directly but the app needs it", needed in declared)

print("== no hardcoded secrets ==")
FORBIDDEN = ["super_secret" + "_synapsift_key_2026", "PGAGI" + "-RECRUITER-2026"]
offenders = [(f, s) for f in os.listdir(BACKEND) if f.endswith(".py") for s in FORBIDDEN if s in read(BACKEND, f)]
check("the old JWT fallback and admin signup code appear in no backend source file", not offenders, offenders)
src = read(BACKEND, "main.py")
check("JWT_SECRET and the admin signup code come from the environment", 'os.getenv("JWT_SECRET")' in src and 'os.getenv("ADMIN_SIGNUP_CODE")' in src)
check("a missing JWT_SECRET falls back to a random per-process key, never a fixed string", "secrets.token_urlsafe" in src)
check("administrator sign-up is refused when no code is configured, and the comparison is constant-time", "Administrator sign-up is not enabled" in src and "hmac.compare_digest" in src)
env = read(BACKEND, ".env.example")
def env_value(name):
    m = re.search(rf"^{name}=(.*)$", env, re.M)
    return None if m is None else m.group(1).strip()
check(".env.example lists JWT_SECRET and ADMIN_SIGNUP_CODE with EMPTY values", env_value("JWT_SECRET") == "" and env_value("ADMIN_SIGNUP_CODE") == "")
check(".env.example holds no real-looking key (provider keys are empty)", all(env_value(k) == "" for k in ("OPENROUTER_API_KEY", "GEMINI_API_KEY", "GEMINI_BACKUP_API_KEY")) and not re.search(r"AIza[0-9A-Za-z_-]{20,}|sk-[A-Za-z0-9-]{20,}", env))
check(".env.example documents the local-model settings", all(env_value(k) for k in ("LOCAL_LLM_MODEL", "OLLAMA_BASE_URL", "LOCAL_LLM_NUM_CTX", "LOCAL_LLM_TIMEOUT_SECONDS")))
gitignore = read(ROOT, ".gitignore")
check(".env, the SQLite database, the Qdrant store, uploads, venv and node_modules are git-ignored", all(p in gitignore for p in (".env", "*.db", "backend/qdrant_db/", "backend/uploads/*", "venv/", "node_modules/")))

print("== README.md ==")
readme_path = os.path.join(ROOT, "README.md")
check("a root README.md exists", os.path.exists(readme_path))
if os.path.exists(readme_path):
    readme = read(ROOT, "README.md")
    prose = re.sub(r"```.*?```", "", readme, flags=re.S)          # ignore '# comments' inside fenced code blocks
    heads = [h.lower() for h in re.findall(r"^#{1,3} (.+)$", prose, re.M)]
    for needed in ["overview", "architecture", "design decisions", "prerequisites", "backend setup", "environment", "ollama", "knowledge base", "ingest", "frontend setup",
                   "running", "tests", "resume", "role", "limitations", "interview flow", "evaluation"]:
        check(f"README has a section on: {needed}", any(needed in h for h in heads), heads)
    for token in ["python ingest.py", "pip install -r requirements.txt", "npm install", "npm run dev", "uvicorn main:app", "ADMIN_SIGNUP_CODE", "JWT_SECRET", "ollama pull", "run_tests.py", ".txt"]:
        check(f"README mentions: {token}", token in readme)
    check("README does not contain real secrets", not re.search(r"AIza[0-9A-Za-z_-]{20,}|sk-[A-Za-z0-9-]{20,}", readme) and "PGAGI" + "-RECRUITER" not in readme)

finish()
