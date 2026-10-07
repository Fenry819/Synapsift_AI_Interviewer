"""Provider layer tests.
STUBBED sections use a local fake HTTP server standing in for Ollama, or patched provider functions.
REAL sections call the actual Ollama instance on this machine (qwen2.5:14b)."""
import os, sys, json, threading, logging, warnings
from http.server import BaseHTTPRequestHandler, HTTPServer
sys.dont_write_bytecode = True
logging.disable(logging.CRITICAL); warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _harness  # noqa: F401  (puts backend/ on sys.path and quiets logging)
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import llm_providers as lp

fails = 0
def check(label, cond, extra=""):
    global fails
    if not cond: fails += 1
    print(("ok   " if cond else "FAIL ") + label + (f"   -> {extra}" if (extra and not cond) else ""))

def clear_env():
    for k in ("LOCAL_LLM_MODEL", "OLLAMA_BASE_URL", "LOCAL_LLM_NUM_CTX", "LOCAL_LLM_TIMEOUT_SECONDS", "OPENROUTER_API_KEY", "OPENROUTER_MODEL"):
        os.environ.pop(k, None)

# ================= STUBBED: configuration =================
print("== [stub] configuration ==")
clear_env()
s = lp.local_llm_settings()
check("default local model is qwen2.5:14b", s.model == "qwen2.5:14b", s)
check("default Ollama URL / ctx / timeout", s.base_url == "http://127.0.0.1:11434" and s.num_ctx == 4096 and s.timeout == 300.0, s)
os.environ.update({"LOCAL_LLM_MODEL": "llama3.1:8b", "OLLAMA_BASE_URL": "http://gpu-box:11434/", "LOCAL_LLM_NUM_CTX": "8192", "LOCAL_LLM_TIMEOUT_SECONDS": "12.5"})
s = lp.local_llm_settings()
check("env overrides model/url (trailing slash stripped)/ctx/timeout", (s.model, s.base_url, s.num_ctx, s.timeout) == ("llama3.1:8b", "http://gpu-box:11434", 8192, 12.5), s)
os.environ.update({"LOCAL_LLM_MODEL": "   ", "LOCAL_LLM_NUM_CTX": "abc"})
s = lp.local_llm_settings()
check("blank model / invalid number fall back to defaults", s.model == "qwen2.5:14b" and s.num_ctx == 4096, s)
clear_env()

# ================= STUBBED: role mapping =================
print("== [stub] message role mapping ==")
msgs = [SystemMessage(content="sys"), HumanMessage(content="hi"), AIMessage(content="yo"), HumanMessage(content="q")]
check("system/human/ai -> system/user/assistant, order and content kept",
      lp.to_ollama_messages(msgs) == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}, {"role": "user", "content": "q"}])

# ================= STUBBED: fake Ollama HTTP server =================
print("== [stub] call_local_llm against a fake Ollama server ==")
class Fake(BaseHTTPRequestHandler):
    mode = "ok"; seen = []
    def log_message(self, *a): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Fake.seen.append((self.path, body))
        if Fake.mode == "ok":
            code, out = 200, json.dumps({"message": {"role": "assistant", "content": "  A fine question?  "}})
        elif Fake.mode == "404":
            code, out = 404, json.dumps({"error": "model 'x' not found"})
        elif Fake.mode == "empty":
            code, out = 200, json.dumps({"message": {"role": "assistant", "content": "   "}})
        elif Fake.mode == "nomsg":
            code, out = 200, json.dumps({})
        else:
            code, out = 200, "this is not json"
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(out.encode())
srv = HTTPServer(("127.0.0.1", 0), Fake); threading.Thread(target=srv.serve_forever, daemon=True).start()
os.environ["OLLAMA_BASE_URL"] = f"http://127.0.0.1:{srv.server_port}"
os.environ["LOCAL_LLM_MODEL"] = "some-other-model:7b"; os.environ["LOCAL_LLM_NUM_CTX"] = "2048"

r = lp.call_local_llm(msgs)
path, body = Fake.seen[-1]
check("posts to /api/chat", path == "/api/chat")
check("model comes from config, not hardcoded", body["model"] == "some-other-model:7b")
check("stream=false, num_ctx pinned from config", body["stream"] is False and body["options"] == {"num_ctx": 2048})
check("roles preserved in the request", [m["role"] for m in body["messages"]] == ["system", "user", "assistant", "user"])
check("result: text stripped, provider=local, model recorded", r == lp.LLMResult("A fine question?", "local", "some-other-model:7b"), r)
for mode, label in [("404", "unknown model (HTTP 404)"), ("empty", "empty reply"), ("nomsg", "missing message"), ("garbage", "non-JSON body")]:
    Fake.mode = mode
    try: lp.call_local_llm(msgs); ok = False
    except lp.ProviderCallError as e: ok = True; detail = str(e)
    check(f"local failure -> ProviderCallError: {label}", ok)
Fake.mode = "404"
try: lp.call_local_llm(msgs)
except lp.ProviderCallError as e: check("404 error names the model so misconfiguration is obvious", "some-other-model:7b" in str(e) and "404" in str(e), str(e))
srv.shutdown()
os.environ["OLLAMA_BASE_URL"] = "http://127.0.0.1:9"      # nothing listens here
try: lp.call_local_llm(msgs); ok = False
except lp.ProviderCallError as e: ok = "unreachable" in str(e)
check("connection refused -> ProviderCallError('unreachable')", ok)
clear_env()

# ================= STUBBED: failover orchestration (patched provider calls) =================
print("== [stub] generate_chat_response failover logic ==")
calls = []
def fake_or_ok(m): calls.append(("openrouter", m)); return lp.LLMResult("from openrouter", "openrouter", "nemotron")
def fake_or_fail(m): calls.append(("openrouter", m)); raise lp.ProviderCallError("429 rate limited")
def fake_local_ok(m): calls.append(("local", m)); return lp.LLMResult("from local", "local", "qwen2.5:14b")
def fake_local_fail(m): calls.append(("local", m)); raise lp.ProviderCallError("ollama down")
orig = (lp.call_openrouter, lp.call_local_llm)
CTX = [HumanMessage(content="FULL CONTEXT: role=AI/ML | profile=PyTorch | ref=CHUNK-123 | answer=I use L2")]

os.environ["OPENROUTER_API_KEY"] = "sk-test"
calls.clear(); lp.call_openrouter, lp.call_local_llm = fake_or_ok, fake_local_ok
r = lp.generate_chat_response(CTX)
check("1) OpenRouter success -> OpenRouter result, local NEVER called", r.provider == "openrouter" and [c[0] for c in calls] == ["openrouter"])
calls.clear(); lp.call_openrouter, lp.call_local_llm = fake_or_fail, fake_local_ok
r = lp.generate_chat_response(CTX)
check("2) OpenRouter failure -> local result", r.provider == "local" and r.model == "qwen2.5:14b" and [c[0] for c in calls] == ["openrouter", "local"])
check("4) provider switch: both providers received the IDENTICAL messages object", calls[0][1] is calls[1][1] and calls[0][1] is CTX)
check("4) ...and identical content (profile, role, RAG chunk, answer all present for the fallback)", all(t in calls[1][1][0].content for t in ("role=AI/ML", "PyTorch", "CHUNK-123", "I use L2")))
calls.clear(); lp.call_openrouter, lp.call_local_llm = fake_or_fail, fake_local_fail
try: lp.generate_chat_response(CTX); ok = False; msg = ""
except lp.ProviderUnavailableError as e: ok = True; msg = str(e)
check("3) both fail -> ProviderUnavailableError (no fake reply)", ok and [c[0] for c in calls] == ["openrouter", "local"], msg)
check("3) error carries both failure reasons", "429 rate limited" in msg and "ollama down" in msg, msg)
calls.clear(); lp.call_openrouter = lambda m: (_ for _ in ()).throw(RuntimeError("unexpected bug in client")); lp.call_local_llm = fake_local_ok
r = lp.generate_chat_response(CTX)
check("unexpected exception type from OpenRouter still falls back", r.provider == "local")
clear_env()
calls.clear(); lp.call_openrouter, lp.call_local_llm = fake_or_ok, fake_local_ok
r = lp.generate_chat_response(CTX)
check("no OPENROUTER_API_KEY -> straight to local, OpenRouter not attempted", r.provider == "local" and [c[0] for c in calls] == ["local"])
lp.call_openrouter, lp.call_local_llm = orig

# ================= REAL: local Ollama / qwen2.5:14b =================
import urllib.request
def _ollama_up():
    try:
        urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3); return True
    except Exception:
        return False
if not _ollama_up():
    print("(skipping the [REAL] Ollama checks: Ollama is not reachable on 127.0.0.1:11434)")
    print(f"\nFAILURES: {fails}"); sys.exit(1 if fails else 0)

print("== [REAL] local Ollama on this machine ==")
clear_env()
real = lp.call_local_llm([SystemMessage(content="You are a concise technical interviewer."),
                          HumanMessage(content="Ask me exactly one open-ended question about overfitting in machine learning. One sentence, no lists.")])
print("   real reply:", repr(real.text))
check("REAL call_local_llm -> non-empty text from provider=local, model=qwen2.5:14b", real.provider == "local" and real.model == "qwen2.5:14b" and len(real.text) > 10, real)
os.environ["OPENROUTER_API_KEY"] = "sk-test"
lp.call_openrouter = fake_or_fail   # simulated OpenRouter failure
real2 = lp.generate_chat_response([HumanMessage(content="Ask one short interview question about decision trees. No lists.")])
lp.call_openrouter = orig[0]
print("   real failover reply:", repr(real2.text))
check("REAL failover: simulated OpenRouter failure -> real Qwen answers", real2.provider == "local" and real2.model == "qwen2.5:14b" and real2.text)
clear_env()
os.environ["LOCAL_LLM_MODEL"] = "this-model-does-not-exist:1b"
try: lp.call_local_llm([HumanMessage(content="hi")]); ok = False; e404 = ""
except lp.ProviderCallError as e: ok = True; e404 = str(e)
check("REAL Ollama: a model that is not installed -> clear ProviderCallError naming it", ok and "this-model-does-not-exist:1b" in e404, e404)
clear_env()
print("\nFAILURES:", fails)
sys.exit(1 if fails else 0)
