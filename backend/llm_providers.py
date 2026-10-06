"""LLM providers for the interviewer.

Chat provider chain:  OpenRouter  ->  local Ollama model  ->  ProviderUnavailableError.

Rules
  * Every provider receives the SAME fully built `messages` list. The backend owns interview state
    (profile, role, RAG context, answer ...) and bakes it into the messages before calling here; no
    provider is ever asked to remember anything, so switching providers loses nothing.
  * The local provider is model-agnostic: its model comes from configuration (LOCAL_LLM_MODEL), and
    nothing in this module names a specific model except the development default below.
  * If no provider produces a usable (non-empty) reply, ProviderUnavailableError is raised. Nothing
    here ever fabricates an interviewer message.

Configuration is read from the environment at CALL time (not import time) because main.py loads
.env after importing this module.
"""
import os
from dataclasses import dataclass

import httpx
from langchain_openai import ChatOpenAI

DEFAULT_LOCAL_LLM_MODEL = "qwen2.5:14b"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_LOCAL_LLM_NUM_CTX = 4096
DEFAULT_LOCAL_LLM_TIMEOUT_SECONDS = 300.0
DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
DEFAULT_OPENROUTER_TIMEOUT_SECONDS = 60.0

PROVIDER_OPENROUTER = "openrouter"
PROVIDER_LOCAL = "local"


class ProviderUnavailableError(Exception):
    """No chat provider produced a usable reply."""


class ProviderCallError(Exception):
    """One specific provider failed (network, HTTP error, empty reply...)."""


@dataclass(frozen=True)
class LLMResult:
    text: str
    provider: str      # PROVIDER_OPENROUTER | PROVIDER_LOCAL
    model: str


@dataclass(frozen=True)
class LocalLLMSettings:
    model: str
    base_url: str
    num_ctx: int
    timeout: float


def _env_text(name: str, default: str) -> str:
    """Stripped env value; unset, empty or whitespace-only falls back to the default."""
    return (os.getenv(name) or "").strip() or default


def _env_number(name: str, default, cast):
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return cast(raw)
    except ValueError:
        print(f"⚠️ Ignoring invalid {name}={raw!r}; using default {default}")
        return default


def local_llm_settings() -> LocalLLMSettings:
    return LocalLLMSettings(
        model=_env_text("LOCAL_LLM_MODEL", DEFAULT_LOCAL_LLM_MODEL),
        base_url=_env_text("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).rstrip("/"),
        num_ctx=_env_number("LOCAL_LLM_NUM_CTX", DEFAULT_LOCAL_LLM_NUM_CTX, int),
        timeout=_env_number("LOCAL_LLM_TIMEOUT_SECONDS", DEFAULT_LOCAL_LLM_TIMEOUT_SECONDS, float),
    )


def _extract_text(content) -> str:
    if isinstance(content, list):
        if len(content) > 0 and isinstance(content[0], dict):
            return content[0].get("text", str(content))
    return str(content)


_OLLAMA_ROLES = {"human": "user", "system": "system", "ai": "assistant"}


def to_ollama_messages(messages: list) -> list:
    """LangChain messages -> Ollama chat messages, keeping each message's role."""
    return [{"role": _OLLAMA_ROLES.get(getattr(m, "type", "human"), "user"), "content": str(m.content)}
            for m in messages]


def call_openrouter(messages: list) -> LLMResult:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ProviderCallError("OPENROUTER_API_KEY is not set")
    model = _env_text("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
    timeout = _env_number("OPENROUTER_TIMEOUT_SECONDS", DEFAULT_OPENROUTER_TIMEOUT_SECONDS, float)
    print(f"🌐 Querying OpenRouter model: {model}")
    try:
        llm = ChatOpenAI(model=model, api_key=api_key, base_url="https://openrouter.ai/api/v1",
                         timeout=timeout, max_retries=1)
        text = _extract_text(llm.invoke(messages).content).strip()
    except Exception as e:
        raise ProviderCallError(f"OpenRouter request failed: {e}") from e
    if not text:
        raise ProviderCallError("OpenRouter succeeded but returned an empty reply")
    return LLMResult(text=text, provider=PROVIDER_OPENROUTER, model=model)


def call_local_llm(messages: list, json_schema: dict | None = None) -> LLMResult:
    """Local Ollama chat call. The model is whatever LOCAL_LLM_MODEL says.

    `json_schema` (optional) is passed as Ollama's `format`, which constrains decoding to that schema. It is a
    reliability aid only: the caller still validates the reply exactly as it does for every other provider."""
    s = local_llm_settings()
    print(f"🤖 Querying local Ollama model: {s.model}")
    payload = {
        "model": s.model,
        "messages": to_ollama_messages(messages),
        "stream": False,
        # Pinned explicitly so a long prompt is never silently cut by an unknown server default.
        "options": {"num_ctx": s.num_ctx},
    }
    if json_schema:
        payload["format"] = json_schema
    try:
        with httpx.Client(timeout=s.timeout) as client:
            response = client.post(f"{s.base_url}/api/chat", json=payload)
    except Exception as e:
        raise ProviderCallError(f"Ollama unreachable at {s.base_url}: {e}") from e
    if response.status_code != 200:
        raise ProviderCallError(f"Ollama returned HTTP {response.status_code} for model '{s.model}': {response.text[:200]}")
    try:
        text = (response.json().get("message") or {}).get("content", "")
    except ValueError as e:
        raise ProviderCallError(f"Ollama returned a non-JSON body: {e}") from e
    if not text or not text.strip():
        raise ProviderCallError(f"Ollama model '{s.model}' returned an empty reply")
    return LLMResult(text=text.strip(), provider=PROVIDER_LOCAL, model=s.model)


def generate_chat_response(messages: list, json_schema: dict | None = None) -> LLMResult:
    """Interviewer reply: OpenRouter first, local model on ANY OpenRouter failure.

    Both calls get the identical `messages` object. Raises ProviderUnavailableError if both fail;
    never returns a made-up reply. `json_schema` only tightens the LOCAL call's decoding (see call_local_llm)."""
    failures = []
    if os.getenv("OPENROUTER_API_KEY"):
        try:
            return call_openrouter(messages)
        except Exception as e:   # any failure, expected or not, hands over to the local model
            print(f"⚠️ OpenRouter failed: {e}. Falling back to the local model.")
            failures.append(f"OpenRouter: {e}")
    else:
        print("ℹ️ OpenRouter is not configured; using the local model.")
        failures.append("OpenRouter is not configured")

    try:
        return call_local_llm(messages, json_schema=json_schema) if json_schema else call_local_llm(messages)
    except Exception as e:
        print(f"🚨 Local model failed: {e}")
        failures.append(f"local: {e}")

    raise ProviderUnavailableError("; ".join(failures))
