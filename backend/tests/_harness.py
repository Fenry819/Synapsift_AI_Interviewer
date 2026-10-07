"""Tiny dependency-free test harness shared by the scripts in this folder.

Each test file is a plain script: `python tests/test_engine.py` (run from `backend/`) prints `ok` / `FAIL` per check
and exits non-zero if anything failed. `python tests/run_tests.py` runs them all.

Only the standard library is needed for the harness itself. Some suites need the packages the app already uses
(pydantic, langchain-core) and the local sentence-transformer model (all-MiniLM-L6-v2) for the semantic checks; if
the model cannot be loaded those suites report SKIPPED instead of failing.
"""
import logging
import os
import sys
import warnings

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")
sys.dont_write_bytecode = True

_fails = 0
_checks = 0


def check(label: str, condition, extra="") -> bool:
    global _fails, _checks
    _checks += 1
    ok = bool(condition)
    if not ok:
        _fails += 1
    print(("ok   " if ok else "FAIL ") + label + (f"   -> {extra}" if (extra and not ok) else ""))
    return ok


def finish() -> None:
    print(f"\nCHECKS: {_checks}  FAILURES: {_fails}")
    sys.exit(1 if _fails else 0)


def skip(reason: str) -> None:
    print(f"SKIPPED: {reason}")
    sys.exit(0)


def load_embedder():
    """The same embedding model the app uses, or SKIP the whole suite if it is not available offline/online."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2").embed_documents
    except Exception as e:  # pragma: no cover - environment dependent
        skip(f"embedding model all-MiniLM-L6-v2 is not available ({str(e)[:80]})")
