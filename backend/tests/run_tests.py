"""Run every test script in this folder with the current interpreter and summarise.

    cd backend
    python tests/run_tests.py            # all suites
    python tests/run_tests.py reactions  # only suites whose name contains 'reactions'

Use the same Python environment as the app (the one with pydantic / langchain installed).
These suites need no running server, no API keys and do not touch synapsift.db or qdrant_db.
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
pattern = sys.argv[1] if len(sys.argv) > 1 else ""
suites = sorted(f for f in os.listdir(HERE) if f.startswith("test_") and f.endswith(".py") and pattern in f)

env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
total_ok = total_fail = 0
worst = 0
for name in suites:
    proc = subprocess.run([sys.executable, os.path.join(HERE, name)], cwd=os.path.dirname(HERE), env=env,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = proc.stdout
    ok = len(re.findall(r"^ok   ", out, re.M))
    bad = len(re.findall(r"^FAIL ", out, re.M))
    skipped = "SKIPPED:" in out
    status = "SKIPPED" if skipped else ("PASS" if proc.returncode == 0 else "FAIL")
    print(f"{name:28s} {status:8s} ok={ok:4d} fail={bad}")
    if proc.returncode != 0 and not skipped:
        worst = 1
        print("   " + "\n   ".join([l for l in out.splitlines() if l.startswith("FAIL")][:10]))
        if proc.stderr.strip():
            print("   stderr: " + proc.stderr.strip().splitlines()[-1][:200])
    total_ok += ok
    total_fail += bad
print(f"\nTOTAL ok={total_ok} fail={total_fail} suites={len(suites)}")
sys.exit(worst)
