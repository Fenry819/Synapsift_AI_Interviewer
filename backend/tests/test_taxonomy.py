import os
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _harness  # noqa: F401  (puts backend/ on sys.path and quiets logging)
import rag_config as cfg

fails = 0
def check(role, want):
    global fails
    got = cfg.resolve_domain(role)
    ok = got == want
    fails += not ok
    print(("ok   " if ok else "FAIL ") + f"{role!r:48s} -> {got}" + ("" if ok else f"   (wanted {want})"))

# dropdown roles exactly as the frontend sends them
check("AI / Machine Learning Role", "ai_ml")
check("Data Science / Applied ML Role", "data_science")
check("Advanced / Theoretical ML", "advanced_ml")
check("Backend Engineering Intern", None)
# synonyms
for r in ["Machine Learning Engineer", "ML Engineer", "AI Engineer", "Artificial Intelligence Engineer", "AI/ML Engineer", "Deep Learning Engineer", "machine learning intern", "  AI   Engineer  "]:
    check(r, "ai_ml")
for r in ["Data Scientist", "Junior Data Scientist", "Data Science Intern", "Applied Machine Learning Engineer", "Applied ML Scientist"]:
    check(r, "data_science")
for r in ["Machine Learning Researcher", "ML Researcher", "Research Scientist", "Theoretical Machine Learning", "Advanced Machine Learning Engineer"]:
    check(r, "advanced_ml")
# NO corpus: must be None, never "closest ML"
for r in ["Backend Engineer", "Frontend Developer", "Full Stack Developer", "DevOps Engineer", "Cloud Security Architect", "Android Developer",
          "Data Analyst", "Data Engineer", "Site Reliability Engineer", "QA Engineer", "Chrome Developer", "Three.js Developer",
          "Salesforce Developer", "Device Driver Engineer", "DevOps Engineer using Chef", "Custom Role", "", "   ", "Yoga Instructor",
          "HTML Developer", "Email Marketing", "Gamma Ray Engineer", "AIML"]:
    check(r, None)
# whole-word matching: fragments never match
for r in ["Maintenance Engineer", "Bail Engineer", "Mainframe ML-less role"]:
    check(r, None)
# every domain used by the taxonomy / manifest has config
used = {d for e in cfg.CORPUS for d in e["domains"]}
dom_ok = used <= set(cfg.DOMAIN_INFO) and {d for d, _ in cfg._DOMAIN_RULES} == set(cfg.DOMAIN_INFO)
print(("ok   " if dom_ok else "FAIL ") + "manifest domains == taxonomy domains == DOMAIN_INFO keys"); fails += not dom_ok
seeds_ok = all(len(i["topic_seeds"]) >= 6 and i["profile_domains"] for i in cfg.DOMAIN_INFO.values())
print(("ok   " if seeds_ok else "FAIL ") + "every domain has topic seeds + profile affinity"); fails += not seeds_ok
dup = [e["file"] for e in cfg.CORPUS]; ok = len(dup) == len(set(dup))
print(("ok   " if ok else "FAIL ") + "no source listed twice in the manifest"); fails += not ok
print("\nFAILURES:", fails); sys.exit(1 if fails else 0)
