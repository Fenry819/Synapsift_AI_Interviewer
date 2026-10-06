"""Deterministic resume text -> structured candidate profile.

Standard library only. Every item in the profile is found by matching a curated lexicon
(or a fixed pattern) against the resume text itself, so nothing can be invented: if a
skill is not written in the resume it can never appear in the profile.
"""
import json
import re

PROFILE_SCHEMA_VERSION = 1
EXTRACTION_METHOD = "deterministic-lexicon-v1"

MAX_RESUME_CHARS = 20000       # hard cap on stored / analysed resume text
MAX_SKILLS = 15
MAX_TECHNOLOGIES = 20
MAX_DOMAINS = 6
MAX_PROJECTS = 8
MAX_PROJECT_TITLE_CHARS = 100
MAX_PROJECT_BODY_CHARS = 600

# --------------------------------------------------------------------------------------
# Lexicon: (canonical name, kind, domains, aliases)
#   kind    "technology" = language / library / framework / tool / platform
#           "skill"      = concept or discipline
#   aliases are matched case-insensitively as whole words ("java" never matches
#   "javascript"); an alias starting with "~" is matched case-SENSITIVELY, used for words
#   that are also ordinary English ("React", "Swift", "Spark").
# --------------------------------------------------------------------------------------
_LEXICON = [
    # --- languages
    ("Python", "technology", (), ("python", "python3")),
    ("Java", "technology", ("backend engineering",), ("java",)),
    ("JavaScript", "technology", ("frontend engineering",), ("javascript", "ecmascript")),
    ("TypeScript", "technology", ("frontend engineering",), ("typescript",)),
    ("C++", "technology", ("algorithms and systems",), ("c++",)),
    ("C#", "technology", ("backend engineering",), ("c#",)),
    (".NET", "technology", ("backend engineering",), (".net", "asp.net", "dotnet")),
    ("Go", "technology", ("backend engineering",), ("golang",)),
    ("Rust", "technology", ("algorithms and systems",), ("~Rust",)),
    ("Kotlin", "technology", ("mobile development",), ("kotlin",)),
    ("Swift", "technology", ("mobile development",), ("~Swift", "swiftui")),
    ("PHP", "technology", ("backend engineering",), ("php",)),
    ("Ruby", "technology", ("backend engineering",), ("ruby", "ruby on rails")),
    ("Scala", "technology", ("data engineering",), ("scala",)),
    ("SQL", "technology", ("databases",), ("sql",)),
    ("R", "technology", ("data science",), ("r programming", "r language", "rstudio", "ggplot2")),
    ("MATLAB", "technology", ("data science",), ("matlab",)),
    # --- ML / data libraries
    ("PyTorch", "technology", ("deep learning", "machine learning"), ("pytorch",)),
    ("TensorFlow", "technology", ("deep learning", "machine learning"), ("tensorflow", "tf.keras")),
    ("Keras", "technology", ("deep learning",), ("keras",)),
    ("scikit-learn", "technology", ("machine learning",), ("scikit-learn", "sklearn")),
    ("Pandas", "technology", ("data science",), ("pandas",)),
    ("NumPy", "technology", ("data science",), ("numpy",)),
    ("SciPy", "technology", ("data science",), ("scipy",)),
    ("Matplotlib", "technology", ("data science",), ("matplotlib",)),
    ("Seaborn", "technology", ("data science",), ("seaborn",)),
    ("XGBoost", "technology", ("machine learning",), ("xgboost",)),
    ("LightGBM", "technology", ("machine learning",), ("lightgbm",)),
    ("OpenCV", "technology", ("computer vision",), ("opencv", "cv2")),
    ("Hugging Face", "technology", ("natural language processing", "generative ai"), ("hugging face", "huggingface")),
    ("LangChain", "technology", ("generative ai",), ("langchain",)),
    ("spaCy", "technology", ("natural language processing",), ("spacy",)),
    ("NLTK", "technology", ("natural language processing",), ("nltk",)),
    ("OpenAI API", "technology", ("generative ai",), ("openai",)),
    ("Ollama", "technology", ("generative ai",), ("ollama",)),
    ("Streamlit", "technology", ("data science",), ("streamlit",)),
    ("Tableau", "technology", ("data science",), ("tableau",)),
    ("Power BI", "technology", ("data science",), ("power bi", "powerbi")),
    # --- data engineering
    ("Apache Spark", "technology", ("data engineering",), ("apache spark", "pyspark", "~Spark")),
    ("Hadoop", "technology", ("data engineering",), ("hadoop",)),
    ("Apache Airflow", "technology", ("data engineering",), ("airflow",)),
    ("Apache Kafka", "technology", ("data engineering", "backend engineering"), ("kafka",)),
    # --- web / backend / frontend
    ("FastAPI", "technology", ("backend engineering",), ("fastapi",)),
    ("Flask", "technology", ("backend engineering",), ("flask",)),
    ("Django", "technology", ("backend engineering",), ("django",)),
    ("Spring Boot", "technology", ("backend engineering",), ("spring boot", "springboot")),
    ("Node.js", "technology", ("backend engineering",), ("node.js", "nodejs")),
    ("Express.js", "technology", ("backend engineering",), ("express.js", "expressjs")),
    ("GraphQL", "technology", ("backend engineering",), ("graphql",)),
    ("React", "technology", ("frontend engineering",), ("~React", "reactjs", "react.js")),
    ("Next.js", "technology", ("frontend engineering",), ("next.js", "nextjs")),
    ("Angular", "technology", ("frontend engineering",), ("angular", "angularjs")),
    ("Vue.js", "technology", ("frontend engineering",), ("vue.js", "vuejs")),
    ("HTML", "technology", ("frontend engineering",), ("html", "html5")),
    ("CSS", "technology", ("frontend engineering",), ("css", "css3")),
    ("Tailwind CSS", "technology", ("frontend engineering",), ("tailwind", "tailwindcss")),
    # --- datastores
    ("PostgreSQL", "technology", ("databases",), ("postgresql", "postgres")),
    ("MySQL", "technology", ("databases",), ("mysql",)),
    ("MongoDB", "technology", ("databases",), ("mongodb",)),
    ("Redis", "technology", ("databases", "backend engineering"), ("redis",)),
    ("SQLite", "technology", ("databases",), ("sqlite",)),
    ("Elasticsearch", "technology", ("databases",), ("elasticsearch",)),
    ("FAISS", "technology", ("generative ai", "databases"), ("faiss",)),
    ("Qdrant", "technology", ("generative ai", "databases"), ("qdrant",)),
    ("Pinecone", "technology", ("generative ai", "databases"), ("pinecone",)),
    # --- devops / cloud
    ("Docker", "technology", ("devops and cloud",), ("docker", "dockerfile")),
    ("Kubernetes", "technology", ("devops and cloud",), ("kubernetes", "k8s")),
    ("AWS", "technology", ("devops and cloud",), ("aws", "amazon web services")),
    ("Azure", "technology", ("devops and cloud",), ("azure",)),
    ("Google Cloud", "technology", ("devops and cloud",), ("gcp", "google cloud")),
    ("Terraform", "technology", ("devops and cloud",), ("terraform",)),
    ("Jenkins", "technology", ("devops and cloud",), ("jenkins",)),
    ("Linux", "technology", ("devops and cloud",), ("linux",)),
    ("Git", "technology", (), ("git",)),
    # --- mobile
    ("Android", "technology", ("mobile development",), ("android",)),
    ("iOS", "technology", ("mobile development",), ("ios",)),
    ("Flutter", "technology", ("mobile development",), ("flutter",)),
    ("React Native", "technology", ("mobile development",), ("react native",)),
    # --- testing
    ("Selenium", "technology", (), ("selenium",)),
    ("Pytest", "technology", (), ("pytest",)),
    # --- skills / concepts
    ("Machine Learning", "skill", ("machine learning",), ("machine learning", "ml")),
    ("Deep Learning", "skill", ("deep learning", "machine learning"), ("deep learning",)),
    ("Neural Networks", "skill", ("deep learning",), ("neural network", "neural networks")),
    ("Natural Language Processing", "skill", ("natural language processing",), ("natural language processing", "nlp")),
    ("Computer Vision", "skill", ("computer vision",), ("computer vision", "image classification", "object detection")),
    ("Reinforcement Learning", "skill", ("machine learning",), ("reinforcement learning",)),
    ("Generative AI", "skill", ("generative ai",), ("generative ai", "genai")),
    ("Large Language Models", "skill", ("generative ai", "natural language processing"), ("large language model", "large language models", "llm", "llms")),
    ("Retrieval-Augmented Generation", "skill", ("generative ai",), ("retrieval augmented generation", "rag")),
    ("Prompt Engineering", "skill", ("generative ai",), ("prompt engineering",)),
    ("Fine-tuning", "skill", ("generative ai", "machine learning"), ("fine tuning", "finetuning")),
    ("Embeddings", "skill", ("generative ai", "natural language processing"), ("embedding", "embeddings", "sentence transformers")),
    ("Vector Databases", "skill", ("generative ai", "databases"), ("vector database", "vector databases", "vector db")),
    ("Transformers", "skill", ("natural language processing", "deep learning"), ("transformers", "transformer models", "transformer architecture")),
    ("Time Series Analysis", "skill", ("data science",), ("time series",)),
    ("Recommendation Systems", "skill", ("machine learning",), ("recommendation system", "recommendation systems", "recommender system", "recommender systems")),
    ("Feature Engineering", "skill", ("machine learning", "data science"), ("feature engineering",)),
    ("Statistics", "skill", ("data science",), ("statistics", "statistical analysis", "hypothesis testing")),
    ("Data Analysis", "skill", ("data science",), ("data analysis", "data analytics", "exploratory data analysis", "eda")),
    ("Data Visualization", "skill", ("data science",), ("data visualization", "data visualisation")),
    ("ETL / Data Pipelines", "skill", ("data engineering",), ("etl", "data pipeline", "data pipelines")),
    ("MLOps", "skill", ("machine learning", "devops and cloud"), ("mlops", "model deployment")),
    ("Data Structures and Algorithms", "skill", ("algorithms and systems",), ("data structures", "dsa", "competitive programming")),
    ("System Design", "skill", ("algorithms and systems", "backend engineering"), ("system design",)),
    ("Distributed Systems", "skill", ("algorithms and systems", "backend engineering"), ("distributed systems",)),
    ("Microservices", "skill", ("backend engineering",), ("microservice", "microservices")),
    ("REST APIs", "skill", ("backend engineering",), ("rest api", "rest apis", "restful", "restful api", "restful apis")),
    ("Object-Oriented Programming", "skill", (), ("oop", "object oriented programming", "object oriented")),
    ("CI/CD", "skill", ("devops and cloud",), ("ci/cd", "cicd", "github actions", "continuous integration")),
    ("Unit Testing", "skill", (), ("unit testing", "unit tests", "test driven development", "tdd")),
]


def _alias_regex(alias: str) -> str:
    # Spaces and hyphens inside an alias are interchangeable ("scikit-learn" == "scikit learn").
    parts = re.split(r"[\s\-]+", alias)
    body = r"[\s\-]+".join(re.escape(p) for p in parts)
    return r"(?<![A-Za-z0-9+#_])" + body + r"(?![A-Za-z0-9+#_])"


def _compile_lexicon():
    compiled = []
    for canonical, kind, domains, aliases in _LEXICON:
        ci = [_alias_regex(a) for a in aliases if not a.startswith("~")]
        cs = [_alias_regex(a[1:]) for a in aliases if a.startswith("~")]
        compiled.append((
            canonical, kind, domains,
            re.compile("|".join(ci), re.IGNORECASE) if ci else None,
            re.compile("|".join(cs)) if cs else None,
        ))
    return compiled


_COMPILED_LEXICON = _compile_lexicon()


def _match_lexicon(text: str) -> dict:
    """canonical -> (kind, domains, occurrence count, first position) for every entry present."""
    found = {}
    for canonical, kind, domains, ci_re, cs_re in _COMPILED_LEXICON:
        positions = []
        if ci_re:
            positions += [m.start() for m in ci_re.finditer(text)]
        if cs_re:
            positions += [m.start() for m in cs_re.finditer(text)]
        if positions:
            found[canonical] = (kind, domains, len(positions), min(positions))
    return found


# --------------------------------------------------------------------------------------
# Text cleaning and section splitting
# --------------------------------------------------------------------------------------
def clean_resume_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").replace("’", "'")
    text = re.sub(r"[^\S\n]+", " ", text)                     # collapse spaces/tabs, keep newlines
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)      # control characters
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:MAX_RESUME_CHARS]


_SECTION_HEADINGS = {
    "projects": ["projects", "personal projects", "academic projects", "key projects", "selected projects",
                 "technical projects", "notable projects", "project experience"],
    "experience": ["experience", "work experience", "professional experience", "employment",
                   "employment history", "work history", "internships", "internship experience",
                   "internship"],
    "education": ["education", "academic background", "education and qualifications", "academics"],
    "skills": ["skills", "technical skills", "key skills", "core competencies", "technologies",
               "tech stack", "skills and tools", "technical proficiency"],
    "certifications": ["certifications", "certification", "certificates", "licenses and certifications",
                       "courses", "courses and certifications"],
    "other": ["summary", "professional summary", "objective", "career objective", "profile", "about me",
              "achievements", "awards", "awards and achievements", "publications", "interests", "hobbies",
              "languages", "references", "extracurricular activities", "extracurriculars",
              "volunteer experience", "contact", "declaration", "positions of responsibility"],
}
_HEADING_TO_SECTION = {h: sec for sec, heads in _SECTION_HEADINGS.items() for h in heads}


def _heading_section(line: str):
    s = line.strip()
    if not s or len(s) > 40:
        return None
    key = re.sub(r"[^a-z& ]", "", s.lower()).replace("&", "and")
    key = re.sub(r"\s+", " ", key).strip()
    return _HEADING_TO_SECTION.get(key)


def _split_sections(text: str) -> dict:
    sections = {"header": []}
    current = "header"
    for line in text.split("\n"):
        section = _heading_section(line)
        if section:
            current = section
            sections.setdefault(current, [])
        else:
            sections[current].append(line)
    return sections


# --------------------------------------------------------------------------------------
# Projects
# --------------------------------------------------------------------------------------
_BULLET_RE = re.compile(r"^(?:[•●▪◦‣·\-*–—]|o)\s+")
_TITLE_SPLIT_RE = re.compile(r"\s+[|–—\-]\s+|\s*[|:]\s*|\s*\(")
_UNSAFE_CHARS_RE = re.compile(r"[\[\]{}<>`\"]")


def _sanitize_snippet(s: str, limit: int) -> str:
    s = _UNSAFE_CHARS_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip(" -–—|:;,.")
    return s[:limit].strip()


def _technologies_in(text: str) -> list:
    found = _match_lexicon(text)
    techs = [(pos, name) for name, (kind, _d, _c, pos) in found.items() if kind == "technology"]
    return [name for _pos, name in sorted(techs)]


def _extract_projects(lines: list) -> list:
    entries = []   # [title, [body lines]]
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        is_bullet = bool(_BULLET_RE.match(line))
        looks_like_title = (
            not is_bullet
            and 3 <= len(line) <= MAX_PROJECT_TITLE_CHARS
            and (line[0].isupper() or line[0].isdigit())     # wrapped continuation lines start lowercase
            and not line.endswith((".", ","))
            and len(line.split()) <= 14
        )
        if looks_like_title:
            entries.append([line, []])
        elif entries:
            entries[-1][1].append(_BULLET_RE.sub("", line))

    projects = []
    for title_line, body_lines in entries:
        title = _sanitize_snippet(_TITLE_SPLIT_RE.split(title_line, maxsplit=1)[0], MAX_PROJECT_TITLE_CHARS)
        if not re.search(r"[A-Za-z]", title):
            continue
        body = " ".join(body_lines)[:MAX_PROJECT_BODY_CHARS]
        projects.append({"title": title, "technologies": _technologies_in(title_line + " " + body)})
        if len(projects) >= MAX_PROJECTS:
            break
    return projects


# --------------------------------------------------------------------------------------
# Experience signals
# --------------------------------------------------------------------------------------
_YEARS_RE = re.compile(
    r"(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:[a-z][a-z/&+\-]*\s+){0,4}?experience",
    re.IGNORECASE,
)
_DOCTORATE_RE = re.compile(r"\b(?:ph\.?\s?d|doctorate|doctoral)\b", re.IGNORECASE)
_MASTERS_RE = re.compile(
    r"\b(?:m\.?\s?tech|m\.?\s?sc|m\.?\s?eng|m\.s|mba|master(?:'s|s)?\s+(?:of|in|degree))\b", re.IGNORECASE)
_BACHELORS_RE = re.compile(
    r"\b(?:b\.?\s?tech|b\.?\s?sc|b\.?\s?eng|b\.s|bachelor(?:'s|s)?\s+(?:of|in|degree))\b", re.IGNORECASE)
_INTERNSHIP_RE = re.compile(r"\bintern(?:ship|ships)?\b", re.IGNORECASE)
_CERTIFIED_RE = re.compile(r"\bcertified\b|\bcertification\b", re.IGNORECASE)


def _experience_signals(text: str, sections: dict, project_count: int) -> dict:
    years = None
    for m in _YEARS_RE.finditer(text):
        value = float(m.group(1))
        if 0 < value <= 40:
            years = max(years or 0.0, value)

    if _DOCTORATE_RE.search(text):
        education = "doctorate"
    elif _MASTERS_RE.search(text):
        education = "masters"
    elif _BACHELORS_RE.search(text):
        education = "bachelors"
    else:
        education = None

    experience_lines = [l for l in sections.get("experience", []) if l.strip()]
    certification_lines = [l for l in sections.get("certifications", []) if l.strip()]

    if years is None:
        seniority = "unknown"
    elif years >= 5:
        seniority = "senior"
    elif years >= 2:
        seniority = "mid"
    else:
        seniority = "entry"

    return {
        "years_of_experience": years,              # only if the resume states it explicitly, else null
        "has_work_experience": len(experience_lines) >= 2,
        "has_internship": bool(_INTERNSHIP_RE.search(text)),
        "project_count": project_count,
        "highest_education": education,
        "has_certifications": bool(certification_lines) or bool(_CERTIFIED_RE.search(text)),
        "seniority_hint": seniority,
    }


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
def empty_candidate_profile(method: str = "none") -> dict:
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "extraction_method": method,
        "skills": [],
        "technologies": [],
        "domains": [],
        "projects": [],
        "experience_signals": {
            "years_of_experience": None,
            "has_work_experience": False,
            "has_internship": False,
            "project_count": 0,
            "highest_education": None,
            "has_certifications": False,
            "seniority_hint": "unknown",
        },
    }


def build_candidate_profile(resume_text: str) -> dict:
    text = clean_resume_text(resume_text)
    sections = _split_sections(text)
    found = _match_lexicon(text)

    def ranked(kind: str, limit: int) -> list:
        items = [(-count, pos, name) for name, (k, _d, count, pos) in found.items() if k == kind]
        return [name for _c, _p, name in sorted(items)[:limit]]

    domain_scores = {}
    for _name, (_kind, domains, count, _pos) in found.items():
        for d in domains:
            domain_scores[d] = domain_scores.get(d, 0) + min(count, 3)
    domains = [d for d, _s in sorted(domain_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:MAX_DOMAINS]]

    projects = _extract_projects(sections.get("projects", []))

    profile = empty_candidate_profile(EXTRACTION_METHOD)
    profile["skills"] = ranked("skill", MAX_SKILLS)
    profile["technologies"] = ranked("technology", MAX_TECHNOLOGIES)
    profile["domains"] = domains
    profile["projects"] = projects
    profile["experience_signals"] = _experience_signals(text, sections, len(projects))
    return profile


def parse_stored_profile(raw) -> dict:
    """Load a profile saved in the DB; anything missing or malformed becomes an empty profile
    (older interviews created before profiles existed have NULL)."""
    base = empty_candidate_profile("none")
    if not raw:
        return base
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return base
    if not isinstance(data, dict):
        return base

    def str_list(value):
        return [v for v in value if isinstance(v, str)] if isinstance(value, list) else []

    base["extraction_method"] = data.get("extraction_method") if isinstance(data.get("extraction_method"), str) else "unknown"
    for key in ("skills", "technologies", "domains"):
        base[key] = str_list(data.get(key))
    projects = []
    for p in data.get("projects", []) if isinstance(data.get("projects"), list) else []:
        if isinstance(p, dict) and isinstance(p.get("title"), str):
            projects.append({"title": p["title"], "technologies": str_list(p.get("technologies"))})
    base["projects"] = projects
    if isinstance(data.get("experience_signals"), dict):
        base["experience_signals"].update(
            {k: v for k, v in data["experience_signals"].items() if k in base["experience_signals"]})
    return base


def profile_is_empty(profile: dict) -> bool:
    return not (profile.get("skills") or profile.get("technologies") or profile.get("domains") or profile.get("projects"))


def profile_to_prompt_text(profile: dict) -> str:
    """Compact, human-readable profile for an LLM prompt (a few hundred characters)."""
    if profile_is_empty(profile):
        return "No verified details could be extracted from the resume."

    lines = []
    if profile.get("skills"):
        lines.append("Skills: " + ", ".join(profile["skills"][:8]))
    if profile.get("technologies"):
        lines.append("Technologies: " + ", ".join(profile["technologies"][:10]))
    if profile.get("domains"):
        lines.append("Domains: " + ", ".join(profile["domains"][:4]))
    if profile.get("projects"):
        parts = []
        for p in profile["projects"][:3]:
            techs = ", ".join(p["technologies"][:3])
            parts.append(f'{p["title"]} ({techs})' if techs else p["title"])
        lines.append("Projects: " + "; ".join(parts))

    sig = profile.get("experience_signals", {})
    exp = []
    if sig.get("years_of_experience"):
        years = sig["years_of_experience"]
        exp.append(f"{years:g} years of experience stated")
    if sig.get("has_internship"):
        exp.append("internship experience")
    if sig.get("has_work_experience"):
        exp.append("work experience section present")
    if sig.get("highest_education"):
        exp.append(f'highest education: {sig["highest_education"]}')
    if exp:
        lines.append("Experience: " + ", ".join(exp))
    return " | ".join(lines)


_CANONICAL_DOMAINS = {canonical: set(domains) for canonical, _kind, domains, _aliases in _LEXICON}


def find_lexicon_terms(text: str) -> set:
    """Canonical names of every lexicon skill/technology mentioned in `text` (used to check that generated
    questions never attribute a skill to the candidate that their resume does not list)."""
    return set(_match_lexicon(text or ""))


def profile_terms_for_domains(profile: dict, domain_names, limit: int = 6) -> list:
    """The candidate's skills/technologies that belong to the given profile domains.

    Used to build retrieval queries: terms unrelated to the knowledge corpus (Docker, AWS, ...)
    are left out instead of diluting the query. Skills (concepts) and technologies each get
    half of `limit`, strongest first (the profile lists are already ranked)."""
    wanted = set(domain_names)
    per_kind = max(limit // 2, 1)
    terms = []
    for key in ("skills", "technologies"):
        picked = [n for n in profile.get(key, []) if _CANONICAL_DOMAINS.get(n, set()) & wanted]
        terms.extend(picked[:per_kind])
    return terms[:limit]
