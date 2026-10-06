"""Single source of truth for RAG: tunable settings, the role -> knowledge-domain taxonomy and the
corpus manifest (which source document feeds which domain). Standard library only, so it can be
imported by the ingestion script, the retrieval module and tests alike."""
import re

# ---------------------------------------------------------------------------------------------
# Tunable settings
# ---------------------------------------------------------------------------------------------
COLLECTION_NAME = "textbook_knowledge"
QDRANT_PATH = "./qdrant_db"
METADATA_KEY = "metadata"          # LangChain-Qdrant stores Document.metadata under this payload key
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

RETRIEVAL_TOP_K = 4                # candidates fetched per query
MAX_EXTRA_CANDIDATES_FOR_DEDUP = 6 # extra candidates fetched so already-used chunks can be replaced
# Cosine SIMILARITY (higher = more similar; identical text = 1.0). Chunks scoring below this are
# discarded and the interviewer is told no reference material is available.
# Measured with the real all-MiniLM-L6-v2 index (7,320 chunks): the best chunk for on-topic queries
# scores 0.54-0.63; for off-topic queries (React/CSS, Kubernetes, SQL, plumbing, recipes...) the best
# chunk scores 0.13-0.30. 0.40 sits in the middle of that gap. Re-measure if the model or corpus changes.
MIN_RELEVANCE_SCORE = 0.40

QUERY_PROFILE_TERMS = 6            # max candidate skills/technologies placed in a query
MIN_ANSWER_CHARS_FOR_QUERY = 40    # shorter answers carry no usable topical signal
MAX_ANSWER_CHARS_IN_QUERY = 200
TRACE_EXCERPT_CHARS = 400          # how much retrieved text is persisted per AI message

# Ingestion chunking (unchanged approach: recursive splitter, 800 chars / 150 overlap)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MIN_CHUNK_CHARS = 120              # shorter chunks are dropped as noise
MIN_ALPHA_RATIO = 0.60             # chunks that are mostly digits/symbols (tables, indexes) are dropped

# ---------------------------------------------------------------------------------------------
# Canonical knowledge domains
# ---------------------------------------------------------------------------------------------
DOMAIN_AI_ML = "ai_ml"
DOMAIN_DATA_SCIENCE = "data_science"
DOMAIN_ADVANCED_ML = "advanced_ml"

# label            : how the domain is described inside retrieval queries
# topic_seeds      : trusted per-domain topics; the interview's progress picks one so topic coverage
#                    advances independently of whatever the candidate happens to write
# profile_domains  : resume-profile domains whose skills are relevant when querying this corpus
DOMAIN_INFO = {
    DOMAIN_AI_ML: {
        "label": "AI and machine learning",
        "topic_seeds": [
            "supervised learning and generalization",
            "overfitting, regularization and model evaluation",
            "decision trees and ensemble methods",
            "neural networks and deep learning",
            "feature engineering and data preparation",
            "unsupervised learning and clustering",
            "gradient descent and optimization",
            "bias and variance trade-off",
        ],
        "profile_domains": {"machine learning", "deep learning", "natural language processing",
                            "computer vision", "generative ai"},
    },
    DOMAIN_DATA_SCIENCE: {
        "label": "data science and applied machine learning",
        "topic_seeds": [
            "data preprocessing and feature scaling",
            "model evaluation and cross-validation",
            "supervised learning algorithms in scikit-learn",
            "unsupervised learning, clustering and dimensionality reduction",
            "pipelines and parameter tuning",
            "representing categorical data and feature engineering",
            "working with text data",
            "choosing the right algorithm for a problem",
        ],
        "profile_domains": {"data science", "machine learning", "natural language processing"},
    },
    DOMAIN_ADVANCED_ML: {
        "label": "advanced and theoretical machine learning",
        "topic_seeds": [
            "probabilistic models and Bayesian inference",
            "linear models for regression and classification",
            "kernel methods and support vector machines",
            "neural networks and backpropagation",
            "mixture models and the EM algorithm",
            "graphical models",
            "sampling methods",
            "model selection and the bias-variance decomposition",
        ],
        "profile_domains": {"machine learning", "deep learning"},
    },
}

# Ordered most-specific first; first rule with a matching phrase wins. A phrase matches only as
# whole words in the normalised role. Roles matching no rule have NO corpus (resolve_domain -> None):
# they are never silently mapped to the closest-looking ML material.
_DOMAIN_RULES = [
    (DOMAIN_ADVANCED_ML, ["advanced theoretical ml", "advanced ml", "advanced machine learning",
                          "theoretical ml", "theoretical machine learning", "ml researcher",
                          "machine learning researcher", "ai researcher", "research scientist"]),
    (DOMAIN_DATA_SCIENCE, ["data scientist", "data science", "applied ml", "applied machine learning",
                           "applied scientist"]),
    (DOMAIN_AI_ML, ["ai ml", "ai and ml", "machine learning", "ml engineer", "ml developer",
                    "ai engineer", "ai developer", "artificial intelligence", "deep learning"]),
]


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower().replace("'", "")).strip()


def resolve_domain(role: str) -> str | None:
    """Canonical knowledge domain for a (stored, user-facing) role, or None if no corpus supports it."""
    padded = f" {_normalize(role)} "
    for domain, phrases in _DOMAIN_RULES:
        if any(f" {phrase} " in padded for phrase in phrases):
            return domain
    return None


# ---------------------------------------------------------------------------------------------
# Corpus manifest: which source document feeds which domain(s).
# Every source is embedded ONCE; a source that serves two domains carries both in its `domains`
# metadata (no duplicated chunks). All sources are machine-learning material, so they back ONLY the
# three ML domains above: there is deliberately no backend / frontend / general-software corpus.
# Files live in "For Rag injesting/" (ingest.py also falls back to backend/).
# ---------------------------------------------------------------------------------------------
CORPUS = [
    {"file": "2019BurkovTheHundred-pageMachineLearning.pdf",
     "title": "The Hundred-Page Machine Learning Book",
     "domains": [DOMAIN_AI_ML, DOMAIN_DATA_SCIENCE], "ingest": True,
     "note": "Concise ML fundamentals; primary AI/ML source and a fit for data science fundamentals."},
    {"file": "Artificial Intelligence, Machine Learning, and Deep Learning.pdf",
     "title": "Artificial Intelligence, Machine Learning, and Deep Learning",
     "domains": [DOMAIN_AI_ML], "ingest": True,
     "note": "Broad AI/ML/DL coverage incl. neural networks, RNN/LSTM, autoencoders. Not previously ingested."},
    {"file": "Machine Learning For Absolute Beginners.pdf",
     "title": "Machine Learning For Absolute Beginners",
     "domains": [DOMAIN_AI_ML], "ingest": True,
     "note": "Plain-English ML fundamentals (introductory level). Not previously ingested."},
    {"file": "Master Machine Learning Algorithms - Discover how they work and Implement Them From Scratch by Jason Brownlee (z-lib.org).pdf",
     "title": "Master Machine Learning Algorithms",
     "domains": [DOMAIN_AI_ML, DOMAIN_DATA_SCIENCE], "ingest": True,
     "note": "Step-by-step explanations of common algorithms (trees, kNN, SVM, ...). Not previously ingested."},
    {"file": "Introduction to Machine Learning with Python ( PDFDrive.com )-min.pdf",
     "title": "Introduction to Machine Learning with Python",
     "domains": [DOMAIN_DATA_SCIENCE], "ingest": True,
     "note": "Applied ML with scikit-learn, 'a guide for data scientists'. Primary data-science source. "
             "Previously ingested twice (also under a Backend role, which was wrong)."},
    {"file": "Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf",
     "title": "Pattern Recognition and Machine Learning",
     "domains": [DOMAIN_ADVANCED_ML], "ingest": True,
     "note": "Theoretical / probabilistic ML. Primary advanced-ML source. Equations extract imperfectly."},
    {"file": "MachineLearningTomMitchell.pdf",
     "title": "Machine Learning (Mitchell)",
     "domains": [DOMAIN_ADVANCED_ML], "ingest": True,
     "note": "Classic theory-oriented textbook (concept learning, Bayesian learning, ...). Not previously ingested."},
]
