"""All tunable values for the interview-intelligence layer in one place (standard library only)."""

# --- difficulty -------------------------------------------------------------------------------
DIFFICULTY_LEVELS = ["beginner", "intermediate", "advanced"]   # index = level; moves at most 1 per answer

# --- memory -----------------------------------------------------------------------------------
RECENT_TURNS = 3                    # Q/A pairs shown to the model (the newest one is the answer being judged)
MAX_HISTORY_QUESTION_CHARS = 300    # per question in the prompt
MAX_HISTORY_ANSWER_CHARS = 400      # per answer in the prompt
MAX_REFERENCE_CHARS = 900           # retrieved chunk placed in the prompt
MAX_TRACKED_TOPICS = 30             # bounds every topic list kept in the stored state
MAX_TRACKED_CHUNKS = 30

# --- flow control -----------------------------------------------------------------------------
MAX_WEAK_PROBES = 1                 # follow-ups allowed on a topic after a weak/irrelevant answer
MAX_PARTIAL_PROBES = 1              # follow-ups allowed on a topic after a partial answer
MIN_TURNS_BEFORE_CONCLUDE = 6       # model-requested conclusion is rejected before this many answers
MIN_TOPICS_BEFORE_CONCLUDE = 3      # ...or before this many distinct topics have been covered
# (the hard cap of 10 answers stays in main.py and is decided by Python alone)

# --- generation / repair ----------------------------------------------------------------------
MAX_REPAIR_ATTEMPTS = 1             # total re-asks per turn, shared by validation AND duplicate failures

# --- semantic duplicate detection (cosine similarity of question embeddings, higher = more alike) --
# Measured with all-MiniLM-L6-v2 on interview-style questions: paraphrases of the same question score
# 0.70-0.95 (median 0.79); different concepts score 0.06-0.30; two same-area-but-distinct pairs scored 0.67 and
# 0.69. 0.70 is the lowest value that catches every paraphrase tried. A false positive only costs one repair.
QUESTION_SIMILARITY_THRESHOLD = 0.70
FOLLOW_UP_SIMILARITY_THRESHOLD = 0.92   # applied to the question being deliberately followed up on

# --- deterministic answer-quality heuristics (interview flow only, never scoring) ---------------
WEAK_MAX_CONTENT_WORDS = 7          # fewer meaningful words than this -> weak
NON_ANSWER_MAX_WORDS = 12           # "I don't know ..." only counts as a non-answer in short replies
NON_ANSWER_MAX_REMAINDER_WORDS = 2  # ...and only if at most this many meaningful words remain besides the phrase
IRRELEVANT_MAX_WORDS = 15           # only short replies can be called irrelevant
# Measured: nonsense replies ("pink pong shoot a gun", "banana pizza dinosaur", ...) score -0.04..0.10 against the
# question; short genuine answers ("add a penalty term", "use dropout", ...) score 0.12..0.50.
IRRELEVANT_MAX_SIMILARITY = 0.11    # answer<->question cosine similarity below this (and no shared terms)
STRONG_MIN_WORDS = 35
STRONG_MIN_RELEVANT_TERMS = 4

# --- question validation ------------------------------------------------------------------------
MIN_QUESTION_CHARS = 15
MAX_QUESTION_CHARS = 450
MAX_QUESTION_SENTENCES = 3          # brief acknowledgement + the question
MAX_TOPIC_CHARS = 60
