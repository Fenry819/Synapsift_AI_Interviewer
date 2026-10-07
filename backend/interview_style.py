"""Candidate-facing wording the BACKEND controls: opening greeting, reactions to the previous answer, closing message.

The model only supplies the technical question. Everything conversational around it is chosen here, deterministically
(from the interview step, the technical quality of the last answer and the candidate's behaviour), so tone never depends
on model compliance and wording varies without randomness. Standard library only.

Rules for every string in this file:
  * firm + neutral + professional: never insulting, shaming, sarcastic, arguing, threatening or mentioning scores,
    pass/fail or hiring; restrained praise only;
  * no technical content (a reaction may say an answer was inaccurate or vague, never what the correct answer is);
  * no mention of internal systems, providers, retrieval, embeddings or scoring;
  * no question mark: the final message must contain exactly one question, and it is the model's;
  * reassurance wording ("No problem", "That's fine", "No worries") appears ONLY in the weak_* families, which are used
    for honest non-answers. It is never used for incorrect, vague, dismissive or confidently wrong answers.
"""
import re

# --- opening (first interviewer turn only) -------------------------------------------------------------------
_OPENING_BODY = ("I'll ask you a few technical questions based on your background and the role you're applying for, "
                 "and we'll adjust the depth as we go.")

_NAME_RE = re.compile(r"[A-Za-zÀ-ɏ][A-Za-zÀ-ɏ'\-]{1,29}")


def candidate_first_name(name) -> str | None:
    """A clean first name for the greeting, or None (-> neutral greeting).

    Account names are user-typed, so nothing is 'repaired': the first word, minus surrounding punctuation, must
    itself look like a name (letters, hyphens, apostrophes). Anything else (markup, digits, underscores, emoji,
    a lone initial...) simply gets the neutral greeting."""
    if not isinstance(name, str):
        return None
    words = name.strip().split()
    if not words:
        return None
    first = words[0].strip(".,;:!?\"()")
    if not _NAME_RE.fullmatch(first):
        return None
    return first[0].upper() + first[1:]


def opening_greeting(name) -> str:
    first = candidate_first_name(name)
    hello = f"Hi {first}, welcome to the interview." if first else "Hello, welcome to the interview."
    return f"{hello} {_OPENING_BODY}"


# --- reactions (every later turn) ------------------------------------------------------------------------------
TRANSITIONS = {
    # ---- strong answer
    "strong_deeper": ["That's a reasonable explanation. Let's go one level deeper.",
                      "Good. Let's take that one step further.",
                      "Thanks, that covers the basics. Let's go a little deeper.",
                      "That's solid so far. Let's push on one aspect."],
    "strong_new": ["Good. Let's move to another topic.",
                   "That's a reasonable explanation. Let's look at another area.",
                   "Thanks. Let's explore a different topic.",
                   "Alright. Let's move on to another area."],
    # ---- partial answer
    "partial_follow": ["You're on the right track, but the explanation is incomplete. Let's clarify one part.",
                       "That's part of it. Let's fill in one missing piece.",
                       "Partly there. Let's look at one aspect more closely.",
                       "That covers some of it. Let's focus on one gap."],
    "partial_new": ["Thanks. Let's move to another topic.",
                    "Understood. Let's explore another area.",
                    "Alright. Let's turn to a different topic."],
    # ---- technically incorrect (recognised misconception): neutral correction, never the answer itself
    "incorrect_follow": ["That isn't technically accurate. Let's clarify the concept.",
                         "That's not quite accurate. Let's revisit the concept.",
                         "I'd challenge that explanation. Let's clarify the idea.",
                         "That doesn't match the standard definition. Let's revisit it."],
    "incorrect_new": ["That still isn't accurate, so let's move on to another topic.",
                      "Let's leave that concept for now and move to another topic.",
                      "That explanation still isn't accurate. Let's move on to a different area."],
    # ---- vague / circular: ask for specifics
    "vague_follow": ["That answer is too vague to demonstrate the concept. Please be more specific.",
                     "That's too general to show understanding. Let's get more specific.",
                     "I need more detail than that. Let's be specific.",
                     "That restates the question without explaining it. Please be more concrete."],
    "vague_new": ["That's still too general, so let's move on to another topic.",
                  "That doesn't demonstrate the concept yet. Let's try a different area.",
                  "Let's move on to another topic; I'd like more concrete detail there."],
    # ---- honest non-answer / too short: calm (the only place reassurance wording is allowed)
    "weak_follow": ["No problem. Let's try a simpler angle.",
                    "That's fine. Let's try a simpler version of the concept.",
                    "Understood. Let's approach it from a more basic angle.",
                    "No worries. Let's take it one step at a time.",
                    "Alright. Let's break that down into something simpler."],
    "weak_new": ["Understood. Let's explore another area.",
                 "Alright, let's move on to a different topic.",
                 "That's fine. Let's turn to another area.",
                 "Okay. Let's look at a different topic.",
                 "Let's move on to another topic."],
    # ---- irrelevant / nonsense: firm refocus
    "redirect": ["That doesn't address the question, so let's refocus.",
                 "That's not quite what I asked, so let's refocus.",
                 "Let's refocus on the question."],
    "irrelevant_new": ["That doesn't address the question, so let's move on to another topic.",
                       "That's not quite what I asked. Let's try a different area.",
                       "Let's set that aside and move on to another topic."],
    # ---- dismissive: a little firmer, still professional
    "dismissive_follow": ["Please answer the technical question directly.",
                          "I'd like a direct answer to the technical question.",
                          "Let's keep to the question and answer it directly.",
                          "Please address the technical question itself."],
    "dismissive_new": ["Please keep your answers focused on the technical questions. Let's move to another topic.",
                       "I need a direct technical answer. Let's try a different topic.",
                       "Let's move on to another topic; please answer the technical questions directly."],
    # ---- mildly unprofessional (slang, laughter, boasting, casual profanity): ONE brief reminder, not every turn
    "reminder_follow": ["Let's keep the interview professional and focused on the technical question.",
                        "Let's keep the discussion focused on the technical content.",
                        "Please keep the interview professional and on topic."],
    "reminder_new": ["Let's keep the interview professional. We'll move on to another topic.",
                     "Please keep the discussion focused on the technical content. Let's try a different area.",
                     "Please keep the interview professional. Let's move to another topic."],
}
_REMINDER_TEXTS = frozenset(TRANSITIONS["reminder_follow"] + TRANSITIONS["reminder_new"])

# Technical-quality labels that count as a poor attempt (used by the planner and by reaction_kind)
POOR_LABELS = ("weak", "vague", "incorrect", "irrelevant")


def reminder_recent(recent: list) -> bool:
    """True if a professionalism reminder was shown in the recently remembered turns (so it is not repeated)."""
    return any(r in _REMINDER_TEXTS for r in recent)


def reaction_kind(step: str, label: str | None, behavior: str, *, reminder_ok: bool = True, confident: bool = False) -> str:
    """Which family of reactions fits. Priority: dismissive behaviour > a (non-repeated) professionalism reminder for a
    poor answer > the technical quality of the answer. A strong answer is never lectured about wording.

    `confident` (bragging / 'easy question') upgrades an otherwise reassuring 'weak' reaction to the neutral 'vague' one:
    a candidate who claims it is easy and then does not answer must not be told 'no problem'."""
    if step == "open":
        return "open"
    if label == "weak" and confident:
        label = "vague"
    follow = step in ("redirect", "follow_up")
    if label != "strong":
        if behavior == "dismissive":
            return "dismissive_follow" if follow else "dismissive_new"
        if behavior == "unprofessional" and reminder_ok and label in POOR_LABELS:
            return "reminder_follow" if follow else "reminder_new"
    if step == "redirect":
        return "redirect"
    if step == "follow_up":
        return {"strong": "strong_deeper", "partial": "partial_follow", "incorrect": "incorrect_follow",
                "vague": "vague_follow"}.get(label, "weak_follow")
    return {"strong": "strong_new", "partial": "partial_new", "incorrect": "incorrect_new", "vague": "vague_new",
            "irrelevant": "irrelevant_new"}.get(label, "weak_new")


def _lead(transition: str) -> str:
    """The opening phrase ('That's fine', 'No problem', 'Understood' ...), which is what makes wording sound repeated."""
    return re.split(r"[.,;]", transition, maxsplit=1)[0].strip().lower()


def _first_word(transition: str) -> str:
    return transition.split(" ", 1)[0].strip(".,;'").lower()


def choose_transition(kind: str, rotation: int, recent: list) -> str:
    """Deterministic and varied. Start at an index that rotates with the interview's progress, then prefer, in order:
       1. an option whose opening phrase AND first word were not used in the last turns (no 'Let's ...' every time),
       2. one whose opening phrase was not used,
       3. one whose text was not used,
       4. the one used longest ago."""
    options = TRANSITIONS[kind]
    start = rotation % len(options)
    ordered = [options[(start + offset) % len(options)] for offset in range(len(options))]
    recent_leads = {_lead(r) for r in recent}
    recent_firsts = {_first_word(r) for r in recent[-2:]}
    for candidate in ordered:
        if candidate not in recent and _lead(candidate) not in recent_leads and _first_word(candidate) not in recent_firsts:
            return candidate
    for candidate in ordered:
        if candidate not in recent and _lead(candidate) not in recent_leads:
            return candidate
    for candidate in ordered:
        if candidate not in recent:
            return candidate
    return min(options, key=lambda o: recent.index(o) if o in recent else -1)


# --- closing ----------------------------------------------------------------------------------------------------
# Shown when the interview ends normally (model-requested conclusion after the minimum, or the answer limit).
# No score, no verdict, no mention of how or by what the answers are evaluated.
CLOSING_MESSAGE = ("Thanks, that concludes the technical interview. "
                   "Your responses will now be evaluated and made available for review.")
