"""Deterministic, FLOW-ONLY signals about how a candidate answered. Nothing here scores anything and nothing here feeds
the final evaluation: it only helps the interviewer choose a fitting reaction and the next step. Standard library only.

Two independent axes are produced elsewhere from these signals:
  * technical quality  -> strong | partial | incorrect | vague | irrelevant | weak   (interview_engine.classify_answer)
  * behaviour          -> normal | dismissive | unprofessional                       (detect_behavior, below)

Everything is a transparent pattern list, so behaviour is explainable and easy to extend. Coverage is deliberately
limited: unknown cases fall back to the conservative label, never to a guess.
"""
import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------------------------------------------
# Behaviour
# --------------------------------------------------------------------------------------------------------------
# Impatience, refusal to engage, deflection. (Explicit resignation such as "I quit" / "stop the interview" is a
# separate deterministic rule in main.py and still ends the interview; these only change the interviewer's reaction.)
_DISMISSIVE = [
    ("already_told", r"\balready (?:told|answered|said|explained|gave|mentioned)\b"),
    ("ask_something_else", r"\bask (?:me )?(?:something|another|a different|some other)\b|\bsomething else\b"),
    ("why_asking", r"\bwhy (?:are|do|would) you (?:asking|ask)\b"),
    ("whatever", r"\bwhatever\b|\bwhatev\b"),
    ("move_on", r"\bnext question\b|\b(?:can|could|let'?s) (?:we )?move on\b|\bskip (?:this|it)\b"),
    ("pointless", r"\b(?:pointless|waste of (?:my )?time|boring|stupid question|silly question|dumb question)\b"),
    ("refuses", r"\b(?:not (?:going to|gonna)|won'?t|will not|refuse to) (?:answer|explain|repeat)\b"),
]
# Confidence / boasting. Not bad on its own: it only matters together with a weak answer (see classify_answer).
_CONFIDENT = [
    ("easy_question", r"\b(?:easy|simple|basic|trivial) (?:question|one|stuff)\b|\b(?:too|so|very) easy\b|\bthis is easy\b|\beasy peasy\b|\bpiece of cake\b"),
    ("obviously", r"\bobviously\b|\bduh\b"),
    ("boasting", r"\b(?:i(?:'m| am)|im) (?:definitely |so |going to |gonna )?(?:get(?:ting)?|got) (?:a|the) job\b|\bhire me\b|"
                 r"\bi(?:'m| am) (?:the best|a genius|an expert)\b|\bi know (?:this|everything)\b|\bget(?:ting)? (?:a|the) job\b"),
]
# Slang, laughter, mockery, casual profanity.
_UNPROFESSIONAL = [
    ("laughter", r"\b(?:hah+a*|haha+h*|haah+|hehe+h*|lol+|lmao|rofl)\b"),
    ("slang", r"\b(?:damn|dang|hell|crap|wtf|omg|wow|bro|dude|bruh|noob|jeez|geez)\b"),
    ("mockery", r"\b(?:yeah right|oh please|as if)\b"),
]
_DISMISSIVE_RE = [(n, re.compile(p, re.I)) for n, p in _DISMISSIVE]
_CONFIDENT_RE = [(n, re.compile(p, re.I)) for n, p in _CONFIDENT]
_UNPROFESSIONAL_RE = [(n, re.compile(p, re.I)) for n, p in _UNPROFESSIONAL]


@dataclass
class Behavior:
    label: str = "normal"                # normal | dismissive | unprofessional
    confident: bool = False              # "easy question" / boasting marker present
    signals: list = field(default_factory=list)


def detect_behavior(text: str, profane: bool = False) -> Behavior:
    """Behaviour label from the candidate's wording. Dismissive outranks unprofessional. Boasting counts as
    unprofessional (casual, off-topic); plain confidence ("easy question") is only recorded as a flag."""
    t = text or ""
    dismissive = [f"dismissive:{n}" for n, rx in _DISMISSIVE_RE if rx.search(t)]
    confident = [f"confident:{n}" for n, rx in _CONFIDENT_RE if rx.search(t)]
    casual = [f"unprofessional:{n}" for n, rx in _UNPROFESSIONAL_RE if rx.search(t)]
    if profane:
        casual.append("unprofessional:profanity")
    boasting = [s for s in confident if s.endswith(":boasting")]

    if dismissive:
        label = "dismissive"
    elif casual or boasting:
        label = "unprofessional"
    else:
        label = "normal"
    return Behavior(label=label, confident=bool(confident), signals=dismissive + confident + casual)


def strip_markers(text: str) -> str:
    """The answer with behaviour chatter (slang, laughter, 'easy question', dismissals) removed, so technical quality is
    judged on the substance alone: 'damn its an easy question sir' must not count as an attempted answer."""
    t = text or ""
    for _n, rx in _DISMISSIVE_RE + _CONFIDENT_RE + _UNPROFESSIONAL_RE:
        t = rx.sub(" ", t)
    t = re.sub(r"\b(?:sir|mate|man)\b", " ", t, flags=re.I)       # address terms that only appear with such chatter
    return re.sub(r"\s+", " ", t).strip(" .,!?-")


# --------------------------------------------------------------------------------------------------------------
# Obvious misconceptions about the core concepts the corpora cover. Each rule: (name, context regex, wrong-claim regex).
# Applied only when the QUESTION/TOPIC matches the context and the ANSWER makes the wrong claim. High confidence or
# nothing: anything not listed is simply not flagged (it will be judged vague/partial instead).
# --------------------------------------------------------------------------------------------------------------
_MISCONCEPTIONS = [
    ("supervised-needs-no-labels", r"(?<!un)supervised",
     r"(?<!un)supervised[^.]{0,60}\b(?:unlabell?ed|without (?:any )?labels?|no labels?|doesn'?t (?:use|need|require) labels?)\b"),
    ("unsupervised-uses-labels", r"unsupervised",
     r"unsupervised[^.]{0,60}\b(?:uses?|using|trained on|requires?|needs?|with) (?:a )?(?:labell?ed|target|output) (?:data|labels?|examples?|pairs?)\b"),
    ("overfitting-too-simple", r"overfit",
     r"overfit\w*[^.]{0,50}\b(?:too (?:simple|small|few)|(?:high|large) bias|not complex enough)\b"),
    ("overfitting-bad-on-training", r"overfit",
     r"overfit\w*[^.]{0,60}\b(?:performs?|does|do|works?) (?:poorly|badly|worse|bad) on (?:the )?training\b"),
    ("regularization-adds-complexity", r"regulari[sz]",
     r"regulari[sz]\w*[^.]{0,60}\b(?:increases?|raises?|adds?) (?:the )?(?:model )?(?:complexity|variance|overfitting)\b"),
    ("cross-validation-speeds-training", r"cross.?valid",
     r"cross.?valid\w*[^.]{0,60}\b(?:train(?:s|ing)?(?: the)?(?: model)? faster|speed(?:s)? up (?:the )?training|(?:make|makes) training faster)\b"),
    ("gradient-descent-maximizes", r"gradient",
     r"gradient (?:descent|decent)[^.]{0,60}\b(?:maximi[sz]\w*|increas\w*)\b[^.]{0,20}\b(?:loss|error|cost)\b"),
    ("learning-rate-is-a-count", r"learning rate",
     r"learning rate[^.]{0,50}\b(?:is|means|refers to) (?:the )?(?:number of|how many) (?:epochs|layers|iterations|samples|features)\b"),
    ("kmeans-is-supervised", r"k.?means",
     r"k.?means[^.]{0,60}(?<!un)\bsupervised\b|k.?means[^.]{0,60}\b(?:labell?ed (?:data|examples)|classif(?:ies|ication))\b"),
    ("test-set-used-for-training", r"test (?:set|data)|train.?test|validation",
     r"\btest (?:set|data)\b[^.]{0,50}\b(?:to train|for training|train(?:s|ing)? the model)\b"),
    ("high-variance-underfits", r"bias|variance",
     r"high variance[^.]{0,40}\bunder.?fit\w*|high bias[^.]{0,40}\bover.?fit\w*"),
]
_MISCONCEPTIONS_RE = [(n, re.compile(c, re.I), re.compile(w, re.I)) for n, c, w in _MISCONCEPTIONS]


def find_misconception(answer: str, context: str) -> str | None:
    """Name of a recognised misconception the answer states about the question's topic, or None.

    Skipped when the answer contrasts concepts (mentions both 'supervised' and 'unsupervised'), because then a clause
    like 'unlike unsupervised learning, which uses unlabeled data' is correct, not a misconception."""
    a = answer or ""
    both = re.search(r"(?<!un)supervised", a, re.I) and re.search(r"unsupervised", a, re.I)
    for name, ctx_rx, wrong_rx in _MISCONCEPTIONS_RE:
        if name.startswith(("supervised", "unsupervised")) and both:
            continue
        if ctx_rx.search(context or "") and wrong_rx.search(a):
            return name
    return None


# Words that add no technical specificity. An answer whose only content words are these, the question's own words or
# the topic's words is 'circular' ("supervised learning is a type of learning").
GENERIC_FILLER = frozenset("""give gives gave make makes made type types kind kinds sort thing things stuff way ways use used uses using
help helps good bad important basically simply actually something someone anything everything know knowledge machine model models data
learn learning learns just really very also like many much lot lots different various certain related called named based part parts
process get gets got go goes one two simple terms term role play plays played""".split())
