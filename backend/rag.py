"""Role-aware retrieval: one entry point that turns trusted interview context into a Qdrant query,
filters by knowledge domain, scores, validates relevance and returns a structured result.

Design rules
  * The domain comes from the stored target role (rag_config.resolve_domain), never from the client.
  * A role without a supported corpus never touches the vector store.
  * A failed or empty filtered search is NEVER widened to an unfiltered/global search.
  * Chunks below the relevance threshold are discarded; "no context" is a first-class result.
"""
from dataclasses import dataclass, field

from qdrant_client import models

import rag_config as cfg
from resume_profile import profile_terms_for_domains

STATUS_OK = "ok"
STATUS_NO_CONTEXT = "no_context"            # filtered search ran, nothing was relevant enough
STATUS_UNSUPPORTED_ROLE = "unsupported_role"  # no corpus exists for this role; vector store not queried
STATUS_ERROR = "error"                      # vector store failed; no fallback retrieval is attempted


@dataclass
class RetrievedChunk:
    text: str
    score: float
    source_file: str | None = None
    source_title: str | None = None
    page: int | None = None
    chunk_id: str | None = None
    domains: list = field(default_factory=list)


@dataclass
class RetrievalResult:
    status: str
    domain: str | None
    query: str | None = None
    accepted: list = field(default_factory=list)   # RetrievedChunk, best first, all >= threshold
    considered: int = 0                            # how many candidates Qdrant returned
    top_score: float | None = None                 # best candidate score, accepted or not (for tuning)
    reason: str = ""
    skipped_used: int = 0                          # relevant chunks passed over because the interview already used them

    @property
    def selected(self) -> RetrievedChunk | None:
        return self.accepted[0] if self.accepted else None


def domain_filter(domain: str) -> models.Filter:
    """Qdrant filter: payload `metadata.domains` (an array) must contain the domain."""
    return models.Filter(must=[
        models.FieldCondition(key=f"{cfg.METADATA_KEY}.domains", match=models.MatchAny(any=[domain])),
    ])


def knowledge_base_status(client) -> tuple:
    """('empty' | 'legacy_schema' | 'ok', total points, points tagged with a known domain).

    'legacy_schema' = the collection holds points but none carry `metadata.domains` (it was built by
    an older ingest.py), so every filtered search would silently find nothing."""
    total = client.count(cfg.COLLECTION_NAME).count
    if total == 0:
        return "empty", 0, 0
    known = models.Filter(must=[models.FieldCondition(
        key=f"{cfg.METADATA_KEY}.domains", match=models.MatchAny(any=list(cfg.DOMAIN_INFO)))])
    tagged = client.count(cfg.COLLECTION_NAME, count_filter=known).count
    return ("ok" if tagged else "legacy_schema"), total, tagged


def build_retrieval_query(domain: str, target_role: str, candidate_profile: dict,
                          answer_number: int, current_answer: str | None = None,
                          topic_hint: str | None = None) -> str:
    """Compose the query from TRUSTED context.

    - the topic to cover next: `topic_hint` when the interview engine has chosen one (an uncovered topic, or the
      current topic for a follow-up), otherwise the domain's topic seed for this point of the interview
      (answer_number 0 = first question). Either way coverage never depends on what the candidate writes;
    - the stored target role and the domain label;
    - only those profile terms that belong to the domain (no header text, no unrelated tools);
    - a short excerpt of the current answer, only when it is substantive. It can nudge retrieval
      but is never the whole query.
    """
    info = cfg.DOMAIN_INFO[domain]
    seeds = info["topic_seeds"]
    parts = [topic_hint or seeds[answer_number % len(seeds)], f'{info["label"]} interview for {target_role}']

    terms = profile_terms_for_domains(candidate_profile or {}, info["profile_domains"], limit=cfg.QUERY_PROFILE_TERMS)
    if terms:
        parts.append("Candidate background: " + ", ".join(terms))

    answer = " ".join((current_answer or "").split())
    if len(answer) >= cfg.MIN_ANSWER_CHARS_FOR_QUERY:
        parts.append("Candidate's last answer: " + answer[: cfg.MAX_ANSWER_CHARS_IN_QUERY])
    return ". ".join(parts)


def _to_chunk(document, score: float) -> RetrievedChunk:
    md = document.metadata or {}
    return RetrievedChunk(
        text=document.page_content,
        score=float(score),
        source_file=md.get("source_file"),
        source_title=md.get("source_title"),
        page=md.get("page"),
        chunk_id=md.get("chunk_id"),
        domains=list(md.get("domains", [])),
    )


def retrieve_context(vector_db, *, target_role: str, candidate_profile: dict, answer_number: int,
                     current_answer: str | None = None, topic_hint: str | None = None,
                     exclude_chunk_ids=None) -> RetrievalResult:
    """The single retrieval interface used by the interview endpoints.

    `vector_db` is a LangChain QdrantVectorStore (anything with `similarity_search_with_score`).

    `exclude_chunk_ids`: chunks this interview already used. A used chunk is never selected again; the search
    simply fetches a few extra candidates so a different RELEVANT chunk can take its place. The relevance
    threshold is never lowered to find novelty: if every relevant chunk was already used the result is
    `no_context`, not an unrelated or repeated chunk.

    Score semantics: the collection uses COSINE distance, and LangChain's
    `similarity_search_with_score` returns Qdrant's raw score, i.e. the cosine SIMILARITY:
    higher is better, 1.0 is identical, ~0 is unrelated. So a chunk is accepted when
    score >= cfg.MIN_RELEVANCE_SCORE.
    """
    domain = cfg.resolve_domain(target_role)
    if domain is None:
        return RetrievalResult(STATUS_UNSUPPORTED_ROLE, None,
                               reason="no knowledge corpus is available for this role")

    exclude = set(exclude_chunk_ids or ())
    fetch_k = cfg.RETRIEVAL_TOP_K + min(len(exclude), cfg.MAX_EXTRA_CANDIDATES_FOR_DEDUP)
    query = build_retrieval_query(domain, target_role, candidate_profile, answer_number, current_answer, topic_hint)
    try:
        scored = vector_db.similarity_search_with_score(query, k=fetch_k, filter=domain_filter(domain))
    except Exception as e:
        # Deliberately no retry without the filter: that would leak other domains' content.
        print(f"RAG retrieval failed for domain '{domain}': {e}")
        return RetrievalResult(STATUS_ERROR, domain, query=query, reason=f"vector store error: {e}")

    candidates = [_to_chunk(doc, score) for doc, score in scored]
    top_score = max((c.score for c in candidates), default=None)
    relevant = sorted((c for c in candidates if c.score >= cfg.MIN_RELEVANCE_SCORE),
                      key=lambda c: c.score, reverse=True)
    accepted = [c for c in relevant if c.chunk_id not in exclude]
    skipped = len(relevant) - len(accepted)
    if not accepted:
        if relevant:
            why = f"all {len(relevant)} sufficiently relevant chunks were already used earlier in this interview"
        elif not candidates:
            why = "no chunks in this domain"
        else:
            why = f"best score {top_score:.3f} is below the relevance threshold {cfg.MIN_RELEVANCE_SCORE}"
        return RetrievalResult(STATUS_NO_CONTEXT, domain, query=query, considered=len(candidates),
                               top_score=top_score, reason=why, skipped_used=skipped)
    return RetrievalResult(STATUS_OK, domain, query=query, accepted=accepted, considered=len(candidates),
                           top_score=top_score, reason="ok", skipped_used=skipped)


def build_rag_trace(result: RetrievalResult | None, target_role: str, *, reason_override: str | None = None) -> dict:
    """JSON-safe record of how (and whether) RAG contributed to one AI message."""
    if result is None:
        return {"rag_used": False, "status": "not_applicable", "target_role": target_role,
                "domain": None, "reason": reason_override or "system message, no retrieval performed"}
    selected = result.selected
    return {
        "rag_used": selected is not None,
        "status": result.status,
        "target_role": target_role,
        "domain": result.domain,
        "query": result.query[:500] if result.query else None,
        "threshold": cfg.MIN_RELEVANCE_SCORE,
        "top_k": cfg.RETRIEVAL_TOP_K,
        "candidates_considered": result.considered,
        "skipped_already_used": result.skipped_used,
        "top_score": round(result.top_score, 4) if result.top_score is not None else None,
        "reason": result.reason,
        "selected": None if selected is None else {
            "chunk_id": selected.chunk_id,
            "source_file": selected.source_file,
            "source_title": selected.source_title,
            "page": selected.page,
            "score": round(selected.score, 4),
            "excerpt": selected.text[: cfg.TRACE_EXCERPT_CHARS],
        },
    }
