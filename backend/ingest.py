"""Explicit knowledge-base ingestion. Run it deliberately; the API server never does this.

    cd backend
    python ingest.py             # rebuild the collection from rag_config.CORPUS
    python ingest.py --dry-run   # parse + clean + chunk + report, write nothing

Behaviour: RECREATE, not upsert. The source -> domain mapping lives in rag_config.CORPUS; if a
source is removed or re-mapped, an upsert would leave stale chunks behind and the collection would
stop matching the manifest. A full rebuild always produces exactly what the manifest describes
(a few minutes of CPU embedding, run on purpose). Chunk ids are deterministic (uuid5 of the chunk
id), so the result is identical on every run. Nothing is dropped until every source has been parsed
and chunked successfully, so a missing PDF can never wipe a working collection.

Stop the API server first: embedded Qdrant allows one process per ./qdrant_db at a time.
"""
import argparse
import collections
import os
import re
import sys
import unicodedata
import uuid

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import rag_config as cfg

HERE = os.path.dirname(os.path.abspath(__file__))
# Where the corpus PDFs are looked for, in order.
SOURCE_DIRS = [os.path.join(HERE, "..", "For Rag injesting"), HERE]
CHUNK_ID_NAMESPACE = uuid.UUID("5a0b3c1e-0000-4000-8000-53796e617073")   # fixed -> stable point ids


def find_source(filename: str) -> str | None:
    for directory in SOURCE_DIRS:
        path = os.path.normpath(os.path.join(directory, filename))
        if os.path.exists(path):
            return path
    return None


# --- cleaning -----------------------------------------------------------------------------------
def _signature(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip().lower())


def strip_page_furniture(pages: list) -> int:
    """Remove running headers/footers and bare page numbers in place. Returns lines removed.

    A line counts as furniture when (digits normalised) it appears among the first/last two lines
    of at least 30% of the pages (min 5). Only edge lines are considered, so body text is safe."""
    counts = collections.Counter()
    for page in pages:
        lines = [l for l in page.page_content.splitlines() if l.strip()]
        for sig in {_signature(l) for l in lines[:2] + lines[-2:]}:
            counts[sig] += 1
    threshold = max(5, int(0.3 * len(pages)))
    furniture = {sig for sig, c in counts.items() if c >= threshold and len(sig) <= 100}

    removed = 0
    for page in pages:
        kept = []
        for line in page.page_content.splitlines():
            if line.strip() and (_signature(line) in furniture or re.fullmatch(r"\s*(page\s+)?\d{1,4}\s*", line, re.I)):
                removed += 1
                continue
            kept.append(line)
        page.page_content = "\n".join(kept)
    return removed


def chunk_rejection(text: str) -> str | None:
    """Why a chunk is useless noise, or None if it should be kept."""
    stripped = text.strip()
    if not stripped:
        return "empty"
    if len(stripped) < cfg.MIN_CHUNK_CHARS:
        return "too_short"
    compact = re.sub(r"\s", "", stripped)
    if sum(c.isalpha() for c in compact) / len(compact) < cfg.MIN_ALPHA_RATIO:
        return "mostly_non_alphabetic"
    if re.search(r"(\.\s*){8,}", stripped):
        return "dot_leaders_toc"
    lines = [l for l in stripped.splitlines() if l.strip()]
    if len(lines) >= 4 and sum(bool(re.search(r"\d\s*$", l)) for l in lines) / len(lines) >= 0.5:
        return "toc_or_index_like"          # most lines end in a page number
    if re.search(r"all rights reserved|\bisbn\b|library of congress|disclaimer of liability|limited warranty|"
                 r"no part of this (?:publication|book)", stripped, re.I):
        return "copyright_boilerplate"
    return None


# --- building documents -------------------------------------------------------------------------
def build_documents(entry: dict, path: str, splitter, stats: collections.Counter) -> list:
    pages = PyPDFLoader(path).load()
    stats["pages"] += len(pages)
    for page in pages:   # fold typographic ligatures (ﬁ, ﬂ, ...) and odd unicode so embeddings see plain words
        page.page_content = unicodedata.normalize("NFKC", page.page_content)
    stats["furniture_lines_removed"] += strip_page_furniture(pages)
    chunks = splitter.split_documents(pages)
    stats["raw_chunks"] += len(chunks)

    slug = re.sub(r"[^a-z0-9]+", "-", entry["title"].lower()).strip("-")[:40]
    per_page = collections.Counter()
    documents = []
    for chunk in chunks:
        reason = chunk_rejection(chunk.page_content)
        if reason:
            stats[f"dropped_{reason}"] += 1
            continue
        page = int(chunk.metadata.get("page", 0)) + 1          # PyPDFLoader pages are 0-based
        per_page[page] += 1
        chunk.page_content = chunk.page_content.strip()
        chunk.metadata = {                                      # explicit schema, no PDF-producer clutter
            "domains": list(entry["domains"]),
            "source_file": entry["file"],
            "source_title": entry["title"],
            "page": page,
            "chunk_id": f"{slug}-p{page:04d}-c{per_page[page]:02d}",
        }
        documents.append(chunk)
    stats["kept_chunks"] += len(documents)
    return documents


def ingest_all_knowledge(qdrant_path: str = cfg.QDRANT_PATH, dry_run: bool = False) -> int:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.CHUNK_SIZE, chunk_overlap=cfg.CHUNK_OVERLAP, separators=["\n\n", "\n", ".", " ", ""])

    all_documents, report = [], []
    for entry in cfg.CORPUS:
        if not entry["ingest"]:
            report.append((entry["file"], "SKIPPED", entry["note"]))
            continue
        path = find_source(entry["file"])
        if path is None:
            print(f"ERROR: source '{entry['file']}' not found in {SOURCE_DIRS}. Nothing was changed.")
            return 1
        stats = collections.Counter()
        print(f"\n📄 {entry['title']}  ->  {entry['domains']}   ({os.path.relpath(path, HERE)})")
        documents = build_documents(entry, path, splitter, stats)
        print("   " + ", ".join(f"{k}={v}" for k, v in sorted(stats.items())))
        all_documents.extend(documents)
        report.append((entry["file"], f"{len(documents)} chunks", ",".join(entry["domains"])))

    if not all_documents:
        print("ERROR: no chunks were produced. Nothing was changed.")
        return 1

    print("\nSource summary:")
    for name, outcome, detail in report:
        print(f"  {name}: {outcome} ({detail})")
    per_domain = collections.Counter(d for doc in all_documents for d in doc.metadata["domains"])
    print("Chunks per domain:", dict(per_domain))
    if dry_run:
        print("Dry run: nothing written.")
        return 0

    # Heavy imports only when we actually write.
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    try:
        client = QdrantClient(path=qdrant_path)
    except Exception as e:
        print(f"ERROR: could not open Qdrant at {qdrant_path} ({e}). Is the API server still running?")
        return 1

    print(f"\n🧠 Loading embedding model {cfg.EMBEDDING_MODEL} ...")
    embeddings = HuggingFaceEmbeddings(model_name=cfg.EMBEDDING_MODEL)

    if client.collection_exists(cfg.COLLECTION_NAME):
        print("🧹 Dropping existing collection (full rebuild)...")
        client.delete_collection(cfg.COLLECTION_NAME)
    client.create_collection(
        collection_name=cfg.COLLECTION_NAME,
        vectors_config=VectorParams(size=cfg.EMBEDDING_DIM, distance=Distance.COSINE),
    )

    print(f"⏳ Embedding and writing {len(all_documents)} chunks...")
    store = QdrantVectorStore(client=client, collection_name=cfg.COLLECTION_NAME, embedding=embeddings)
    ids = [str(uuid.uuid5(CHUNK_ID_NAMESPACE, doc.metadata["chunk_id"])) for doc in all_documents]
    store.add_documents(documents=all_documents, ids=ids, batch_size=64)

    stored = client.count(cfg.COLLECTION_NAME, exact=True).count
    print(f"✅ Done. {stored} points stored (expected {len(all_documents)}).")
    client.close()
    return 0 if stored == len(all_documents) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild the SynapSift knowledge base.")
    parser.add_argument("--dry-run", action="store_true", help="parse, clean and chunk only; write nothing")
    parser.add_argument("--path", default=cfg.QDRANT_PATH, help="Qdrant storage directory (default: %(default)s)")
    args = parser.parse_args()
    sys.exit(ingest_all_knowledge(args.path, args.dry_run))
