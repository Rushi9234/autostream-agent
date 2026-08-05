"""
RAG pipeline: loads the markdown KB, builds a FAISS index, retrieves chunks.

This is where I experimented with retrieval-augmented generation instead of
just stuffing the whole knowledge base into every prompt. The index is now
cached to disk (index/) and only rebuilt when the knowledge base file actually
changes, so restarting the app doesn't re-embed the same text every single time.
"""

import hashlib
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter

KB_PATH = Path(__file__).parent / "knowledge_base" / "autostream_kb.md"
INDEX_DIR = Path(__file__).parent / "index"
INDEX_HASH_FILE = INDEX_DIR / "kb.hash"

# Split the KB on markdown headers so pricing, features, and policies end up
# as separate chunks. This works well because the KB is small and structured.
HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

# Cache the vector store in-process too, so a single run doesn't even hit disk
# more than once.
_vector_store = None


def _load_and_split():
    """Read the KB file and return a list of chunks."""
    if not KB_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge base not found at {KB_PATH}. "
            "Make sure knowledge_base/autostream_kb.md exists."
        )

    text = KB_PATH.read_text(encoding="utf-8")
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)
    return splitter.split_text(text)


def _kb_hash():
    """Content hash of the KB file, used to detect whether the index is stale."""
    return hashlib.sha256(KB_PATH.read_bytes()).hexdigest()


def _load_cached_index(embeddings):
    """Return the on-disk FAISS index if it exists and still matches the KB file."""
    if not (INDEX_DIR / "index.faiss").exists() or not INDEX_HASH_FILE.exists():
        return None
    if INDEX_HASH_FILE.read_text().strip() != _kb_hash():
        return None
    return FAISS.load_local(
        str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
    )


def build_vector_store(force_rebuild=False):
    """
    Build the FAISS index, or load it from disk / an in-process cache.

    Originally this rebuilt (and re-embedded) the KB from scratch on every
    process start, which doesn't scale past a toy knowledge base. Now the
    index is saved under index/ and only rebuilt when the KB content hash
    changes, or when force_rebuild=True.
    """
    global _vector_store
    if _vector_store is not None and not force_rebuild:
        return _vector_store

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    if not force_rebuild:
        cached = _load_cached_index(embeddings)
        if cached is not None:
            _vector_store = cached
            return _vector_store

    chunks = _load_and_split()
    _vector_store = FAISS.from_documents(chunks, embeddings)

    INDEX_DIR.mkdir(exist_ok=True)
    _vector_store.save_local(str(INDEX_DIR))
    INDEX_HASH_FILE.write_text(_kb_hash())

    return _vector_store


def retrieve(query, k=3):
    """Return top-k chunks joined into a single context string."""
    store = build_vector_store()
    docs = store.similarity_search(query, k=k)
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    # `python rag.py` forces a rebuild and prints a quick retrieval sanity check.
    build_vector_store(force_rebuild=True)
    print(retrieve("How much does the Pro plan cost?"))
