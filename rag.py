"""RAG pipeline: loads the markdown KB, builds a FAISS index, retrieves chunks."""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter


KB_PATH = Path(__file__).parent / "knowledge_base" / "autostream_kb.md"

# Split the KB on markdown headers so pricing, features, and policies end up
# as separate chunks. This works well because the KB is small and structured.
HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

# Cache the vector store so we don't rebuild it every turn.
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


def build_vector_store():
    """Build the FAISS index (or return the cached one)."""
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    chunks = _load_and_split()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    _vector_store = FAISS.from_documents(chunks, embeddings)
    return _vector_store


def retrieve(query, k=3):
    """Return top-k chunks joined into a single context string."""
    store = build_vector_store()
    docs = store.similarity_search(query, k=k)
    return "\n\n---\n\n".join(doc.page_content for doc in docs)
