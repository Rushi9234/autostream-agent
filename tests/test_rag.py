"""
Tests for the parts of rag.py that don't need an embeddings API call —
chunking and the KB-change hash used to decide whether to rebuild the index.
"""

import rag


def test_load_and_split_produces_chunks_with_expected_headers():
    chunks = rag._load_and_split()
    assert len(chunks) > 0

    headers_seen = {chunk.metadata.get("h2") for chunk in chunks if chunk.metadata.get("h2")}
    assert "Pricing Plans" in headers_seen or any("Pricing" in h for h in headers_seen)


def test_kb_hash_is_stable_for_unchanged_file():
    assert rag._kb_hash() == rag._kb_hash()


def test_format_chunk_section_hierarchy():
    class DummyDoc:
        def __init__(self, metadata):
            self.metadata = metadata

    doc = DummyDoc({"h1": "AutoStream KB", "h2": "Pricing Plans", "h3": "Pro Plan"})
    assert rag.format_chunk_section(doc) == "Pricing Plans → Pro Plan"

    doc_h2_only = DummyDoc({"h1": "AutoStream KB", "h2": "Company Policies"})
    assert rag.format_chunk_section(doc_h2_only) == "Company Policies"


def test_retrieve_with_sources_filters_off_topic_high_distance(monkeypatch):
    class DummyDoc:
        def __init__(self, content, metadata):
            self.page_content = content
            self.metadata = metadata

    class FakeStore:
        def similarity_search_with_score(self, query, k=3):
            if "weather" in query:
                # Distance 1.35 > threshold 0.52
                return [(DummyDoc("Some text", {"h1": "KB"}), 1.35)]
            # Distance 0.35 <= threshold 0.52
            return [(DummyDoc("Pro Plan is $79/mo", {"h1": "KB", "h2": "Pricing Plans", "h3": "Pro Plan"}), 0.35)]

    monkeypatch.setattr(rag, "build_vector_store", lambda: FakeStore())

    # Relevant query returns context & source
    ctx, sources = rag.retrieve_with_sources("How much is Pro?")
    assert "[Section: Pricing Plans → Pro Plan]" in ctx
    assert len(sources) == 1
    assert sources[0]["section"] == "Pricing Plans → Pro Plan"

    # Off-topic query returns empty context & no sources
    off_ctx, off_sources = rag.retrieve_with_sources("What is the weather in Tokyo?")
    assert off_ctx == ""
    assert off_sources == []

