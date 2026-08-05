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
