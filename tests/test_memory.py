"""tests/test_memory.py — Unit tests for FAISSMemory and Embedder."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.embedder import StubEmbedder, Embedder
from src.memory.faiss_memory import FAISSMemory

# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def stub_cfg():
    return {
        "embeddings": {"backend": "stub", "dimension": 64},
        "memory": {"index_path": "vector_store/test.index",
                   "metadata_path": "vector_store/test_meta.pkl",
                   "top_k": 3},
    }


@pytest.fixture
def memory(stub_cfg):
    embedder = Embedder.from_config(stub_cfg)
    return FAISSMemory(embedder=embedder, cfg=stub_cfg)


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestStubEmbedder:

    def test_single_embed_shape(self):
        emb = StubEmbedder(dim=64)
        vec = emb.embed("hello world")
        assert vec.shape == (1, 64), f"Expected (1, 64), got {vec.shape}"

    def test_batch_embed_shape(self):
        emb = StubEmbedder(dim=64)
        vecs = emb.embed(["text a", "text b", "text c"])
        assert vecs.shape == (3, 64)

    def test_deterministic(self):
        emb = StubEmbedder(dim=32)
        v1 = emb.embed("same text")
        v2 = emb.embed("same text")
        np.testing.assert_array_equal(v1, v2)

    def test_unit_norm(self):
        emb = StubEmbedder(dim=128)
        vec = emb.embed("unit norm test")[0]
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"


class TestFAISSMemory:

    def test_add_increases_size(self, memory):
        assert memory.size == 0
        memory.add(["document one", "document two"])
        assert memory.size == 2

    def test_search_returns_results(self, memory):
        texts = [
            "neural networks learn representations",
            "FAISS is a similarity search library",
            "RAG improves LLM accuracy",
            "transformers use self-attention",
            "prompt engineering is important",
        ]
        memory.add(texts)
        results = memory.search("vector similarity search", k=3)
        assert len(results) > 0, "Expected at least 1 result"
        assert "text" in results[0]
        assert "score" in results[0]
        assert "rank" in results[0]

    def test_search_rank_ordering(self, memory):
        memory.add(["a", "b", "c", "d"])
        results = memory.search("query", k=4)
        ranks = [r["rank"] for r in results]
        assert ranks == sorted(ranks), "Results should be rank-ordered"

    def test_search_empty_returns_empty(self, memory):
        results = memory.search("anything", k=5)
        assert results == []

    def test_large_index_simulation(self, memory):
        """Simulate a small version of the 100K benchmark for CI speed."""
        N = 500
        memory.simulate_large_index(n=N, batch_size=100)
        assert memory.size == N, f"Expected {N}, got {memory.size}"
        results = memory.search("neural network", k=5)
        assert len(results) == 5
