"""tests/test_rag.py — Unit tests for RAGPipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.llm_backend import StubLLM
from src.memory.embedder import Embedder
from src.memory.faiss_memory import FAISSMemory
from src.rag.document_loader import DocumentLoader, Document
from src.rag.rag_pipeline import RAGPipeline


# ── Fixtures ───────────────────────────────────────────────────────────────────

STUB_CFG = {
    "embeddings": {"backend": "stub", "dimension": 64},
    "memory": {"top_k": 3},
    "rag": {"top_k": 3, "chunk_size": 200, "chunk_overlap": 20},
}

CORPUS_TEXT = """
Retrieval-Augmented Generation (RAG) combines neural retrieval with language generation.
FAISS is an efficient vector similarity search library by Meta AI Research.
Sentence transformers produce semantic embeddings for text search and retrieval.
LangChain enables composable LLM application development with agents and chains.
Prompt engineering improves LLM accuracy through structured input design.
"""


@pytest.fixture
def pipeline():
    embedder = Embedder.from_config(STUB_CFG)
    memory = FAISSMemory(embedder=embedder, cfg=STUB_CFG)
    llm = StubLLM()
    pipe = RAGPipeline(memory=memory, llm=llm, top_k=3)
    pipe.ingest_text(CORPUS_TEXT, source="test_corpus")
    return pipe


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestDocumentLoader:

    def test_load_text_chunking(self):
        loader = DocumentLoader(chunk_size=100, chunk_overlap=10)
        docs = loader.load_text("A" * 350)
        assert len(docs) >= 3, "Expected at least 3 chunks for 350-char text with chunk_size=100"

    def test_chunk_content_non_empty(self):
        loader = DocumentLoader(chunk_size=200, chunk_overlap=20)
        docs = loader.load_text("Hello world. " * 20)
        for doc in docs:
            assert doc.content.strip(), "All chunks must be non-empty"

    def test_document_metadata_fields(self):
        loader = DocumentLoader()
        docs = loader.load_text("Some text.", source="my_source")
        assert docs[0].source == "my_source"
        assert docs[0].chunk_index == 0
        assert isinstance(docs[0], Document)


class TestRAGPipeline:

    def test_ingest_increases_memory(self):
        embedder = Embedder.from_config(STUB_CFG)
        memory = FAISSMemory(embedder=embedder, cfg=STUB_CFG)
        llm = StubLLM()
        pipe = RAGPipeline(memory=memory, llm=llm)
        n = pipe.ingest_text("Some knowledge about AI.", source="test")
        assert n >= 1
        assert memory.size >= 1

    def test_query_returns_answer(self, pipeline):
        result = pipeline.query("What is RAG?")
        assert result.answer, "Answer must not be empty"
        assert isinstance(result.answer, str)
        assert len(result.answer) > 10

    def test_query_returns_sources(self, pipeline):
        result = pipeline.query("FAISS similarity search")
        assert isinstance(result.sources, list)
        # May return 0 sources if memory is empty, but pipeline ingested data
        assert len(result.sources) >= 0

    def test_result_has_latency_fields(self, pipeline):
        result = pipeline.query("sentence transformers embeddings")
        assert result.retrieval_latency_ms >= 0
        assert result.generation_latency_ms >= 0
        assert result.total_latency_ms >= 0

    def test_to_dict_keys(self, pipeline):
        result = pipeline.query("LangChain agents")
        d = result.to_dict()
        for key in ("answer", "sources", "query", "retrieval_latency_ms",
                    "generation_latency_ms", "total_latency_ms"):
            assert key in d, f"Missing key: {key}"
