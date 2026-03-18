"""tests/test_agent.py — Integration tests for MultimodalAgent (stub mode)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.agent import MultimodalAgent
from src.agent.planner import TaskPlanner, Step
from src.llm.llm_backend import StubLLM


# ── Fixtures ───────────────────────────────────────────────────────────────────

STUB_CONFIG = {
    "llm": {"backend": "stub", "stub": {"response_prefix": "[TEST]"}},
    "vision": {"backend": "stub"},
    "embeddings": {"backend": "stub", "dimension": 64},
    "memory": {"top_k": 3},
    "rag": {"top_k": 3, "corpus_path": "data/sample_documents.txt"},
    "agent": {
        "max_iterations": 5,
        "enable_search_tool": False,  # stub search in tests
        "enable_memory_tool": True,
        "enable_vision_tool": True,
    },
    "logging": {"level": "WARNING", "rich_console": False},
    "metrics": {"output_path": "logs/test_metrics.json"},
}


@pytest.fixture(scope="module")
def agent(tmp_path_factory):
    """Create one agent instance shared across the test module."""
    import yaml
    tmp = tmp_path_factory.mktemp("cfg")
    cfg_file = tmp / "config.yaml"
    cfg_file.write_text(yaml.dump(STUB_CONFIG), encoding="utf-8")
    return MultimodalAgent(config_path=str(cfg_file))


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestTaskPlanner:

    def test_fallback_plan_length(self):
        llm = StubLLM()
        planner = TaskPlanner(llm=llm, max_steps=6)
        steps = planner.plan("Summarize the latest AI research papers")
        assert len(steps) >= 1, "Expected at least 1 step"

    def test_step_has_required_fields(self):
        llm = StubLLM()
        planner = TaskPlanner(llm=llm)
        steps = planner.plan("Analyze an image and explain the content")
        for s in steps:
            assert isinstance(s, Step)
            assert s.tool in {"VISION", "MEMORY", "SEARCH", "LLM", "NONE"}
            assert isinstance(s.description, str) and s.description

    def test_steps_are_indexed(self):
        llm = StubLLM()
        planner = TaskPlanner(llm=llm)
        steps = planner.plan("A multimodal task")
        indices = [s.index for s in steps]
        assert len(indices) == len(set(indices)), "Step indices must be unique"


class TestMultimodalAgent:

    def test_run_returns_expected_keys(self, agent):
        result = agent.run("What is retrieval augmented generation?")
        required_keys = {"task", "answer", "rag_answer", "sources",
                         "plan_steps", "step_results", "metrics"}
        for key in required_keys:
            assert key in result, f"Missing key in result: {key}"

    def test_run_answer_is_string(self, agent):
        result = agent.run("Explain the concept of vector embeddings")
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    def test_run_with_image_path(self, agent):
        result = agent.run(
            task="Describe what you see in the image",
            image_path="data/sample_images/test.jpg",
        )
        assert result["image_path"] == "data/sample_images/test.jpg"
        assert result["answer"]

    def test_plan_steps_list(self, agent):
        result = agent.run("Analyze visual content and summarize")
        steps = result["plan_steps"]
        assert isinstance(steps, list)
        assert len(steps) >= 1
        for s in steps:
            assert "index" in s and "tool" in s and "description" in s

    def test_metrics_structure(self, agent):
        result = agent.run("A simple test task")
        m = result["metrics"]
        assert "total_latency_ms" in m
        assert "goal_completed" in m
        assert m["goal_completed"] is True

    def test_query_rag_only(self, agent):
        result = agent.query_rag("What is FAISS?")
        assert "answer" in result
        assert isinstance(result["answer"], str)
