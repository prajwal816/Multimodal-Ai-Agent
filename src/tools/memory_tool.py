"""
src/tools/memory_tool.py
─────────────────────────
MemoryRetrievalTool — LangChain BaseTool querying FAISSMemory.
"""

from __future__ import annotations

import logging
from typing import Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from src.memory.faiss_memory import FAISSMemory

logger = logging.getLogger("multimodal_agent.tools.memory")


class MemoryInput(BaseModel):
    query: str = Field(description="Semantic query to retrieve relevant knowledge from memory.")
    k: int = Field(default=5, description="Number of results to retrieve.")


class MemoryRetrievalTool(BaseTool):
    """
    Semantic memory retrieval tool backed by FAISS.
    Returns the top-k most relevant stored text chunks.
    """

    name: str = "memory_retrieval"
    description: str = (
        "Retrieve semantically relevant knowledge from the agent's memory store. "
        "Use this when you need background information, prior context, or domain knowledge. "
        "Input should be a natural-language query string."
    )
    args_schema: Type[BaseModel] = MemoryInput
    memory: FAISSMemory = None  # type: ignore[assignment]

    class Config:
        arbitrary_types_allowed = True

    def _run(self, query: str, k: int = 5) -> str:
        if self.memory is None or self.memory.size == 0:
            return "Memory store is empty. No results retrieved."

        results = self.memory.search(query, k=k)
        if not results:
            return f"No relevant memory found for: {query!r}"

        lines = [f"Memory retrieval for '{query}' — top {len(results)} results:\n"]
        for r in results:
            score_pct = min(100.0, r["score"] * 100)
            lines.append(
                f"  [{r['rank']}] (relevance: {score_pct:.1f}%)\n"
                f"      {r['text'][:200].strip()}"
            )
        output = "\n".join(lines)
        logger.debug(f"Memory retrieved {len(results)} results for '{query[:60]}'")
        return output

    async def _arun(self, query: str, k: int = 5) -> str:
        return self._run(query, k)
