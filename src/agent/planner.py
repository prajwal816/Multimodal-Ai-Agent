"""
src/agent/planner.py
─────────────────────
TaskPlanner — decomposes a high-level task into executable sub-steps.

Uses a chain-of-thought LLM prompt that returns a numbered list.
Parses the list into structured Step objects consumed by TaskExecutor.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from src.llm.llm_backend import BaseLLM
from src.llm.prompt_templates import build_planner_prompt

logger = logging.getLogger("multimodal_agent.planner")

# Tool labels the planner can assign
VALID_TOOLS = {"VISION", "MEMORY", "SEARCH", "LLM", "NONE"}


@dataclass
class Step:
    index: int
    description: str
    tool: str          # one of VALID_TOOLS
    completed: bool = False
    result: Optional[str] = None

    def __repr__(self) -> str:
        status = "✅" if self.completed else "⏳"
        return f"Step {self.index} [{self.tool}] {status}: {self.description[:60]}"


class TaskPlanner:

    def __init__(self, llm: BaseLLM, max_steps: int = 8) -> None:
        self._llm = llm
        self._max_steps = max_steps

    def plan(self, task: str) -> List[Step]:
        """
        Decompose *task* into an ordered list of Steps.

        If the LLM returns unparseable output, falls back to a sensible
        default 4-step plan so the executor can still run.
        """
        prompt = build_planner_prompt(task, max_steps=self._max_steps)
        raw = self._llm.generate(prompt)
        logger.debug(f"Raw plan output:\n{raw}")

        steps = self._parse(raw)
        if not steps:
            logger.warning("Planner produced no parseable steps — using fallback plan.")
            steps = self._fallback_plan(task)

        logger.info(f"Plan created: {len(steps)} steps for task: '{task[:80]}'")
        for s in steps:
            logger.debug(f"  {s}")
        return steps

    # ── Parsing ────────────────────────────────────────────────────────────────

    _STEP_RE = re.compile(
        r"^\s*(\d+)[.):]\s*"           # leading number
        r"(?:\[([A-Z]+)\]\s*)?"        # optional [TOOL]
        r"(.+)$",                       # description
        re.MULTILINE,
    )
    _TOOL_INLINE_RE = re.compile(r"\[(VISION|MEMORY|SEARCH|LLM|NONE)\]", re.IGNORECASE)

    def _parse(self, raw: str) -> List[Step]:
        steps: List[Step] = []
        for m in self._STEP_RE.finditer(raw):
            idx = int(m.group(1))
            tool_tag = (m.group(2) or "").upper()
            description = m.group(3).strip()

            # Try to extract tool from inline tag if not in group 2
            if tool_tag not in VALID_TOOLS:
                inline = self._TOOL_INLINE_RE.search(description)
                tool_tag = inline.group(1).upper() if inline else "LLM"
                description = self._TOOL_INLINE_RE.sub("", description).strip()

            tool_tag = tool_tag if tool_tag in VALID_TOOLS else "LLM"
            steps.append(Step(index=idx, description=description, tool=tool_tag))

        return steps

    def _fallback_plan(self, task: str) -> List[Step]:
        return [
            Step(1, "Analyse any provided image or visual content", "VISION"),
            Step(2, f"Retrieve relevant background knowledge for: {task[:60]}", "MEMORY"),
            Step(3, "Search the web for the most current information", "SEARCH"),
            Step(4, "Synthesize retrieved information and generate final answer", "LLM"),
        ]
