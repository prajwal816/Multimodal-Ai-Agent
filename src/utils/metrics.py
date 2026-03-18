"""
src/utils/metrics.py
────────────────────
MetricsTracker — records latency, retrieval stats, and goal completion.
Writes a structured JSON report at the end of each agent run.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RetrievalEvent:
    query: str
    retrieved_k: int
    top_score: float
    latency_ms: float


@dataclass
class StepEvent:
    step_index: int
    tool_name: str
    input_summary: str
    output_summary: str
    latency_ms: float
    success: bool


@dataclass
class AgentRunMetrics:
    task: str
    image_path: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    # Aggregate counters
    total_steps: int = 0
    successful_steps: int = 0
    failed_steps: int = 0
    goal_completed: bool = False

    # Sub-event lists
    retrieval_events: List[RetrievalEvent] = field(default_factory=list)
    step_events: List[StepEvent] = field(default_factory=list)

    # Derived (filled on finalise())
    total_latency_ms: float = 0.0
    avg_retrieval_latency_ms: float = 0.0
    avg_retrieval_score: float = 0.0
    goal_completion_rate: float = 0.0

    def record_retrieval(
        self,
        query: str,
        retrieved_k: int,
        top_score: float,
        latency_ms: float,
    ) -> None:
        self.retrieval_events.append(
            RetrievalEvent(query, retrieved_k, top_score, latency_ms)
        )

    def record_step(
        self,
        step_index: int,
        tool_name: str,
        input_summary: str,
        output_summary: str,
        latency_ms: float,
        success: bool,
    ) -> None:
        self.step_events.append(
            StepEvent(step_index, tool_name, input_summary, output_summary, latency_ms, success)
        )
        self.total_steps += 1
        if success:
            self.successful_steps += 1
        else:
            self.failed_steps += 1

    def finalise(self, goal_completed: bool = True) -> "AgentRunMetrics":
        self.end_time = time.time()
        self.goal_completed = goal_completed
        self.total_latency_ms = (self.end_time - self.start_time) * 1000

        if self.retrieval_events:
            self.avg_retrieval_latency_ms = sum(
                e.latency_ms for e in self.retrieval_events
            ) / len(self.retrieval_events)
            self.avg_retrieval_score = sum(
                e.top_score for e in self.retrieval_events
            ) / len(self.retrieval_events)

        self.goal_completion_rate = (
            1.0 if goal_completed else self.successful_steps / max(self.total_steps, 1)
        )
        return self

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert dataclass nested lists
        d["retrieval_events"] = [asdict(e) for e in self.retrieval_events]
        d["step_events"] = [asdict(e) for e in self.step_events]
        return d

    def save(self, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def summary_str(self) -> str:
        return (
            f"Task: {self.task}\n"
            f"  Goal completed  : {'✅' if self.goal_completed else '❌'}\n"
            f"  Steps           : {self.successful_steps}/{self.total_steps} succeeded\n"
            f"  Total latency   : {self.total_latency_ms:.1f} ms\n"
            f"  Retrieval events: {len(self.retrieval_events)}\n"
            f"  Avg ret. score  : {self.avg_retrieval_score:.3f}\n"
            f"  Goal completion : {self.goal_completion_rate:.1%}"
        )


class MetricsTracker:
    """Factory / context manager for AgentRunMetrics."""

    def __init__(self, output_path: str = "logs/metrics.json"):
        self.output_path = output_path
        self._current: Optional[AgentRunMetrics] = None

    def start_run(self, task: str, image_path: Optional[str] = None) -> AgentRunMetrics:
        self._current = AgentRunMetrics(task=task, image_path=image_path)
        return self._current

    def finish_run(self, goal_completed: bool = True) -> Optional[AgentRunMetrics]:
        if self._current is None:
            return None
        self._current.finalise(goal_completed)
        self._current.save(self.output_path)
        return self._current

    @property
    def current(self) -> Optional[AgentRunMetrics]:
        return self._current
