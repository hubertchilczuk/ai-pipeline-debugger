"""Evaluation metrics: accuracy, latency, cost."""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean


# Approximate $/1K tokens — update when models change. Source of truth: vendor pricing pages.
PRICING_USD_PER_1K: dict[str, tuple[float, float]] = {
    # model: (input, output)
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.005, 0.015),
}


@dataclass(slots=True)
class EvalSample:
    case_id: str
    expected_error_type: str
    predicted_error_type: str
    latency_ms: int
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int


@dataclass(slots=True)
class EvalReport:
    samples: list[EvalSample] = field(default_factory=list)

    def accuracy(self) -> float:
        if not self.samples:
            return 0.0
        correct = sum(
            1
            for s in self.samples
            if s.predicted_error_type.lower() == s.expected_error_type.lower()
        )
        return correct / len(self.samples)

    def avg_latency_ms(self) -> float:
        return mean(s.latency_ms for s in self.samples) if self.samples else 0.0

    def total_cost_usd(self) -> float:
        total = 0.0
        for s in self.samples:
            if pricing := PRICING_USD_PER_1K.get(s.model):
                in_cost, out_cost = pricing
                total += (s.prompt_tokens / 1000) * in_cost
                total += (s.completion_tokens / 1000) * out_cost
        return round(total, 4)
