"""Run budget: a token and/or USD cap with a per-run reserve (moved from beacon_eval.runner)."""

from __future__ import annotations

from dataclasses import dataclass

#: Tokens set aside for each run; a run that passes it is killed and stops the matrix.
DEFAULT_RUN_RESERVE = 400_000


class BudgetExhausted(RuntimeError):
    """The next run's reserve would not fit under the cap, or a run passed its reserve."""


@dataclass
class Budget:
    cap: int | None
    reserve: int | None = DEFAULT_RUN_RESERVE
    used: int = 0
    cap_usd: float | None = None
    reserve_usd: float = 0.0
    used_usd: float = 0.0
    #: Runs admitted but not finished; each still holds its reserve (parallel matrix).
    in_flight: int = 0

    def check(self) -> None:
        held = self.in_flight + 1
        flying = f", {self.in_flight} run(s) in flight" if self.in_flight else ""
        if self.cap is not None and self.used + (self.reserve or 0) * held > self.cap:
            raise BudgetExhausted(
                f"the next run's {self.reserve:,}-token reserve would exceed the "
                f"{self.cap:,}-token cap ({self.used:,} used{flying})"
            )
        if self.cap_usd is not None and self.used_usd + self.reserve_usd * held > self.cap_usd:
            raise BudgetExhausted(
                f"the next run's ${self.reserve_usd:.2f} reserve would exceed the "
                f"${self.cap_usd:.2f} cap (${self.used_usd:.4f} spent{flying})"
            )

    def spend(self, tokens: int, usd: float = 0.0) -> None:
        self.used += tokens
        self.used_usd += usd
