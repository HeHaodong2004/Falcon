"""Cooperative, atomic inference deadlines; these are not hard real-time timers."""
import math
import time
from contextvars import ContextVar
from dataclasses import dataclass


class InferenceBudgetExceeded(TimeoutError):
    def __init__(self, stage, elapsed_ms, budget_ms, pending_observation=False):
        self.stage = stage
        self.elapsed_ms = float(elapsed_ms)
        self.budget_ms = float(budget_ms)
        self.pending_observation = bool(pending_observation)
        self.status = 'PENDING_OBSERVATION' if pending_observation else 'TIMEOUT'
        super().__init__('{} at {} ({:.3f}/{:.3f} ms); no partial probability result'
                         .format(self.status, stage, elapsed_ms, budget_ms))


class UnsupportedConditioning(ValueError):
    """An operation would require an unimplemented change of probability measure."""
    status = 'UNSUPPORTED_CONTINUOUS_REFINEMENT'

    def __init__(self,message,status=None):
        if status is not None:self.status=status
        super().__init__(message)


@dataclass
class Deadline:
    budget_ms: float
    start: float
    pending_observation: bool = False

    @classmethod
    def create(cls, budget_ms, pending_observation=False):
        budget_ms = float(budget_ms)
        if not math.isfinite(budget_ms) or budget_ms < 0:
            raise ValueError('budget_ms must be finite and nonnegative')
        return cls(budget_ms, time.perf_counter(), pending_observation)

    def check(self, stage):
        elapsed = (time.perf_counter() - self.start) * 1000.
        if elapsed >= self.budget_ms:
            raise InferenceBudgetExceeded(stage, elapsed, self.budget_ms, self.pending_observation)


ACTIVE_DEADLINE = ContextVar('l2_structural_deadline', default=None)


def check_budget(stage):
    deadline = ACTIVE_DEADLINE.get()
    if deadline is not None:
        deadline.check(stage)
