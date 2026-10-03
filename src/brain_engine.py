"""brain_engine — the single public entry point for the Brain.

    from brain_engine import BrainEngine
    brain = BrainEngine(sdb)

Layering (each module has one responsibility; imports only flow downward):
    brain_helpers             storage-key constants, pure helpers
    brain_report              report rendering (pure functions)
    brain_recommend_baseline  baseline recommendation analytics
    brain                     BrainCore: persistence, verdicts, kill switch
    brain_recommend_full      prescriptive recommendation pipeline
    brain_enhanced            BrainEngineV2: orchestration, plans, apply/rollback
    brain_engine              this facade

`brain.BrainEngine`, `brain.BaseBrainEngine` and `brain_enhanced.BrainEngineV2`
keep working as aliases, but new code should import from here.
"""
from __future__ import annotations

from brain_enhanced import BrainEngineV2 as BrainEngine
from state import RedisStateStore

__all__ = ["BrainEngine", "create_brain_engine"]


def create_brain_engine(sdb: RedisStateStore) -> BrainEngine:
    """Build the one Brain engine. Prefer this over instantiating classes directly."""
    return BrainEngine(sdb)
