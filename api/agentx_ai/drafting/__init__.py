"""
Drafting Models Framework for AgentX.

This module implements multi-model generation strategies:
- Speculative Decoding: Fast draft model + accurate target model
- Model Pipelines: Multi-stage processing with different models
- Candidate Generation: N-best selection with scoring/ranking
"""

from .base import DraftingStrategy, DraftResult
from .speculative import SpeculativeDecoder
from .pipeline import ModelPipeline, PipelineStage
from .candidate import CandidateGenerator, ScoringMethod

__all__ = [
    "DraftingStrategy",
    "DraftResult",
    "SpeculativeDecoder",
    "ModelPipeline",
    "PipelineStage",
    "CandidateGenerator",
    "ScoringMethod",
]
