"""
Model Pipeline implementation.

Model pipelines route content through multiple stages, each handled
by a potentially different model. This enables specialized processing
like: analyze → draft → review → refine.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from ..providers.base import Message, MessageRole, CompletionResult
from ..providers.registry import get_registry
from .base import DraftingConfig, DraftingStrategy, DraftResult, DraftStatus

logger = logging.getLogger(__name__)


class StageRole(str, Enum):
    """Predefined roles for pipeline stages."""
    ANALYZE = "analyze"
    DRAFT = "draft"
    REVIEW = "review"
    REFINE = "refine"
    SUMMARIZE = "summarize"
    CRITIQUE = "critique"
    CODE = "code"
    TEST = "test"
    CUSTOM = "custom"


@dataclass
class PipelineStage:
    """Configuration for a single pipeline stage."""
    name: str
    model: str
    role: StageRole = StageRole.CUSTOM
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    
    # Transform functions (optional)
    # These allow preprocessing/postprocessing at each stage
    preprocess: Optional[Callable[[str], str]] = None
    postprocess: Optional[Callable[[str], str]] = None
    
    # Whether to include previous stage output in context
    include_previous: bool = True
    
    # Custom instructions added to the prompt
    instructions: Optional[str] = None


# Default system prompts for common roles
DEFAULT_PROMPTS = {
    StageRole.ANALYZE: "Analyze the following input carefully. Identify key points, potential issues, and areas that need attention.",
    StageRole.DRAFT: "Create an initial draft based on the requirements provided. Focus on completeness over perfection.",
    StageRole.REVIEW: "Review the following content critically. Identify errors, inconsistencies, and areas for improvement.",
    StageRole.REFINE: "Refine and improve the following content based on the feedback provided. Make it more polished and professional.",
    StageRole.SUMMARIZE: "Summarize the following content concisely while preserving key information.",
    StageRole.CRITIQUE: "Provide constructive criticism of the following. Be specific about what works and what doesn't.",
    StageRole.CODE: "Generate code based on the requirements. Follow best practices and include comments.",
    StageRole.TEST: "Generate test cases for the following code or specification.",
}


@dataclass
class PipelineConfig:
    """Configuration for a model pipeline."""
    name: str
    stages: list[PipelineStage]
    pass_full_context: bool = True  # Pass all previous outputs or just the last
    stop_on_failure: bool = True  # Stop pipeline if a stage fails


class ModelPipeline(DraftingStrategy):
    """
    Multi-stage model pipeline.
    
    Routes content through multiple stages, each potentially handled
    by a different model. Useful for complex workflows like:
    - Code review: generate → review → incorporate feedback
    - Writing: outline → draft → edit → polish
    - Analysis: decompose → analyze parts → synthesize
    
    Example usage:
        pipeline = ModelPipeline(PipelineConfig(
            name="code-review",
            stages=[
                PipelineStage(name="generate", model="gpt-4-turbo", role=StageRole.CODE),
                PipelineStage(name="review", model="claude-3-opus", role=StageRole.REVIEW),
                PipelineStage(name="refine", model="gpt-4-turbo", role=StageRole.REFINE),
            ],
        ))
        result = await pipeline.generate(messages)
    """
    
    def __init__(self, config: PipelineConfig):
        self.pipeline_config = config
        super().__init__(DraftingConfig(
            name=config.name,
            strategy_type="pipeline",
            extra={"pipeline_config": config},
        ))
        self._registry = None
    
    @property
    def name(self) -> str:
        return self.pipeline_config.name
    
    @property
    def strategy_type(self) -> str:
        return "pipeline"
    
    @property
    def registry(self):
        """Lazy-load the provider registry."""
        if self._registry is None:
            self._registry = get_registry()
        return self._registry
    
    async def generate(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> DraftResult:
        """
        Generate content by routing through all pipeline stages.
        """
        start_time = time.time()
        
        models_used = []
        stage_outputs: list[dict[str, Any]] = []
        total_input_tokens = 0
        total_output_tokens = 0
        current_content = ""
        
        # Extract initial user content
        for msg in reversed(messages):
            if msg.role == MessageRole.USER:
                current_content = msg.content
                break
        
        for i, stage in enumerate(self.pipeline_config.stages):
            stage_start = time.time()
            logger.info(f"Pipeline stage {i + 1}/{len(self.pipeline_config.stages)}: {stage.name}")
            
            try:
                provider, model_id = self.registry.get_provider_for_model(stage.model)
            except ValueError as e:
                logger.error(f"Stage {stage.name} failed: {e}")
                if self.pipeline_config.stop_on_failure:
                    return DraftResult(
                        content=current_content,
                        strategy=self.name,
                        status=DraftStatus.FAILED,
                        models_used=models_used,
                        stages_completed=i,
                        raw_results=stage_outputs,
                    )
                continue
            
            models_used.append(stage.model)
            
            # Build stage messages
            stage_messages = self._build_stage_messages(
                stage, messages, stage_outputs, current_content
            )
            
            # Preprocess if defined
            if stage.preprocess and current_content:
                current_content = stage.preprocess(current_content)
            
            # Execute stage
            try:
                result = await provider.complete(
                    stage_messages,
                    model_id,
                    temperature=stage.temperature,
                    max_tokens=stage.max_tokens or kwargs.get("max_tokens"),
                )
            except Exception as e:
                logger.error(f"Stage {stage.name} execution failed: {e}")
                if self.pipeline_config.stop_on_failure:
                    return DraftResult(
                        content=current_content,
                        strategy=self.name,
                        status=DraftStatus.FAILED,
                        models_used=models_used,
                        stages_completed=i,
                        raw_results=stage_outputs,
                    )
                continue
            
            stage_time = (time.time() - stage_start) * 1000
            
            # Track tokens
            if result.usage:
                total_input_tokens += result.usage.get("prompt_tokens", 0)
                total_output_tokens += result.usage.get("completion_tokens", 0)
            
            # Postprocess if defined
            stage_output = result.content
            if stage.postprocess:
                stage_output = stage.postprocess(stage_output)
            
            current_content = stage_output
            
            stage_outputs.append({
                "stage": stage.name,
                "model": stage.model,
                "role": stage.role.value,
                "output": stage_output,
                "time_ms": stage_time,
            })
            
            logger.debug(f"Stage {stage.name} completed in {stage_time:.0f}ms")
        
        total_time = (time.time() - start_time) * 1000
        
        # Calculate estimated cost
        estimated_cost = 0.0
        for stage in self.pipeline_config.stages:
            try:
                provider, model_id = self.registry.get_provider_for_model(stage.model)
                caps = provider.get_capabilities(model_id)
                if caps.cost_per_1k_input and caps.cost_per_1k_output:
                    # Rough estimate: divide tokens evenly across stages
                    stage_input = total_input_tokens / len(self.pipeline_config.stages)
                    stage_output = total_output_tokens / len(self.pipeline_config.stages)
                    estimated_cost += (stage_input / 1000) * caps.cost_per_1k_input
                    estimated_cost += (stage_output / 1000) * caps.cost_per_1k_output
            except ValueError:
                pass
        
        return DraftResult(
            content=current_content,
            strategy=self.name,
            status=DraftStatus.COMPLETE,
            total_time_ms=total_time,
            models_used=models_used,
            stages_completed=len(self.pipeline_config.stages),
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            estimated_cost=estimated_cost,
            raw_results=stage_outputs,
        )
    
    def _build_stage_messages(
        self,
        stage: PipelineStage,
        original_messages: list[Message],
        previous_outputs: list[dict[str, Any]],
        current_content: str,
    ) -> list[Message]:
        """Build the messages for a pipeline stage."""
        messages = []
        
        # System prompt
        system_prompt = stage.system_prompt
        if not system_prompt and stage.role in DEFAULT_PROMPTS:
            system_prompt = DEFAULT_PROMPTS[stage.role]
        
        if system_prompt:
            messages.append(Message(role=MessageRole.SYSTEM, content=system_prompt))
        
        # Include context based on configuration
        if self.pipeline_config.pass_full_context and previous_outputs:
            # Include all previous stage outputs
            context_parts = []
            for output in previous_outputs:
                context_parts.append(
                    f"[{output['stage']} ({output['role']})]:\n{output['output']}"
                )
            
            if context_parts:
                context = "\n\n---\n\n".join(context_parts)
                messages.append(Message(
                    role=MessageRole.USER,
                    content=f"Previous stages:\n\n{context}\n\n---\n\nOriginal request:\n{current_content}",
                ))
        elif stage.include_previous and previous_outputs:
            # Include only the last stage output
            last_output = previous_outputs[-1]
            messages.append(Message(
                role=MessageRole.USER,
                content=f"Previous output from {last_output['stage']}:\n\n{last_output['output']}\n\n---\n\nPlease continue with your task.",
            ))
        else:
            # Just the current content
            messages.append(Message(role=MessageRole.USER, content=current_content))
        
        # Add any custom instructions
        if stage.instructions:
            messages.append(Message(
                role=MessageRole.USER,
                content=f"Additional instructions: {stage.instructions}",
            ))
        
        return messages
    
    async def validate(self) -> bool:
        """Validate that all stage models are accessible."""
        for stage in self.pipeline_config.stages:
            try:
                self.registry.get_provider_for_model(stage.model)
            except ValueError as e:
                logger.error(f"Stage {stage.name} validation failed: {e}")
                return False
        return True
    
    def get_description(self) -> str:
        stage_desc = " → ".join(
            f"{s.name}({s.model})" for s in self.pipeline_config.stages
        )
        return f"Pipeline: {stage_desc}"
