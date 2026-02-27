"""
Tree-of-Thought (ToT) reasoning implementation.

ToT explores multiple reasoning paths simultaneously,
evaluating and pruning branches to find the best solution.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional
import uuid

from ..providers.base import Message, MessageRole
from ..providers.registry import get_registry
from ..prompts.loader import get_prompt_loader
from .base import (
    ReasoningConfig,
    ReasoningResult,
    ReasoningStatus,
    ReasoningStrategy,
    ThoughtStep,
    ThoughtType,
)

logger = logging.getLogger(__name__)


@dataclass
class TreeNode:
    """A node in the thought tree."""
    id: str
    content: str
    parent_id: Optional[str] = None
    depth: int = 0
    score: float = 0.0
    is_terminal: bool = False
    is_pruned: bool = False
    children: list["TreeNode"] = field(default_factory=list)


@dataclass
class ToTConfig:
    """Configuration for Tree-of-Thought reasoning."""
    model: str
    
    # Tree structure
    branching_factor: int = 3  # Number of branches per node
    max_depth: int = 4  # Maximum tree depth
    
    # Search strategy
    search_method: str = "bfs"  # "bfs", "dfs", "beam"
    beam_width: int = 3  # For beam search
    
    # Evaluation
    evaluator_model: Optional[str] = None  # Separate model for evaluation
    evaluation_prompt: Optional[str] = None
    pruning_threshold: float = 0.3  # Prune branches below this score
    
    # Generation
    thought_prompt: str = "Generate {n} different approaches to continue solving this problem."


class TreeOfThought(ReasoningStrategy):
    """
    Tree-of-Thought reasoning strategy.
    
    Explores multiple reasoning paths by:
    1. Generating multiple candidate thoughts at each step
    2. Evaluating each thought's promise
    3. Expanding promising branches, pruning poor ones
    4. Selecting the best complete path
    
    Example usage:
        tot = TreeOfThought(ToTConfig(
            model="gpt-4-turbo",
            branching_factor=3,
            max_depth=4,
            search_method="beam",
        ))
        result = await tot.reason("Design a marketing strategy for a new product")
    """
    
    def __init__(self, config: ToTConfig):
        self.tot_config = config
        super().__init__(ReasoningConfig(
            name=f"tot-{config.search_method}",
            strategy_type="tot",
            model=config.model,
            extra={"tot_config": config},
        ))
        self._registry = None
    
    @property
    def name(self) -> str:
        return self.config.name
    
    @property
    def strategy_type(self) -> str:
        return "tot"
    
    @property
    def registry(self):
        """Lazy-load the provider registry."""
        if self._registry is None:
            self._registry = get_registry()
        return self._registry
    
    def reason(
        self,
        task: str,
        context: Optional[list[Message]] = None,
        **kwargs: Any,
    ) -> ReasoningResult:
        """
        Apply Tree-of-Thought reasoning to a task.
        """
        start_time = time.time()

        logger.info(f"ToT reasoning ({self.tot_config.search_method}): {task[:50]}...")

        # Initialize the tree with root node
        root = TreeNode(
            id="root",
            content=task,
            depth=0,
        )

        # Explore the tree
        all_nodes = [root]
        steps = []
        total_tokens = 0

        if self.tot_config.search_method == "bfs":
            all_nodes, steps, total_tokens = self._bfs_search(root, task, context)
        elif self.tot_config.search_method == "dfs":
            all_nodes, steps, total_tokens = self._dfs_search(root, task, context)
        else:  # beam search
            all_nodes, steps, total_tokens = self._beam_search(root, task, context)
        
        # Find the best terminal node
        terminal_nodes = [n for n in all_nodes if n.is_terminal and not n.is_pruned]
        
        if not terminal_nodes:
            # Fall back to highest-scoring node
            terminal_nodes = [n for n in all_nodes if not n.is_pruned]
        
        if terminal_nodes:
            best_node = max(terminal_nodes, key=lambda n: n.score)
            best_path = self._get_path_to_node(best_node, all_nodes)
            answer = self._synthesize_answer(best_path, task)
            best_branch = best_node.id
        else:
            answer = "Unable to find a solution through tree exploration."
            best_branch = None
        
        total_time = (time.time() - start_time) * 1000
        branches_explored = len([n for n in all_nodes if n.id != "root"])
        
        return ReasoningResult(
            answer=answer,
            strategy=self.name,
            status=ReasoningStatus.COMPLETE,
            steps=steps,
            total_steps=len(steps),
            branches_explored=branches_explored,
            best_branch=best_branch,
            total_tokens=total_tokens,
            total_time_ms=total_time,
            models_used=[self.tot_config.model],
            raw_trace=[{"tree": self._serialize_tree(root)}],
        )
    
    def _bfs_search(
        self,
        root: TreeNode,
        task: str,
        context: Optional[list[Message]],
    ) -> tuple[list[TreeNode], list[ThoughtStep], int]:
        """Breadth-first search through the thought tree."""
        all_nodes = [root]
        steps = []
        total_tokens = 0
        step_num = 0

        current_level = [root]

        for depth in range(self.tot_config.max_depth):
            if not current_level:
                break

            next_level = []

            for node in current_level:
                if node.is_pruned:
                    continue

                # Generate children
                children, tokens = self._expand_node(node, task, context)
                total_tokens += tokens

                # Evaluate and add children
                for child in children:
                    child.score, eval_tokens = self._evaluate_node(child, task, context)
                    total_tokens += eval_tokens

                    step_num += 1
                    steps.append(ThoughtStep(
                        step_number=step_num,
                        thought_type=ThoughtType.HYPOTHESIS,
                        content=child.content,
                        confidence=child.score,
                        parent_step=node.depth,
                        branch_id=child.id,
                        model=self.tot_config.model,
                    ))

                    if child.score < self.tot_config.pruning_threshold:
                        child.is_pruned = True
                    else:
                        next_level.append(child)

                    all_nodes.append(child)
                    node.children.append(child)

            current_level = next_level

        # Mark leaf nodes as terminal
        for node in all_nodes:
            if not node.children and not node.is_pruned:
                node.is_terminal = True

        return all_nodes, steps, total_tokens
    
    def _dfs_search(
        self,
        root: TreeNode,
        task: str,
        context: Optional[list[Message]],
    ) -> tuple[list[TreeNode], list[ThoughtStep], int]:
        """Depth-first search through the thought tree."""
        all_nodes = [root]
        steps = []
        total_tokens = 0
        step_num = [0]  # Use list for closure

        def explore(node: TreeNode, depth: int):
            nonlocal total_tokens
            if depth >= self.tot_config.max_depth or node.is_pruned:
                node.is_terminal = True
                return

            children, tokens = self._expand_node(node, task, context)
            total_tokens += tokens

            for child in children:
                child.score, eval_tokens = self._evaluate_node(child, task, context)
                total_tokens += eval_tokens

                step_num[0] += 1
                steps.append(ThoughtStep(
                    step_number=step_num[0],
                    thought_type=ThoughtType.HYPOTHESIS,
                    content=child.content,
                    confidence=child.score,
                    parent_step=node.depth,
                    branch_id=child.id,
                    model=self.tot_config.model,
                ))

                all_nodes.append(child)
                node.children.append(child)

                if child.score < self.tot_config.pruning_threshold:
                    child.is_pruned = True
                else:
                    explore(child, depth + 1)

        explore(root, 0)
        return all_nodes, steps, total_tokens
    
    def _beam_search(
        self,
        root: TreeNode,
        task: str,
        context: Optional[list[Message]],
    ) -> tuple[list[TreeNode], list[ThoughtStep], int]:
        """Beam search through the thought tree."""
        all_nodes = [root]
        steps = []
        total_tokens = 0
        step_num = 0

        beam = [root]

        for depth in range(self.tot_config.max_depth):
            if not beam:
                break

            candidates = []

            for node in beam:
                children, tokens = self._expand_node(node, task, context)
                total_tokens += tokens

                for child in children:
                    child.score, eval_tokens = self._evaluate_node(child, task, context)
                    total_tokens += eval_tokens

                    step_num += 1
                    steps.append(ThoughtStep(
                        step_number=step_num,
                        thought_type=ThoughtType.HYPOTHESIS,
                        content=child.content,
                        confidence=child.score,
                        parent_step=node.depth,
                        branch_id=child.id,
                        model=self.tot_config.model,
                    ))

                    all_nodes.append(child)
                    node.children.append(child)
                    candidates.append(child)

            # Keep top beam_width candidates
            candidates.sort(key=lambda n: n.score, reverse=True)
            beam = candidates[:self.tot_config.beam_width]

            # Prune the rest
            for c in candidates[self.tot_config.beam_width:]:
                c.is_pruned = True

        # Mark beam nodes as terminal
        for node in beam:
            node.is_terminal = True

        return all_nodes, steps, total_tokens
    
    def _expand_node(
        self,
        node: TreeNode,
        task: str,
        context: Optional[list[Message]],
    ) -> tuple[list[TreeNode], int]:
        """Generate child nodes by exploring different thought paths."""
        provider, model_id = self.registry.get_provider_for_model(
            self.tot_config.model
        )
        loader = get_prompt_loader()

        # Build prompt for generating thoughts
        path = self._get_path_content(node)
        prompt = self.tot_config.thought_prompt.format(n=self.tot_config.branching_factor)

        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content=loader.get("reasoning.tree_of_thought.expand"),
            ),
            Message(
                role=MessageRole.USER,
                content=(
                    f"Problem: {task}\n\n"
                    f"Current reasoning path:\n{path}\n\n"
                    f"{prompt}\n\n"
                    f"Format: List each approach on a new line, starting with a number."
                )
            ),
        ]

        if context:
            messages = messages[:1] + context + messages[1:]

        try:
            result = provider.complete(
                messages,
                model_id,
                temperature=0.9,  # Higher temperature for diversity
                max_tokens=1000,
            )
        except Exception as e:
            logger.error(f"Node expansion failed: {e}")
            return [], 0
        
        tokens = result.usage.get("total_tokens", 0) if result.usage else 0
        
        # Parse the generated thoughts
        children = []
        lines = result.content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering
                content = line.lstrip('0123456789.-) ').strip()
                if content:
                    child = TreeNode(
                        id=str(uuid.uuid4())[:8],
                        content=content,
                        parent_id=node.id,
                        depth=node.depth + 1,
                    )
                    children.append(child)
        
        # Limit to branching factor
        return children[:self.tot_config.branching_factor], tokens
    
    def _evaluate_node(
        self,
        node: TreeNode,
        task: str,
        context: Optional[list[Message]],
    ) -> tuple[float, int]:
        """Evaluate a node's promise for solving the task."""
        eval_model = self.tot_config.evaluator_model or self.tot_config.model
        provider, model_id = self.registry.get_provider_for_model(eval_model)
        loader = get_prompt_loader()

        prompt = self.tot_config.evaluation_prompt or loader.get(
            "reasoning.tree_of_thought.evaluate"
        )

        messages = [
            Message(role=MessageRole.SYSTEM, content=prompt),
            Message(
                role=MessageRole.USER,
                content=f"Problem: {task}\n\nApproach: {node.content}"
            ),
        ]

        try:
            result = provider.complete(
                messages,
                model_id,
                temperature=0.0,
                max_tokens=10,
            )

            tokens = result.usage.get("total_tokens", 0) if result.usage else 0

            # Parse score
            try:
                score = float(result.content.strip().split()[0])
                score = min(max(score, 0.0), 1.0)
            except (ValueError, IndexError):
                score = 0.5

            return score, tokens
        except Exception as e:
            logger.error(f"Node evaluation failed: {e}")
            return 0.5, 0
    
    def _get_path_to_node(
        self,
        node: TreeNode,
        all_nodes: list[TreeNode],
    ) -> list[TreeNode]:
        """Get the path from root to a node."""
        path = []
        current = node
        
        while current:
            path.append(current)
            if current.parent_id:
                current = next(
                    (n for n in all_nodes if n.id == current.parent_id),
                    None
                )
            else:
                break
        
        return list(reversed(path))
    
    def _get_path_content(self, node: TreeNode) -> str:
        """Get the content of the path to a node as a string."""
        if node.id == "root":
            return "(Starting point)"
        return f"Step {node.depth}: {node.content}"
    
    def _synthesize_answer(self, path: list[TreeNode], task: str) -> str:
        """Synthesize a final answer from the best path."""
        if len(path) <= 1:
            return "No solution path found."
        
        steps = []
        for i, node in enumerate(path[1:], 1):  # Skip root
            steps.append(f"Step {i}: {node.content}")
        
        return (
            "Solution approach:\n\n" +
            "\n".join(steps) +
            f"\n\nThis approach was selected with confidence score: {path[-1].score:.2f}"
        )
    
    def _serialize_tree(self, root: TreeNode) -> dict:
        """Serialize the tree for storage/visualization."""
        return {
            "id": root.id,
            "content": root.content[:100],
            "depth": root.depth,
            "score": root.score,
            "is_terminal": root.is_terminal,
            "is_pruned": root.is_pruned,
            "children": [self._serialize_tree(c) for c in root.children],
        }
    
    def get_description(self) -> str:
        return (
            f"Tree-of-Thought ({self.tot_config.search_method}) "
            f"with branching={self.tot_config.branching_factor}, "
            f"depth={self.tot_config.max_depth}"
        )
