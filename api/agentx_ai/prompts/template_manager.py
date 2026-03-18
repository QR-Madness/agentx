"""
Prompt Template Manager for AgentX.

Manages prompt template CRUD operations and YAML persistence.
Replaces the PromptProfile/PromptSection system with a unified template model.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from .models import PromptTemplate, TemplateType
from .template_defaults import DEFAULT_TEMPLATES

logger = logging.getLogger(__name__)


class PromptTemplateManager:
    """
    Manages prompt templates with YAML persistence.

    Handles:
    - Loading/saving templates from YAML configuration
    - Template management (CRUD operations)
    - Tag-based filtering and search
    - Rollback to default content
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the PromptTemplateManager.

        Args:
            config_path: Path to prompt_templates.yaml (defaults to data/prompt_templates.yaml)
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent.parent / "data" / "prompt_templates.yaml"

        self.config_path = config_path
        self._templates: dict[str, PromptTemplate] = {}

        # Load from file or initialize defaults
        if config_path.exists():
            self._load_config(config_path)
        else:
            self._init_defaults()

    def _init_defaults(self) -> None:
        """Initialize with default templates."""
        for template in DEFAULT_TEMPLATES:
            self._templates[template.id] = template

        # Save defaults to disk
        self.save_config()
        logger.info(f"Initialized prompt templates with {len(self._templates)} defaults")

    def _load_config(self, path: Path) -> None:
        """Load templates from YAML file."""
        try:
            with open(path, "r") as f:
                config = yaml.safe_load(f)

            if not config or "templates" not in config:
                self._init_defaults()
                return

            for template_data in config["templates"]:
                # Handle datetime fields
                if "created_at" in template_data and isinstance(template_data["created_at"], str):
                    template_data["created_at"] = datetime.fromisoformat(template_data["created_at"])
                if "updated_at" in template_data and isinstance(template_data["updated_at"], str):
                    template_data["updated_at"] = datetime.fromisoformat(template_data["updated_at"])

                # Handle enum conversion
                if "type" in template_data and isinstance(template_data["type"], str):
                    template_data["type"] = TemplateType(template_data["type"])

                template = PromptTemplate(**template_data)
                self._templates[template.id] = template

            # Ensure all builtin templates exist (may have been added in new versions)
            for default_template in DEFAULT_TEMPLATES:
                if default_template.id not in self._templates:
                    self._templates[default_template.id] = default_template
                    logger.info(f"Added missing builtin template: {default_template.id}")

            logger.info(f"Loaded {len(self._templates)} prompt templates from {path}")

        except Exception as e:
            logger.error(f"Failed to load prompt templates: {e}")
            self._init_defaults()

    def save_config(self, path: Optional[Path] = None) -> None:
        """Save current templates to YAML file."""
        save_path = path or self.config_path
        if not save_path:
            raise ValueError("No config path specified")

        # Ensure parent directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        config = {
            "templates": [
                {
                    "id": t.id,
                    "name": t.name,
                    "content": t.content,
                    "default_content": t.default_content,
                    "tags": t.tags,
                    "placeholders": t.placeholders,
                    "type": t.type if isinstance(t.type, str) else t.type.value,
                    "is_builtin": t.is_builtin,
                    "description": t.description,
                    "created_at": t.created_at.isoformat() if t.created_at else None,
                    "updated_at": t.updated_at.isoformat() if t.updated_at else None,
                }
                for t in self._templates.values()
            ]
        }

        # Filter out None values from each template dict
        config["templates"] = [
            {k: v for k, v in t.items() if v is not None}
            for t in config["templates"]
        ]

        with open(save_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        logger.info(f"Saved prompt templates to {save_path}")

    # =========================================================================
    # Query Operations
    # =========================================================================

    def list_templates(
        self,
        type_filter: Optional[TemplateType] = None,
        tag_filter: Optional[str] = None,
        search: Optional[str] = None,
    ) -> list[PromptTemplate]:
        """
        List templates with optional filtering.

        Args:
            type_filter: Filter by template type
            tag_filter: Filter by tag (templates containing this tag)
            search: Search text in name, description, or content

        Returns:
            List of matching templates
        """
        templates = list(self._templates.values())

        if type_filter:
            type_value = type_filter if isinstance(type_filter, str) else type_filter.value
            templates = [t for t in templates if t.type == type_value]

        if tag_filter:
            templates = [t for t in templates if tag_filter in t.tags]

        if search:
            search_lower = search.lower()
            templates = [
                t for t in templates
                if search_lower in t.name.lower()
                or (t.description and search_lower in t.description.lower())
                or search_lower in t.content.lower()
            ]

        # Sort: builtin first, then by name
        templates.sort(key=lambda t: (not t.is_builtin, t.name.lower()))

        return templates

    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """Get a template by ID."""
        return self._templates.get(template_id)

    def list_tags(self) -> list[str]:
        """Get all unique tags across all templates."""
        tags = set()
        for template in self._templates.values():
            tags.update(template.tags)
        return sorted(tags)

    def get_tag_counts(self) -> dict[str, int]:
        """Get tag counts for all templates."""
        counts: dict[str, int] = {}
        for template in self._templates.values():
            for tag in template.tags:
                counts[tag] = counts.get(tag, 0) + 1
        return dict(sorted(counts.items()))

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    def create_template(self, template: PromptTemplate) -> PromptTemplate:
        """
        Create a new template.

        Args:
            template: The template to create

        Returns:
            The created template
        """
        if template.id in self._templates:
            raise ValueError(f"Template with ID '{template.id}' already exists")

        template.created_at = datetime.utcnow()
        template.updated_at = datetime.utcnow()
        self._templates[template.id] = template
        self.save_config()
        return template

    def update_template(self, template_id: str, updates: dict) -> Optional[PromptTemplate]:
        """
        Update an existing template.

        Args:
            template_id: ID of the template to update
            updates: Dictionary of fields to update

        Returns:
            The updated template, or None if not found
        """
        if template_id not in self._templates:
            return None

        current = self._templates[template_id]

        # Build updated template data
        updated_data = {
            "id": current.id,
            "name": current.name,
            "content": current.content,
            "default_content": current.default_content,
            "tags": current.tags,
            "placeholders": current.placeholders,
            "type": current.type,
            "is_builtin": current.is_builtin,
            "description": current.description,
            "created_at": current.created_at,
        }

        # Apply updates (except protected fields)
        protected_fields = {"id", "default_content", "is_builtin", "created_at"}
        for key, value in updates.items():
            if key not in protected_fields:
                updated_data[key] = value

        updated_data["updated_at"] = datetime.utcnow()

        self._templates[template_id] = PromptTemplate(**updated_data)
        self.save_config()
        return self._templates[template_id]

    def delete_template(self, template_id: str) -> bool:
        """
        Delete a template by ID.

        Args:
            template_id: ID of the template to delete

        Returns:
            True if deleted, False if not found

        Raises:
            ValueError: If attempting to delete a builtin template
        """
        if template_id not in self._templates:
            return False

        template = self._templates[template_id]
        if template.is_builtin:
            raise ValueError(f"Cannot delete builtin template '{template_id}'")

        del self._templates[template_id]
        self.save_config()
        return True

    def reset_to_default(self, template_id: str) -> Optional[PromptTemplate]:
        """
        Reset a template's content to its default_content.

        Args:
            template_id: ID of the template to reset

        Returns:
            The reset template, or None if not found
        """
        if template_id not in self._templates:
            return None

        template = self._templates[template_id]
        template.reset_to_default()
        self.save_config()
        return template

    # =========================================================================
    # Rendering
    # =========================================================================

    def render_template(self, template_id: str, **variables: str) -> Optional[str]:
        """
        Render a template with variable substitution.

        Args:
            template_id: ID of the template to render
            **variables: Variable values to substitute

        Returns:
            Rendered content, or None if template not found
        """
        template = self.get_template(template_id)
        if not template:
            return None
        return template.render(**variables)


# =============================================================================
# Singleton instance
# =============================================================================

_template_manager: Optional[PromptTemplateManager] = None


def get_template_manager() -> PromptTemplateManager:
    """Get or create the global PromptTemplateManager instance."""
    global _template_manager
    if _template_manager is None:
        _template_manager = PromptTemplateManager()
        logger.info("PromptTemplateManager initialized")
    return _template_manager
