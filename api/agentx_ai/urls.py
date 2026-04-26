from django.urls import path

from . import views
from .auth import views as auth_views

urlpatterns = [
    path("index", views.index, name="index"),
    path("health", views.health, name="health"),
    path("version", views.version, name="version"),
    # Authentication endpoints (Phase 17)
    path("auth/status", auth_views.auth_status, name="auth-status"),
    path("auth/login", auth_views.auth_login, name="auth-login"),
    path("auth/logout", auth_views.auth_logout, name="auth-logout"),
    path("auth/session", auth_views.auth_session, name="auth-session"),
    path("auth/change-password", auth_views.auth_change_password, name="auth-change-password"),
    path("auth/setup", auth_views.auth_setup, name="auth-setup"),
    # Tool endpoints
    path("tools/language-detect-20", views.language_detect, name="language-detect"),
    path("tools/translate", views.translate, name="translate"),
    # MCP endpoints
    path("mcp/servers", views.mcp_servers, name="mcp-servers"),
    path("mcp/tools", views.mcp_tools, name="mcp-tools"),
    path("mcp/resources", views.mcp_resources, name="mcp-resources"),
    path("mcp/connect", views.mcp_connect, name="mcp-connect"),
    path("mcp/disconnect", views.mcp_disconnect, name="mcp-disconnect"),
    # Provider endpoints
    path("providers", views.providers_list, name="providers-list"),
    path("providers/models", views.providers_models, name="providers-models"),
    path("providers/health", views.providers_health, name="providers-health"),
    # Agent endpoints
    path("agent/run", views.agent_run, name="agent-run"),
    path("agent/chat", views.agent_chat, name="agent-chat"),
    path("agent/chat/stream", views.agent_chat_stream, name="agent-chat-stream"),
    path("agent/status", views.agent_status, name="agent-status"),
    # Tool output storage endpoints
    path("tool-outputs", views.tool_outputs_list, name="tool-outputs-list"),
    path("tool-outputs/<str:storage_key>", views.tool_outputs_detail, name="tool-outputs-detail"),
    # Agent profile endpoints
    path("agent/profiles", views.agent_profiles_list, name="agent-profiles-list"),
    path("agent/profiles/<str:profile_id>", views.agent_profile_detail, name="agent-profile-detail"),
    path("agent/profiles/<str:profile_id>/set-default", views.agent_profile_set_default, name="agent-profile-set-default"),
    # Prompt management endpoints
    path("prompts/profiles", views.prompts_profiles, name="prompts-profiles"),
    path("prompts/profiles/<str:profile_id>", views.prompts_profile_detail, name="prompts-profile-detail"),
    path("prompts/global", views.prompts_global, name="prompts-global"),
    path("prompts/global/update", views.prompts_global_update, name="prompts-global-update"),
    path("prompts/sections", views.prompts_sections, name="prompts-sections"),
    path("prompts/compose", views.prompts_compose, name="prompts-compose"),
    path("prompts/mcp-tools", views.prompts_mcp_tools, name="prompts-mcp-tools"),
    # Prompt template endpoints
    path("prompts/templates", views.prompts_templates_list, name="prompts-templates-list"),
    path("prompts/templates/tags", views.prompts_templates_tags, name="prompts-templates-tags"),
    path("prompts/templates/<str:template_id>", views.prompts_template_detail, name="prompts-template-detail"),
    path("prompts/templates/<str:template_id>/reset", views.prompts_template_reset, name="prompts-template-reset"),
    path("prompts/enhance", views.prompts_enhance, name="prompts-enhance"),
    # Conversation history endpoints
    path("conversations", views.conversations_list, name="conversations-list"),
    path("conversations/<str:conversation_id>/messages", views.conversations_messages, name="conversations-messages"),
    # Memory channel management endpoints
    path("memory/channels", views.memory_channels, name="memory-channels"),
    path("memory/channels/<str:name>", views.memory_channel_delete, name="memory-channel-delete"),
    path("memory/conversations/<str:conversation_id>", views.memory_conversation_delete, name="memory-conversation-delete"),
    # Memory explorer endpoints
    path("memory/entities", views.memory_entities, name="memory-entities"),
    path("memory/entities/<str:entity_id>", views.memory_entity_detail, name="memory-entity-detail"),
    path("memory/entities/<str:entity_id>/graph", views.memory_entity_graph, name="memory-entity-graph"),
    path("memory/facts", views.memory_facts, name="memory-facts"),
    path("memory/facts/<str:fact_id>", views.memory_fact_detail, name="memory-fact-detail"),
    path("memory/strategies", views.memory_strategies, name="memory-strategies"),
    path("memory/stats", views.memory_stats, name="memory-stats"),
    path("memory/consolidate", views.memory_consolidate, name="memory-consolidate"),
    path("memory/consolidate/stream", views.consolidate_stream, name="consolidate-stream"),
    path("memory/reset", views.memory_reset, name="memory-reset"),
    path("memory/settings", views.memory_settings, name="memory-settings"),
    path("memory/recall-settings", views.recall_settings, name="recall-settings"),
    # Job monitoring endpoints
    path("jobs", views.jobs_list, name="jobs-list"),
    path("jobs/clear-stuck", views.jobs_clear_stuck, name="jobs-clear-stuck"),
    path("jobs/<str:job_name>", views.job_detail, name="job-detail"),
    path("jobs/<str:job_name>/run", views.job_run, name="job-run"),
    path("jobs/<str:job_name>/toggle", views.job_toggle, name="job-toggle"),
    # Config management endpoints
    path("config", views.config_get, name="config-get"),
    path("config/update", views.config_update, name="config-update"),
    path("config/context-limits", views.context_limits, name="context-limits"),
]