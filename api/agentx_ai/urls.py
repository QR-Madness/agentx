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
    path("tools/search-health", views.search_health, name="search-health"),
    # MCP endpoints
    path("mcp/servers", views.mcp_servers, name="mcp-servers"),
    path("mcp/servers/validate", views.mcp_server_validate, name="mcp-server-validate"),
    path("mcp/servers/<str:name>", views.mcp_server_detail, name="mcp-server-detail"),
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
    path("agent/chat/stream/attach", views.agent_chat_attach, name="agent-chat-attach"),
    path("agent/chat/runs", views.agent_chat_runs, name="agent-chat-runs"),
    path("agent/chat/runs/<str:run_id>/cancel", views.agent_chat_run_cancel, name="agent-chat-run-cancel"),
    path("agent/chat/runs/<str:run_id>/steer", views.agent_chat_run_steer, name="agent-chat-run-steer"),
    # Ambassador (16.6) — specific routes BEFORE the <conversation_id> catch-all.
    path("agent/ambassador/brief-turn", views.ambassador_brief_turn, name="ambassador-brief-turn"),
    path("agent/ambassador/ask", views.ambassador_ask, name="ambassador-ask"),
    path("agent/ambassador/draft", views.ambassador_draft, name="ambassador-draft"),
    path("agent/ambassador/stream", views.ambassador_stream, name="ambassador-stream"),
    path("agent/ambassador/<str:conversation_id>", views.ambassador_briefings, name="ambassador-briefings"),
    path("chat/background", views.chat_background, name="chat-background"),
    path("chat/background/<str:job_id>", views.chat_background_detail, name="chat-background-detail"),
    path("agent/status", views.agent_status, name="agent-status"),
    path("agent/plans/cancel", views.agent_plan_cancel, name="agent-plan-cancel"),
    path("agent/plans/<str:plan_id>/status", views.agent_plan_status, name="agent-plan-status"),
    path("agent/plans/<str:plan_id>/resume", views.agent_plan_resume, name="agent-plan-resume"),
    # Tool output storage endpoints
    path("tool-outputs", views.tool_outputs_list, name="tool-outputs-list"),
    path("tool-outputs/<str:storage_key>", views.tool_outputs_detail, name="tool-outputs-detail"),
    # Agent profile endpoints
    path("agent/profiles", views.agent_profiles_list, name="agent-profiles-list"),
    path("agent/profiles/<str:profile_id>", views.agent_profile_detail, name="agent-profile-detail"),
    path("agent/profiles/<str:profile_id>/set-default", views.agent_profile_set_default, name="agent-profile-set-default"),
    # Agent Alloy (multi-agent workflow) endpoints
    path("alloy/workflows", views.alloy_workflows_list, name="alloy-workflows-list"),
    path("alloy/workflows/<str:workflow_id>", views.alloy_workflow_detail, name="alloy-workflow-detail"),
    # Prompt management endpoints
    path("prompts/profiles", views.prompts_profiles, name="prompts-profiles"),
    path("prompts/profiles/<str:profile_id>", views.prompts_profile_detail, name="prompts-profile-detail"),
    path("prompts/global", views.prompts_global, name="prompts-global"),
    path("prompts/global/update", views.prompts_global_update, name="prompts-global-update"),
    path("prompts/sections", views.prompts_sections, name="prompts-sections"),
    path("prompts/compose", views.prompts_compose, name="prompts-compose"),
    path("prompts/mcp-tools", views.prompts_mcp_tools, name="prompts-mcp-tools"),
    # Prompt layer (stack) endpoints — `reorder` before the <layer_id> catch-all
    path("prompts/layers", views.prompts_layers, name="prompts-layers"),
    path("prompts/layers/reorder", views.prompts_layers_reorder, name="prompts-layers-reorder"),
    path("prompts/layers/<str:layer_id>", views.prompts_layer_detail, name="prompts-layer-detail"),
    path("prompts/layers/<str:layer_id>/reset", views.prompts_layer_reset, name="prompts-layer-reset"),
    path("prompts/layers/<str:layer_id>/acknowledge", views.prompts_layer_acknowledge, name="prompts-layer-acknowledge"),
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
    path("memory/facts/<str:fact_id>/remember", views.memory_fact_remember, name="memory-fact-remember"),
    path("memory/facts/<str:fact_id>/forget", views.memory_fact_forget, name="memory-fact-forget"),
    path("memory/facts/<str:fact_id>/entities", views.memory_fact_entities, name="memory-fact-entities"),
    path("memory/facts/<str:fact_id>/provenance", views.memory_fact_provenance, name="memory-fact-provenance"),
    path("memory/strategies", views.memory_strategies, name="memory-strategies"),
    path("memory/procedures", views.memory_procedures, name="memory-procedures"),
    path("memory/stats", views.memory_stats, name="memory-stats"),
    path("metrics/usage", views.usage_metrics, name="usage-metrics"),
    path("memory/checkpoints", views.memory_checkpoints, name="memory-checkpoints"),
    path("memory/user-history", views.memory_user_history, name="memory-user-history"),
    path("memory/consolidate", views.memory_consolidate, name="memory-consolidate"),
    path("memory/consolidate/stream", views.consolidate_stream, name="consolidate-stream"),
    path("memory/reset", views.memory_reset, name="memory-reset"),
    path("memory/export", views.memory_export, name="memory-export"),
    path("memory/import", views.memory_import, name="memory-import"),
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