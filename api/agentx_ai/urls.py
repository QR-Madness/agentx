from django.urls import path

from . import views

urlpatterns = [
    path("index", views.index, name="index"),
    path("health", views.health, name="health"),
    path("tools/language-detect-20", views.language_detect, name="language-detect"),
    path("tools/translate", views.translate, name="translate"),
    # MCP endpoints
    path("mcp/servers", views.mcp_servers, name="mcp-servers"),
    path("mcp/tools", views.mcp_tools, name="mcp-tools"),
    path("mcp/resources", views.mcp_resources, name="mcp-resources"),
    # Provider endpoints
    path("providers", views.providers_list, name="providers-list"),
    path("providers/models", views.providers_models, name="providers-models"),
    path("providers/health", views.providers_health, name="providers-health"),
    # Agent endpoints
    path("agent/run", views.agent_run, name="agent-run"),
    path("agent/chat", views.agent_chat, name="agent-chat"),
    path("agent/chat/stream", views.agent_chat_stream, name="agent-chat-stream"),
    path("agent/status", views.agent_status, name="agent-status"),
    # Prompt management endpoints
    path("prompts/profiles", views.prompts_profiles, name="prompts-profiles"),
    path("prompts/profiles/<str:profile_id>", views.prompts_profile_detail, name="prompts-profile-detail"),
    path("prompts/global", views.prompts_global, name="prompts-global"),
    path("prompts/global/update", views.prompts_global_update, name="prompts-global-update"),
    path("prompts/sections", views.prompts_sections, name="prompts-sections"),
    path("prompts/compose", views.prompts_compose, name="prompts-compose"),
    path("prompts/mcp-tools", views.prompts_mcp_tools, name="prompts-mcp-tools"),
]