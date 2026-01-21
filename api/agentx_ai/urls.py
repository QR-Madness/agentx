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
]