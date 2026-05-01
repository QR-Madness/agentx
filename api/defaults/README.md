# api/defaults/

Canonical seed files for a fresh AgentX deployment. These are baked into the
Docker image and copied into a cluster's `config/` directory by
`task cluster:new` (and on demand by `task cluster:seed CLUSTER=<name>`).

| File | Purpose |
|------|---------|
| `agent_profiles.yaml` | Default agent profiles (AgentX/Muse/Sage) shipped with new installs. |
| `prompt_templates.yaml` | Canonical prompt templates (global assistant, etc.). |
| `workflows.yaml` | Empty Alloy workflow list — clusters add their own. |
| `memory_settings.json` | Default recall layer + memory tuning settings. |

`config.json` is intentionally not shipped: when missing, `ConfigManager` synthesizes the full default schema from `DEFAULT_CONFIG` in `api/agentx_ai/config.py`.

To change a default, edit the file here and rerun `task cluster:seed CLUSTER=<name>` to overwrite the cluster's copy (with `FORCE=1`).
