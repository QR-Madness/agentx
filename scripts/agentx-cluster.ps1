#
# agentx-cluster.ps1 - Multi-cluster launcher for AgentX (Windows)
#
# Usage:
#   .\agentx-cluster.ps1 <cluster-dir> <command> [args...]
#
# Examples:
#   .\agentx-cluster.ps1 C:\clusters\production up -d
#   .\agentx-cluster.ps1 C:\clusters\staging logs -f api
#

param(
    [Parameter(Position=0)]
    [string]$ClusterDir,

    [Parameter(Position=1, ValueFromRemainingArguments=$true)]
    [string[]]$Command
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$SourceDir = Split-Path -Parent $ScriptDir

function Show-Usage {
    Write-Host "agentx-cluster - Multi-cluster launcher for AgentX"
    Write-Host ""
    Write-Host "Usage: .\agentx-cluster.ps1 <cluster-dir> <command> [args...]"
    Write-Host ""
    Write-Host "Commands (passed to docker compose):"
    Write-Host "  up [-d]       Start cluster services"
    Write-Host "  down          Stop cluster services"
    Write-Host "  logs [-f]     View logs"
    Write-Host "  ps            Show status"
    Write-Host "  restart       Restart services"
    Write-Host "  exec          Execute command in container"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\agentx-cluster.ps1 C:\clusters\prod up -d"
    Write-Host "  .\agentx-cluster.ps1 C:\clusters\staging logs -f api"
    Write-Host ""
    exit 1
}

# Show usage if no args
if (-not $ClusterDir -or $Command.Count -eq 0) {
    Show-Usage
}

# Resolve to absolute path
$ClusterDir = Resolve-Path $ClusterDir -ErrorAction SilentlyContinue
if (-not $ClusterDir) {
    Write-Host "Error: Cluster directory not found: $ClusterDir"
    exit 1
}

$EnvFile = Join-Path $ClusterDir ".env.production"
if (-not (Test-Path $EnvFile)) {
    Write-Host "Error: Missing .env.production in $ClusterDir"
    Write-Host ""
    Write-Host "Create it with at least:"
    Write-Host "  AGENTX_CLUSTER_NAME=your-cluster-name"
    Write-Host "  API_PORT=12319"
    Write-Host "  NEO4J_HTTP_PORT=7474"
    Write-Host "  NEO4J_BOLT_PORT=7687"
    Write-Host "  POSTGRES_PORT=5432"
    Write-Host "  REDIS_PORT=6379"
    Write-Host ""
    Write-Host "Or copy from the template:"
    Write-Host "  Copy-Item $SourceDir\.env.production.example $EnvFile"
    exit 1
}

# Set cluster data directory
$DataDir = Join-Path $ClusterDir "data"
$env:AGENTX_DATA_DIR = $DataDir

# Ensure data directories exist
$dirs = @(
    (Join-Path $DataDir "neo4j\data"),
    (Join-Path $DataDir "neo4j\logs"),
    (Join-Path $DataDir "neo4j\plugins"),
    (Join-Path $DataDir "postgres"),
    (Join-Path $DataDir "redis")
)
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

Write-Host "Cluster: $ClusterDir"
Write-Host "Data:    $DataDir"
Write-Host ""

# Run docker compose
$ComposeFile = Join-Path $SourceDir "docker-compose.yml"
Push-Location $SourceDir
try {
    & docker compose -f $ComposeFile --env-file $EnvFile --profile production @Command
} finally {
    Pop-Location
}
