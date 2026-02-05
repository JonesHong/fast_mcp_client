$ErrorActionPreference = "Stop"

$name = $env:FAST_MCP_CONTAINER
if (-not $name) { $name = "fast-mcp-client" }

Write-Host "[docker-stop] stopping $name"
docker stop $name

