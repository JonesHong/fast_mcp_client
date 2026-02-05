$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path
Set-Location $root

Write-Host "[dev] cwd=$root"
Write-Host "[dev] starting gateway (reload=true)"

python -m uvicorn fast_mcp_client.agent_gateway:app --host 0.0.0.0 --port 8081 --reload

