$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path
Set-Location $root

Write-Host "[run] cwd=$root"
Write-Host "[run] starting gateway (reload=false)"

python -m uvicorn fast_mcp_client.agent_gateway:app --host 0.0.0.0 --port 8081

