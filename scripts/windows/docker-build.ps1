$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path
Set-Location $root

$image = $env:FAST_MCP_IMAGE
if (-not $image) { $image = "fast-mcp-client:local" }

Write-Host "[docker-build] image=$image"
docker build -t $image .

