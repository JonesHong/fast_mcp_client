$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path
Set-Location $root

$image = $env:FAST_MCP_IMAGE
if (-not $image) { $image = "fast-mcp-client:local" }

$name = $env:FAST_MCP_CONTAINER
if (-not $name) { $name = "fast-mcp-client" }

$port = $env:FAST_MCP_PORT
if (-not $port) { $port = "8081" }

if (-not (Test-Path "$root\\config.yaml")) {
  Write-Host "[docker-run] missing config.yaml; copy config.example.yaml -> config.yaml first" -ForegroundColor Yellow
  exit 1
}

Write-Host "[docker-run] name=$name image=$image port=$port"
Write-Host "[docker-run] mounting config.yaml -> /app/fast_mcp_client/config.yaml"

docker run --rm -it `
  --name $name `
  -p "$port`:8081" `
  -e "FAST_MCP_CONFIG=/app/fast_mcp_client/config.yaml" `
  -e "OPENAI_API_KEY=$env:OPENAI_API_KEY" `
  -v "$root\\config.yaml:/app/fast_mcp_client/config.yaml:ro" `
  $image

