$ErrorActionPreference = "Stop"

$base = $env:FAST_MCP_BASEURL
if (-not $base) { $base = "http://localhost:8081" }
$base = $base.TrimEnd("/")

Write-Host "[smoke] base=$base"

curl.exe -s "$base/agent/health"
Write-Host ""
curl.exe -s "$base/agent/tools"
Write-Host ""

