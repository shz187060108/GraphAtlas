$ProjectRoot = Split-Path -Parent $PSScriptRoot
$base = Join-Path $ProjectRoot "outputs\logs\overnight"
$latestRun = Join-Path $base "latest_run.txt"
$latestPid = Join-Path $base "latest_pid.txt"
if (-not (Test-Path $latestRun)) { throw "No overnight run has been started." }
$run = (Get-Content $latestRun -Raw).Trim()
$pidValue = if (Test-Path $latestPid) { [int](Get-Content $latestPid -Raw).Trim() } else { 0 }
$alive = [bool](Get-Process -Id $pidValue -ErrorAction SilentlyContinue)
Write-Host "PID: $pidValue alive=$alive"
Write-Host "Log directory: $run"
$status = Join-Path $run "status.json"
if (Test-Path $status) { Get-Content $status }
$pipeline = Join-Path $run "pipeline.log"
if (Test-Path $pipeline) { Get-Content $pipeline -Wait -Tail 40 }
