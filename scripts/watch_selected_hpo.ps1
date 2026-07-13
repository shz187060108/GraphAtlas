param(
    [ValidateRange(1, 1440)]
    [int]$IntervalMinutes = 30
)

$Root = Split-Path -Parent $PSScriptRoot
$StatusDirectory = Join-Path $Root "outputs\.work\status"
$LogPath = Join-Path $StatusDirectory "selected_hpo_monitor.log"
New-Item -ItemType Directory -Force -Path $StatusDirectory | Out-Null

while ($true) {
    $now = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $processes = Get-Process -Name python, pythonw -ErrorAction SilentlyContinue |
        Where-Object { $_.Path -eq "C:\Users\24953\.conda\envs\dl\python.exe" } |
        ForEach-Object { "pid=$($_.Id), cpu=$([math]::Round($_.CPU, 1)), ram_gb=$([math]::Round($_.WorkingSet64 / 1GB, 2))" }
    $latestHistory = Get-ChildItem (Join-Path $Root "outputs\selected_optuna") -Recurse -Filter history.csv -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    $historyStatus = if ($latestHistory) {
        $last = Get-Content $latestHistory.FullName -Tail 1
        "history=$($latestHistory.FullName.Substring($Root.Length + 1)); last=$last"
    } else {
        "history=none"
    }
    $processStatus = if ($processes) { $processes -join "; " } else { "no dl Python process" }
    Add-Content -LiteralPath $LogPath -Value "[$now] $processStatus | $historyStatus"
    Start-Sleep -Seconds ($IntervalMinutes * 60)
}
