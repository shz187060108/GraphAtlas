param(
    [string[]]$SearchRoot = @("C:\Users\24953\PycharmProjects"),
    [string]$OutputRoot = "outputs\manifests\local_data",
    [ValidateSet("screening", "confirmatory", "all")]
    [string]$Stage = "screening",
    [int]$Limit = 0,
    [switch]$SkipInstall,
    [switch]$SkipDownload,
    [switch]$SkipImport,
    [switch]$WithOptionalGraphPackages,
    [switch]$ContinueOnError,
    [switch]$IncludeTextHeterophily,
    [switch]$IncludeH2GB,
    [switch]$IncludeLargeNonHomophily,
    [switch]$IncludeFrontierText
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

function Invoke-Step([string]$Name, [scriptblock]$Action) {
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    & $Action
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
}

if (-not $SkipInstall) {
    Invoke-Step "Install GraphAtlas" { python -m pip install -e . }
    if ($WithOptionalGraphPackages) {
        Invoke-Step "Install optional graph packages" { python -m pip install -e ".[graph,ogb]" }
    }
}

$discoverArgs = @("scripts/discover_local_data.py", "--output", "$OutputRoot/local_dataset_candidates.json")
foreach ($root in $SearchRoot) { $discoverArgs += @("--search-root", $root) }
Invoke-Step "Discover local datasets" { python @discoverArgs }

if (-not $SkipImport) {
    Invoke-Step "Import local datasets" {
        python scripts/import_local_data.py --manifest "$OutputRoot/local_dataset_candidates.json" --destination data --prefer first
    }
}

if (-not $SkipDownload) {
    Invoke-Step "Download missing core datasets" { python scripts/download_data.py --main --root data }
}

$datasetArgs = @(
    "roman_empire", "amazon_ratings", "minesweeper", "tolokers", "questions", "actor",
    "chameleon_filtered", "squirrel_filtered", "cornell", "texas", "wisconsin",
    "cora", "citeseer", "pubmed", "wikics", "dblp", "coauthor_cs", "coauthor_physics"
)
$auditArgs = @("scripts/audit_datasets.py", "--root", "data", "--output-dir", "outputs/reports/data_audit")
foreach ($dataset in $datasetArgs) {
    if (Test-Path "data/$dataset/raw/$dataset.npz") { $auditArgs += @("--dataset", $dataset) }
}
Invoke-Step "Audit datasets" { python @auditArgs }

$common = @()
if ($Limit -gt 0) { $common += @("--limit", "$Limit") }
if ($ContinueOnError) { $common += "--continue-on-error" }

if ($Stage -eq "screening" -or $Stage -eq "all") {
    Invoke-Step "Run GraphAtlas screening matrix" { python scripts/run_pipeline.py --preset screening @common }
}
if ($Stage -eq "confirmatory" -or $Stage -eq "all") {
    Invoke-Step "Run GraphAtlas confirmatory matrix" { python scripts/run_pipeline.py --preset confirmatory @common }
}
if ($Stage -eq "all") {
    Invoke-Step "Run Atlas-Het grid" { python scripts/run_pipeline.py --preset atlas_grid @common }
    Invoke-Step "Run link prediction" { python scripts/run_pipeline.py --preset link @common }
}
if ($IncludeTextHeterophily) {
    Invoke-Step "Run available HeTGB datasets" { python scripts/run_optional_preset.py --preset text_heterophily @common }
}
if ($IncludeH2GB) {
    Invoke-Step "Run available H2GB datasets" { python scripts/run_optional_preset.py --preset heterogeneous_heterophily @common }
}
if ($IncludeLargeNonHomophily) {
    Invoke-Step "Run available large non-homophily datasets" { python scripts/run_optional_preset.py --preset large_nonhomophily @common }
}
if ($IncludeFrontierText) {
    Invoke-Step "Run available frontier heterogeneous-text datasets" { python scripts/run_optional_preset.py --preset frontier_heterogeneous_text @common }
}

Invoke-Step "Validate published result registry" { python scripts/validate_published_results.py }

Write-Host "`nBenchmark workflow complete. GraphAtlas is trained locally; external baselines are imported from cited papers." -ForegroundColor Green
