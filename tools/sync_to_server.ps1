$ErrorActionPreference = "Stop"

$Server = "ustc"
$Remote = "/data/gauss/ldh/projects/norm-route"

$Items = @(
    "AGENTS.md",
    "README.md",
    "pyproject.toml",
    "requirements.lock.txt",
    "configs",
    "docs",
    "src",
    "scripts",
    "tests"
)

foreach ($item in $Items) {
    if (Test-Path $item) {
        Write-Host "Syncing $item ..."
        scp -r $item "${Server}:${Remote}/"
    }
}

Write-Host "Sync finished."