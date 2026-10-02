$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $Root
try {
    python exp243_long_closed_loop.py --output EXP243_long_closed_loop.json
    python analyze_exp243_long.py
    python plot_exp243_long.py
    Write-Host "EXP243-long complete: results/EXP243_long_closed_loop.json"
} finally {
    Pop-Location
}
