$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

python .\exp244_complexity_scaling.py --seeds 12 --budgets 600 1200 2400
python .\plot_exp244.py

Write-Host "EXP244 completed. Results are under .\results\."
