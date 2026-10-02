$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

python exp245_probe_lifecycle.py --seeds 12 --candidate-budget 1800 --output EXP245_probe_lifecycle.json
python plot_exp245.py
