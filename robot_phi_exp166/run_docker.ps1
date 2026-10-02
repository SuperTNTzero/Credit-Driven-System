param(
    [string]$Seeds = "0,1,2,3,4",
    [int]$DamageEpisodes = 8
)

$ErrorActionPreference = "Stop"
$ExperimentRoot = (Resolve-Path $PSScriptRoot).Path
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LicensePath = (Resolve-Path (Join-Path $ProjectRoot "continualworld_phi\mjkey.txt")).Path

docker image inspect credit-continualworld:latest | Out-Null
docker run --rm `
    -v "${ExperimentRoot}:/workspace/project/robot_phi_exp166" `
    -v "${LicensePath}:/opt/.mujoco/mjkey.txt:ro" `
    credit-continualworld:latest `
    python3 robot_phi_exp166/run_exp166.py `
    --seeds $Seeds `
    --damage-episodes $DamageEpisodes

if ($LASTEXITCODE -ne 0) {
    throw "EXP166 Docker run failed with exit code $LASTEXITCODE"
}
