param(
    [string]$HostAddress = "0.0.0.0",
    [int]$Port = 18765,
    [int]$Seed = 0,
    [ValidateSet("sparse_success", "factory")]
    [string]$RewardMode = "sparse_success",
    [ValidateSet("none", "spacemouse")]
    [string]$InterventionDevice = "none",
    [double]$SpaceMouseDeadzone = 0.05,
    [double]$SpaceMouseTranslationScale = 1.0,
    [double]$SpaceMouseRotationScale = 1.0
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$isaacLabRoot = Join-Path $repoRoot "submodules\IsaacLab"
$pythonExe = "C:\Isaac\envs\arena-py311\python.exe"
$serverScript = Join-Path $repoRoot "tools\run_windows_factory_rpc_server.py"
$experience = Join-Path $isaacLabRoot "apps\isaaclab.python.headless.kit"

if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    throw "Isaac/Arena Python not found: $pythonExe"
}
if (-not (Test-Path -LiteralPath $experience -PathType Leaf)) {
    throw "Isaac Lab experience not found: $experience"
}

$sourcePaths = Get-ChildItem -LiteralPath (Join-Path $isaacLabRoot "source") -Directory |
    ForEach-Object { $_.FullName }
$env:PYTHONPATH = (($sourcePaths + $repoRoot) -join ";")
$env:OMNI_KIT_ACCEPT_EULA = "Y"
$env:PYTHONUNBUFFERED = "1"

Write-Host "Factory RPC initialization can take many minutes on the first run."
Push-Location $repoRoot
try {
    & $pythonExe -u $serverScript `
        --task Isaac-Factory-PegInsert-Direct-v0 `
        --host $HostAddress `
        --port $Port `
        --seed $Seed `
        --reward_mode $RewardMode `
        --intervention_device $InterventionDevice `
        --spacemouse_deadzone $SpaceMouseDeadzone `
        --spacemouse_translation_scale $SpaceMouseTranslationScale `
        --spacemouse_rotation_scale $SpaceMouseRotationScale `
        --headless `
        --device cuda:0 `
        --experience $experience
    if ($LASTEXITCODE -ne 0) {
        throw "Factory RPC server exited with code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
