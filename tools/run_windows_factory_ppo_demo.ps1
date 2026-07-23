param(
    [string]$Checkpoint = "",
    [int]$VideoLength = 600,
    [int]$NumEnvs = 1,
    [int]$Seed = 0,
    [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$isaacLabRoot = Join-Path $repoRoot "submodules\IsaacLab"
$pythonExe = "C:\Isaac\envs\arena-py311\python.exe"
$checkpointConverter = Join-Path $repoRoot "tools\convert_rl_games_checkpoint_numpy_compat.py"
$player = Join-Path $repoRoot "tools\play_factory_ppo_with_camera.py"
$videoValidator = Join-Path $repoRoot "tools\validate_factory_ppo_video.py"
$experience = Join-Path $isaacLabRoot "apps\isaaclab.python.headless.rendering.kit"

if ([string]::IsNullOrWhiteSpace($Checkpoint)) {
    $Checkpoint = Join-Path $isaacLabRoot ".pretrained_checkpoints\rl_games\Isaac-Factory-PegInsert-Direct-v0\Assets\Isaac\6.0\Isaac\IsaacLab\PretrainedCheckpoints\rl_games\Isaac-Factory-PegInsert-Direct-v0\checkpoint.pth"
}
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $repoRoot "logs\factory_ppo_demo"
}

if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    throw "Isaac/Arena Python not found: $pythonExe"
}
if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
    throw "Factory PPO checkpoint not found: $Checkpoint"
}
if (-not (Test-Path -LiteralPath $experience -PathType Leaf)) {
    throw "Isaac Lab experience not found: $experience"
}

$sourcePaths = Get-ChildItem -LiteralPath (Join-Path $isaacLabRoot "source") -Directory |
    ForEach-Object { $_.FullName }
$env:PYTHONPATH = (($sourcePaths + $repoRoot) -join ";")
$env:OMNI_KIT_ACCEPT_EULA = "Y"
$env:PYTHONUNBUFFERED = "1"

Write-Host "Factory PPO demo can spend many minutes initializing assets and PhysX."
Write-Host "Checkpoint: $Checkpoint"

$checkpointDirectory = Split-Path -Parent $Checkpoint
$checkpointStem = [System.IO.Path]::GetFileNameWithoutExtension($Checkpoint)
$compatibleCheckpoint = Join-Path $checkpointDirectory "$checkpointStem-numpy1.pth"
if (
    -not (Test-Path -LiteralPath $compatibleCheckpoint -PathType Leaf) -or
    (Get-Item -LiteralPath $compatibleCheckpoint).LastWriteTime -lt (Get-Item -LiteralPath $Checkpoint).LastWriteTime
) {
    Write-Host "Converting the official checkpoint for NumPy 1.x..."
    & $pythonExe -u $checkpointConverter $Checkpoint $compatibleCheckpoint
    if ($LASTEXITCODE -ne 0) {
        throw "Checkpoint conversion exited with code $LASTEXITCODE"
    }
}
else {
    Write-Host "Using compatible checkpoint: $compatibleCheckpoint"
}

Push-Location $repoRoot
try {
    $checkpointParent = Split-Path -Parent (Split-Path -Parent $compatibleCheckpoint)
    $recordedVideo = Join-Path $checkpointParent "videos\play\rl-video-step-0.mp4"
    if (Test-Path -LiteralPath $recordedVideo -PathType Leaf) {
        Remove-Item -LiteralPath $recordedVideo -Force
    }
    $runStartedAt = Get-Date
    & $pythonExe -u $player `
        --task Isaac-Factory-PegInsert-Direct-v0 `
        --num_envs $NumEnvs `
        --seed $Seed `
        --checkpoint $compatibleCheckpoint `
        --video `
        --video_length $VideoLength `
        --headless `
        --enable_cameras `
        --experience $experience
    if ($LASTEXITCODE -ne 0) {
        throw "Factory PPO player exited with code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

if (-not (Test-Path -LiteralPath $recordedVideo -PathType Leaf)) {
    throw "Factory PPO player finished, but the recorded video was not found: $recordedVideo"
}
if ((Get-Item -LiteralPath $recordedVideo).LastWriteTime -lt $runStartedAt) {
    throw "Factory PPO player did not create a fresh video: $recordedVideo"
}

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$outputVideo = Join-Path $OutputDirectory "factory-ppo-peg-insert.mp4"
Copy-Item -LiteralPath $recordedVideo -Destination $outputVideo -Force
$minimumFrames = [Math]::Max(1, $VideoLength - 2)
$validatorArguments = @($videoValidator, $outputVideo, "--min-frames", $minimumFrames)
if ($VideoLength -ge 30) {
    $validatorArguments += "--require-motion"
}
& $pythonExe -u @validatorArguments
if ($LASTEXITCODE -ne 0) {
    throw "Factory PPO video validation exited with code $LASTEXITCODE"
}
Write-Host "Factory PPO video: $outputVideo"
