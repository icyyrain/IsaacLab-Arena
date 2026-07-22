[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$isaacLabRoot = Join-Path $repoRoot "submodules\IsaacLab"
$patchPath = Join-Path $repoRoot "patches\isaaclab\windows-native.patch"
$expectedCommit = "55df2c34390ba94b22d41879514c5485c5115462"

if (-not (Test-Path -LiteralPath (Join-Path $isaacLabRoot ".git"))) {
    throw "IsaacLab is not initialized. Run: git submodule update --init -- submodules/IsaacLab"
}

$currentCommit = (& git -C $isaacLabRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Unable to read the IsaacLab submodule commit."
}
if ($currentCommit -ne $expectedCommit) {
    throw "Windows-native patch expects IsaacLab $expectedCommit, but found $currentCommit."
}

$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& git -C $isaacLabRoot apply --reverse --check --unidiff-zero $patchPath 2>$null
$reverseCheckExitCode = $LASTEXITCODE
$ErrorActionPreference = $previousErrorActionPreference
if ($reverseCheckExitCode -eq 0) {
    Write-Host "IsaacLab Windows-native patch is already applied."
    exit 0
}

& git -C $isaacLabRoot diff --quiet --
if ($LASTEXITCODE -ne 0) {
    throw "IsaacLab has unrelated tracked changes. Refusing to apply the patch."
}

& git -C $isaacLabRoot diff --cached --quiet --
if ($LASTEXITCODE -ne 0) {
    throw "IsaacLab has staged changes. Refusing to apply the patch."
}

& git -C $isaacLabRoot apply --check --unidiff-zero $patchPath
if ($LASTEXITCODE -ne 0) {
    throw "Windows-native patch does not apply cleanly."
}

& git -C $isaacLabRoot apply --unidiff-zero $patchPath
if ($LASTEXITCODE -ne 0) {
    throw "Failed to apply the Windows-native patch."
}

Write-Host "Applied IsaacLab Windows-native patch to $isaacLabRoot"
