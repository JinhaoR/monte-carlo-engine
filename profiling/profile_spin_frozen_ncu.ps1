param(
    [int]$L = 64,
    [int]$Temps = 8,
    [double]$TMin = 0.4,
    [double]$TMax = 1.2,
    [double]$K = 1.0,
    [double]$FieldStep = 0.35,
    [int]$WarmupSweeps = 5,
    [int]$ProfileSweeps = 4,
    [int]$SwapsBetween = 1000000,
    [int]$ThreadsPerBlock = 128,
    [ValidateSet("sweeps", "sweeps-and-observables")]
    [string]$Mode = "sweeps",
    [string]$MetricSet = "default"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command ncu -ErrorAction SilentlyContinue)) {
    throw "ncu was not found on PATH. Install NVIDIA Nsight Compute or open a shell where ncu is available."
}

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$OutDir = Join-Path $PSScriptRoot "results"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$KToken = ($K.ToString([Globalization.CultureInfo]::InvariantCulture)).Replace(".", "p")
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutputBase = Join-Path $OutDir "spin_frozen_ncu_L${L}_K${KToken}_${Stamp}"

Push-Location $Root
try {
    & ncu `
        --force-overwrite `
        --profile-from-start off `
        --target-processes all `
        --set $MetricSet `
        --export "$OutputBase" `
        python profiling\profile_spin_frozen.py `
            --L $L `
            --n-temps $Temps `
            --T-min $TMin `
            --T-max $TMax `
            --K $K `
            --field-step $FieldStep `
            --warmup-sweeps $WarmupSweeps `
            --profile-sweeps $ProfileSweeps `
            --swaps-between $SwapsBetween `
            --threads-per-block $ThreadsPerBlock `
            --mode $Mode `
            --cuda-profile-api
}
finally {
    Pop-Location
}

Write-Host "Nsight Compute report base: $OutputBase"
