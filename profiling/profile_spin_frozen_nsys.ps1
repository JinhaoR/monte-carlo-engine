param(
    [int]$L = 64,
    [int]$Temps = 32,
    [double]$TMin = 0.4,
    [double]$TMax = 1.2,
    [double]$K = 1.0,
    [double]$FieldStep = 0.35,
    [int]$WarmupSweeps = 20,
    [int]$ProfileSweeps = 200,
    [int]$SwapsBetween = 10,
    [int]$ThreadsPerBlock = 128,
    [ValidateSet("sweeps", "sweeps-and-observables")]
    [string]$Mode = "sweeps",
    [string]$Trace = "cuda,nvtx",
    [ValidateSet("none", "process-tree", "system-wide")]
    [string]$Sample = "none",
    [ValidateSet("true", "false")]
    [string]$Stats = "false"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command nsys -ErrorAction SilentlyContinue)) {
    throw "nsys was not found on PATH. Install NVIDIA Nsight Systems or open a shell where nsys is available."
}

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$OutDir = Join-Path $PSScriptRoot "results"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$KToken = ($K.ToString([Globalization.CultureInfo]::InvariantCulture)).Replace(".", "p")
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutputBase = Join-Path $OutDir "spin_frozen_nsys_L${L}_K${KToken}_${Stamp}"

Push-Location $Root
try {
    & nsys profile `
        --force-overwrite=true `
        --trace="$Trace" `
        --capture-range=cudaProfilerApi `
        --capture-range-end=stop `
        --sample=$Sample `
        --stats=$Stats `
        --output="$OutputBase" `
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

Write-Host "Nsight Systems report base: $OutputBase"
