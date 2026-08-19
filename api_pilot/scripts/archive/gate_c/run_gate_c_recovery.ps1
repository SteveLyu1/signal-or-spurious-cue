param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('openai', 'mistral', 'gemini')]
    [string]$Provider
)

$ErrorActionPreference = 'Stop'
$workspace = Resolve-Path (Join-Path $PSScriptRoot '..\..\..\..')
Set-Location -LiteralPath $workspace

$proxyPort = 7897
$proxy = "http://127.0.0.1:$proxyPort"
if (-not (Get-NetTCPConnection -State Listen -LocalPort $proxyPort -ErrorAction SilentlyContinue)) {
    throw "The local Vortex proxy is not listening on port $proxyPort."
}
$env:HTTPS_PROXY = $proxy
$env:HTTP_PROXY = $proxy
if ($Provider -eq 'openai') {
}

$mainCheckpoint = 'api_pilot/restricted_outputs/gate_c_v3_five_systems.sqlite'
$output = "api_pilot/restricted_outputs/gate_c_v3_$Provider-recovery.jsonl"
$checkpoint = "api_pilot/restricted_outputs/gate_c_v3_$Provider-recovery.sqlite"
$progress = "api_pilot/restricted_outputs/gate_c_v3_$Provider-recovery-progress.md"

if (-not (Test-Path -LiteralPath $checkpoint)) {
    & python api_pilot/prepare_provider_recovery.py `
        --checkpoint $mainCheckpoint --provider $Provider --output $output
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$maxAttempts = if ($Provider -eq 'gemini') { '1' } else { '3' }
$quotaThreshold = if ($Provider -eq 'gemini') { '1' } else { '3' }
$resume = "powershell -ExecutionPolicy Bypass -File api_pilot/scripts/archive/gate_c/run_gate_c_recovery.ps1 -Provider $Provider"

$providerArguments = @('--providers', $Provider)
if ($Provider -eq 'openai') {
    $providerArguments += @(
        '--openai-model', 'gpt-5.4',
        '--allow-nonofficial-openai-endpoint'
    )
}

& python api_pilot/survey_pilot_runner.py `
    --profiles dataset/derived/gate_c_profiles_6x12.jsonl `
    --prompt-bundle api_pilot/prompts/survey-confirmatory-v3.json `
    --max-profiles-per-country 12 `
    --target-batch-size 5 `
    @providerArguments `
    --workers-per-key 1 `
    --max-attempts $maxAttempts `
    --quota-stop-threshold $quotaThreshold `
    --output $output `
    --checkpoint $checkpoint `
    --progress-doc $progress `
    --progress-every 12 `
    --run-label "Gate C low-concurrency recovery: $Provider" `
    --resume-command $resume `
    --acknowledge-respondent-data-processing
exit $LASTEXITCODE
