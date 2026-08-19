param(
    [int]$WorkersPerKey = 4
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

& python api_pilot/survey_pilot_runner.py `
    --profiles dataset/derived/gate_c_profiles_6x12.jsonl `
    --prompt-bundle api_pilot/prompts/survey-confirmatory-v3.json `
    --max-profiles-per-country 12 `
    --target-batch-size 5 `
    --providers gemini `
    --gemini-model gemini-3.6-flash `
    --workers-per-key $WorkersPerKey `
    --max-attempts 3 `
    --quota-stop-threshold 3 `
    --output api_pilot/restricted_outputs/gate_c_v3_gemini-3.6-flash.jsonl `
    --checkpoint api_pilot/restricted_outputs/gate_c_v3_gemini-3.6-flash.sqlite `
    --progress-doc api_pilot/restricted_outputs/gate_c_v3_gemini-3.6-flash-progress.md `
    --progress-every 12 `
    --run-label "Gate C Gemini replacement: gemini-3.6-flash" `
    --resume-command "powershell -ExecutionPolicy Bypass -File api_pilot/scripts/archive/gate_c/run_gate_c_gemini_replacement.ps1" `
    --acknowledge-respondent-data-processing
exit $LASTEXITCODE
