param(
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$workspace = Resolve-Path (Join-Path $PSScriptRoot '..\..\..\..')
Set-Location -LiteralPath $workspace

$proxyPort = 7897
$proxy = "http://127.0.0.1:$proxyPort"
$listener = Get-NetTCPConnection -State Listen -LocalPort $proxyPort -ErrorAction SilentlyContinue
if (-not $listener) {
    throw "The local Vortex proxy is not listening on port $proxyPort. Start it before inference."
}

# The relay, Mistral, and Gemini require the local proxy on this workstation.
# DeepSeek and DashScope remain direct because their direct paths passed smoke tests.
$env:HTTPS_PROXY = $proxy
$env:HTTP_PROXY = $proxy
$env:NO_PROXY = 'api.deepseek.com,dashscope.aliyuncs.com,127.0.0.1,localhost'

$runnerArgs = @(
    'api_pilot/survey_pilot_runner.py',
    '--profiles', 'dataset/derived/gate_c_profiles_6x12.jsonl',
    '--prompt-bundle', 'api_pilot/prompts/survey-confirmatory-v3.json',
    '--max-profiles-per-country', '12',
    '--target-batch-size', '5',
    '--providers', 'openai', 'deepseek', 'mistral', 'gemini', 'qwen',
    '--openai-model', 'gpt-5.4',
    '--workers-per-key', '4',
    '--max-attempts', '3',
    '--quota-stop-threshold', '3',
    '--output', 'api_pilot/restricted_outputs/gate_c_v3_five_systems.jsonl',
    '--checkpoint', 'api_pilot/restricted_outputs/gate_c_v3_five_systems.sqlite',
    '--progress-doc', 'api_pilot/progress/EXPERIMENT_PROGRESS.md',
    '--progress-every', '12',
    '--run-label', 'Gate C core pilot: four official providers plus RayinAI GPT relay',
    '--resume-command', 'powershell -ExecutionPolicy Bypass -File api_pilot/scripts/archive/gate_c/run_gate_c.ps1',
    '--allow-nonofficial-openai-endpoint',
    '--acknowledge-respondent-data-processing'
)

if ($DryRun) {
    $runnerArgs += '--dry-run'
}

& python @runnerArgs
exit $LASTEXITCODE
