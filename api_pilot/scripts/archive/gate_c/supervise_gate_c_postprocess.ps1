$ErrorActionPreference = 'Continue'
$workspace = Resolve-Path (Join-Path $PSScriptRoot '..\..\..\..')
Set-Location -LiteralPath $workspace

$intervalSeconds = 600
$status = 'api_pilot/outputs/gate_c_v3_postprocess_status.json'
$log = 'api_pilot/restricted_outputs/gate_c_postprocess-supervisor.log'
$arguments = @(
    'api_pilot/finalize_gate_c.py',
    '--main-checkpoint', 'api_pilot/restricted_outputs/gate_c_v3_five_systems.sqlite',
    '--overlay-checkpoint', 'api_pilot/restricted_outputs/gate_c_v3_mistral-recovery.sqlite',
    '--overlay-checkpoint', 'api_pilot/restricted_outputs/gate_c_v3_gemini-3.6-flash.sqlite',
    '--overlay-checkpoint', 'api_pilot/restricted_outputs/gate_c_v3_openai-recovery.sqlite',
    '--overlay-checkpoint', 'api_pilot/restricted_outputs/gate_c_v3_mistral-normalized-repair.sqlite',
    '--model-override', 'gemini=gemini-3.6-flash',
    '--profiles', 'dataset/derived/gate_c_profiles_6x12.jsonl',
    '--prompt', 'api_pilot/prompts/survey-confirmatory-v3.json',
    '--manifest', 'dataset/analysis/gate_c_manifest_v1.json',
    '--combined-results', 'api_pilot/restricted_outputs/gate_c_v3_complete.jsonl',
    '--analysis-summary', 'api_pilot/outputs/gate_c_v3_analysis_summary.json',
    '--gate-report', 'api_pilot/outputs/gate_c_v3_gate_report.json',
    '--report-markdown', 'api_pilot/reports/archive/GATE_C_EVALUATION.md',
    '--status', $status
)

Add-Content -LiteralPath $log -Value "$(Get-Date -Format o) supervisor started" -Encoding UTF8
while ($true) {
    if (Test-Path -LiteralPath $status) {
        try {
            $state = (Get-Content -Raw -LiteralPath $status | ConvertFrom-Json).state
            if ($state -eq 'complete') {
                Add-Content -LiteralPath $log -Value "$(Get-Date -Format o) already complete; exiting" -Encoding UTF8
                exit 0
            }
        } catch {
            Add-Content -LiteralPath $log -Value "$(Get-Date -Format o) status parse failed: $_" -Encoding UTF8
        }
    }
    & python @arguments *>> $log
    $code = $LASTEXITCODE
    if ($code -eq 0) {
        Add-Content -LiteralPath $log -Value "$(Get-Date -Format o) postprocess complete" -Encoding UTF8
        exit 0
    }
    if ($code -ne 10) {
        Add-Content -LiteralPath $log -Value "$(Get-Date -Format o) postprocess error exit_code=$code" -Encoding UTF8
    }
    Start-Sleep -Seconds $intervalSeconds
}
