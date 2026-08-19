$ErrorActionPreference = 'Stop'
$workspace = Resolve-Path (Join-Path $PSScriptRoot '..\..\..\..')
Set-Location -LiteralPath $workspace

$arguments = @(
    'api_pilot/monitor_experiment.py',
    '--checkpoint', 'api_pilot/restricted_outputs/gate_c_v3_five_systems.sqlite',
    '--html', 'api_pilot/EXPERIMENT_MONITOR.html',
    '--interval-seconds', '600',
    '--overlay-checkpoint', 'api_pilot/restricted_outputs/gate_c_v3_mistral-recovery.sqlite',
    '--replacement-checkpoint', 'api_pilot/restricted_outputs/gate_c_v3_gemini-3.6-flash.sqlite',
    '--overlay-checkpoint', 'api_pilot/restricted_outputs/gate_c_v3_openai-recovery.sqlite',
    '--overlay-checkpoint', 'api_pilot/restricted_outputs/gate_c_v3_mistral-normalized-repair.sqlite',
    '--postprocess-status', 'api_pilot/outputs/gate_c_v3_postprocess_status.json',
    '--pid-file', 'api_pilot/restricted_outputs/gate_c_monitor.pid'
)

& python @arguments
exit $LASTEXITCODE
