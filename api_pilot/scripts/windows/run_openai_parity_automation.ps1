$ErrorActionPreference = 'Continue'
$workspace = Resolve-Path (Join-Path $PSScriptRoot '..\..\..')
Set-Location -LiteralPath $workspace

$python = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
$pidPath = Join-Path $workspace 'api_pilot\restricted_outputs\openai_parity_automation.pid'
$logPath = Join-Path $workspace 'api_pilot\logs\remaining_experiment\openai_parity_supervisor.log'
$phases = @(
    [pscustomobject]@{ Manifest='dataset/analysis/repeat_identical_openai_parity_manifest_v1.json'; StopAfter=0 },
    [pscustomobject]@{ Manifest='dataset/analysis/single_item_openai_parity_manifest_v1.json'; StopAfter=0 },
    [pscustomobject]@{ Manifest='dataset/analysis/wording_b_openai_parity_manifest_v1.json'; StopAfter=0 },
    [pscustomobject]@{ Manifest='dataset/analysis/option_reverse_openai_parity_manifest_v1.json'; StopAfter=0 },
    [pscustomobject]@{ Manifest='dataset/analysis/confirmatory_sentinel_before_v1_openai_parity_manifest_v1.json'; StopAfter=0 },
    [pscustomobject]@{ Manifest='dataset/analysis/confirmatory_core_v2_raw_openai_parity_manifest_v1.json'; StopAfter=4032 },
    [pscustomobject]@{ Manifest='dataset/analysis/confirmatory_sentinel_mid_v1_openai_parity_manifest_v1.json'; StopAfter=0 },
    [pscustomobject]@{ Manifest='dataset/analysis/confirmatory_core_v2_raw_openai_parity_manifest_v1.json'; StopAfter=0 },
    [pscustomobject]@{ Manifest='dataset/analysis/confirmatory_batch_context_replication_v1_openai_parity_manifest_v1.json'; StopAfter=0 },
    [pscustomobject]@{ Manifest='dataset/analysis/confirmatory_sentinel_after_v1_openai_parity_manifest_v1.json'; StopAfter=0 }
)

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $logPath) | Out-Null
if (Test-Path -LiteralPath $pidPath) {
    $existing = [int]((Get-Content -LiteralPath $pidPath -Raw).Trim())
    if ($existing -and (Get-Process -Id $existing -ErrorAction SilentlyContinue)) { exit 0 }
}
Set-Content -LiteralPath $pidPath -Value $PID -Encoding ascii
try {
    while ($true) {
        $allComplete = $true
        foreach ($phase in $phases) {
            $manifestPath = $phase.Manifest
            $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
            $checkpoint = Join-Path $workspace $manifest.immutable_checkpoint_path
            $success = 0
            if (Test-Path -LiteralPath $checkpoint) {
                $success = & $python -c "import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); print(c.execute('select count(*) from results where success=1').fetchone()[0])" $checkpoint
            }
            $target = if ([int]$phase.StopAfter -gt 0) { [int]$phase.StopAfter } else { [int]$manifest.expected.total_calls }
            if ([int]$success -ge $target) { continue }
            $allComplete = $false
            Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) start $($manifest.experiment_id) success=$success/$($manifest.expected.total_calls)"
            $extraArguments = @()
            if ($manifest.source.analysis_manifest_file) {
                $extraArguments += @('--analysis-manifest', $manifest.source.analysis_manifest_file)
            }
            if ($manifest.runner_configuration.reverse_target_options) {
                $extraArguments += '--reverse-target-options'
            }
            if ([int]$phase.StopAfter -gt 0) {
                $extraArguments += @('--stop-after-successes-per-provider', [string]$phase.StopAfter)
            }
            & $python api_pilot/survey_pilot_runner.py `
                --profiles $manifest.source.profile_file `
                --prompt-bundle $manifest.prompt.file `
                --max-profiles-per-country $manifest.source.profiles_per_country `
                --providers openai `
                --openai-model gpt-5.4 `
                --allow-custom-openai-base-url `
                --workers-per-key 4 `
                --max-attempts 1 `
                --quota-stop-threshold 3 `
                --target-batch-size $manifest.runner_configuration.target_batch_size `
                --task-manifest $manifestPath `
                @extraArguments `
                --output $manifest.immutable_output_path `
                --checkpoint $manifest.immutable_checkpoint_path `
                --progress-doc "api_pilot/progress/$($manifest.experiment_id).md" `
                --progress-every 48 `
                --run-label $manifest.experiment_id `
                --resume-command 'Managed by api_pilot/scripts/windows/run_openai_parity_automation.ps1' `
                --acknowledge-respondent-data-processing *>> $logPath
            if ($LASTEXITCODE -ne 0) { break }
        }
        if ($allComplete) { break }
        Start-Sleep -Seconds 60
    }
}
finally {
    Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
}
