$ErrorActionPreference = 'Continue'
$workspace = Resolve-Path (Join-Path $PSScriptRoot '..\..\..')
Set-Location -LiteralPath $workspace

$python = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
$manifestPath = 'dataset/analysis/checklist_correction_supplement_manifest_v1.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$pidPath = Join-Path $workspace 'api_pilot\restricted_outputs\checklist_correction_automation.pid'
$logPath = Join-Path $workspace 'api_pilot\logs\checklist_correction_automation.log'
$statusPath = Join-Path $workspace 'api_pilot\outputs\checklist_correction_automation_status.json'

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $logPath) | Out-Null
if (Test-Path -LiteralPath $pidPath) {
    $existing = [int]((Get-Content -LiteralPath $pidPath -Raw).Trim())
    if ($existing -and (Get-Process -Id $existing -ErrorAction SilentlyContinue)) { exit 0 }
}
Set-Content -LiteralPath $pidPath -Value $PID -Encoding ascii

function Get-SuccessCounts {
    $checkpoint = Join-Path $workspace $manifest.immutable_checkpoint_path
    if (-not (Test-Path -LiteralPath $checkpoint)) { return @{} }
    $raw = & $python -c "import json,sqlite3,sys; c=sqlite3.connect(sys.argv[1]); print(json.dumps(dict(c.execute('select provider,count(*) from results where success=1 group by provider').fetchall())))" $checkpoint
    if (-not $raw) { return @{} }
    $parsed = $raw | ConvertFrom-Json
    $counts = @{}
    foreach ($property in $parsed.psobject.Properties) {
        $counts[$property.Name] = [int]$property.Value
    }
    return $counts
}

function Write-Status([string]$phase, [string]$message) {
    $counts = Get-SuccessCounts
    $done = 0
    foreach ($value in $counts.Values) { $done += [int]$value }
    $status = [ordered]@{
        updated_at = (Get-Date).ToString('o')
        phase = $phase
        message = $message
        expected_total = [int]$manifest.expected.total_calls
        successful_total = $done
        progress_percent = [math]::Round(100.0 * $done / [int]$manifest.expected.total_calls, 2)
        successful_by_provider = $counts
        api_output = $manifest.immutable_output_path
        analysis_output = 'api_pilot/outputs/checklist_correction_analysis.json'
        report = 'api_pilot/reports/final/CHECKLIST_CORRECTION_SUPPLEMENT.md'
    }
    $status | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $statusPath -Encoding utf8
}

try {
    while ($true) {
        $counts = Get-SuccessCounts
        $complete = $true
        foreach ($provider in $manifest.provider_models.psobject.Properties.Name) {
            $value = if ($counts.ContainsKey($provider)) { [int]$counts[$provider] } else { 0 }
            if ($value -lt [int]$manifest.expected.calls_per_provider) { $complete = $false }
        }
        if ($complete) { break }

        Write-Status 'api_inference' 'Running or resuming corrected checklist prompts.'
        Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) inference cycle start"
        & $python api_pilot/survey_pilot_runner.py `
            --profiles $manifest.source.profile_file `
            --prompt-bundle $manifest.prompt.file `
            --max-profiles-per-country $manifest.source.profiles_per_country `
            --providers openai deepseek mistral gemini qwen `
            --openai-model gpt-5.4 `
            --deepseek-model deepseek-v4-pro `
            --mistral-model mistral-medium-3-5 `
            --gemini-model gemini-3.6-flash `
            --qwen-model qwen3.7-max-2026-06-08 `
            --allow-custom-openai-base-url `
            --provider-workers openai=4 `
            --provider-workers deepseek=6 `
            --provider-workers mistral=4 `
            --provider-workers gemini=4 `
            --provider-workers qwen=32 `
            --max-attempts 2 `
            --quota-stop-threshold 3 `
            --target-batch-size $manifest.runner_configuration.target_batch_size `
            --task-manifest $manifestPath `
            --analysis-manifest $manifest.source.analysis_manifest_file `
            --output $manifest.immutable_output_path `
            --checkpoint $manifest.immutable_checkpoint_path `
            --progress-doc api_pilot/progress/CHECKLIST_CORRECTION_PROGRESS.md `
            --progress-every 24 `
            --run-label checklist_correction_supplement_v1 `
            --resume-command 'Managed by api_pilot/scripts/windows/run_checklist_correction_automation.ps1' `
            --acknowledge-respondent-data-processing *>> $logPath
        Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) inference cycle exit=$LASTEXITCODE"
        Write-Status 'api_retry_wait' "Inference cycle exited with code $LASTEXITCODE; incomplete calls will resume."
        Start-Sleep -Seconds 45
    }

    Write-Status 'local_analysis' 'API panel complete; running correction and exclusion analyses.'
    Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) analysis start"
    & $python api_pilot/analyze_checklist_correction_supplement.py `
        --bootstrap-replicates 2000 `
        --permutations 20000 *>> $logPath
    if ($LASTEXITCODE -ne 0) {
        Write-Status 'analysis_error' "Local analysis exited with code $LASTEXITCODE."
        exit $LASTEXITCODE
    }
    Write-Status 'complete' 'Corrected checklist supplement and exclusion sensitivity are complete.'
    Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) complete"
}
finally {
    Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
}
