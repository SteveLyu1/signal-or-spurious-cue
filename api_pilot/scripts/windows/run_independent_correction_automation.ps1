$ErrorActionPreference = 'Continue'
$workspace = Resolve-Path (Join-Path $PSScriptRoot '..\..\..')
Set-Location -LiteralPath $workspace
$python = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
$manifestPath = 'dataset/analysis/independent_correction_replication_manifest_v2.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$logPath = 'api_pilot/logs/independent_correction_automation_v2.log'
$statusPath = 'api_pilot/outputs/independent_correction_automation_v2_status.json'
$pidPath = 'api_pilot/restricted_outputs/independent_correction_automation_v2.pid'

if (Test-Path $pidPath) {
    $old = [int]((Get-Content $pidPath -Raw).Trim())
    $oldProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$old" -ErrorAction SilentlyContinue
    if ($oldProcess -and $oldProcess.CommandLine -match 'run_independent_correction_automation.ps1') { exit 0 }
}
Set-Content $pidPath $PID -Encoding ascii

function Write-Status([string]$phase, [string]$message) {
    $checkpoint = $manifest.immutable_checkpoint_path
    $counts = @{}
    if (Test-Path $checkpoint) {
        $raw = & $python -c "import json,sqlite3,sys;c=sqlite3.connect(sys.argv[1]);print(json.dumps(dict(c.execute('select provider,count(*) from results where success=1 group by provider').fetchall())))" $checkpoint
        if ($raw) { $obj = $raw | ConvertFrom-Json; foreach ($p in $obj.psobject.Properties) { $counts[$p.Name] = [int]$p.Value } }
    }
    $done = 0; foreach ($n in $counts.Values) { $done += $n }
    [ordered]@{updated_at=(Get-Date).ToString('o');phase=$phase;message=$message;expected_total=[int]$manifest.expected.total_calls;successful_total=$done;progress_percent=[math]::Round(100*$done/[int]$manifest.expected.total_calls,2);successful_by_provider=$counts} | ConvertTo-Json -Depth 4 | Set-Content $statusPath -Encoding utf8
}

try {
    while ($true) {
        Write-Status 'api_inference' 'Running or resuming the independent corrected-wording replication.'
        & $python api_pilot/survey_pilot_runner.py --profiles $manifest.source.profile_file --prompt-bundle $manifest.prompt.file --max-profiles-per-country 12 --providers openai deepseek mistral gemini qwen --openai-model gpt-5.4 --deepseek-model deepseek-v4-pro --mistral-model mistral-medium-3-5 --gemini-model gemini-3.6-flash --qwen-model qwen3.7-max-2026-06-08 --allow-custom-openai-base-url --provider-workers openai=4 --provider-workers deepseek=6 --provider-workers mistral=4 --provider-workers gemini=4 --provider-workers qwen=16 --max-attempts 2 --quota-stop-threshold 3 --target-batch-size 5 --task-manifest $manifestPath --analysis-manifest $manifest.source.analysis_manifest_file --output $manifest.immutable_output_path --checkpoint $manifest.immutable_checkpoint_path --progress-doc api_pilot/progress/INDEPENDENT_CORRECTION_PROGRESS.md --progress-every 24 --run-label independent_correction_replication_v2 --resume-command 'Managed automatically' --acknowledge-respondent-data-processing *>> $logPath
        $runnerCode = $LASTEXITCODE
        # survey_pilot_runner: 0=complete, 1=retryable calls remain,
        # 2=configuration error, 3=explicit phase cap, 130=interrupted.
        if ($runnerCode -notin @(0, 1)) {
            Write-Status 'configuration_error' "Runner exited with code $runnerCode. Automatic retries stopped; inspect the version-specific log."
            exit $runnerCode
        }
        $analysisPath = 'api_pilot/outputs/independent_correction_replication_v2_analysis.json'
        Remove-Item $analysisPath -Force -ErrorAction SilentlyContinue
        & $python api_pilot/analyze_independent_correction_replication.py --bootstrap-replicates 10000 --permutations 100000 *>> $logPath
        $analysisCode = $LASTEXITCODE
        if ($analysisCode -ne 0) {
            Write-Status 'analysis_error' "Fail-closed analysis exited with code $analysisCode. Automatic retries stopped."
            exit $analysisCode
        }
        & $python api_pilot/verify_independent_correction_replication.py *>> $logPath
        if ($LASTEXITCODE -ne 0) {
            Write-Status 'verification_error' 'Completion verifier failed. Automatic retries stopped.'
            exit 3
        }
        $completion = Get-Content api_pilot/outputs/independent_correction_replication_v2_completion.json -Raw | ConvertFrom-Json
        if ($completion.complete -eq $true) {
            & $python api_pilot/verify_five_model_completion.py *>> $logPath
            if ($LASTEXITCODE -ne 0) { Write-Status 'postprocess_error' 'Final completion contract failed.'; exit 4 }
            & $python api_pilot/export_final_results_tables.py *>> $logPath
            if ($LASTEXITCODE -ne 0) { Write-Status 'postprocess_error' 'Final table export failed.'; exit 5 }
            Write-Status 'complete' 'API inference, fail-closed H1/H2 analysis, final contract, and tables complete.'
            break
        }
        $failureClass = & $python -c "import json,sqlite3,sys;c=sqlite3.connect(sys.argv[1]);types={str(json.loads(r[0]).get('error_type') or '') for r in c.execute('select result_json from results where success=0')};allowed={'Arrearage','429','503','transport_error','408','500','502','504'};print('retryable' if not types or types<=allowed else 'structural:'+','.join(sorted(types-allowed)))" $manifest.immutable_checkpoint_path
        if ($failureClass -like 'structural:*') {
            Write-Status 'configuration_error' "Non-retryable checkpoint failure: $failureClass"
            exit 6
        }
        $arrears = $false
        if (Test-Path $manifest.immutable_checkpoint_path) {
            $arrearsRaw = & $python -c "import json,sqlite3,sys;c=sqlite3.connect(sys.argv[1]);print(any(json.loads(r[0]).get('error_type')=='Arrearage' for r in c.execute('select result_json from results where success=0')))" $manifest.immutable_checkpoint_path
            $arrears = ($arrearsRaw -eq 'True')
        }
        $cooldown = if ($arrears) { 600 } else { 60 }
        Write-Status 'api_retry_wait' "Incomplete calls retained in checkpoint; retrying after $cooldown seconds."
        Start-Sleep -Seconds $cooldown
    }
} finally { Remove-Item $pidPath -Force -ErrorAction SilentlyContinue }
