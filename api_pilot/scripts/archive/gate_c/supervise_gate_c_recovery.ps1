param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('openai', 'mistral', 'gemini')]
    [string]$Provider,

    [int]$IntervalSeconds = 60
)

$ErrorActionPreference = 'Continue'
$workspace = Resolve-Path (Join-Path $PSScriptRoot '..\..\..\..')
Set-Location -LiteralPath $workspace
$log = "api_pilot/restricted_outputs/gate_c_$Provider-supervisor.log"
if ($Provider -eq 'gemini') {
    $proxyPort = 7897
    if (-not (Get-NetTCPConnection -State Listen -LocalPort $proxyPort -ErrorAction SilentlyContinue)) {
        throw "The local Vortex proxy is not listening on port $proxyPort."
    }
    $env:HTTPS_PROXY = "http://127.0.0.1:$proxyPort"
    $env:HTTP_PROXY = "http://127.0.0.1:$proxyPort"
}

function Write-SupervisorLog([string]$Message) {
    $line = "$(Get-Date -Format o) $Message"
    Add-Content -LiteralPath $log -Value $line -Encoding UTF8
}

Write-SupervisorLog "started interval_seconds=$IntervalSeconds"
while ($true) {
    Start-Sleep -Seconds $IntervalSeconds
    $needle = "gate_c_v3_$Provider-recovery.sqlite"
    $active = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -eq 'python.exe' -and
            $_.CommandLine -like '*survey_pilot_runner.py*' -and
            $_.CommandLine -like "*$needle*"
        }
    if ($active) {
        Write-SupervisorLog "skip active_pid=$($active.ProcessId -join ',')"
        continue
    }
    if ($Provider -eq 'gemini') {
        & python api_pilot/probe_gemini_capacity.py *>> $log
        if ($LASTEXITCODE -ne 0) {
            Write-SupervisorLog "capacity probe unavailable exit_code=$LASTEXITCODE"
            continue
        }
        Write-SupervisorLog 'capacity probe passed'
    }
    Write-SupervisorLog 'resume attempt started'
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
        api_pilot/scripts/archive/gate_c/run_gate_c_recovery.ps1 -Provider $Provider *>> $log
    Write-SupervisorLog "resume attempt finished exit_code=$LASTEXITCODE"
}
