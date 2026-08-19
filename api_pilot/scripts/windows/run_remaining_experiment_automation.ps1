$ErrorActionPreference = 'Continue'
$workspace = Resolve-Path (Join-Path $PSScriptRoot '..\..\..')
Set-Location -LiteralPath $workspace

$python = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
$pidPath = Join-Path $workspace 'api_pilot\restricted_outputs\remaining_experiment_automation.pid'
$logPath = Join-Path $workspace 'api_pilot\logs\remaining_experiment\supervisor.log'
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $logPath) | Out-Null

if (Test-Path -LiteralPath $pidPath) {
    $existing = [int]((Get-Content -LiteralPath $pidPath -Raw).Trim())
    if ($existing -and (Get-Process -Id $existing -ErrorAction SilentlyContinue)) {
        Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) supervisor already running pid=$existing"
        exit 0
    }
}

Set-Content -LiteralPath $pidPath -Value $PID -Encoding ascii
try {
    while ($true) {
        Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) cycle start"
        & $python 'api_pilot/automate_remaining_experiment.py' '--poll-seconds' '30' *>> $logPath
        $code = $LASTEXITCODE
        Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) cycle exit=$code"
        if ($code -eq 0) { break }
        if ($code -eq 14) {
            Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) report-only scientific warning; continuing next cycle"
        }
        if ($code -eq 13) {
            Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) machine work complete; awaiting human review"
            break
        }
        Start-Sleep -Seconds 60
    }
}
finally {
    Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
}
