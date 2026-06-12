# CheckSure v3 - one-command launch (Windows PowerShell)
$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "=== CheckSure v3 ===" -ForegroundColor Cyan

# 1. Check Ollama
try {
    $null = Get-Command ollama -ErrorAction Stop
    ollama list | Out-Null
    Write-Host "OK: Ollama reachable" -ForegroundColor Green
}
catch {
    Write-Host "WARN: Ollama not found or not running - start Ollama Desktop first" -ForegroundColor Yellow
}

# 2. Check Tavily key
$envPath = Join-Path $ProjectRoot ".env"
$hasTavily = [bool]$env:TAVILY_API_KEY -or (Test-Path $envPath)
if (-not $hasTavily) {
    Write-Host "WARN: No TAVILY_API_KEY - copy .env.example to .env and add your key" -ForegroundColor Yellow
}
if ($hasTavily) {
    Write-Host "OK: Tavily config found (.env or env var)" -ForegroundColor Green
}

# 3. Python venv + deps
$venvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Creating Python venv..."
    python -m venv .venv
}
& $venvPython -m pip install -q -r requirements.txt

# 4. Start FastAPI
$portInUse = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($portInUse) {
    $pid = $portInUse[0].OwningProcess
    Write-Host "ERROR: Port 8000 is already in use (PID $pid)." -ForegroundColor Red
    Write-Host "       Stop the other server (Ctrl+C in that terminal), or run:" -ForegroundColor Yellow
    Write-Host "       taskkill /PID $pid /F" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "Starting server at http://localhost:8000" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop" -ForegroundColor Gray
& $venvPython -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
