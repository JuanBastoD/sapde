<#
.SYNOPSIS
    Lanza los servicios de SAPDE (API y Dashboard) en Windows.
.PARAMETER Servicio
    "api" | "dashboard" | "ambos" (default: ambos)
.EXAMPLE
    .\scripts\run.ps1 -Servicio ambos
    .\scripts\run.ps1 -Servicio api
#>

param(
    [ValidateSet("api","dashboard","ambos")]
    [string]$Servicio = "ambos"
)

$base   = $PSScriptRoot | Split-Path -Parent
$python = Join-Path $base ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Host "Entorno no encontrado. Ejecuta primero: .\scripts\setup.ps1" -ForegroundColor Red
    exit 1
}

Set-Location $base

if ($Servicio -in @("api","ambos")) {
    Write-Host "Levantando API FastAPI en http://localhost:8000 ..." -ForegroundColor Cyan
    $apiJob = Start-Job -ScriptBlock {
        param($p, $b)
        Set-Location $b
        & $p -m uvicorn sapde.api.main:app --host 0.0.0.0 --port 8000 --reload
    } -ArgumentList $python, $base
    Write-Host "  API iniciada (Job ID: $($apiJob.Id))" -ForegroundColor Green
}

if ($Servicio -in @("dashboard","ambos")) {
    Write-Host "Levantando Dashboard Streamlit en http://localhost:8501 ..." -ForegroundColor Cyan
    # Pequeno delay para que la API arranque primero
    Start-Sleep -Seconds 2
    & (Join-Path $base ".venv\Scripts\streamlit.exe") run (Join-Path $base "dashboard\app.py")
}

if ($Servicio -eq "api") {
    Write-Host "Presiona Ctrl+C para detener la API."
    Wait-Job $apiJob
}
