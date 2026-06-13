<#
.SYNOPSIS
    Configura el entorno de desarrollo de SAPDE en Windows.
.DESCRIPTION
    1. Verifica Python 3.12
    2. Crea entorno virtual
    3. Instala dependencias
    4. Copia .env.example -> .env
    5. Genera el dataset sintetico
.EXAMPLE
    cd sapde
    .\scripts\setup.ps1
#>

$ErrorActionPreference = "Stop"
$base = $PSScriptRoot | Split-Path -Parent

Write-Host "`n=== SAPDE — Setup de entorno ===" -ForegroundColor Cyan

# ----------------------------------------------------------
# 1. Verificar Python 3.12
# ----------------------------------------------------------
Write-Host "`n[1/5] Verificando Python 3.12..." -ForegroundColor Yellow

$pythonCmd = $null
foreach ($cmd in @("py -3.12", "python3.12", "python")) {
    try {
        $ver = & cmd /c "$cmd --version" 2>&1
        if ($ver -match "3\.12") {
            $pythonCmd = $cmd
            Write-Host "  OK: $ver" -ForegroundColor Green
            break
        }
    } catch { }
}

if (-not $pythonCmd) {
    Write-Host "  ERROR: Python 3.12 no encontrado." -ForegroundColor Red
    Write-Host "  Descargar: https://www.python.org/downloads/release/python-3120/" -ForegroundColor Red
    exit 1
}

# ----------------------------------------------------------
# 2. Crear entorno virtual
# ----------------------------------------------------------
Write-Host "`n[2/5] Creando entorno virtual (.venv)..." -ForegroundColor Yellow

$venvPath = Join-Path $base ".venv"
if (-not (Test-Path $venvPath)) {
    & cmd /c "$pythonCmd -m venv `"$venvPath`""
    Write-Host "  Entorno creado en: $venvPath" -ForegroundColor Green
} else {
    Write-Host "  Entorno ya existe, omitiendo creacion." -ForegroundColor Gray
}

$pip = Join-Path $venvPath "Scripts\pip.exe"
$python = Join-Path $venvPath "Scripts\python.exe"

# ----------------------------------------------------------
# 3. Instalar dependencias
# ----------------------------------------------------------
Write-Host "`n[3/5] Instalando dependencias..." -ForegroundColor Yellow

& "$pip" install --upgrade pip --quiet
& "$pip" install -r (Join-Path $base "requirements-dev.txt")

Write-Host "  Dependencias instaladas." -ForegroundColor Green

# ----------------------------------------------------------
# 4. Configurar archivo .env
# ----------------------------------------------------------
Write-Host "`n[4/5] Configurando .env..." -ForegroundColor Yellow

$envFile    = Join-Path $base ".env"
$envExample = Join-Path $base ".env.example"

if (-not (Test-Path $envFile)) {
    Copy-Item $envExample $envFile
    Write-Host "  .env creado desde .env.example" -ForegroundColor Green
    Write-Host "  IMPORTANTE: edita .env con tu contrasena de PostgreSQL" -ForegroundColor Yellow
} else {
    Write-Host "  .env ya existe, no se sobreescribe." -ForegroundColor Gray
}

# ----------------------------------------------------------
# 5. Generar dataset sintetico
# ----------------------------------------------------------
Write-Host "`n[5/5] Generando dataset sintetico..." -ForegroundColor Yellow

Set-Location $base
& "$python" -m sapde.data.synthetic

Write-Host "`n=== Setup completado ===" -ForegroundColor Cyan
Write-Host "Activa el entorno con: .venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "Prueba con:            pytest tests\" -ForegroundColor White
