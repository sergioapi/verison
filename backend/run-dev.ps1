$ErrorActionPreference = "Stop"

$backendDir = $PSScriptRoot
$repoRoot = Split-Path -Parent $backendDir

$pythonPath = Join-Path $repoRoot ".venv\Scripts\python.exe"
$modelPath = Join-Path $repoRoot "data\models\mfcc_svm_baseline.joblib"

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    [Console]::Error.WriteLine("No se encontro el interprete Python del entorno virtual: $pythonPath")
    exit 1
}

if (-not (Test-Path -LiteralPath $modelPath -PathType Leaf)) {
    [Console]::Error.WriteLine("No se encontro el artefacto del modelo: $modelPath")
    exit 1
}

$env:MODEL_PATH = (Resolve-Path -LiteralPath $modelPath).Path
$env:CORS_ALLOWED_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"

& $pythonPath -m uvicorn app.main:app --app-dir $backendDir --host 127.0.0.1 --port 8000
exit $LASTEXITCODE
