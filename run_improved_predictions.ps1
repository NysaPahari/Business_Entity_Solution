param(
    [int]$Workers = 6,
    [int]$RetrievalBatchSize = 8000
)
$ErrorActionPreference = 'Stop'
$pipelinePython = Join-Path $PSScriptRoot '.pipeline-venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pipelinePython)) {
    $pipelinePython = (Get-Command python -ErrorAction Stop).Source
}
Push-Location $PSScriptRoot
try {
    & $pipelinePython -u -m src.pipeline --stage predict `
        --artifact-dir artifacts/accuracy/selected `
        --work-dir work_accuracy_test --output-dir output_improved `
        --blocking-strategy composite --max-df 4096 --pool-size 128 --top-k 128 `
        --memory 4GB --threads 1 --retrieval-batch-size $RetrievalBatchSize --workers $Workers
    if ($LASTEXITCODE -ne 0) { throw "Prediction or validation failed (exit $LASTEXITCODE)." }
    & $pipelinePython -m src.finalize_improved
    if ($LASTEXITCODE -ne 0) { throw "Output provenance verification failed (exit $LASTEXITCODE)." }
} finally {
    Pop-Location
}
