<#
.SYNOPSIS
    RLT Model MLOps Script for Windows PowerShell
.DESCRIPTION
    Automates training and evaluation of RLT models
.EXAMPLE
    .\run.ps1 train -Dataset breast_cancer
    .\run.ps1 evaluate
    .\run.ps1 all -Dataset sonar
#>

param(
    [Parameter(Position=0)]
    [ValidateSet("train", "evaluate", "all", "clean", "setup", "serve", "help", "list-datasets")]
    [string]$Command = "help",
    
    [string]$Dataset = "",
    [string]$CsvPath = "",
    [string]$Target = "",
    [ValidateSet("classification", "regression")]
    [string]$TaskType = "classification",
    [int]$NRltTrees = 10,
    [int]$NExtraTrees = 50,
    [double]$MutingRate = 0.1,
    [int]$K = 3,
    [int]$MinProtected = 5,
    [int]$MinSamplesSplit = 2,
    [double]$TestSize = 0.2,
    [int]$RandomState = 42,
    [int]$NJobs = -1,
    [string]$ModelDir = "models",
    [string]$ModelName = ""
)

$ErrorActionPreference = "Stop"
$Python = ".\.venv\Scripts\python.exe"

function Write-Header($text) {
    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor Cyan
    Write-Host $text -ForegroundColor Yellow
    Write-Host ("=" * 60) -ForegroundColor Cyan
}

function Invoke-Train {
    Write-Header "Training RLT Model"
    
    # Build arguments list
    $trainArgs = @("scripts/train.py")
    
    if ($CsvPath -ne "") {
        Write-Host "CSV File: $CsvPath" -ForegroundColor Green
        Write-Host "Target: $Target"
        Write-Host "Task Type: $TaskType"
        $trainArgs += "--csv-path", $CsvPath
        $trainArgs += "--target", $Target
        $trainArgs += "--task-type", $TaskType
    } elseif ($Dataset -ne "") {
        Write-Host "Dataset: $Dataset" -ForegroundColor Green
        $trainArgs += "--dataset", $Dataset
    } else {
        Write-Host "Error: Either -Dataset or -CsvPath must be provided" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "N_RLT_Trees: $NRltTrees"
    Write-Host "N_Extra_Trees: $NExtraTrees"
    Write-Host "Muting_Rate: $MutingRate"
    Write-Host "K: $K"
    Write-Host ("-" * 40)
    
    $trainArgs += "--n-rlt-trees", $NRltTrees
    $trainArgs += "--n-extra-trees", $NExtraTrees
    $trainArgs += "--muting-rate", $MutingRate
    $trainArgs += "--k", $K
    $trainArgs += "--min-protected", $MinProtected
    $trainArgs += "--min-samples-split", $MinSamplesSplit
    $trainArgs += "--test-size", $TestSize
    $trainArgs += "--random-state", $RandomState
    $trainArgs += "--n-jobs", $NJobs
    $trainArgs += "--output-dir", $ModelDir
    
    & $Python @trainArgs
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Training failed!" -ForegroundColor Red
        exit 1
    }
}

function Invoke-Evaluate {
    Write-Header "Evaluating RLT Model"
    
    $args = @(
        "scripts/evaluate.py",
        "--model-dir", $ModelDir,
        "--save-results"
    )
    
    if ($ModelName -ne "") {
        $args += "--model-name"
        $args += $ModelName
    }
    
    & $Python @args
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Evaluation failed!" -ForegroundColor Red
        exit 1
    }
}

function Invoke-Clean {
    Write-Header "Cleaning Model Artifacts"
    
    if (Test-Path $ModelDir) {
        Remove-Item -Recurse -Force $ModelDir
        Write-Host "Removed $ModelDir directory" -ForegroundColor Green
    } else {
        Write-Host "Nothing to clean" -ForegroundColor Yellow
    }
}

function Invoke-Setup {
    Write-Header "Setting Up Virtual Environment"
    
    if (-not (Test-Path ".venv")) {
        Write-Host "Creating virtual environment..."
        python -m venv .venv
    }
    
    Write-Host "Installing dependencies..."
    & .\.venv\Scripts\pip.exe install -e .
    & .\.venv\Scripts\pip.exe install scikit-learn numpy pandas matplotlib seaborn flask joblib
    
    Write-Host ""
    Write-Host "Setup complete!" -ForegroundColor Green
    Write-Host "Activate with: .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
}

function Invoke-Serve {
    Write-Header "Starting Flask Application"
    & $Python app/app.py
}

function Show-Datasets {
    Write-Header "Available Datasets"
    $datasets = @(
        "breast_cancer",
        "sonar",
        "winequality_red",
        "winequality_white",
        "housing",
        "concrete",
        "auto_mpg",
        "parkinson",
        "eighthr"
    )
    
    foreach ($ds in $datasets) {
        Write-Host "  - $ds" -ForegroundColor Cyan
    }
}

function Show-Help {
    Write-Host @"

========================================
RLT Model MLOps Script
========================================

Usage: .\run.ps1 <command> [options]

Commands:
  train           Train the RLT model
  evaluate        Evaluate the latest model
  all             Train and evaluate
  clean           Remove model artifacts
  setup           Setup virtual environment
  serve           Run Flask web app
  list-datasets   Show available datasets
  help            Show this help

Data Source (choose one):
  -Dataset          Dataset name from dataset_wrapper
  -CsvPath          Path to external CSV file
  -Target           Target column name (required with -CsvPath)
  -TaskType         'classification' or 'regression' (default: classification)

Model Parameters:
  -NRltTrees        Number of RLT trees (default: 10)
  -NExtraTrees      Extra trees per RLT (default: 50)
  -MutingRate       Muting rate (default: 0.1)
  -K                K parameter (default: 3)
  -MinProtected     Min protected (default: 5)
  -MinSamplesSplit  Min samples split (default: 2)
  -TestSize         Test set fraction (default: 0.2)
  -RandomState      Random state (default: 42)
  -NJobs            Number of jobs (default: -1)
  -ModelDir         Model output directory (default: models)
  -ModelName        Specific model to evaluate (optional)

Examples with dataset_wrapper:
  .\run.ps1 train -Dataset breast_cancer
  .\run.ps1 train -Dataset sonar -NRltTrees 15 -NExtraTrees 75
  .\run.ps1 all -Dataset breast_cancer

Examples with external CSV:
  .\run.ps1 train -CsvPath "data/iris.csv" -Target "species"
  .\run.ps1 train -CsvPath "data/prices.csv" -Target "price" -TaskType regression
  .\run.ps1 all -CsvPath "my_data.csv" -Target "label" -NRltTrees 20

Other:
  .\run.ps1 evaluate
  .\run.ps1 evaluate -ModelName "rlt_breast_cancer_20240115_143022"
  .\run.ps1 clean
  .\run.ps1 serve

"@ -ForegroundColor White
}

# Main execution
switch ($Command) {
    "train" { Invoke-Train }
    "evaluate" { Invoke-Evaluate }
    "all" { 
        Invoke-Train
        Invoke-Evaluate
    }
    "clean" { Invoke-Clean }
    "setup" { Invoke-Setup }
    "serve" { Invoke-Serve }
    "list-datasets" { Show-Datasets }
    "help" { Show-Help }
    default { Show-Help }
}
