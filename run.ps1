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
    [ValidateSet("train", "evaluate", "all", "clean", "setup", "serve", "help", "list-datasets", "docker-build", "docker-run", "docker-run-hub", "docker-stop", "docker-logs", "docker-push", "docker-pull", "docker-clean")]
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

# Docker configuration
$DockerImage = "rlt-app"
$DockerTag = "latest"
$DockerUser = "kousay763"
$DockerContainer = "rlt-container"
$DockerPort = 5000

function Invoke-DockerBuild {
    Write-Header "Building Docker Image: ${DockerImage}:${DockerTag}"
    docker build -t "${DockerImage}:${DockerTag}" .
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Docker image built successfully!" -ForegroundColor Green
    } else {
        Write-Host "Docker build failed!" -ForegroundColor Red
        exit 1
    }
}

function Invoke-DockerRun {
    Write-Header "Running Docker Container (Local): $DockerContainer"
    Write-Host "Access the app at: http://localhost:${DockerPort}" -ForegroundColor Green
    docker run -d --name $DockerContainer -p "${DockerPort}:5000" "${DockerImage}:${DockerTag}"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Container started successfully!" -ForegroundColor Green
    } else {
        Write-Host "Failed to start container!" -ForegroundColor Red
    }
}

function Invoke-DockerRunHub {
    Write-Header "Running Docker Container from Docker Hub"
    Write-Host "Image: ${DockerUser}/${DockerImage}:${DockerTag}" -ForegroundColor Cyan
    Write-Host "Access the app at: http://localhost:${DockerPort}" -ForegroundColor Green
    Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
    docker run -p "${DockerPort}:5000" "${DockerUser}/${DockerImage}:${DockerTag}"
}

function Invoke-DockerStop {
    Write-Header "Stopping Docker Container: $DockerContainer"
    docker stop $DockerContainer 2>$null
    docker rm $DockerContainer 2>$null
    Write-Host "Container stopped and removed." -ForegroundColor Green
}

function Invoke-DockerLogs {
    Write-Header "Docker Container Logs"
    docker logs -f $DockerContainer
}

function Invoke-DockerPush {
    Write-Header "Pushing to Docker Hub: ${DockerUser}/${DockerImage}:${DockerTag}"
    Write-Host "Tagging image..." -ForegroundColor Yellow
    docker tag "${DockerImage}:${DockerTag}" "${DockerUser}/${DockerImage}:${DockerTag}"
    Write-Host "Pushing to Docker Hub..." -ForegroundColor Yellow
    docker push "${DockerUser}/${DockerImage}:${DockerTag}"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Push complete! Image available at: https://hub.docker.com/r/${DockerUser}/${DockerImage}" -ForegroundColor Green
    } else {
        Write-Host "Push failed! Make sure you are logged in: docker login" -ForegroundColor Red
    }
}

function Invoke-DockerPull {
    Write-Header "Pulling from Docker Hub: ${DockerUser}/${DockerImage}:${DockerTag}"
    docker pull "${DockerUser}/${DockerImage}:${DockerTag}"
}

function Invoke-DockerClean {
    Write-Header "Cleaning Docker Artifacts"
    
    # Stop and remove named container (suppress all errors)
    try { docker stop $DockerContainer 2>&1 | Out-Null } catch {}
    try { docker rm $DockerContainer 2>&1 | Out-Null } catch {}
    
    # Remove all containers using gunicorn (RLT app containers)
    Write-Host "Removing RLT app containers..." -ForegroundColor Yellow
    try {
        $rltContainers = docker ps -a --format "{{.ID}}|{{.Command}}" | Where-Object { $_ -match "gunicorn" }
        if ($rltContainers) {
            $rltContainers | ForEach-Object {
                $containerId = ($_ -split '\|')[0]
                Write-Host "  Removing container: $containerId" -ForegroundColor DarkGray
                docker rm -f $containerId 2>&1 | Out-Null
            }
        }
    } catch {}
    
    # Remove local images (suppress all errors)
    Write-Host "Removing images..." -ForegroundColor Yellow
    try { docker rmi "${DockerImage}:${DockerTag}" -f 2>&1 | Out-Null } catch {}
    try { docker rmi "${DockerUser}/${DockerImage}:${DockerTag}" -f 2>&1 | Out-Null } catch {}
    
    # Remove dangling images from this project (images with <none> tag)
    Write-Host "Removing dangling RLT images..." -ForegroundColor Yellow
    try {
        $danglingImages = docker images --filter "reference=${DockerUser}/${DockerImage}" --filter "dangling=true" -q
        if ($danglingImages) {
            $danglingImages | ForEach-Object {
                Write-Host "  Removing image: $_" -ForegroundColor DarkGray
                docker rmi -f $_ 2>&1 | Out-Null
            }
        }
        # Also check local images
        $localDangling = docker images --filter "reference=${DockerImage}" --filter "dangling=true" -q
        if ($localDangling) {
            $localDangling | ForEach-Object {
                docker rmi -f $_ 2>&1 | Out-Null
            }
        }
    } catch {}
    
    Write-Host "Docker cleanup complete." -ForegroundColor Green
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

Docker Commands:
  .\run.ps1 docker-build    Build Docker image
  .\run.ps1 docker-run      Run Docker container (local image)
  .\run.ps1 docker-run-hub  Run from Docker Hub (kousay763/rlt-app)
  .\run.ps1 docker-stop     Stop and remove container
  .\run.ps1 docker-logs     View container logs
  .\run.ps1 docker-push     Push image to Docker Hub (kousay763)
  .\run.ps1 docker-pull     Pull image from Docker Hub
  .\run.ps1 docker-clean    Remove all Docker artifacts

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
    "docker-build" { Invoke-DockerBuild }
    "docker-run" { Invoke-DockerRun }
    "docker-run-hub" { Invoke-DockerRunHub }
    "docker-stop" { Invoke-DockerStop }
    "docker-logs" { Invoke-DockerLogs }
    "docker-push" { Invoke-DockerPush }
    "docker-pull" { Invoke-DockerPull }
    "docker-clean" { Invoke-DockerClean }
    "help" { Show-Help }
    default { Show-Help }
}
