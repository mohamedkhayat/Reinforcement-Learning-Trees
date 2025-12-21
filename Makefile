# ==============================================================================
# RLT Model - MLOps Makefile
# ==============================================================================
# Usage (PowerShell):
#   make train DATASET=breast_cancer
#   make evaluate
#   make all DATASET=sonar
#   make clean
#
# Docker Usage:
#   make docker-build
#   make docker-run
#   make docker-stop
#   make docker-push DOCKER_USER=yourusername
# ==============================================================================

# Default values
DATASET ?= breast_cancer
N_RLT_TREES ?= 10
N_EXTRA_TREES ?= 50
MUTING_RATE ?= 0.1
K ?= 3
MIN_PROTECTED ?= 5
MIN_SAMPLES_SPLIT ?= 2
TEST_SIZE ?= 0.2
RANDOM_STATE ?= 42
N_JOBS ?= -1
MODEL_DIR ?= models

# Docker configuration
DOCKER_IMAGE ?= rlt-app
DOCKER_TAG ?= latest
DOCKER_USER ?= kousay763
DOCKER_CONTAINER ?= rlt-container
DOCKER_PORT ?= 5000

# Python executable - adjust path if needed
# For Windows, use the venv Python directly
PYTHON = .venv\Scripts\python.exe

# ==============================================================================
# Main targets
# ==============================================================================

.PHONY: all train evaluate clean help setup info docker-build docker-run docker-stop docker-push docker-clean docker-logs

# Default target
all: train evaluate

# Train the model
train:
	@echo "========================================"
	@echo "Training RLT Model"
	@echo "========================================"
	@echo "Dataset: $(DATASET)"
	@echo "N_RLT_Trees: $(N_RLT_TREES)"
	@echo "N_Extra_Trees: $(N_EXTRA_TREES)"
	@echo "Muting_Rate: $(MUTING_RATE)"
	@echo "K: $(K)"
	@echo "----------------------------------------"
	$(PYTHON) scripts/train.py \
		--dataset $(DATASET) \
		--n-rlt-trees $(N_RLT_TREES) \
		--n-extra-trees $(N_EXTRA_TREES) \
		--muting-rate $(MUTING_RATE) \
		--k $(K) \
		--min-protected $(MIN_PROTECTED) \
		--min-samples-split $(MIN_SAMPLES_SPLIT) \
		--test-size $(TEST_SIZE) \
		--random-state $(RANDOM_STATE) \
		--n-jobs $(N_JOBS) \
		--output-dir $(MODEL_DIR)

# Evaluate the latest model
evaluate:
	@echo "========================================"
	@echo "Evaluating RLT Model"
	@echo "========================================"
	$(PYTHON) scripts/evaluate.py \
		--model-dir $(MODEL_DIR) \
		--save-results

# Evaluate a specific model
evaluate-model:
	@echo "========================================"
	@echo "Evaluating Model: $(MODEL_NAME)"
	@echo "========================================"
	$(PYTHON) scripts/evaluate.py \
		--model-dir $(MODEL_DIR) \
		--model-name $(MODEL_NAME) \
		--save-results

# ==============================================================================
# Dataset-specific shortcuts
# ==============================================================================

train-breast-cancer:
	$(MAKE) train DATASET=breast_cancer N_RLT_TREES=10 N_EXTRA_TREES=50

train-sonar:
	$(MAKE) train DATASET=sonar N_RLT_TREES=15 N_EXTRA_TREES=75

train-wine-red:
	$(MAKE) train DATASET=winequality_red N_RLT_TREES=10 N_EXTRA_TREES=50

train-wine-white:
	$(MAKE) train DATASET=winequality_white N_RLT_TREES=10 N_EXTRA_TREES=50

train-housing:
	$(MAKE) train DATASET=housing N_RLT_TREES=10 N_EXTRA_TREES=50

train-concrete:
	$(MAKE) train DATASET=concrete N_RLT_TREES=10 N_EXTRA_TREES=50

train-auto-mpg:
	$(MAKE) train DATASET=auto_mpg N_RLT_TREES=10 N_EXTRA_TREES=50

train-parkinson:
	$(MAKE) train DATASET=parkinson N_RLT_TREES=10 N_EXTRA_TREES=50

train-eighthr:
	$(MAKE) train DATASET=eighthr N_RLT_TREES=10 N_EXTRA_TREES=50

# Train all datasets
train-all: train-breast-cancer train-sonar train-wine-red train-wine-white \
           train-housing train-concrete train-auto-mpg train-parkinson train-eighthr

# ==============================================================================
# Experiment targets (grid search)
# ==============================================================================

# Run experiments with different hyperparameters
experiment-trees:
	@echo "Running tree count experiments..."
	$(MAKE) train N_RLT_TREES=5 N_EXTRA_TREES=25
	$(MAKE) evaluate
	$(MAKE) train N_RLT_TREES=10 N_EXTRA_TREES=50
	$(MAKE) evaluate
	$(MAKE) train N_RLT_TREES=15 N_EXTRA_TREES=75
	$(MAKE) evaluate
	$(MAKE) train N_RLT_TREES=20 N_EXTRA_TREES=100
	$(MAKE) evaluate

experiment-muting:
	@echo "Running muting rate experiments..."
	$(MAKE) train MUTING_RATE=0.05
	$(MAKE) evaluate
	$(MAKE) train MUTING_RATE=0.1
	$(MAKE) evaluate
	$(MAKE) train MUTING_RATE=0.15
	$(MAKE) evaluate
	$(MAKE) train MUTING_RATE=0.2
	$(MAKE) evaluate

experiment-k:
	@echo "Running K parameter experiments..."
	$(MAKE) train K=1
	$(MAKE) evaluate
	$(MAKE) train K=3
	$(MAKE) evaluate
	$(MAKE) train K=5
	$(MAKE) evaluate
	$(MAKE) train K=7
	$(MAKE) evaluate

# ==============================================================================
# Utility targets
# ==============================================================================

# Setup virtual environment and install dependencies
setup:
	@echo "Setting up virtual environment..."
	python -m venv .venv
	.venv\Scripts\pip.exe install -e .
	.venv\Scripts\pip.exe install scikit-learn numpy pandas matplotlib seaborn flask joblib
	@echo "Setup complete! Activate with: .\.venv\Scripts\Activate.ps1"

# Install additional dependencies
install-deps:
	$(PYTHON) -m pip install scikit-learn numpy pandas matplotlib seaborn flask joblib

# Clean model artifacts
clean:
	@echo "Cleaning model artifacts..."
	@if exist $(MODEL_DIR) rmdir /s /q $(MODEL_DIR)
	@echo "Clean complete!"

# Clean and retrain
rebuild: clean train evaluate

# Show available datasets
list-datasets:
	@echo "Available datasets:"
	@echo "  - breast_cancer"
	@echo "  - sonar"
	@echo "  - winequality_red"
	@echo "  - winequality_white"
	@echo "  - housing"
	@echo "  - concrete"
	@echo "  - auto_mpg"
	@echo "  - parkinson"
	@echo "  - eighthr"

# Show model information
info:
	@echo "========================================"
	@echo "Model Directory: $(MODEL_DIR)"
	@echo "========================================"
	@if exist $(MODEL_DIR)\latest_model.txt ( \
		type $(MODEL_DIR)\latest_model.txt \
	) else ( \
		echo "No trained model found" \
	)

# Run Flask app
serve:
	@echo "Starting Flask application..."
	$(PYTHON) app/app.py

# ==============================================================================
# Help
# ==============================================================================

help:
	@echo "=============================================="
	@echo "RLT Model MLOps Makefile"
	@echo "=============================================="
	@echo ""
	@echo "Main Commands:"
	@echo "  make train            Train the RLT model"
	@echo "  make evaluate         Evaluate the latest model"
	@echo "  make all              Train and evaluate"
	@echo "  make serve            Run Flask web app"
	@echo ""
	@echo "Dataset Shortcuts:"
	@echo "  make train-breast-cancer"
	@echo "  make train-sonar"
	@echo "  make train-wine-red"
	@echo "  make train-wine-white"
	@echo "  make train-housing"
	@echo "  make train-concrete"
	@echo "  make train-auto-mpg"
	@echo "  make train-parkinson"
	@echo "  make train-eighthr"
	@echo "  make train-all        Train all datasets"
	@echo ""
	@echo "Experiments:"
	@echo "  make experiment-trees  Test different tree counts"
	@echo "  make experiment-muting Test different muting rates"
	@echo "  make experiment-k      Test different K values"
	@echo ""
	@echo "Utility:"
	@echo "  make setup            Setup virtual environment"
	@echo "  make clean            Remove model artifacts"
	@echo "  make rebuild          Clean and retrain"
	@echo "  make list-datasets    Show available datasets"
	@echo "  make info             Show current model info"
	@echo "  make help             Show this help"
	@echo ""
	@echo "Parameters (use with train):"
	@echo "  DATASET=name          Dataset name (default: breast_cancer)"
	@echo "  N_RLT_TREES=n         Number of RLT trees (default: 10)"
	@echo "  N_EXTRA_TREES=n       Extra trees per RLT (default: 50)"
	@echo "  MUTING_RATE=r         Muting rate (default: 0.1)"
	@echo "  K=k                   K parameter (default: 3)"
	@echo "  MIN_PROTECTED=n       Min protected (default: 5)"
	@echo "  MIN_SAMPLES_SPLIT=n   Min samples split (default: 2)"
	@echo "  TEST_SIZE=f           Test set fraction (default: 0.2)"
	@echo ""
	@echo "Examples:"
	@echo "  make train DATASET=sonar N_RLT_TREES=15"
	@echo "  make all DATASET=breast_cancer"
	@echo "  make evaluate-model MODEL_NAME=breast_cancer_20240115_143022"

# ==============================================================================
# Docker targets
# ==============================================================================

# Build Docker image
docker-build:
	@echo "========================================"
	@echo "Building Docker Image: $(DOCKER_IMAGE):$(DOCKER_TAG)"
	@echo "========================================"
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

# Run Docker container (local image)
docker-run:
	@echo "========================================"
	@echo "Running Docker Container: $(DOCKER_CONTAINER)"
	@echo "========================================"
	@echo "Access the app at: http://localhost:$(DOCKER_PORT)"
	@echo "----------------------------------------"
	docker run -d --name $(DOCKER_CONTAINER) -p $(DOCKER_PORT):5000 $(DOCKER_IMAGE):$(DOCKER_TAG)
	@echo "Container started successfully!"

# Run Docker container from Docker Hub
docker-run-hub:
	@echo "========================================"
	@echo "Running Docker Container from Docker Hub"
	@echo "========================================"
	@echo "Image: $(DOCKER_USER)/$(DOCKER_IMAGE):$(DOCKER_TAG)"
	@echo "Access the app at: http://localhost:$(DOCKER_PORT)"
	@echo "----------------------------------------"
	docker run -p $(DOCKER_PORT):5000 $(DOCKER_USER)/$(DOCKER_IMAGE):$(DOCKER_TAG)

# Run Docker container (interactive mode)
docker-run-it:
	@echo "========================================"
	@echo "Running Docker Container (Interactive)"
	@echo "========================================"
	docker run -it --rm -p $(DOCKER_PORT):5000 $(DOCKER_IMAGE):$(DOCKER_TAG)

# Stop Docker container
docker-stop:
	@echo "========================================"
	@echo "Stopping Docker Container: $(DOCKER_CONTAINER)"
	@echo "========================================"
	-docker stop $(DOCKER_CONTAINER)
	-docker rm $(DOCKER_CONTAINER)
	@echo "Container stopped and removed."

# View Docker container logs
docker-logs:
	@echo "========================================"
	@echo "Docker Container Logs"
	@echo "========================================"
	docker logs -f $(DOCKER_CONTAINER)

# Tag and push to Docker Hub
docker-push:
	@echo "========================================"
	@echo "Pushing to Docker Hub"
	@echo "========================================"
	@echo "Tagging image as $(DOCKER_USER)/$(DOCKER_IMAGE):$(DOCKER_TAG)"
	docker tag $(DOCKER_IMAGE):$(DOCKER_TAG) $(DOCKER_USER)/$(DOCKER_IMAGE):$(DOCKER_TAG)
	@echo "Pushing to Docker Hub..."
	docker push $(DOCKER_USER)/$(DOCKER_IMAGE):$(DOCKER_TAG)
	@echo "Push complete!"

# Pull from Docker Hub
docker-pull:
	@echo "========================================"
	@echo "Pulling from Docker Hub"
	@echo "========================================"
	docker pull $(DOCKER_USER)/$(DOCKER_IMAGE):$(DOCKER_TAG)

# Clean Docker artifacts
docker-clean:
	@echo "========================================"
	@echo "Cleaning Docker Artifacts"
	@echo "========================================"
	-docker stop $(DOCKER_CONTAINER) 2>/dev/null || true
	-docker rm $(DOCKER_CONTAINER) 2>/dev/null || true
	@echo "Removing all containers using image $(DOCKER_IMAGE)..."
	-docker ps -a -q --filter ancestor=$(DOCKER_IMAGE):$(DOCKER_TAG) | ForEach-Object { docker rm -f $_ } 2>/dev/null || true
	-docker rmi $(DOCKER_IMAGE):$(DOCKER_TAG) -f 2>/dev/null || true
	-docker rmi $(DOCKER_USER)/$(DOCKER_IMAGE):$(DOCKER_TAG) -f 2>/dev/null || true
	@echo "Docker cleanup complete."

# Full Docker workflow: build, run
docker-deploy: docker-build docker-run
	@echo "========================================"
	@echo "Deployment Complete!"
	@echo "========================================"
	@echo "App running at: http://localhost:$(DOCKER_PORT)"
