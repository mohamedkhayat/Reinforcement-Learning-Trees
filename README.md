# Reinforcement Learning Trees (RLT) Implementation

> **Re-implementation of "Reinforcement Learning Trees" (Zhu et al., 2015)**
> *A novel tree-based method that uses reinforcement learning to identifying strong signals in high-dimensional, sparse data.*

---

## 📖 Project Overview

This project implements **Reinforcement Learning Trees (RLT)** from scratch in Python. RLT improves upon Random Forests by introducing an "embedded model" at each split node. Instead of greedily choosing the best immediate split, RLT uses reinforcement learning to look ahead, selecting variables that maximize future rewards. This makes it particularly effective for **high-dimensional, sparse datasets** where traditional methods often fail to distinguish signal from noise.

This project follows the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology to ensure a rigorous, reproducible data science lifecycle.

### 📄 Reference Paper
*   **Title:** Reinforcement Learning Trees
*   **Authors:** Ruoqing Zhu, Donglin Zeng, & Michael R. Kosorok
*   **Journal:** Journal of the American Statistical Association (2015)
*   **Original Paper:** https://www.tandfonline.com/doi/full/10.1080/01621459.2015.1036994?scroll=top&needAccess=true

---

## 🎯 Data Science Objectives (DSOs)

This project is structured around four key objectives:

### 1️⃣ DSO 1: Strategy Re-implementation
**Goal:** Faithfully reproduce the RLT algorithm and validate it on the 4 synthetic scenarios described in the original paper.

*   **Key Features Implemented:**
    *   Custom `ReinforcementLearningTrees` class with **Embedded Model** (Extremely Randomized Trees)
    *   **Variable Muting** mechanism (0%, 50%, 80% rates)
    *   **Linear Combination Splits** with parameter `k` (k=1, 2, 5)
    *   **Protected Set** for preserving important features
*   **Validation:** Successfully replicated all 4 scenarios across p=200, 500, 1000 dimensions:
    *   *Scenario 1:* Sparse Classification
    *   *Scenario 2:* Non-linear Regression
    *   *Scenario 3:* Checkerboard (High correlation & Interaction effects)
    *   *Scenario 4:* Linear signals

### 2️⃣ DSO 2: Benchmark Comparison
**Goal:** Compare RLT against industry-standard models on 10 real-world UCI datasets (augmented with noise to p=500).

*   **Competitors:** Random Forest (sklearn), Gradient Boosting (sklearn), XGBoost
*   **Metrics:** Error Rate (Classification), MSE (Regression), Training Time
*   **Key Finding:** RLT achieves **best or near-best accuracy** on most datasets, with a trade-off in training speed.

### 3️⃣ DSO 3: Explainability & Diagnosis
**Goal:** Diagnose *why* RLT outperforms RF in sparse settings using XAI techniques.

*   **Techniques Applied:**
    *   **Global Feature Importance:** Comparison plots showing RLT's superior noise filtering
    *   **Global SHAP (Beeswarm Plot):** Validates Protected Set mechanism
    *   **Local SHAP (Waterfall Plot):** Confirms muted features contribute +0.00 to predictions
*   **Key Finding:** RLT achieves **Model Sparsity** by zeroing out >50% of noisy predictors.

### 4️⃣ DSO 4: Innovation (Architectural Improvements)
**Goal:** Propose and test two architectural improvements to the original RLT.

*   **Experiment 4.1 - LightGBM Embedded Model:**
    *   Replacing ExtraTrees with **LightGBM** for signal extraction
    *   **Result:** Better accuracy on complex problems, but 10-12x slower

*   **Experiment 4.2 - K-Armed Bandit (UCB1):**
    *   Using **UCB1 algorithm** for intelligent feature selection
    *   **Result:** 15-25% faster training with comparable accuracy

---
## 🛠️ Installation & Environment Setup

### Requirements

* Python **3.9+**
* pip
* Virtual environment support

---

### 1️⃣ Create a virtual environment

From the project root:

```bash
python -m venv .venv
```

---

### 2️⃣ Activate the environment

**Linux / macOS**

```bash
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
.venv\Scripts\Activate.ps1
```

You should now see `(.venv)` in your terminal.

---

### 3️⃣ Upgrade pip (recommended)

```bash
pip install --upgrade pip
```

---

### 4️⃣ Install the project and dependencies

Install directly from `pyproject.toml`:

```bash
pip install .
```

This installs:

* all required dependencies
* the `rlt` package itself

---

### 🔧 Development Installation (optional)

If you plan to modify the source code or run experiments:

```bash
pip install -e .
```

Editable installs ensure code changes are reflected immediately.

---

### 🧪 Verify installation

```bash
python -c "from rlt import ReinforcementLearningTree; print('RLT ready')"
```

---

## 🚀 Usage Example

```python
from rlt import ReinforcementLearningTree

model = ReinforcementLearningTree(
    n_estimators=100,
    max_depth=6,
    embedded_model="extratrees",
    exploration_rate=0.1
)

model.fit(X_train, y_train)
preds = model.predict(X_test)
```
## 🌐 Flask Web Application
```markdown
### Install Flask Requirements

```bash
pip install -r app/requirements.txt
```

---

## 🚀 MLOps Commands (Windows PowerShell)

Use run.ps1 for all MLOps operations:

### Train a Model

```powershell
.\run.ps1 train -Dataset breast_cancer
```

With custom parameters:

```powershell
.\run.ps1 train -Dataset sonar -NRltTrees 15 -NExtraTrees 75 -MutingRate 0.5
```

### Evaluate a Model

```powershell
.\run.ps1 evaluate
```

### Train + Evaluate

```powershell
.\run.ps1 all -Dataset breast_cancer
```

### Launch Flask Web App

```powershell
.\run.ps1 serve
```

Then open: **http://127.0.0.1:5000**

### List Available Datasets

```powershell
.\run.ps1 list-datasets
```

### Show Help

```powershell
.\run.ps1 help
```
## 📚 Citation

If you use this implementation in academic work:

```bibtex
@article{zhu2015reinforcement,
  title={Reinforcement Learning Trees},
  author={Zhu, Ruoqing and Zeng, Donglin and Kosorok, Michael R.},
  journal={Journal of the American Statistical Association},
  year={2015}
}
```

---

## ⚠️ Disclaimer

This is an **independent re-implementation** for research and educational purposes.
It is not an official reproduction by the original authors.

---
