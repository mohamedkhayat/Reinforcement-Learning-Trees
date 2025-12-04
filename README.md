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
*   **Key Feature:** Custom `ReinforcementLearningTree` class with "Embedded Model" (Extremely Randomized Trees) and "Variable Muting" logic.
*   **Validation:** Successfully replicated scenarios:
    *   *Scenario 1:* Sparse Classification
    *   *Scenario 2:* Non-linear relationships
    *   *Scenario 3:* Checkerboard (High correlation/Interaction)
    *   *Scenario 4:* Linear signals

### 2️⃣ DSO 2: Benchmark Comparison
**Goal:** Compare RLT against industry-standard models on 10 real-world UCI datasets (augmented with noise to $p=500$).
*   **Competitors:** Random Forest (sklearn), Gradient Boosting (sklearn), XGBoost.
*   **Metrics:** MSE (Regression), Accuracy (Classification), Training Time.

### 3️⃣ DSO 3: Explainability & Diagnosis
**Goal:** diagnose *why* RLT outperforms RF in sparse settings.
*   **Global Explainability:** Comparison of Variable Importance (VI) plots to show RLT's superior noise filtering.
*   **Local Explainability:** LIME analysis on individual predictions.

### 4️⃣ DSO 4: Innovation
**Goal:** Propose and test architectural improvements to the original RLT.
*   **Experiment:** Replacing the standard embedded model with **LightGBM** to improve training speed without sacrificing accuracy.
