# 📚 Documentation Détaillée du Pipeline RLT

**Projet:** Réimplémentation et Amélioration de l'algorithme RLT (Reinforcement Learning Trees)  
**Date:** Novembre 2025  
**Auteurs:** Kousay Najar, Hamza Farhani, Taoufik Krid, Wiem Ben M'Sahel, Rawen Mezzi, Mohamed Khayat

---

## 🎯 Vue d'Ensemble du Projet

Ce projet implémente l'algorithme **Reinforcement Learning Trees (RLT)**, une méthode d'ensemble basée sur les arbres de décision, conçue pour gérer efficacement les données **haute dimension avec sparsité** (p >> p1, où p1 = nombre de variables signaux).

### Objectifs Principaux (DSOs)

1. **DSO1:** Implémenter l'algorithme RLT avec ses 3 stratégies innovantes
2. **DSO2:** Comparer RLT avec les méthodes classiques (RF, GBM, XGBoost)
3. **DSO3:** Rendre les décisions explicables (XAI)
4. **DSO4:** Optimiser l'algorithme RLT

---

## 📁 Architecture du Projet

```
Reinforcement-Learning-Trees/
│
├── datasets/                          # Données brutes (CSV)
│   ├── breast_cancer.csv
│   ├── concrete_data.csv
│   ├── Parkinsson disease.csv
│   └── ... (7 autres datasets)
│
├── src/
│   ├── utils/
│   │   └── dataset_wrapper.py        # ⭐ Classe pour standardiser les datasets
│   │
│   ├── scripts/
│       ├── data_understanding.py      # ⭐ Exploration et visualisation
│       └── data_preparation.py        # ⭐ Nettoyage et préparation
│   └── RLT/
│       ├── Node.py                    # Structure d'un nœud de l'arbre RLT
│       ├── RLT.py                     # Base commune aux variantes RLT
│       ├── RLTClassification.py       # Spécialisation pour les tâches BO1/DSO1 (classification)
│       └── RLTRegression.py           # Spécialisation pour les tâches BO1/DSO1 (régression)
│
├── notebooks/
│   └── notebook.ipynb                 # Pipeline complet (Phases 1-6)
│
└── DOCUMENTATION_PIPELINE.md          # 📖 Ce fichier
```

---

## 🔧 Composants Python Détaillés

### 1️⃣ `dataset_wrapper.py` - Le Gestionnaire Universel

#### 🎯 **Rôle et Utilité**

Ce fichier résout un **problème critique** : nous travaillons avec **10 datasets différents** provenant de sources variées (UCI ML Repository). Chaque dataset a :

- Des **noms de colonnes différents** (ex: "diagnosis" vs "status" vs "Label")
- Des **types de cibles différents** (classification binaire, régression, classification multi-classes)
- Des **colonnes d'ID** parfois présentes, parfois absentes
- Des **valeurs manquantes** encodées différemment (`?`, `NaN`, `nan`, etc.)

**Solution :** `DatasetWrapper` **unifie** tous ces datasets dans une interface commune.

#### 📋 **Structure de Données - `datasets_dict`**

```python
datasets_dict = {
    "breast_cancer": {
        "path": "breast_cancer.csv",          # Chemin relatif du fichier
        "target": "diagnosis",                 # Nom de la colonne cible
        "id_col": "id",                        # Colonne d'identifiant (à exclure)
        "type": "Categorical",                 # Type de tâche
    },
    "concrete": {
        "path": "concrete_data.csv",
        "target": "Strength",
        "type": "Continuous",                  # Pas d'id_col ici
    },
    # ... 8 autres datasets
}
```

**Points clés :**

- `target` : identifie la variable Y à prédire
- `id_col` : optionnel, pour exclure les colonnes non-informatives
- `type` : `"Categorical"` (classification) ou `"Continuous"` (régression)

#### 🔍 **Fonctionnement de la Classe `DatasetWrapper`**

```python
class DatasetWrapper:
    def __init__(self, name):
        # 1. Récupérer la configuration du dataset
        config = datasets_dict[name]

        # 2. Charger le CSV avec gestion intelligente des valeurs manquantes
        self.df = pd.read_csv(
            "datasets/" + config["path"],
            na_values=["?", "nan", "NaN", ""]  # Unification
        )

        # 3. Supprimer les duplicatas
        self.df = self.df.drop_duplicates()

        # 4. Identifier automatiquement les variables quantitatives
        all_numerics = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_exclude = [self.target]
        if self.id_col:
            cols_to_exclude.append(self.id_col)

        self.quantitatives_variables = [
            c for c in all_numerics if c not in cols_to_exclude
        ]

        # 5. Identifier les variables catégorielles (par déduction)
        all_columns = self.df.columns.tolist()
        self.categorical_variables = [
            c for c in all_columns
            if (c not in self.quantitatives_variables and c not in cols_to_exclude)
        ]
```

#### ✅ **Avantages de cette Architecture**

| Problème                         | Solution DatasetWrapper                                  |
| -------------------------------- | -------------------------------------------------------- |
| **10 datasets différents**       | Interface unifiée : `wrapper.df`, `wrapper.target`, etc. |
| **Types de variables variés**    | Classification automatique en quantitatif/catégoriel     |
| **Valeurs manquantes multiples** | Normalisation lors du chargement (`na_values=...`)       |
| **Colonnes ID parasites**        | Exclusion automatique via `id_col`                       |
| **Code dupliqué**                | Un seul code pour tous les datasets                      |

#### 🔄 **Utilisation Pratique**

```python
# Au lieu de :
# df1 = pd.read_csv("breast_cancer.csv")
# target1 = "diagnosis"
# X1 = df1.drop(columns=["id", "diagnosis"])
# df2 = pd.read_csv("concrete.csv")
# target2 = "Strength"
# X2 = df2.drop(columns=["Strength"])
# ... répéter 10 fois

# Avec DatasetWrapper :
wrapper = DatasetWrapper("breast_cancer")
# wrapper.df         → DataFrame nettoyé
# wrapper.target     → "diagnosis"
# wrapper.quantitatives_variables → ['radius_mean', 'texture_mean', ...]
# wrapper.categorical_variables   → []
```

**Impact :** Code **10x plus court** et **maintenable**.

---

### 2️⃣ `data_understanding.py` - L'Explorateur Visuel

#### 🎯 **Rôle et Utilité**

Ce script implémente la **Phase 2 du CRISP-DM** (Data Understanding). Il génère automatiquement un **rapport d'analyse exploratoire complet** pour chaque dataset.

#### 📊 **Fonction Principale : `understand_data(wrapper)`**

**Entrée :** Un objet `DatasetWrapper`  
**Sortie :** Affichage de 7 analyses + visualisations

#### 🔬 **Les 7 Étapes d'Analyse**

##### **1. Statistiques Descriptives Basiques**

```python
print(f"Number of Rows: {len(df)}")           # Ex: 569 (breast_cancer)
print(f"Number of Columns: {df.shape[1]}")   # Ex: 31 colonnes
```

**Utilité :** Comprendre la taille du dataset (important pour RLT : n^(1/3) = nmin)

##### **2. Classification des Variables**

```python
print("Qualitatives Columns:")
for col in categorical_variables:
    print(col)  # Ex: aucune pour breast_cancer

print("Quantitatives Columns:")
for col in quantitatives_variables:
    print(col)  # Ex: radius_mean, texture_mean, ...
```

**Utilité :** RLT ne traite que les variables numériques (pour l'instant)

##### **3. Identification de la Cible**

```python
print(f"Target: {target_variable} (Type: {type_target})")
# Ex: diagnosis (Type: Categorical)
```

**Utilité :** Détermine si on fait de la classification ou régression

##### **4. Détection des Valeurs Manquantes**

```python
missing_pct = (df.isna().sum() / len(df)) * 100
for name, val in zip(missing_pct.index, missing_pct):
    if val != 0:
        print(f"{name}: {val:.2f}% missing")
```

**Utilité :** Décider de la stratégie d'imputation (KNN dans data_preparation)

##### **5. Détection des Duplicatas**

```python
duplicated_pct = (df.duplicated().sum() / len(df)) * 100
print(f"{duplicated_pct:.2f}% duplicated rows")
```

**Utilité :** Les duplicatas sont déjà supprimés dans `DatasetWrapper`

##### **6. Détection des Outliers (Méthode IQR)**

```python
def has_outliers_iqr(data_column):
    q1 = data_column.quantile(0.25)
    q3 = data_column.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = (data_column < lower_bound) | (data_column > upper_bound)
    return outliers.any()
```

**Visualisation :** Boxplots pour chaque variable avec outliers  
**Utilité :** RLT est **robuste aux outliers** (arbres de décision), mais bon à documenter

##### **7. Visualisations Complètes**

**a) Histogrammes (Variables Quantitatives)**

```python
def plot_quantitative_histogram(df, col, bins=30):
    sns.histplot(data=df, x=col, bins=bins, kde=True, color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.show()
```

**Utilité :** Voir si distributions sont normales, bimodales, etc.

**b) Countplots (Variables Catégorielles + Cible)**

```python
def plot_qualitative_countplot(df, col, top_n=10):
    category_counts = df[col].value_counts().nlargest(top_n)
    sns.countplot(data=df, y=col, order=category_counts.index)
    plt.show()
```

**Utilité :** Détecter les **déséquilibres de classes** (important pour classification)

**c) Pairplot (Relations 2D)**

```python
def plot_pairplot(df, quant_vars, target, max_features=10):
    cols = quant_vars[:max_features] + [target]
    sns.pairplot(df[cols], hue=target, diag_kind='kde')
    plt.show()
```

**Utilité :** Visualiser les **corrélations** et **séparabilité** des classes

#### 🎨 **Exemple de Sortie (Breast Cancer)**

```
================================================================================
📌 ANALYZING DATASET: BREAST_CANCER
================================================================================

Number of Rows: 569
Number of Columns: 31

Qualitatives Columns: None
Quantitatives Columns: 30 variables (radius_mean, texture_mean, ...)

Target: diagnosis (Type: Categorical)

No missing values

No duplicated values

Outliers detected in 15 features: [radius_mean, area_mean, ...]
[📊 Boxplots affichés]

[📈 Histogrammes pour 5 premières variables]
[📊 Countplot pour la cible : M=212, B=357]
[🔗 Pairplot des 10 premières variables colorées par diagnosis]
```

#### 🚀 **Optimisations Implémentées**

1. **Limitation des graphiques :**

   - Max 10 outliers boxplots (éviter surcharge)
   - Max 5 histogrammes
   - Max 10 variables dans pairplot

2. **Gestion des erreurs :**

   - `try/except` pour pairplot (peut crasher avec trop de catégories)
   - Vérification `dropna()` avant calcul IQR

3. **Performance :**
   - `plt.close('all')` pour libérer mémoire
   - Pas de calculs inutiles sur colonnes vides

---

### 3️⃣ `data_preparation.py` - Le Nettoyeur Intelligent

#### 🎯 **Rôle et Utilité**

Ce script implémente la **Phase 3 du CRISP-DM** (Data Preparation). Il transforme les données **brutes** en données **prêtes pour l'entraînement**.

#### 🔧 **Fonction Principale : `prepare_data(wrapper)`**

**Pipeline en 6 Étapes :**

```
Données brutes (wrapper.df)
        ↓
[1] Extraction X, y
        ↓
[2] Suppression colonnes > 60% missing
        ↓
[3] Suppression lignes > 50% missing
        ↓
[4] Split Train/Test (80/20)
        ↓
[5] Imputation KNN (k=5)
        ↓
[6] Standardisation (μ=0, σ=1)
        ↓
X_train, X_test, y_train, y_test
```

#### 📋 **Détail des Étapes**

##### **Étape 1 : Extraction X et y**

```python
X = df[wrapper.quantitatives_variables].copy()  # Features
y = df[wrapper.target].copy()                   # Target
```

**Pourquoi `.copy()` ?** Éviter les `SettingWithCopyWarning` de pandas

##### **Étape 2 : Suppression des Colonnes Trop Manquantes**

```python
missing_pct = X.isnull().mean() * 100
cols_to_drop = missing_pct[missing_pct > 60].index
X_clean = X.drop(columns=cols_to_drop)
```

**Seuil :** 60% de valeurs manquantes  
**Justification :** Une colonne avec >60% de NaN apporte peu d'information

##### **Étape 3 : Suppression des Lignes Trop Manquantes**

```python
row_missing_pct = X_clean.isnull().mean(axis=1)
X_clean = X_clean[row_missing_pct <= 0.5]
y_clean = y.loc[X_clean.index]  # ⚠️ Synchroniser y avec X
```

**Seuil :** 50% de valeurs manquantes  
**Important :** Toujours synchroniser `y` avec les lignes conservées de `X`

##### **Étape 4 : Split Train/Test Stratifié**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean,
    test_size=0.2,      # 80% train, 20% test
    shuffle=True,       # Mélange aléatoire
    random_state=42     # Reproductibilité
)
```

**Pourquoi 80/20 ?** Standard pour datasets de taille moyenne (n=200-1000)  
**Note :** Pour RLT, on pourrait ajouter `stratify=y_clean` pour classification

##### **Étape 5 : Imputation par K-Nearest Neighbors**

```python
imputer = KNNImputer(n_neighbors=5)
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)  # ⚠️ Pas de fit sur test !
```

**Pourquoi KNN plutôt que médiane/moyenne ?**

- KNN capture les **relations locales** entre variables
- Médiane = imputation naïve (ignore corrélations)

**Exemple :** Si `age=NaN` mais `height=180cm, weight=80kg`, KNN trouvera les 5 personnes les plus similaires et utilisera leur `age` moyen.

**⚠️ Data Leakage Prevention :** `fit()` sur train uniquement, `transform()` sur test

##### **Étape 6 : Standardisation (Scaling)**

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)
```

**Formule :**  
$$X_{scaled} = \frac{X - \mu}{\sigma}$$

**Pourquoi standardiser ?**

- **Arbres de décision (RLT) :** Techniquement pas nécessaire (splits invariants à l'échelle)
- **Mais utile pour :**
  - Comparaison avec modèles linéaires (Lasso, Ridge)
  - Stabilité numérique des calculs de distance (si on ajoute des méthodes basées distance)
  - Interprétabilité des coefficients (Linear Combination Splits dans RLT)

**⚠️ Data Leakage Prevention :** Utiliser `μ` et `σ` du **train** sur le **test**

#### 📊 **Sortie Console**

```python
print(f"Shape of X_train: {X_train_scaled.shape}")  # Ex: (455, 30)
print(f"Shape of X_test: {X_test_scaled.shape}")    # Ex: (114, 30)
print(f"Shape of y_train: {y_train.shape}")         # Ex: (455,)
print(f"Shape of y_test: {y_test.shape}")           # Ex: (114,)
```

---

### 4️⃣ Répertoire `src/RLT/` – Cœur de l'Algorithme RLT

#### 🎯 **Rôle global et lien avec les objectifs (BO/DSO)**

Le dossier `src/RLT/` regroupe l'implémentation incrémentale du modèle demandé par le **BO1** (« Réimplémenter la stratégie ») et son pendant technique **DSO1** (« Implémenter l'algorithme RLT »). Les spécialisations y préparent aussi les comparaisons de **DSO2** (benchmarks classification/régression) et les futurs modules d'explicabilité (BO3/DSO3) en conservant une structure de nœuds riche en méta-données.

| Fichier                | Responsabilité principale                                                                                                                                                  | Objectifs couverts                                                                                                 |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `EmbeddedModel.py`     | Implémentation de l'« embedded model » (Extra Trees) utilisé à chaque nœud pour estimer l'importance des variables et alimenter la stratégie look-ahead.                  | BO1 / DSO1 – innovation n°1 (renforcement via variable importance) ; prépare les métriques utilisées pour la muting list (BO4/DSO4). |
| `Node.py`              | Définition de la structure minimale d'un nœud RLT (enfants, statistiques, récompense estimée).                                                                             | BO1 / DSO1 – supporte les trois innovations RLT (look-ahead, muting, combinaisons linéaires).                      |
| `RLT.py`               | Classe de base factorisant la construction d'arbres, la stratégie de renforcement et l'orchestration des sous-modules (muting, splits linéaires, politique de récompense). | BO1 / DSO1 – colonne vertébrale de l'implémentation ; prépare l'instrumentation pour BO3 (stockage des décisions). |
| `RLTClassification.py` | Spécialisation de `RLT` pour les tâches supervisées discrètes : calcul de récompenses basé sur la pureté, gestion du vote majoritaire, métriques de gain.                  | BO1 & DSO1 – scénarios de classification ; BO2 / DSO2 – comparaisons avec RF, GBM, XGB.                            |
| `RLTRegression.py`     | Spécialisation de `RLT` pour les cibles continues : erreurs quadratiques, moyenne locale, ajustement des coefficients de combinaisons linéaires.                           | BO1 & DSO1 – scénarios de régression ; BO2 / DSO2 – benchmarks continus.                                           |

#### 🧩 **Interactions internes**

```mermaid
flowchart TD
    A[DatasetWrapper / Data Preparation] --> |X_train, y_train| B[RLTClassification]
    A --> |X_train, y_train| C[RLTRegression]
    B --> D{Node}
    C --> D
    B --> E[RLT (Base)]
    C --> E
    E --> D
```

- **`EmbeddedModel.py`** encapsule un mini-modèle *Extremely Randomized Tree* (classification ou régression) exécuté avec bagging/OOB pour produire des scores d'importance par permutation. Ces scores alimentent directement `RLT.select_action` (look-ahead) et mettent à jour la liste des variables actives pour le muting.
- **`Node.py`** centralise les attributs utilisés par les stratégies RLT : profondeur, sous-échantillons, récompenses Q(s, a) et liste des variables actives (utile pour le **Progressive Muting**, donc BO1/innovation n°2).
- **`RLT.py`** encapsule les trois composantes clés décrites dans le notebook (look-ahead, muting, combinaisons linéaires) et expose des hooks (`select_action`, `evaluate_reward`, `update_policy`) que les variantes utilisent. Cette classe servira d'ancrage aux modules d'explicabilité (BO3/DSO3) en conservant l'historique des décisions.
- **`RLTClassification.py`** implémente les heuristiques propres aux scénarios 1 & 2 de l'article (cibles binaires / multi-classes) et prépare la pipeline de comparaison demandée dans BO2/DSO2 (scores d'exactitude, log-loss, courbes ROC).
- **`RLTRegression.py`** couvre les scénarios 3 & 4 (signal linéaire / interaction continue) en fournissant la logique MSE/RMSE et la sélection de variables pour les combinaisons linéaires ; indispensable pour comparer RLT à GBM/XGBoost sur les jeux continus (BO2/DSO2).

#### 🛠️ **Utilisation prévue dans le notebook (`notebook.ipynb`)**

1. **Phase 3 → Phase 4** : après la préparation des données, le notebook instanciera `RLTClassification` ou `RLTRegression` selon le type de `wrapper.type_target`.
2. **BO1 / DSO1 (Cellules Phase 4)** : les expérimentations sur scénarios synthétiques utiliseront directement ces classes pour construire un ensemble de M arbres (`M=100` défini dans le plan DSO1).
3. **BO2 / DSO2** : lors des comparaisons avec RF/GBM/XGB, `RLTClassification` et `RLTRegression` fourniront les métriques RLT et enregistreront le temps d'entraînement pour aligner le tableau de benchmark prévu.
4. **BO3 / DSO3** : les objets `Node` maintiennent les variables sélectionnées à chaque profondeur, information réutilisable pour générer importances globales et explications locales.

> 📌 **À retenir :** le répertoire `src/RLT/` est la pièce maîtresse qui connecte les données préparées (Phases 2-3) aux objectifs métiers BO1-BO4 définis dans le notebook. Chaque fichier y contribue en séparant clairement la logique structurelle (nœuds), la mécanique générique (classe de base), le moteur d'importance embarqué et les spécialisations par type de cible.

#### 🗺️ **Prochaines étapes pour boucler DSO1**

1. **Brancher l'embedded model dans `RLT.py`** : sécuriser l'appel à `EmbeddedModel.get_variables_importances`, ajouter la gestion des seeds (`random_state`) et des cas limites (samples < `min_samples_split`).
2. **Programmer les 4 scénarios synthétiques** : créer un module `src/RLT/scenarios.py` (ou équivalent) qui génère les jeux contrôlés décrits dans la documentation (classification linéaire/non-linéaire, interactions corrélées, régression linéaire).
3. **Orchestration d'entraînement** : implémenter une fonction utilitaire (ex. `run_experiment(config)`) qui enchaîne préparation des données, apprentissage de `M=100` arbres RLT, application du muting progressif et des splits linéaires (`k` dans {1,2,5}).
4. **Logging des métriques** : consigner dans un dataframe les scores (accuracy/MSE), le temps d'entraînement, la profondeur moyenne et la taille effective des variables actives pour comparer aux résultats de l'article de référence.
5. **Préparer l'explicabilité (pont vers DSO3)** : enrichir `Node` pour stocker les importances renvoyées par l'embedded model à chaque split, facilitant la production des rapports XAI prévus.

#### 🎯 **Retour de Fonction**

```python
return X_train_scaled, X_test_scaled, y_train, y_test
```

Format compatible avec **scikit-learn** et futures implémentations RLT

---

## 🔄 Pipeline Complet dans le Notebook

### Phase 2 : Data Understanding (Cellule 11)

```python
for dataset_name in dataset_wrapper.datasets_dict.keys():
    wrapped_ds = dataset_wrapper.DatasetWrapper(dataset_name)
    data_understanding.understand_data(wrapped_ds)
```

**Ce qui se passe :**

1. Boucle sur les 7 datasets configurés
2. Pour chaque dataset :
   - Chargement via `DatasetWrapper` (normalisation)
   - Génération du rapport complet (7 analyses + visualisations)
3. Sortie : ~50 graphiques + statistiques

### Phase 3 : Data Preparation (Cellule 13)

```python
for dataset_name in dataset_wrapper.datasets_dict.keys():
    wrapped_ds = dataset_wrapper.DatasetWrapper(dataset_name)
    _ = data_preparation.prepare_data(wrapped_ds)
```

**Ce qui se passe :**

1. Même boucle sur 7 datasets
2. Pour chaque dataset :
   - Pipeline de nettoyage (6 étapes)
   - Affichage des shapes finales
3. Sortie : 7 × 4 matrices (X_train, X_test, y_train, y_test)

**Note :** `_ =` signifie qu'on n'utilise pas encore les données (juste validation)

---

## 🧠 DSO1 : Implémentation RLT (Phase 4)

### 🎯 Objectif de DSO1

**Reproduire** les expériences du paper original sur **4 scénarios simulés** :

| Scénario | Type           | Variables Signal      | Variables Bruit | Particularité                  |
| -------- | -------------- | --------------------- | --------------- | ------------------------------ |
| **1**    | Classification | X₁, X₂                | p-2 (indép.)    | Linéaire simple                |
| **2**    | Classification | X₁, X₂                | p-2 (indép.)    | **Non-linéaire** (sin, exp)    |
| **3**    | Classification | X₅₀, X₁₀₀, X₁₅₀, X₂₀₀ | p-4 (corrélés)  | **Checkerboard** (interaction) |
| **4**    | Régression     | X₅₀, X₁₀₀, X₁₅₀       | p-3 (corrélés)  | Linéaire avec corrélation      |

**Tests :** p ∈ {200, 500, 1000} pour chaque scénario

### 🔧 Ce qu'il Faut Implémenter

#### **1. Générateur de Données Synthétiques**

```python
def generate_scenario(scenario_id, n=1000, p=500, random_state=42):
    """
    Génère un dataset selon les spécifications du paper.

    Args:
        scenario_id: 1, 2, 3, ou 4
        n: nombre d'échantillons
        p: dimension totale
        random_state: reproductibilité

    Returns:
        X: (n, p) array
        y: (n,) array (labels ou valeurs continues)
    """
    rng = np.random.default_rng(random_state)

    if scenario_id == 1:
        # Signal linéaire sur X₁, X₂
        X = rng.normal(size=(n, p))
        signal = X[:, 0] + X[:, 1]
        y = (signal > np.median(signal)).astype(int)

    elif scenario_id == 2:
        # Signal non-linéaire
        X = rng.normal(size=(n, p))
        signal = np.sin(X[:, 0]) + np.exp(0.1 * X[:, 1])
        y = (signal > np.median(signal)).astype(int)

    elif scenario_id == 3:
        # Checkerboard avec corrélation
        # Créer facteurs latents pour corrélation
        latent = rng.normal(size=(n, 10))
        loadings = rng.normal(size=(10, p))
        X = latent @ loadings + rng.normal(scale=0.1, size=(n, p))

        # Signal d'interaction (checkerboard)
        signal = (X[:, 49] > 0) != (X[:, 99] > 0)  # XOR pattern
        y = signal.astype(int)

    elif scenario_id == 4:
        # Régression linéaire avec corrélation
        latent = rng.normal(size=(n, 10))
        loadings = rng.normal(size=(10, p))
        X = latent @ loadings + rng.normal(scale=0.1, size=(n, p))

        # Signal linéaire
        y = X[:, 49] + X[:, 99] + X[:, 149] + rng.normal(scale=0.5, size=n)

    return X, y
```

#### **2. Classe RLT (Structure de Base)**

```python
class RLT:
    def __init__(self,
                 n_trees=100,              # M dans le paper
                 min_samples_leaf=None,    # nmin = n^(1/3)
                 muting_rate=0.5,          # p_d (0, 0.5, 0.8)
                 k_linear_comb=1,          # k pour linear combination (1, 2, 5)
                 embedded_model='ET',      # Extremely Randomized Trees
                 random_state=None):

        self.n_trees = n_trees
        self.min_samples_leaf = min_samples_leaf
        self.muting_rate = muting_rate
        self.k_linear_comb = k_linear_comb
        self.embedded_model = embedded_model
        self.random_state = random_state
        self.trees_ = []  # Liste d'arbres entraînés

    def fit(self, X, y):
        """Entraîner M arbres avec les 3 stratégies RLT"""
        n, p = X.shape

        # Calcul nmin si non fourni
        if self.min_samples_leaf is None:
            self.min_samples_leaf = int(n ** (1/3))

        for tree_idx in range(self.n_trees):
            # Bootstrap sample
            boot_indices = np.random.choice(n, size=n, replace=True)
            X_boot, y_boot = X[boot_indices], y[boot_indices]

            # Construire arbre avec stratégies RLT
            tree = self._build_tree(X_boot, y_boot, depth=0, muted_vars=set())
            self.trees_.append(tree)

        return self

    def _build_tree(self, X, y, depth, muted_vars):
        """Récursion pour construire un arbre RLT"""
        n, p = X.shape

        # Condition d'arrêt
        if n < self.min_samples_leaf or len(np.unique(y)) == 1:
            return {'type': 'leaf', 'value': np.mean(y)}

        # === STRATÉGIE 1 : REINFORCEMENT LEARNING (VI calculation) ===
        active_vars = [i for i in range(p) if i not in muted_vars]
        vi_scores = self._calculate_VI(X[:, active_vars], y)

        # === STRATÉGIE 3 : LINEAR COMBINATION SPLITS ===
        if self.k_linear_comb > 1:
            # Sélectionner top-k variables
            top_k_indices = np.argsort(vi_scores)[-self.k_linear_comb:]
            # Créer split linéaire : β₁X₁ + β₂X₂ + ... > 0
            split_var, split_val = self._linear_combination_split(
                X[:, active_vars[top_k_indices]], y, vi_scores[top_k_indices]
            )
        else:
            # Split classique sur 1 variable
            split_var = active_vars[np.argmax(vi_scores)]
            split_val = np.median(X[:, split_var])

        # Diviser les données
        left_mask = X[:, split_var] <= split_val

        # === STRATÉGIE 2 : PROGRESSIVE MUTING ===
        # Muter les variables avec faible VI
        threshold_vi = np.quantile(vi_scores, self.muting_rate)
        newly_muted = {active_vars[i] for i, vi in enumerate(vi_scores) if vi < threshold_vi}
        muted_vars_left = muted_vars.union(newly_muted)

        # Récursion
        left_child = self._build_tree(X[left_mask], y[left_mask], depth+1, muted_vars_left)
        right_child = self._build_tree(X[~left_mask], y[~left_mask], depth+1, muted_vars_left)

        return {
            'type': 'split',
            'var': split_var,
            'val': split_val,
            'left': left_child,
            'right': right_child
        }

    def _calculate_VI(self, X, y):
        """Calculer Variable Importance via embedded model (ET)"""
        from sklearn.ensemble import ExtraTreesClassifier

        # Entraîner petit modèle embarqué
        et = ExtraTreesClassifier(n_estimators=10, max_depth=5, random_state=42)
        et.fit(X, y)

        # VI = feature_importances_ de ExtraTrees
        return et.feature_importances_

    def _linear_combination_split(self, X_top_k, y, vi_scores):
        """Créer split de type β₁X₁ + ... + βₖXₖ > threshold"""
        # β_j = sign(corr(X_j, y)) × VI(j)
        beta = np.array([
            np.sign(np.corrcoef(X_top_k[:, j], y)[0, 1]) * vi_scores[j]
            for j in range(X_top_k.shape[1])
        ])

        # Calculer scores linéaires
        linear_scores = X_top_k @ beta
        threshold = np.median(linear_scores)

        return linear_scores, threshold  # Simplification (retourner scores)

    def predict(self, X):
        """Prédiction par vote majoritaire (classification) ou moyenne (régression)"""
        predictions = np.array([self._predict_tree(tree, X) for tree in self.trees_])

        # Vote majoritaire si classification, moyenne si régression
        if len(np.unique(predictions)) <= 10:  # Heuristique
            return np.round(np.mean(predictions, axis=0))
        else:
            return np.mean(predictions, axis=0)

    def _predict_tree(self, tree, X):
        """Prédiction pour un arbre unique (récursif)"""
        if tree['type'] == 'leaf':
            return np.full(X.shape[0], tree['value'])

        left_mask = X[:, tree['var']] <= tree['val']
        predictions = np.empty(X.shape[0])

        if left_mask.any():
            predictions[left_mask] = self._predict_tree(tree['left'], X[left_mask])
        if (~left_mask).any():
            predictions[~left_mask] = self._predict_tree(tree['right'], X[~left_mask])

        return predictions
```

#### **3. Script d'Expérimentation**

```python
# Dans notebook Phase 4 - DSO1

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

results = []

for scenario_id in [1, 2, 3, 4]:
    for p in [200, 500, 1000]:
        # Générer données
        X, y = generate_scenario(scenario_id, n=1000, p=p, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # === RLT avec différentes configs ===
        for muting in [0, 0.5, 0.8]:
            for k in [1, 2, 5]:
                rlt = RLT(n_trees=100, muting_rate=muting, k_linear_comb=k)
                rlt.fit(X_train, y_train)
                y_pred = rlt.predict(X_test)
                acc_rlt = accuracy_score(y_test, y_pred)

                results.append({
                    'scenario': scenario_id,
                    'p': p,
                    'model': f'RLT_muting{muting}_k{k}',
                    'accuracy': acc_rlt
                })

        # === Baseline Random Forest ===
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        acc_rf = accuracy_score(y_test, y_pred_rf)

        results.append({
            'scenario': scenario_id,
            'p': p,
            'model': 'RandomForest',
            'accuracy': acc_rf
        })

# Analyser résultats
results_df = pd.DataFrame(results)
pivot = results_df.pivot_table(
    index=['scenario', 'p'],
    columns='model',
    values='accuracy'
)
print(pivot)
```

#### **4. Résultats Attendus (selon Paper)**

**Scénario 3 (Checkerboard) avec p=1000 :**

- Random Forest : ~50-55% accuracy (comme hasard)
- RLT (k=2, muting=0.5) : ~85-90% accuracy ✅

**Pourquoi ?** RLT détecte l'interaction X₅₀ ⊕ X₁₀₀ grâce au VI + linear combination

---

## 📈 Métriques de Validation

### Pour DSO1 (Scénarios Simulés)

1. **Accuracy** (classification) ou **MSE** (régression)
2. **Comparaison RLT vs RF** : RLT devrait gagner sur scénarios 2-4
3. **Sensibilité à p** : performance RLT stable quand p↑, RF dégradé

### Pour DSO2 (Datasets Réels)

1. **Accuracy / MSE**
2. **Temps d'entraînement**
3. **Temps de prédiction**
4. **Stabilité (cross-validation)**

---

## 🎓 Concepts Clés à Retenir

### 1. DatasetWrapper : Le Pattern "Adapter"

**Problème :** 10 datasets → 10 formats différents  
**Solution :** 1 interface unifiée  
**Bénéfice :** Code réutilisable, maintenable

### 2. Data Understanding : Le Fondement

**Sans comprendre les données :**

- Impossible de choisir bon preprocessing
- Impossible d'interpréter résultats
- Risque de data leakage

### 3. Data Preparation : Le Pré-requis

**Sans préparation correcte :**

- Modèles instables (NaN → crash)
- Biais (data leakage)
- Mauvaise généralisation

### 4. RLT : L'Innovation

**3 Stratégies complémentaires :**

1. **VI (Reinforcement)** : Look-ahead pour trouver vrais signaux
2. **Muting** : Éliminer bruit progressivement
3. **Linear Comb** : Splits plus expressifs

---

## 🔮 Prochaines Étapes

### DSO2 : Benchmarking

- Implémenter RF, GBM, XGBoost avec mêmes données
- Comparer 4 métriques (accuracy, time, memory, stability)

### DSO3 : Explicabilité

- SHAP values pour RLT
- Feature importance global
- LIME pour prédictions locales

### DSO4 : Optimisation

- Hyperparameter tuning (GridSearch)
- Parallélisation (joblib)
- Optimisation mémoire (sparse arrays)

---

## 📚 Références

1. **Paper Original RLT :** Zhu, R., Zeng, D., & Kosorok, M. R. (2015). _Reinforcement Learning Trees_
2. **CRISP-DM :** Cross-Industry Standard Process for Data Mining
3. **Scikit-learn :** Documentation pour RandomForestClassifier, StandardScaler, etc.

---

**📝 Note Finale :** Ce document sera mis à jour au fur et à mesure de l'avancement du projet. Chaque modification majeure du code devrait être reflétée ici.
