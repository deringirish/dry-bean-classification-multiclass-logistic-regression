# Dry Bean Variety Classification Using Multiclass Logistic Regression

[![Python 3.14](https://img.shields.io/badge/Python-3.14-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.9+-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![uv](https://img.shields.io/badge/managed%20by-uv-DE5FE9.svg?style=flat)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An applied machine learning case study that classifies **seven commercial varieties of dry beans (*Phaseolus vulgaris L.*)** using computer vision-derived morphological and shape features. The solution implements an end-to-end scikit-learn pipeline featuring feature standardization and regularized **Multinomial Logistic Regression**.

---

## 📌 Executive Summary & Key Results

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Test Accuracy** | **91.92%** | Evaluated on held-out 20% stratified test set (2,709 samples) |
| **5-Fold Stratified CV Accuracy** | **92.42% ± 0.35%** | Consistent performance across all training folds |
| **Training Accuracy** | **92.60%** | Tight generalization gap (< 0.70%), zero overfitting |
| **Macro-Average ROC-AUC** | **0.995** | Exceptional class separation across all decision thresholds |
| **Weighted-Average ROC-AUC** | **0.993** | Robust against class imbalance |
| **Best Performing Class** | **Bombay (100% P / 100% R)** | 100% precision & recall due to distinct large grain size |

---

## 🌾 Problem Context & Agricultural Domain

Quality inspection, purity verification, and variety sorting of dry bean seeds are vital for certified seed production and food processing. Because commercial bean cultivars command different market prices and exhibit distinct culinary characteristics (cooking times, canning properties), automated classification using digital imaging replaces subjective, labor-intensive manual sorting with high-throughput computer vision.

### Target Varieties (7 Classes)
1. **SEKER** (Round, spherical)
2. **BARBUNYA** (Oval, speckled pattern)
3. **BOMBAY** (Exceptionally large white bean)
4. **CALI** (Large kidney bean)
5. **DERMASON** (Small, flat white bean; largest class)
6. **HOROZ** (Long, cylindrical white bean)
7. **SIRA** (Small, flat, medium-light bean)

---

## 📊 Dataset & Feature Description

The dataset was acquired from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) (Koklu & Ozkan, 2020). It consists of **13,611 grain images** captured with a high-resolution camera, from which **16 geometric and morphological features** were extracted:

| Feature Name | Description |
| :--- | :--- |
| **Area ($A$)** | Number of pixels within the bean boundary |
| **Perimeter ($P$)** | Circumference length of the bean margin |
| **MajorAxisLength ($L$)** | Distance between the ends of the longest line through the bean |
| **MinorAxisLength ($l$)** | Longest line perpendicular to the major axis |
| **AspectRation ($K$)** | Ratio between major and minor axis lengths ($L / l$) |
| **Eccentricity ($e$)** | Eccentricity of the equivalent ellipse |
| **ConvexArea ($C$)** | Pixel area of the smallest convex polygon enclosing the bean |
| **EquivDiameter ($d$)** | Diameter of a circle with the same area ($\sqrt{4A / \pi}$) |
| **Extent ($Ex$)** | Ratio of bean pixels to pixels in the bounding box |
| **Solidity ($S$)** | Ratio of bean area to convex area ($A / C$) |
| **roundness ($R$)** | Circularity metric: $(4\pi A) / P^2$ |
| **Compactness ($CO$)** | Compactness index: $d / L$ |
| **ShapeFactor1 - 4** | Four dimension-invariant affine shape descriptors |
| **Class** | Ground-truth cultivar label (7 categories) |

---

## 🧠 Methodology

### 1. Data Cleaning & Deduplication
- **68 duplicate records (0.50%)** were identified and purged, leaving **13,543 unique grains** for analysis.
- Zero missing values were found across all 16 feature columns.

### 2. Exploratory Data Analysis & Multicollinearity
- **Extreme Multicollinearity:** Size dimensions (`Area`, `Perimeter`, `ConvexArea`, `EquivDiameter`) exhibit Pearson correlations $\ge 0.95$.
- **Collinearity Mitigation:** Handled via regularized **$L_2$ shrinkage penalty** in Logistic Regression, preventing variance inflation without loss of predictive signal.

### 3. Leakage-Free Pipeline Architecture
- Built with `sklearn.pipeline.make_pipeline`:
  ```python
  model = make_pipeline(
      StandardScaler(),
      LogisticRegression(max_iter=1000, random_state=42)
  )
  ```
- Fitting the `StandardScaler` strictly on `X_train` prevents information leakage from the test distribution.

### 4. Multiclass Formulation (Softmax)
$$P(y = k \mid \mathbf{x}) = \frac{\exp(\mathbf{w}_k^T \mathbf{x} + b_k)}{\sum_{j=1}^K \exp(\mathbf{w}_j^T \mathbf{x} + b_j)}$$

Optimized using the L-BFGS solver with categorical cross-entropy loss and $L_2$ regularization.

---

## 📈 Detailed Performance Breakdown

```text
              precision    recall  f1-score   support

    BARBUNYA     0.9328    0.8906    0.9112       265
      BOMBAY     1.0000    1.0000    1.0000       104
        CALI     0.9110    0.9417    0.9261       326
    DERMASON     0.9319    0.9069    0.9192       709
       HOROZ     0.9643    0.9435    0.9538       372
       SEKER     0.9341    0.9433    0.9387       406
        SIRA     0.8646    0.8842    0.8743       527

    accuracy                         0.9192      2709
   macro avg     0.9341    0.9300    0.9319      2709
weighted avg     0.9197    0.9192    0.9193      2709
```

### Key Insights from Confusion Matrix:
- **Bombay:** 100% precision and recall; easily separated on pixel area alone.
- **Horoz:** 96.4% precision; highly distinguishable by its pronounced elongation (Aspect Ratio).
- **Dermason vs. Sira Confusion:** 53 Dermason samples misclassified as Sira, and 41 Sira misclassified as Dermason. Both cultivars are small, round-to-flat white beans with overlapping morphological boundaries that challenge linear decision surfaces.

---

## 📂 Project Structure

```text
Dry Bean Variety Classification Using Multiclass Logistic Regression/
├── .gitignore                                      # Comprehensive Git ignore rules
├── .python-version                                 # Python runtime version pinning
├── pyproject.toml                                  # Dependency & uv environment configuration
├── README.md                                       # Project documentation
├── uv.lock                                         # Cryptographically pinned lockfile
│
├── data/
│   └── Dry_Bean_Dataset.xlsx                       # Original UCI Dry Bean dataset
│
└── notebook/
    └── dry_bean_multiclass_logistic_regression.ipynb # End-to-end executed & documented notebook
```

---

## 🚀 Getting Started

### Prerequisites
- [uv](https://github.com/astral-sh/uv) (recommended) or Python 3.10+

### Installation & Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/dry-bean-variety-classification.git
   cd "Dry Bean Variety Classification Using Multiclass Logistic Regression"
   ```

2. **Sync dependencies with uv:**
   ```bash
   uv sync
   ```

3. **Launch the Jupyter Notebook:**
   ```bash
   uv run jupyter notebook notebook/dry_bean_multiclass_logistic_regression.ipynb
   ```

---

## 📚 References & Citation

* **Dataset Source:**  
  Koklu, M., & Ozkan, I. A. (2020). *Multiclass classification of dry beans using computer vision and machine learning techniques.* Computers and Electronics in Agriculture, 174, 105507.  
  DOI: [10.1016/j.compag.2020.105507](https://doi.org/10.1016/j.compag.2020.105507)
* **UCI Machine Learning Repository:**  
  [https://archive.ics.uci.edu/dataset/602/dry+bean+dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)
