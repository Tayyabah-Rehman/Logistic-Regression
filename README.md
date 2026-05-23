<div style="background: linear-gradient(135deg, #1a1a2e, #0f3460); border-radius: 12px; padding: 2rem; color: white; font-family: monospace;">

<h1 style="color:#fff; margin:0 0 6px 0;">🚢 Titanic Survival Prediction</h1>
<h3 style="color:#9FE1CB; margin:0 0 12px 0; font-weight:400;">Logistic Regression — Binary Classification Pipeline</h3>
<p style="color:rgba(255,255,255,0.55); margin: 0 0 18px 0;">Machine Learning · Task 4 &nbsp;|&nbsp; Tayyabah Rehman &nbsp;|&nbsp; MPhil Artificial Intelligence &nbsp;|&nbsp; May 2026</p>

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-2ea44f?style=for-the-badge)

</div>

---

## 📋 Project Summary

| | |
|---|---|
| **Dataset** | Titanic preprocessed (from Task 2) — 891 samples, 12 features |
| **Goal** | Predict passenger survival using Logistic Regression (binary classification) |
| **Input** | `titanic_preprocessed.csv` — scaled, encoded, feature-engineered |
| **Output** | Trained model + metrics + confusion matrix + ROC curve |
| **Split** | 80% Train (712 samples) / 20% Test (179 samples) |

---

## 📊 Model Results

| Metric | Score | Meaning |
|--------|-------|---------|
| ✅ **Accuracy** | `0.7989` | Model correctly predicted 79.9% of cases |
| 🎯 **Precision** | `0.7568` | 75.7% of predicted survivors were actual survivors |
| 🔍 **Recall** | `0.7568` | Found 75.7% of all actual survivors |
| ⚖️ **F1-Score** | `0.7568` | Balanced precision-recall score |
| 📈 **AUC** | `0.8902` | 89.0% ability to distinguish survivors from non-survivors |

### Confusion Matrix Breakdown

| | Predicted: 0 | Predicted: 1 |
|---|---|---|
| **Actual: 0** | ✅ TN = 87 | ❌ FP = 18 |
| **Actual: 1** | ❌ FN = 18 | ✅ TP = 56 |

---

## 🗺️ Pipeline Overview

```
Preprocessed CSV  →  Define Features & Target  →  Train/Test Split (80/20)
                 →  Train LogisticRegression    →  Predict on Test Set
                 →  Calculate Metrics           →  Plot Confusion Matrix + ROC Curve
```

---

## 📦 Dataset Source

> Uses the preprocessed Titanic dataset output from Task 2 — no raw data cleaning needed.

```python
df = pd.read_csv('/content/sample_data/titanic_preprocessed.csv')
# Shape: (891, 13)
# Features: Pclass, Age, SibSp, Parch, Fare, FamilySize,
#           Sex_female, Sex_male, Embarked_C, Embarked_Q, Embarked_S, Title_encoded
# Target: Survived (0 = No, 1 = Yes)
```

---

## 🔧 Methodology

### 1️⃣ Features & Target Split
```python
X = df.drop('Survived', axis=1)   # Shape: (891, 12)
y = df['Survived']
# Class distribution: 0 → 549  |  1 → 342
```

### 2️⃣ Train / Test Split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Train: 712 samples | Test: 179 samples
```

### 3️⃣ Train Logistic Regression
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
# Intercept: 2.9218
```

### 4️⃣ Predictions
```python
y_pred       = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # probabilities for ROC
```

### 5️⃣ Performance Metrics
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy  = accuracy_score(y_test, y_pred)   # 0.7989
precision = precision_score(y_test, y_pred)  # 0.7568
recall    = recall_score(y_test, y_pred)     # 0.7568
f1        = f1_score(y_test, y_pred)         # 0.7568
```

### 6️⃣ Confusion Matrix
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
# TN=87 | FP=18 | FN=18 | TP=56
```

### 7️⃣ ROC Curve & AUC
```python
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc_score   = roc_auc_score(y_test, y_pred_proba)  # 0.8902

plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier')
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
```

---

## 🔍 Interpretation

- **Accuracy = 79.9%** — Strong overall performance for a baseline logistic model on this dataset
- **Precision = Recall = F1 = 0.757** — Perfectly balanced; the model is equally good at avoiding false alarms and catching true survivors
- **AUC = 0.890** — Excellent discriminative ability; the model correctly ranks a survivor above a non-survivor 89% of the time
- **Confusion Matrix** — Equal FP and FN (both = 18) confirms the balanced precision/recall; no systematic bias toward over- or under-predicting survival

---

## 📁 Project Structure

```
logistic-regression-titanic/
│
├── Task4_ML_Tayyabah_Rehman.ipynb   # Main Jupyter Notebook
├── titanic_preprocessed.csv         # Input — output from Task 2
├── confusion_matrix.png             # Output plot (300 DPI)
├── roc_curve.png                    # Output plot (300 DPI)
├── Task4_Description.docx           # Written task description
└── README.md                        # This file
```

---

## 🛠️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/logistic-regression-titanic.git
cd logistic-regression-titanic
```

### 2. Install Required Libraries
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### `requirements.txt`
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

---

## ▶️ How to Run

**Option A — Jupyter Notebook**
```bash
jupyter notebook Task4_ML_Tayyabah_Rehman.ipynb
```
Run all cells: **Kernel → Restart & Run All**

**Option B — Google Colab**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Zso2o0lOVn-GxkEqf1VMLy9JRFAkilLn)

> Upload `titanic_preprocessed.csv` from Task 2 to `/content/sample_data/` before running.

---

## 🧰 Tools & Libraries

| Tool | Version | Purpose |
|------|---------|---------|
| `pandas` | ≥1.3 | Data loading and manipulation |
| `numpy` | ≥1.21 | Numerical operations |
| `scikit-learn` | ≥0.24 | Model training, splitting, all metrics |
| `matplotlib` | ≥3.4 | Plot rendering and saving |
| `seaborn` | ≥0.11 | Enhanced confusion matrix heatmap |
| `jupyter` | ≥1.0 | Interactive notebook environment |

---

## 👩‍💻 Author

**Tayyabah Rehman**
MPhil Artificial Intelligence · Machine Learning Task 4 · May 2026

---
