# Spam Email Classification – Project 1

This repository contains the code and report for **Project 1: Classification on the Spambase dataset**.  
The goal is to implement several classification models (including from-scratch implementations) and compare their performance using 5-fold cross-validation.

In this project, the term **“test data”** refers to the **held-out validation folds in 5-fold cross-validation**.  
For each fold, models are trained on the training subset and evaluated on the validation subset, and we report accuracy, precision, recall, and F1-score.

---

## 1. Dataset

We use the **Spambase** dataset from the UCI Machine Learning Repository.

- Source: UCI Spambase (downloaded programmatically using [`ucimlrepo`](https://pypi.org/project/ucimlrepo/))
- Number of samples: 4,601
- Number of features: 57 numeric features
- Target variable: `spam` vs `non-spam` (binary classification)

The dataset is automatically loaded in the notebook via:

```python
from ucimlrepo import fetch_ucirepo
spambase = fetch_ucirepo(id=94)
X_df = spambase.data.features
y_df = spambase.data.targets
```

No manual download is required.

---

## 2. Environment & Dependencies

- Python >= 3.8
- Required packages:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `ucimlrepo`

You can install all dependencies with:

```bash
pip install numpy pandas scikit-learn ucimlrepo
```

---

## 3. Code Structure

The project is implemented in a **single Jupyter Notebook**:

- `project1_spam.ipynb`  
  Main Jupyter Notebook for the project. It contains:
  - Data loading and preprocessing  
  - From-scratch implementations of KNN and Perceptron  
  - From-scratch implementation of Gaussian Naive Bayes (bonus)  
  - Definitions of other `scikit-learn` classifiers  
  - A unified 5-fold cross-validation evaluation function  
  - A performance comparison table and model selection  

This notebook is the only code file submitted and is fully executable on its own.

---

## 4. How to Run the Code

1. Make sure all dependencies are installed.
2. Open the notebook in Jupyter:

   ```bash
   jupyter notebook project1_spam.ipynb
   ```

3. Run all cells from top to bottom.  
   The notebook will:

   - Download and load the **Spambase** dataset using `ucimlrepo`
   - Standardize features using `StandardScaler`
   - Perform 5-fold cross-validation on all models
   - Print per-fold and averaged performance metrics
   - Display a summary table of all models sorted by F1-score

No additional test file (such as `test.csv`) is required to run the notebook.

---

## 5. Models Implemented

### 5.1 From-scratch implementations

1. **KNNClassifier (from scratch)**
   - Custom implementation of k-Nearest Neighbors
   - Uses Euclidean distance on standardized features
   - Majority voting over the `k` nearest neighbors

2. **PerceptronClassifier (from scratch)**
   - Binary linear classifier with labels mapped to {+1, -1} during training
   - Parameters updated using the classic Perceptron update rule
   - Predictions mapped back to {0, 1} (non-spam / spam)

3. **GaussianNB_Custom (from scratch, bonus)**
   - Gaussian Naive Bayes implementation
   - Computes per-class mean and variance for each feature
   - Uses log joint likelihood for numerical stability
   - Compared against `sklearn.naive_bayes.GaussianNB`

### 5.2 `scikit-learn` models

1. **Logistic Regression**
2. **SVM with RBF kernel**
3. **Random Forest**
4. **GaussianNB (sklearn version)**

### 5.3 Ensemble model (bonus)

- **Soft Voting Ensemble**
  - Combines Logistic Regression, SVM (RBF) and Random Forest
  - Uses soft voting based on predicted class probabilities
  - Aims to improve overall performance and robustness

---

## 6. Data Preprocessing

Before model training, we apply:

- **Standardization of features** using `StandardScaler`:

  ```python
  from sklearn.preprocessing import StandardScaler

  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)
  ```

- All models in this project are trained on the standardized features `X_scaled`.  
  This is especially important for distance-based models (KNN) and margin-based models (SVM, Perceptron).

---

## 7. Cross-Validation & Performance Evaluation

We use **5-fold cross-validation** with the following fixed configuration:

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
```

For each model, the following steps are performed:

1. Split the standardized data into 5 folds.
2. For each fold:
   - Train the model on 4 folds (training set).
   - Evaluate it on the remaining 1 fold (validation set).
3. Collect per-fold metrics:
   - Accuracy
   - Precision
   - Recall
   - F1-score
4. Compute the mean and standard deviation of each metric across folds.

The final results are summarized in a table, for example:

- `Acc_mean`, `Acc_std`
- `Prec_mean`, `Prec_std`
- `Rec_mean`, `Rec_std`
- `F1_mean`, `F1_std`

The **best model** is selected based on the **highest mean F1-score**.

---

## 8. Interpreting “Test Data”

In this project, **no separate external test set file (e.g., `test.csv`) is provided**.  
Instead, the term **“test data”** in the assignment is interpreted as the **held-out validation fold** in each round of 5-fold cross-validation.

Concretely:

- For each fold, the validation fold serves as the **test data** for that fold.
- We train on the remaining 4 folds and evaluate on this validation fold.
- Performance metrics reported in the summary table are computed over these validation folds.

This design ensures that:

- The notebook is **fully executable** without manual test file preparation.
- Performance is evaluated in a statistically sound way across different splits.

---

## 9. Reproducibility

To reproduce the reported results:

1. Use the same environment (Python version and package versions) as listed above.
2. Keep the same random seed in `KFold`:
   - `random_state=42`
3. Run the notebook from start to finish without modifying the data-splitting logic.

Because the data loading, preprocessing, and evaluation are all deterministic under this configuration, you should obtain the same (or extremely similar) performance numbers.

---
