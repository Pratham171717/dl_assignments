# Multilayer Perceptron (MLP) for Wine Classification

A Multilayer Perceptron (scikit-learn `MLPClassifier`) that classifies wines into 3 classes from 13 chemical measurements, evaluated with accuracy and a confusion matrix.

## Problem Statement
Design and implement a Multilayer Perceptron (MLP) for classification of the Iris or Wine dataset, and evaluate its performance using accuracy and a confusion matrix.

## Dataset
- **Source:** Wine recognition dataset from scikit-learn (`load_wine`), originally from the UCI Machine Learning Repository
- **Size:** 178 samples, 13 numeric features (alcohol, malic acid, ash, flavanoids, proline, etc.), 3 classes (class_0: 59, class_1: 71, class_2: 48)
- **Quality:** no missing values and no duplicate rows

## Approach
1. Explore the data: statistics, class distribution, histograms, box plot, pair plot, correlation heatmap and a scatter plot.
2. Split 80% / 20% with a stratified split (142 training / 36 testing samples, `random_state=42`).
3. Standardize the features with `StandardScaler` (fitted on the training data only).
4. Train the MLP: 13 inputs → 10 neurons (ReLU) → 10 neurons (ReLU) → 3 outputs, Adam optimizer, `max_iter=1000`, `random_state=42`.
5. Evaluate with accuracy, confusion matrix and classification report.

## Requirements
- Python 3
- NumPy, pandas, matplotlib, seaborn, scikit-learn
- Google Colab (runs in seconds, no GPU needed)

## How to Run
1. Open `assignment_2.ipynb` in Google Colab.
2. Click **Runtime → Run all**.

## Results (test set, 36 samples)
**Accuracy: 97.22%** (35 of 36 correct)

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| class_0 | 1.00 | 1.00 | 1.00 | 12 |
| class_1 | 0.93 | 1.00 | 0.97 | 14 |
| class_2 | 1.00 | 0.90 | 0.95 | 10 |
| Macro avg | 0.98 | 0.97 | 0.97 | 36 |

**Confusion matrix**

| Actual \ Predicted | class_0 | class_1 | class_2 |
|---|---|---|---|
| class_0 | 12 | 0 | 0 |
| class_1 | 0 | 14 | 0 |
| class_2 | 0 | 1 | 9 |

## Observations
- Only one wine was misclassified: a class_2 wine predicted as class_1.
- The training loss decreased smoothly and converged after roughly 475 iterations, well before the 1,000 limit.
- Flavanoids (-0.85), OD280/OD315 (-0.79) and total phenols (-0.72) are the features most strongly related to the wine class.
- The test set is small (36 samples), so one mistake changes the accuracy by about 2.8 percentage points.

## Possible Improvements
- Use cross-validation for a more reliable accuracy estimate.
- Tune the hidden layer sizes, activation and regularization (`alpha`).
- Compare the MLP with other classifiers (e.g. SVM, Random Forest).

## Files
- `assignment_2.ipynb`: notebook with code and outputs
- `Assignment_MLP_Wine_Classification.docx`: assignment report
