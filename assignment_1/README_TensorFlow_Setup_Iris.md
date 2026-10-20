# Installing and Configuring TensorFlow/Keras in Google Colab: Data Preprocessing and Visualization

Sets up TensorFlow/Keras in Google Colab and performs data exploration, normalization, train-test splitting and visualization on the Iris dataset.

## Problem Statement
Install and configure TensorFlow/Keras in Google Colab. Perform data preprocessing, normalization, train-test splitting, and data visualization on a sample dataset.

## Dataset
- **Source:** Iris dataset from scikit-learn (`load_iris`)
- **Size:** 150 samples, 4 numeric features (sepal length/width, petal length/width), 3 classes (species 0, 1, 2)
- **Quality:** no missing values

## Steps
1. Open Google Colab (TensorFlow and Keras come pre-installed) and print the TensorFlow version to confirm the setup.
2. Load the dataset into a pandas DataFrame and explore it (shape, data types, summary statistics, missing values).
3. Normalize the features with `StandardScaler` (mean 0, standard deviation 1).
4. Split the data 80% / 20% with `train_test_split` (`random_state=42`): 120 training and 30 testing samples.
5. Visualize the data: pair plot, correlation heatmap, histograms and box plot.
6. Build a small Keras model (`Dense(8, ReLU)` → `Dense(3, softmax)`, 67 parameters) and print its summary to confirm Keras works.

## Requirements
- Python 3, TensorFlow / Keras (2.20.0 in the notebook run)
- pandas, NumPy, scikit-learn, matplotlib, seaborn
- Google Colab (no GPU needed)

## How to Run
1. Open `Assignment_1.ipynb` in Google Colab.
2. Click **Runtime → Run all**.

## Results
| Item | Value |
|---|---|
| TensorFlow version | 2.20.0 |
| Dataset shape | (150, 4) |
| Training data | (120, 4) |
| Testing data | (30, 4) |
| Missing values | 0 |

## Observations
- The features have different ranges (e.g. petal length 1.0–6.9, petal width 0.1–2.5), so scaling was applied before training.
- Species 0 (setosa) is clearly separated from the other two species in the pair plot; species 1 and 2 overlap slightly.
- Petal length and petal width are very strongly correlated (0.96), and both are strongly correlated with the species (0.95 and 0.96).

## Possible Improvements
- Fit the scaler on the training data only (after the split) to avoid using test-set information.
- Train the model and evaluate it on the test set.

## Files
- `Assignment_1.ipynb`: notebook with code and outputs
- `Assignment_Install_Configure_TensorFlow.docx`: assignment report
