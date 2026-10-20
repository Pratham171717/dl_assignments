# RNN vs LSTM vs GRU for Sequence Classification

Compares three recurrent neural networks (SimpleRNN, LSTM and GRU) on IMDB movie-review sentiment classification.

## Problem Statement
Implement and compare RNN, LSTM, and GRU models for sequence classification, and analyze their performance using appropriate evaluation metrics.

## Dataset
- **Source:** IMDB movie reviews, loaded from `keras.datasets.imdb`
- **Size:** 25,000 training and 25,000 test reviews (balanced: 12,500 positive / 12,500 negative)
- **Preprocessing:** vocabulary limited to 10,000 words; every review padded or truncated to 200 words
- **Task:** binary classification (0 = negative, 1 = positive)

## Model Architecture
All three models share the same structure and differ only in the recurrent layer:

`Embedding(10000, 64)` → `SimpleRNN(64)` / `LSTM(64)` / `GRU(64)` → `Dense(1, sigmoid)`

Training: Adam optimizer, binary cross-entropy, batch size 64, 3 epochs, 20% of the training data used for validation.

## Requirements
- Python 3, TensorFlow / Keras
- NumPy, pandas, matplotlib, seaborn, scikit-learn
- Google Colab (a GPU speeds up training)

## How to Run
1. Open `LSTM_RNN_GRU.ipynb` in Google Colab.
2. Click **Runtime → Run all** (training takes about 8–10 minutes).

## Results (test set, 25,000 reviews)
| Model | Accuracy | Precision | Recall | F1 Score | Test Loss |
|---|---|---|---|---|---|
| RNN | 0.4974 | 0.4952 | 0.2718 | 0.3510 | 1.0860 |
| LSTM | 0.5493 | 0.5298 | 0.8760 | 0.6603 | 0.6800 |
| GRU | **0.8572** | **0.8328** | 0.8937 | **0.8622** | **0.3391** |

**Best model: GRU.**

## Observations
- GRU generalized best and learned fastest (validation accuracy 55% → 80% → 86%).
- RNN stayed near 50% (chance level) on validation and test data despite about 77% training accuracy, so it did not generalize.
- LSTM was biased toward the positive class (positive recall 0.88, negative recall 0.22) and had not converged after 3 epochs.
- These results apply to this setup only (3 epochs, padding at the end of each review). LSTM usually matches GRU with more epochs, so this should not be read as LSTM being generally worse.

## Possible Improvements
- Train for more epochs and use pre-padding (padding="pre").
- Try bidirectional LSTM/GRU, stacked layers, or a larger embedding size.

## Files
- `LSTM_RNN_GRU.ipynb`: notebook with code and outputs
- `Assignment_RNN_LSTM_GRU_Comparison.docx`: assignment report
