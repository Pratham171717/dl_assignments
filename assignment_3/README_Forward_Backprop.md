# Forward Propagation and Backpropagation with TensorFlow/Keras: Effect of Learning Rate and Epochs

A neural network trained on Fashion-MNIST to demonstrate forward propagation and backpropagation, and to analyze how the learning rate and the number of epochs affect performance.

## Problem Statement
Implement forward propagation and backpropagation using TensorFlow/Keras. Analyze the effect of different learning rates and the number of epochs on model performance.

## Dataset
- **Source:** Fashion-MNIST, loaded from `keras.datasets.fashion_mnist`
- **Size:** 60,000 training and 10,000 test images (28 × 28 grayscale), 10 clothing classes
- **Preprocessing:** pixel values divided by 255 (range 0–1); 20% of the training data used for validation

## Model Architecture
`Flatten(784)` → `Dense(128, ReLU)` → `Dense(64, ReLU)` → `Dense(10, softmax)` (109,386 parameters)

- **Forward propagation:** each layer computes `activation(weights × input + bias)`; the softmax layer outputs the class probabilities, and the loss is sparse categorical cross-entropy.
- **Backpropagation:** the gradients of the loss are computed from the output layer back to the input layer (chain rule), and the Adam optimizer updates the weights. Keras does both automatically in every batch of `model.fit()`.

## Requirements
- Python 3, TensorFlow / Keras
- NumPy, pandas, matplotlib, seaborn, scikit-learn
- Google Colab (a GPU is optional)

## How to Run
1. Open `DL_3.ipynb` in Google Colab.
2. Click **Runtime → Run all** (takes a few minutes).

## Results
**Main model** (Adam, learning rate 0.001, 10 epochs): **test accuracy 88.34%**, test loss 0.3497, validation accuracy 88.99%.

**Effect of the learning rate** (10 epochs):

| Learning rate | Validation accuracy |
|---|---|
| 0.1 | 9.83% |
| 0.01 | 85.34% |
| 0.001 | **87.88%** |

**Effect of the number of epochs** (learning rate 0.001):

| Epochs | Validation accuracy |
|---|---|
| 5 | 87.38% |
| 10 | 88.69% |
| 20 | **89.12%** |

## Observations
- A learning rate of 0.1 was too large: the model could not learn and stayed at about chance level (10 classes). 0.001 gave the best and most stable result.
- More epochs improved accuracy, but with diminishing returns (+1.31 points from 5 to 10 epochs, +0.43 from 10 to 20) and a higher risk of overfitting. In the main run the validation loss stopped improving at about epoch 9 while the training loss kept falling.
- Best classes: Trouser (F1 0.98), Sandal (0.96). Weakest class: Shirt (F1 0.70), often confused with similar clothing such as T-shirt, Pullover and Coat.
- Results vary slightly between runs because of random weight initialization.

## Possible Improvements
- Add early stopping, dropout or regularization.
- Try a CNN, which usually performs better on image data.
- Test more learning rates between 0.01 and 0.0001, and learning-rate schedules.

## Files
- `DL_3.ipynb`: notebook with code and outputs
- `Assignment_Forward_Backpropagation.docx`: assignment report
