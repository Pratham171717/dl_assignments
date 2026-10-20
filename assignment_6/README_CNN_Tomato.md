# CNN for Tomato Leaf Disease Classification

A Convolutional Neural Network (CNN) that classifies tomato leaf images into 10 disease / healthy classes.

## Problem Statement
Design and implement a Convolutional Neural Network (CNN) for image classification using the Tomato or Soybean disease dataset.

## Dataset
- **Source:** [PlantVillage dataset](https://github.com/spMohanty/PlantVillage-Dataset) (`raw/color`), cloned automatically in the notebook
- **Classes used:** the 10 Tomato classes (Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites, Target Spot, Yellow Leaf Curl Virus, Mosaic virus, healthy)
- **Size:** 18,160 images, split 80% training (14,528) / 20% validation (3,632)
- **Preprocessing:** resized to 128 × 128, pixel values scaled to 0–1

## Model Architecture
| Layer | Details |
|---|---|
| Conv2D + MaxPooling | 32 filters, 3 × 3, ReLU, 2 × 2 pool |
| Conv2D + MaxPooling | 64 filters, 3 × 3, ReLU, 2 × 2 pool |
| Conv2D + MaxPooling | 128 filters, 3 × 3, ReLU, 2 × 2 pool |
| Flatten | |
| Dense | 128 units, ReLU |
| Dropout | 0.5 |
| Dense | 10 units, softmax |

Total parameters: 3,305,930. Training: Adam optimizer, sparse categorical cross-entropy, batch size 32, 3 epochs.

## Requirements
- Python 3, TensorFlow / Keras
- NumPy, pandas, matplotlib, seaborn, scikit-learn
- Google Colab with GPU recommended (the dataset clone is about 2 GB)

## How to Run
1. Open `CNN.ipynb` in Google Colab.
2. Switch to a GPU: **Runtime → Change runtime type → T4 GPU**.
3. Click **Runtime → Run all**. The dataset clone and copying take a few minutes.

## Results
| Epoch | Train Acc | Train Loss | Val Acc | Val Loss |
|---|---|---|---|---|
| 1 | 0.5891 | 1.2160 | 0.7971 | 0.6363 |
| 2 | 0.7761 | 0.6564 | 0.8544 | 0.4156 |
| 3 | 0.8287 | 0.5032 | 0.8824 | 0.3364 |

**Final validation accuracy: 88.24%** (3,205 of 3,632 images correct). Macro F1 = 0.84, weighted F1 = 0.88.

## Observations
- Best classes: Yellow Leaf Curl Virus (F1 0.97), healthy (0.95), Bacterial spot (0.92).
- Weakest classes: Early blight (recall 0.49), Target Spot (F1 0.78), Leaf Mold (F1 0.79).
- Accuracy and loss were still improving at epoch 3; the model was trained for only 3 epochs because each epoch took about 9 minutes.

## Possible Improvements
- Train for more epochs and add data augmentation.
- Handle class imbalance (Yellow Leaf Curl Virus has 5,357 images, Mosaic virus only 373).
- Try transfer learning (e.g. ResNet50, EfficientNet).

## Files
- `CNN.ipynb`: notebook with code and outputs
- `Assignment_CNN_Tomato_Disease.docx`: assignment report
