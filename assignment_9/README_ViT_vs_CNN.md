# Image Classification with a Pre-trained Vision Transformer (ViT) vs a CNN

Transfer learning on CIFAR-10 with a pre-trained **ViT-B/16** (Vision Transformer) and a pre-trained **ResNet50** (CNN). The two models are compared on accuracy, loss, precision, recall, F1-score, training time, inference time and number of parameters.

## Problem Statement
Perform image classification using a pre-trained Vision Transformer (ViT) model and compare its performance with a CNN model.

## Dataset
- **Source:** CIFAR-10, loaded with `torchvision.datasets.CIFAR10` (downloaded automatically, about 170 MB)
- **Full size:** 50,000 training and 10,000 test images (32 × 32 RGB), 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Used here:** a random subset of 5,000 training and 2,000 test images, to keep training fast (change `TRAIN_SIZE` and `TEST_SIZE` in the notebook to use more)
- **Preprocessing:** resized to 224 × 224, random horizontal flip (training only), ImageNet normalization

## Models
| Model | Type | Pre-trained weights |
|---|---|---|
| ViT-B/16 | Vision Transformer (image split into 16 × 16 patches, self-attention) | ImageNet |
| ResNet50 | Convolutional neural network with residual connections | ImageNet |

Transfer learning: all pre-trained layers are frozen and only a new 10-class final layer is trained. Training: Adam optimizer (learning rate 1e-3), cross-entropy loss, batch size 64, 3 epochs, mixed precision on the GPU.

## Requirements
- Python 3, PyTorch, torchvision
- scikit-learn, pandas, matplotlib, seaborn
- Google Colab with a **T4 GPU** is recommended

## How to Run
1. Open `vit_vs_cnn_image_classification.ipynb` in Google Colab.
2. Go to **Runtime → Change runtime type → T4 GPU**.
3. Click **Runtime → Run all**.

## Results (test set, 2,000 images)
| Metric | ViT-B/16 | ResNet50 (CNN) |
|---|---|---|
| Test accuracy | **0.9400** | 0.8860 |
| Test loss | **0.1922** | 0.3912 |
| Precision (macro) | **0.9395** | 0.8855 |
| Recall (macro) | **0.9400** | 0.8859 |
| F1-score (macro) | **0.9396** | 0.8855 |
| Training time (s) | 65.26 | **39.24** |
| Inference time (ms/image) | 3.03 | **1.19** |
| Total parameters (M) | 85.81 | **23.53** |
| Trainable parameters | 7,690 | 20,490 |

## Observations
- ViT-B/16 was clearly more accurate (94.0% vs 88.6%) and had the higher F1-score on all ten classes. It was already at 93.2% test accuracy after the first epoch (ResNet50: 84.9%).
- The better accuracy costs more: ViT trained about 1.7× slower, was about 2.6× slower at inference and has about 3.6× more parameters.
- Both models found cat and dog the hardest classes (ViT: 22 cat/dog confusions; ResNet50: 39). ResNet50 also mixed up airplane with ship and automobile with truck (12 each).
- ResNet50 was still improving at epoch 3, so the gap could shrink with more epochs or fine-tuning.
- Limitations: only the final layer was trained, on 5,000 images for 3 epochs in a single run, and the ResNet50 weights (ImageNet V2) have a different training recipe from the ViT weights (ImageNet V1), so the comparison is indicative rather than exact.

## Possible Improvements
- Fine-tune the last few blocks of each model with a lower learning rate (e.g. 1e-5).
- Use the full CIFAR-10 dataset and more epochs.
- Add a CNN trained from scratch as a third baseline.

## Files
- `vit_vs_cnn_image_classification.ipynb`: notebook with code and outputs
- `Assignment_ViT_vs_CNN.docx`: assignment report
