# Transfer Learning for Cats vs Dogs Image Classification

Binary image classification (cat / dog) using transfer learning with four pretrained CNNs: **AlexNet, VGG16, ResNet50 and EfficientNetB0**. The models are trained and compared on accuracy, loss and training time.

## Problem Statement
Implement transfer learning for Cats vs Dogs image classification using AlexNet, VGG16, ResNet50 and EfficientNetB0. Train and evaluate the models, then compare their accuracy, loss and training performance.

## Dataset
- **Source:** [`microsoft/cats_vs_dogs`](https://huggingface.co/datasets/microsoft/cats_vs_dogs) on Hugging Face (loaded automatically)
- **Sample used:** 3,000 random images, split into 2,000 training and 1,000 validation images
- **Preprocessing:** resize to 224 × 224, random horizontal flip (training only), ImageNet normalization

## Approach
1. Load ImageNet-pretrained weights for each model.
2. Freeze all pretrained layers.
3. Replace the final layer with a new 2-class linear layer.
4. Train only the new layer: Adam optimizer, learning rate 1e-3, cross-entropy loss, batch size 32, 5 epochs, mixed precision.
5. Record train/validation loss and accuracy for every epoch, plus total training time.

## Requirements
- Python 3, PyTorch, torchvision
- `datasets` (Hugging Face), pandas, matplotlib
- Google Colab with a **T4 GPU** is recommended

## How to Run
1. Open `cats_vs_dogs_transfer_learning.ipynb` in Google Colab.
2. Go to **Runtime → Change runtime type → T4 GPU**.
3. Click **Runtime → Run all**.

## Results

| Model | Final Train Acc | Final Val Acc | Best Val Acc | Final Train Loss | Final Val Loss | Time (s) | Trainable Params |
|---|---|---|---|---|---|---|---|
| AlexNet | 0.9845 | 0.9560 | 0.9560 | 0.0418 | 0.1193 | 54.65 | 8,194 |
| VGG16 | 0.9940 | 0.9900 | 0.9900 | 0.0171 | 0.0308 | 61.70 | 8,194 |
| ResNet50 | 0.9990 | 0.9940 | 0.9940 | 0.0144 | 0.0221 | 57.16 | 4,098 |
| EfficientNetB0 | 0.9950 | 0.9940 | 0.9950 | 0.0228 | 0.0222 | 57.11 | 2,562 |

## Observations
- **EfficientNetB0** reached the highest best validation accuracy (99.5%), followed closely by **ResNet50** (99.4%) and **VGG16** (99.0%).
- **AlexNet** was the weakest (95.6% accuracy, highest validation loss).
- ResNet50 and EfficientNetB0 had the lowest validation loss (about 0.022).
- Training times were similar (55–62 s) because only the final layer was trained.
- **Conclusion:** ResNet50 and EfficientNetB0 give the best balance of accuracy and loss; EfficientNetB0 does so with the fewest trainable parameters.

## Possible Improvements
- Unfreeze the last few blocks and fine-tune with a lower learning rate (e.g. 1e-5).
- Train on the full dataset (about 23,000 images) for more epochs.

## Files
- `cats_vs_dogs_transfer_learning.ipynb`: notebook with code and outputs
- `Assignment_CatsVsDogs_TransferLearning.docx`: assignment report
