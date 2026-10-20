# BERT for Sentiment Analysis (IMDB)

Fine-tunes a pre-trained **`bert-base-uncased`** model for binary sentiment classification (positive / negative) on a sample of the IMDB movie-review dataset.

## Problem Statement
Fine-tune a pre-trained BERT model for sentiment classification on a sample of the IMDB dataset, evaluate it using Accuracy, Precision, Recall and F1-score, and test it on new sentences.

## Dataset
- **Source:** [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb) on Hugging Face (loaded automatically)
- **Sample used:** 2,000 training reviews (balanced, 1,000 per class) and 500 test reviews
- **Labels:** 0 = Negative, 1 = Positive

## Approach
1. Tokenize reviews with the BERT tokenizer (max length 256, dynamic padding).
2. Load `bert-base-uncased` with a new 2-class classification head.
3. Fine-tune with the Hugging Face `Trainer`: 2 epochs, batch size 16, learning rate 2e-5, weight decay 0.01, mixed precision (fp16).
4. Evaluate on the test set with Accuracy, Precision, Recall, F1, a classification report and a confusion matrix.
5. Predict the sentiment of new sentences with a text-classification pipeline.

## Requirements
- Python 3, PyTorch
- `transformers`, `datasets`, `scikit-learn`, `accelerate`, matplotlib
- Google Colab with a **T4 GPU** is recommended

## How to Run
1. Open `bert_imdb_sentiment_analysis.ipynb` in Google Colab.
2. Go to **Runtime → Change runtime type → T4 GPU**.
3. Click **Runtime → Run all** (training takes about one minute).

## Results

**Test set (500 reviews)**

| Metric | Score |
|---|---|
| Accuracy | 0.8940 |
| Precision | 0.8726 |
| Recall | 0.9187 |
| F1-score | 0.8950 |

**Per epoch**

| Epoch | Training Loss | Validation Loss | Accuracy | F1 |
|---|---|---|---|---|
| 1 | 0.3953 | 0.2852 | 0.8760 | 0.8808 |
| 2 | 0.1892 | 0.2876 | 0.8940 | 0.8951 |

**Sample predictions**

| Sentence | Prediction | Confidence |
|---|---|---|
| This movie was absolutely fantastic, I loved every minute of it! | POSITIVE | 97.42% |
| Terrible plot and awful acting. A complete waste of time. | NEGATIVE | 97.10% |
| The visuals were stunning, but the story felt a bit slow. | NEGATIVE | 66.71% |
| I would happily watch this film again with my family. | POSITIVE | 93.79% |
| Boring, predictable, and way too long. | NEGATIVE | 94.17% |

## Observations
- The model reached 89.4% accuracy after only 2 epochs on 2,000 training samples.
- Positive-class recall (0.92) is higher than precision (0.87): it rarely misses a positive review but sometimes labels a negative one as positive.
- Validation loss stayed near 0.286 while training loss kept falling, so more data or epochs would be needed to improve further.
- The mixed-sentiment sentence got much lower confidence (66.71%), as expected.

## Possible Improvements
- Increase `TRAIN_SIZE` / `TEST_SIZE` or use the full 25,000-review dataset.
- Train for more epochs and compare with other models such as DistilBERT or RoBERTa.

## Files
- `bert_imdb_sentiment_analysis.ipynb`: notebook with code and outputs
- `Assignment_BERT_Sentiment_Analysis.docx`: assignment report
