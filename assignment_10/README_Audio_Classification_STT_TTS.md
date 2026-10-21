# Audio Classification with Speech-to-Text (STT) and Text-to-Speech (TTS)

A deep learning audio application with three components: a **CNN** that classifies spoken digits from mel spectrograms, **Whisper** for speech-to-text and **MMS-TTS** for text-to-speech. Each component is evaluated with suitable metrics.

## Problem Statement
Design and implement a deep learning application for audio classification that incorporates Speech-to-Text (STT) and Text-to-Speech (TTS) using an appropriate deep learning architecture, and evaluate its performance using suitable metrics.

## Dataset
- **Source:** [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset), cloned automatically from GitHub in the notebook
- **Content:** 3,000 recordings (300 per digit) of the digits 0–9 spoken by 6 speakers (8 kHz WAV files, 0.44 s on average); the digit is taken from the file name
- **Split:** stratified 70% training (2,099) / 15% validation (451) / 15% test (450)

## Components
| Component | Model | Details |
|---|---|---|
| Audio classification | CNN (trained from scratch, 98,922 parameters) | Input: log-mel spectrogram (64 mel bands × 80 frames). Four Conv2D–BatchNorm–ReLU–MaxPool blocks (16, 32, 64, 128 filters), global average pooling, Dropout 0.3, Dense(10). Adam, learning rate 1e-3, 20 epochs |
| Speech-to-Text | Whisper (`openai/whisper-tiny.en`), pre-trained | Recordings resampled to 16 kHz; text normalized before scoring |
| Text-to-Speech | MMS-TTS (`facebook/mms-tts-eng`, VITS), pre-trained | Evaluated by transcribing the generated speech again with Whisper (round trip) |

## Application Pipeline
Input recording → **CNN** predicts the digit → **Whisper** transcribes the speech → **TTS** speaks "the predicted digit is …" → **Whisper** checks the spoken answer.

## Metrics
- **Classification:** accuracy, precision, recall, F1-score (macro), confusion matrix
- **STT:** Word Error Rate (WER), Character Error Rate (CER), digit recognition accuracy
- **TTS:** round-trip WER (TTS → STT) and real-time factor

## Requirements
- Python 3, PyTorch, librosa, Hugging Face `transformers`, `jiwer`
- scikit-learn, pandas, matplotlib, seaborn
- Google Colab with a **T4 GPU** is recommended

## How to Run
1. Open `audio_classification_stt_tts.ipynb` in Google Colab.
2. Go to **Runtime → Change runtime type → T4 GPU**.
3. Click **Runtime → Run all**. The first run downloads the dataset and the Whisper and MMS-TTS models.

## Results
| Component | Metric | Value |
|---|---|---|
| CNN audio classifier | Test accuracy | **0.9756** |
| CNN audio classifier | Macro F1-score | 0.9752 |
| CNN audio classifier | Inference time (ms/recording) | 0.08 |
| Whisper STT | Word Error Rate (WER) | 0.6250 |
| Whisper STT | Character Error Rate (CER) | 0.5934 |
| Whisper STT | Digit recognition accuracy | 0.5800 (CNN on the same 200 recordings: 0.9800) |
| Whisper STT | Time (ms/recording) | 64.3 |
| MMS-TTS | Round-trip WER (all texts) | 0.3167 (words: 1.00, sentences: 0.18) |
| MMS-TTS | Real-time factor | 0.098 |

## Observations
- The CNN classified 439 of 450 test recordings correctly. Most errors were the digit "one" (7 of 45 misclassified); training took only 10.1 s. Validation loss rose slightly at the end, so the best checkpoint (epoch 16) was kept.
- Whisper performed much worse (58% digit accuracy). For many recordings it returned unrelated phrases such as "Bye." or "Thank you very much.", probably because the clips are extremely short (0.44 s on average, 8 kHz) and Whisper is built for longer natural speech. It was best on "one" and "zero" and worst on "five".
- The CNN comparison favours the CNN: it is trained on this dataset for 10 fixed words (the same 6 speakers appear in training and test), while Whisper is a general model that was not trained on this data.
- TTS sentences were understood well (WER 0.18), but isolated words were not (only "three" and "eight" were recognized). The round trip also depends on Whisper's weakness on short clips, so these errors cannot be assigned to the TTS model alone. Speech was generated about 10× faster than real time.
- In the end-to-end demo, all 3 recordings were classified and transcribed correctly, and the correct digit was recognized in each spoken reply.

## Possible Improvements
- Use a larger dataset such as Google Speech Commands (more words and speakers) and a speaker-independent train/test split.
- Add data augmentation (noise, time shift, pitch shift) to the CNN.
- Try a larger Whisper model (base, small) or fine-tune Whisper on the dataset.

## Files
- `audio_classification_stt_tts.ipynb`: notebook with code and outputs
- `Assignment_Audio_Classification_STT_TTS.docx`: assignment report
