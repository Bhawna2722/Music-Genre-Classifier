# 🎵 Music Genre Classification from Cross-Song Audio Mashups

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Transformers](https://img.shields.io/badge/HuggingFace-AST-yellow)
![Kaggle](https://img.shields.io/badge/Kaggle-0.905-success)

Deep Learning project for **BSDA2001P – Introduction to DL & GenAI** focused on classifying **music genres from cross-song audio mashups**.

🏆 **Kaggle Score:** **0.905**  
🎯 **Best Validation Macro F1:** **≈ 0.9883 (AST)**  
🌐 **Live Demo:** *https://huggingface.co/spaces/Bhawnasharma27/Music-Genre-Classifier*  
📄 **Project Report:** Included in repository

---

# 📌 Project Overview

Traditional music genre classification uses complete songs.

This project tackles a much harder challenge:

Each audio clip is created by **mixing stems from four different songs of the same genre**, meaning the model never hears a full song. Instead, it must predict genre from a **cross-song mashup** containing:

- 🎤 Vocals
- 🥁 Drums
- 🎸 Bass
- 🎼 Other instruments

This creates significant genre ambiguity and makes classification considerably more difficult.

---

# 🎯 Objectives

The project aimed to:

- Achieve strong Kaggle and validation performance
- Compare multiple architectures:
  - Audio Spectrogram Transformer (AST)
  - ResNet-18 Transfer Learning
  - Custom CNN trained from scratch
- Match training generation with test mashup creation
- Improve robustness using Test-Time Augmentation (TTA)

---

# 📂 Dataset

## Dataset Overview

| Component | Details |
|---|---|
| Songs | 1000 |
| Genres | 10 |
| Songs per genre | 100 |
| Test Mashups | 3020 |
| Audio Format | 16 kHz Mono |
| Clip Length | 10 sec |
| Extra Dataset | ESC-50 |

Each song contains four stems:

```python
drums.wav
vocals.wav
bass.wav
other.wav
```

Test clips were created by mixing stems from **different songs of the same genre**.

---

# ⚙️ Data Preprocessing & Augmentation

## Preprocessing

- Volume normalization
- Spectrogram generation
- Audio balancing
- Log-Mel feature extraction

## Data Augmentation

### 1. Cross-Song Stem Mixing

Training data was generated using the same process as the test set:

- One stem selected from four songs
- Same-genre mixing
- Fresh combinations every epoch

### 2. Background Noise Injection

Environmental noise from **ESC-50** was injected at low volume to improve robustness against noisy real-world audio.

---

# 🔬 Feature Extraction

Two feature pipelines were used.

## AST Pipeline

Raw Audio

↓

Log-Mel Spectrogram

↓

16×16 Patch Embeddings

↓

Transformer Encoder

↓

Genre Prediction

AST processes audio similarly to how language models process tokens.

## CNN / ResNet Pipeline

Mel spectrograms generated using:

```python
librosa.feature.melspectrogram(
    n_mels=128,
    hop_length=512
)
```

Output size:

```python
128 × 313
```

treated as image input.

---

# 🧠 Models Evaluated

## 1️⃣ Audio Spectrogram Transformer (AST) — Best Model

AudioSet-pretrained transformer fine-tuned for genre classification.

### Configuration

- 12 Transformer layers
- NVIDIA T4 GPU
- 20 epochs
- Cosine LR schedule
- 10-class classifier

### Performance

✅ Validation Macro F1 ≈ **0.9883**

---

## 2️⃣ Custom 3-Block CNN

Scratch baseline model.

Architecture:

```python
Conv → BN → ReLU → MaxPool
32 → 64 → 128 filters
```

Followed by:

- Global Average Pooling
- FC(256)
- FC(128)
- FC(10)
- Dropout 0.5

Training:

```python
Optimizer = Adam
LR = 1e-3
Epochs = 5
```

Purpose:

Evaluate learning without pre-training.

---

## 3️⃣ ResNet-18 Transfer Learning

ImageNet-pretrained ResNet-18.

Mel spectrograms replicated into 3-channel images.

Final layer:

```python
Linear(512 → 10)
```

### Two-Stage Training

### Stage 1

Frozen backbone

```python
2 epochs
LR = 1e-3
```

### Stage 2

Full fine-tuning

```python
3 epochs
LR = 1e-4
```

---

# 📊 Performance Comparison

| Model | Validation Macro F1 |
|---|---:|
| Custom CNN | 0.3309 |
| ResNet-18 | 0.6005 |
| AST | ≈ 0.9883 |

AST substantially outperformed both alternatives due to audio-specific pretraining.

---

# 🚀 Test-Time Augmentation (TTA)

Inference used **5 prediction passes per clip**.

Included:

- Clean inference
- Gaussian noise
- Random gain adjustment

Predictions were averaged before final classification.

Estimated gain:

```python
+0.02 → +0.04 F1
```

Inference time:

~26 minutes on NVIDIA T4.

---

# 📈 Prediction Insights

Most frequently predicted genres:

- Pop
- Disco
- Metal

Least predicted:

- Classical
- Country

Common confusions:

- Rock ↔ Metal
- Jazz ↔ Blues
- Pop ↔ Disco

The model relied heavily on rhythm-heavy stems such as drums and bass.

---

# 🌐 Deployment

Try the deployed application here:

🔗 **Live Demo:**  
**https://huggingface.co/spaces/Bhawnasharma27/Music-Genre-Classifier**

Example deployment platforms:

- Hugging Face Spaces
- Streamlit Cloud
- Render
- Vercel

### Features

- Audio processing
- Genre prediction
- AST inference
- Mashup classification

---

# 💡 Key Learnings

This project highlighted:

✅ Audio pretraining is extremely powerful  
✅ Train/test generation alignment matters  
✅ Noise injection improves robustness  
✅ TTA improves inference stability  
✅ Small datasets limit scratch CNN performance

---

# ⚠️ Challenges

Some key challenges:

- Silent stems
- Genre overlap
- Country/Classical confusion
- GPU memory limits (16GB T4)
- Small batch size slowing training

---

# 🔮 Future Work

Possible improvements:

- AST + temporal hybrid models
- Pitch shifting
- Time stretching
- Wav2Vec2 / CLAP experimentation
- Genre-aware loss functions
- Cross-validation

---

# 🛠 Tech Stack

### Languages & Libraries

- Python
- PyTorch
- Transformers (Hugging Face)
- Librosa
- NumPy
- Weights & Biases

### Deep Learning Models

- Audio Spectrogram Transformer (AST)
- ResNet-18
- Custom CNN

---

# 📁 Repository Structure

```bash
├── data/
├── notebooks/
├── models/
├── src/
├── report/
├── requirements.txt
├── README.md
```

---

# 📖 References

- PyTorch Documentation
- Hugging Face AST
- Librosa
- Weights & Biases
- Audio Classification Learning Resources

---
