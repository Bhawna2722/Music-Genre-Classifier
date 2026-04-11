import gradio as gr
import torch
import librosa
import numpy as np
import random
from transformers import ASTFeatureExtractor, ASTForAudioClassification

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR = 16000
DURATION = 10

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
ID2GENRE = {i: g for i, g in enumerate(GENRES)}

# Load model and feature extractor
print("Loading model and feature extractor...")
feature_extractor = ASTFeatureExtractor.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593')
model = ASTForAudioClassification.from_pretrained(
    'MIT/ast-finetuned-audioset-10-10-0.4593',
    num_labels=10,
    ignore_mismatched_sizes=True
)

# Load the fine-tuned weights
print("Loading fine-tuned weights...")
try:
    state_dict = torch.load("best_model.pt", map_location=DEVICE)
    model.load_state_dict(state_dict)
    print("Fine-tuned weights loaded successfully.")
except Exception as e:
    print(f"Error loading best_model.pt: {e}. Make sure the file exists in the directory.")

model.to(DEVICE)
model.eval()

def pad_or_trim(audio, target_len):
    if len(audio) < target_len:
        return np.pad(audio, (0, target_len - len(audio)))
    return audio[:target_len]

def predict(audio_path):
    if audio_path is None:
        return "Please upload an audio file."
    
    try:
        audio, _ = librosa.load(audio_path, sr=SR, duration=DURATION, mono=True)
        audio = pad_or_trim(audio.astype(np.float32), SR * DURATION)
    except Exception as e:
        return {f"Error loading audio: {e}": 1.0}

    n_tta = 5
    logits_sum = None
    
    for i in range(n_tta):
        if i == 0:
            aug = audio.copy()   
        else:
            # Slight noise + gain variation
            aug = audio + np.random.randn(len(audio)).astype(np.float32) * 0.003
            aug = aug * random.uniform(0.9, 1.1)
            aug = aug / (np.abs(aug).max() + 1e-8)

        inputs = feature_extractor(aug, sampling_rate=SR, return_tensors='pt')
        with torch.no_grad():
            logits = model(input_values=inputs['input_values'].to(DEVICE)).logits
        logits_sum = logits if logits_sum is None else logits_sum + logits

    # Compute probabilities
    probs = torch.nn.functional.softmax(logits_sum[0], dim=-1).cpu().numpy()
    result = {ID2GENRE[i]: float(probs[i]) for i in range(len(GENRES))}

    return result

title = "Audio Genre Classification"
description = ("Upload a music track or record audio to predict its genre. "
               "The model uses an AST (Audio Spectrogram Transformer) fine-tuned on custom genres.")

iface = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath"),
    outputs=gr.Label(num_top_classes=10),
    title=title,
    description=description
)

if __name__ == "__main__":
    iface.launch()
