import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
import sounddevice as sd
import numpy as np
import soundfile as sf  # ✅ plus robuste que torchaudio.load

# ========================
# ⚙️ Paramètres
# ========================
classes = ["avance", "recule", "droite", "gauche", "stop"]
MODEL_PATH = "voice_model_1.pth"
SR = 16000
N_MELS = 64
T_FIXED = 81          # ✅ ton modèle ONNX attendait 81 -> on fixe pareil
TARGET_LEN = 16000    # 1 seconde

# ========================
# 🧠 Classe VoiceNet (corrigée : dummy T=81)
# ========================
class VoiceNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, N_MELS, T_FIXED)   # ✅ 81
            out = self.conv(dummy)
            flattened_size = out.view(out.size(0), -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flattened_size, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ========================
# ✅ Chargement du modèle
# ========================
model = VoiceNet(num_classes=len(classes))
state = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state)
model.eval()

mel = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_mels=N_MELS)
db = torchaudio.transforms.AmplitudeToDB()

# ========================
# 🎧 Préprocess commun : waveform -> mel (1,1,64,81)
# ========================
def preprocess_waveform(waveform: torch.Tensor) -> torch.Tensor:
    # waveform: (channels, N) ou (1, N)
    waveform = torch.mean(waveform, dim=0, keepdim=True)  # mono

    # 1 seconde
    if waveform.shape[1] < TARGET_LEN:
        waveform = F.pad(waveform, (0, TARGET_LEN - waveform.shape[1]))
    else:
        waveform = waveform[:, :TARGET_LEN]

    mel_spec = db(mel(waveform))  # (1,64,Tm)

    # ✅ force T=81 pour matcher le modèle
    Tm = mel_spec.shape[-1]
    if Tm < T_FIXED:
        mel_spec = F.pad(mel_spec, (0, T_FIXED - Tm))
    else:
        mel_spec = mel_spec[:, :, :T_FIXED]

    return mel_spec.unsqueeze(0)  # (1,1,64,81)

# ========================
# 🧠 Inférence
# ========================
def predict_from_waveform(waveform: torch.Tensor) -> str:
    x = preprocess_waveform(waveform)
    with torch.no_grad():
        logits = model(x)
        pred = int(torch.argmax(logits, dim=1).item())
    return classes[pred]

# ========================
# Silence detection
# ========================
def is_silent(audio_np: np.ndarray, threshold=4e-7) -> bool:
    energy = float(np.mean(audio_np**2))
    print(f"🔍 Énergie mesurée : {energy:.3e}")
    return energy < threshold

# ========================
# 🎤 Test micro
# ========================
def test_micro():
    duration = 3.0
    print("🎙️ Dites une commande ('avance', 'stop', etc.)...")

    audio = sd.rec(int(SR * duration), samplerate=SR, channels=1, dtype='float32')
    sd.wait()

    audio_np = audio.flatten()

    if is_silent(audio_np):
        print("🤫 Silence détecté (aucune commande reconnue).")
        return

    waveform = torch.tensor(audio.T)  # (1, N)
    cmd = predict_from_waveform(waveform)
    print(f"🗣️ Commande détectée : {cmd}")

# ========================
# 📁 Test fichier .wav (robuste avec soundfile)
# ========================
def test_fichier(path: str):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # si sr != 16000, idéalement resampler (simple warning pour l’instant)
    if sr != SR:
        print(f"⚠️ Attention: sr={sr}, attendu {SR}. Idéalement il faut resampler.")

    waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, N)
    cmd = predict_from_waveform(waveform)
    print(f"🗣️ Commande détectée : {cmd}")

# ========================
# 🔽 Menu principal
# ========================
if __name__ == "__main__":
    while True:
        choix = input("Tapez 'm' pour micro, 'f' pour fichier, ou 'q' pour quitter : ").lower()

        if choix == 'm':
            test_micro()
        elif choix == 'f':
            path = input("Chemin du fichier .wav à tester : ")
            test_fichier(path)
        elif choix == 'q':
            print("👋 Fin du programme.")
            break
        else:
            print("❌ Choix invalide.")

        print("\n--- Nouveau test ---\n")
