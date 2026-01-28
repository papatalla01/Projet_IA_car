import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
import sounddevice as sd
import numpy as np

# ========================
# 🧠 Classe VoiceNet (ta version)
# ========================
class VoiceNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )

        # Calcul dynamique du flatten size
        with torch.no_grad():
            dummy_input_shape = (1, 1, 64, 80)
            dummy_input = torch.zeros(dummy_input_shape)
            dummy_output = self.conv(dummy_input)
            flattened_size = dummy_output.view(dummy_output.size(0), -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flattened_size, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ========================
# ⚙️ Chargement du modèle
# ========================
classes = ["avance", "recule", "droite", "gauche", "stop"]

model = VoiceNet(num_classes=len(classes))
model.load_state_dict(torch.load("voice_model.pth", map_location="cpu"))
model.eval()

mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=64)
db = torchaudio.transforms.AmplitudeToDB()

# ========================
# 🎧 Fonction de prédiction
# ========================
def predict_from_audio(waveform, sr=16000):
    waveform = torch.mean(waveform, dim=0, keepdim=True)
    target_len = 16000
    if waveform.shape[1] < target_len:
        pad = target_len - waveform.shape[1]
        waveform = F.pad(waveform, (0, pad))
    else:
        waveform = waveform[:, :target_len]
    mel_spec = db(mel(waveform))
    mel_spec = mel_spec.unsqueeze(0)
    with torch.no_grad():
        output = model(mel_spec)
        pred = torch.argmax(output, dim=1).item()
    return classes[pred]
# ========================
# Silence detection
# ========================

def is_silent(audio, threshold=4e-7):
    """
    audio : numpy array audio (float32)
    threshold : énergie minimale pour considérer que quelqu'un parle
    """
    energy = np.mean(audio**2)
    print(f"🔍 Énergie mesurée : {energy}")

    return energy < threshold


# ========================
# 🎤 Test micro
# ========================
def test_micro():
    sr = 16000
    duration = 3.0
    print("🎙️ Dites une commande ('avance', 'stop', etc.)...")
    
    audio = sd.rec(int(sr * duration), samplerate=sr, channels=1, dtype='float32')
    sd.wait()

    audio_np = audio.flatten()

    # 🔇 Vérifie si c'est du silence
    if is_silent(audio_np):
        print("🤫 Silence détecté (aucune commande reconnue).")
        return

    # 🧠 Sinon on fait l'inférence
    waveform = torch.tensor(audio.T)
    cmd = predict_from_audio(waveform, sr)
    print(f"🗣️ Commande détectée : {cmd}")


# ========================
# 📁 Test fichier .wav
# ========================
def test_fichier(path):
    waveform, sr = torchaudio.load(path)
    cmd = predict_from_audio(waveform, sr)
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

