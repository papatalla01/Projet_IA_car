import os
import sounddevice as sd
from scipy.io.wavfile import write

# --- PARAMÈTRES ---
sr = 16000         # fréquence d'échantillonnage (Hz)
duration = 1.5     # durée par enregistrement (s)
commands = ["avance", "recule", "droite", "gauche", "stop"]
samples_per_word = 30

base_path = "voice_dataset"
os.makedirs(base_path, exist_ok=True)

# --- ENREGISTEMENT ---

for cmd in commands:
    path = os.path.join(base_path, cmd)
    os.makedirs(path, exist_ok=True)

    # Trouver combien de fichiers existent déjà
    existing = len([f for f in os.listdir(path) if f.endswith(".wav")])
    print(f"\n🎙️ Enregistrements existants pour '{cmd}' : {existing}")

    for i in range(existing, existing + samples_per_word):
        input(f"Appuie sur Entrée et dis '{cmd}' ({i+1})...")
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        filename = os.path.join(path, f"{cmd}_{i:03d}.wav")
        write(filename, sr, audio)
        print(f"✅ Sauvegardé : {filename}")
