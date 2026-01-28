import os
import random
import soundfile as sf
import librosa
import numpy as np
from tqdm import tqdm

# ========================
# ⚙️ Paramètres
# ========================
INPUT_DIR = "voice_dataset"
OUTPUT_DIR = "voice_dataset_augmented"
AUG_PER_FILE = 3
NOISE_FACTOR = 0.005
SPEED_RANGE = (0.95, 1.05)
PITCH_RANGE = (-0.5, 0.5)

# ========================
# 🚀 Fonction d'augmentation
# ========================
def augment_audio(input_file, output_file):
    # Lecture du fichier audio
    y, sr = sf.read(input_file, dtype='float32')

    # Si stéréo → mono
    if y.ndim > 1:
        y = y[:, 0]

    # 1️⃣ Time-stretch (variation de vitesse)
    speed = random.uniform(*SPEED_RANGE)
    try:
        y = librosa.effects.time_stretch(y, rate=speed)
    except Exception:
        pass

    # 2️⃣ Pitch-shift
    n_steps = random.uniform(*PITCH_RANGE)
    try:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    except Exception:
        pass

    # 3️⃣ Ajout de bruit
    #noise = np.random.randn(len(y)) * NOISE_FACTOR
    #y = y + noise

    # 4️⃣ Normalisation
    y = y / np.max(np.abs(y))

    # 5️⃣ Sauvegarde sans torchcodec
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    sf.write(output_file, y, sr)
    # print(f"✅ {output_file}")

# ========================
# 🗂️ Boucle principale
# ========================
for cmd in os.listdir(INPUT_DIR):
    cmd_path = os.path.join(INPUT_DIR, cmd)
    if not os.path.isdir(cmd_path):
        continue

    output_cmd_path = os.path.join(OUTPUT_DIR, cmd)
    os.makedirs(output_cmd_path, exist_ok=True)

    files = [f for f in os.listdir(cmd_path) if f.endswith(".wav")]
    print(f"\n🎙️ Commande : {cmd} ({len(files)} fichiers)")

    for file in tqdm(files, desc=f"Augmentation {cmd}"):
        src = os.path.join(cmd_path, file)
        for i in range(AUG_PER_FILE):
            dst_name = file.replace(".wav", f"_aug{i+1}.wav")
            dst = os.path.join(output_cmd_path, dst_name)
            try:
                augment_audio(src, dst)
            except Exception as e:
                print(f"⚠️ Erreur sur {file}: {e}")
