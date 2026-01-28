import onnxruntime as ort
import numpy as np

# Charger le modèle
session = ort.InferenceSession("voice_model_single.onnx")

# Voir les dimensions d'entrée attendues
input_info = session.get_inputs()[0]
print(f"Nom de l'entrée : {input_info.name}")
print(f"Format attendu : {input_info.shape}")

# Créer une donnée factice (dummy data) pour vérifier que le fichier fonctionne
# Remplacez les dimensions par celles affichées au-dessus (ex: [1, 1, 64, 64])
dummy_input = np.random.randn(1, 1, 64, 81).astype(np.float32) 

outputs = session.run(None, {input_info.name: dummy_input})
print("Sortie du modèle :", outputs)