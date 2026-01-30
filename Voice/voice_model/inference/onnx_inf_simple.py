import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("voice_model.onnx")

# même shape que pendant l'export
T = 80  # adapte si nécessaire
x = np.random.randn(1, 1, 64, T).astype(np.float32)

y = sess.run(None, {"input": x})[0]
print("Output shape:", y.shape)
print("Logits:", y)
