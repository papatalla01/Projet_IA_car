import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Initialise automatiquement le contexte CUDA
import numpy as np
import cv2
import time

# Logger pour TensorRT 
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class DepthEstimatorTRT:
    def __init__(self, engine_path):
        print(f"Chargement du moteur TensorRT : {engine_path}")
        
        # 1. Charger le fichier .engine
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        # 2. Créer le contexte d'exécution
        self.context = self.engine.create_execution_context()
        
        # 3. Allouer la mémoire (Host = CPU, Device = GPU)
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream() # Pour l'exécution asynchrone

        for binding in self.engine:
            # Récupérer taille et le type du tenseur
            shape = self.engine.get_binding_shape(binding)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            size = trt.volume(shape) * np.dtype(dtype).itemsize
            
            # Allouer la mémoire "pagelocked" sur le CPU (plus rapide pour le transfert)
            host_mem = cuda.pagelocked_empty(trt.volume(shape), dtype)
            # Allouer la mémoire sur le GPU
            device_mem = cuda.mem_alloc(size)
            
            # Ajouter aux listes
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
        
        print("Moteur TensorRT prêt !")

    def preprocess(self, img_rgb, lidar_sparse):
        """
        Prépare l'image et le lidar pour le modèle (Resize + Normalisation + Concat)
        """
        # Dimensions attendues par le modèle 
        target_h, target_w = 256, 832 
        
        # 1. Resize
        img_resized = cv2.resize(img_rgb, (target_w, target_h))
        lidar_resized = cv2.resize(lidar_sparse, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        
        # 2. Normalisation et transformation en Tensor
        # Image : (H,W,3) -> (3,H,W) / 255.0
        x_img = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        
        # Lidar : (H,W) -> (1,H,W) / 80.0 
        x_lidar = lidar_resized.astype(np.float32) / 80.0
        x_lidar = np.expand_dims(x_lidar, axis=0)
        
        # 3. Concaténation (4 canaux)
        x_input = np.concatenate((x_img, x_lidar), axis=0)
        
        # 4. Aplatir pour TensorRT (1D array)
        return np.ascontiguousarray(x_input.ravel())

    def predict(self, img_rgb, lidar_sparse):
        # 1. Prétraitement
        processed_input = self.preprocess(img_rgb, lidar_sparse)
        
        # Copier l'entrée dans le buffer CPU
        np.copyto(self.inputs[0]['host'], processed_input)
        
        # --- INFERENCE ---
        # 2. Transfert CPU -> GPU (Asynchrone)
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # 3. Exécution du modèle
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # 4. Transfert GPU -> CPU (Résultat)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        
        # Synchroniser (attendre la fin du calcul)
        self.stream.synchronize()
        
        # 5. Post-traitement (Reshape)
        output_data = self.outputs[0]['host']
        depth_map = output_data.reshape((256, 832))
        
        # Dénormaliser
        return depth_map * 80.0

# =========================================================
# Main pour tester l'inférence TensorRT
# =========================================================
if __name__ == "__main__":
    ENGINE_FILE = "unet_depth.engine"
    
    # Initialiser le modèle
    model = DepthEstimatorTRT(ENGINE_FILE)
    
    # --- Création de fausses données pour tester (Dummy Data) ---
    print("Test avec une image noire et un lidar vide...")
    dummy_img = np.zeros((375, 1242, 3), dtype=np.uint8) # Taille KITTI originale
    dummy_lidar = np.zeros((375, 1242), dtype=np.float32)
    
    # Simuler une ligne de lidar au milieu (obstacle à 10m)
    dummy_lidar[200:205, :] = 10.0
    
    # --- Mesure du temps (Benchmark) ---
    _ = model.predict(dummy_img, dummy_lidar)
    
    start = time.time()
    for i in range(50):
        depth_map = model.predict(dummy_img, dummy_lidar)
    end = time.time()
    
    fps = 50 / (end - start)
    print(f"Vitesse d'inférence : {fps:.2f} FPS")
    print(f"Shape de sortie : {depth_map.shape}")
    print(f"Valeur prédite au centre : {depth_map[128, 416]:.2f} m")