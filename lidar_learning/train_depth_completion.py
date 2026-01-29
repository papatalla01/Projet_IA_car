import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List

# ============================================================
# 0) CONFIGURATION & CHEMINS
# ============================================================
DATA_ROOT = "/home/ilyes/Documents/COURS/IA_embarquee/LIDAR/archive"
TRAIN_ROOT = os.path.join(DATA_ROOT, "training")
IMG_DIR   = os.path.join(TRAIN_ROOT, "image_2")
LIDAR_DIR = os.path.join(TRAIN_ROOT, "velodyne")
CALIB_DIR = os.path.join(TRAIN_ROOT, "calib")

# Paramètres de l'image pour le réseau (Redimensionnement nécessaire pour UNet standard)
IMG_H, IMG_W = 256, 832  
MAX_DEPTH = 80.0         # On ignore les points au delà de 80m

# Simulation LiDAR 2D
LIDAR_2D_Z_MIN = -0.5    # On garde une tranche autour de z (hauteur capteur)
LIDAR_2D_Z_MAX = 0.5     

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Working on {device}")

# ============================================================
# 1) UTILITAIRES KITTI
# ============================================================
def load_kitti_calib(idx: int) -> Dict[str, np.ndarray]:
    path = os.path.join(CALIB_DIR, f"{idx:06d}.txt")
    calib = {}
    with open(path, "r") as f:
        for line in f:
            if not line.strip(): continue
            key, val = line.split(":", 1)
            calib[key] = np.array([float(x) for x in val.strip().split()], dtype=np.float32)
    return calib

def project_lidar_to_depth_map(lidar_points, calib, img_shape):
    """
    Projette le lidar sur le plan image et retourne une Depth Map (H, W).
    """
    P2 = calib["P2"].reshape(3, 4)
    Tr = calib["Tr_velo_to_cam"].reshape(3, 4)
    
    # 1. Passage en ref caméra
    pts_h = np.hstack((lidar_points[:, :3], np.ones((lidar_points.shape[0], 1))))
    pts_cam = (Tr @ pts_h.T).T
    
    # 2. Filtrer ce qui est derrière la caméra
    pts_cam = pts_cam[pts_cam[:, 2] > 0]
    
    # 3. Projection image
    pts_img = (P2 @ np.hstack((pts_cam, np.ones((pts_cam.shape[0], 1)))).T).T
    u = (pts_img[:, 0] / pts_img[:, 2]).astype(np.int32)
    v = (pts_img[:, 1] / pts_img[:, 2]).astype(np.int32)
    depths = pts_cam[:, 2]

    # 4. Filtrer bornes image
    h_img, w_img = img_shape[:2]
    mask = (u >= 0) & (u < w_img) & (v >= 0) & (v < h_img) & (depths < MAX_DEPTH)
    
    u = u[mask]
    v = v[mask]
    d = depths[mask]

    # 5. Créer la map (on prend le min depth si conflit, ou dernier écrasé)
    depth_map = np.zeros((h_img, w_img), dtype=np.float32)
    depth_map[v, u] = d
    
    return depth_map

# ============================================================
# 2) DATASET
# ============================================================
class KittiDepthDataset(Dataset):
    def __init__(self, indices, augment=False):
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        
        # --- A. Chargement ---
        img_path = os.path.join(IMG_DIR, f"{idx:06d}.png")
        lidar_path = os.path.join(LIDAR_DIR, f"{idx:06d}.bin")
        
        img = cv2.imread(img_path) # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        calib = load_kitti_calib(idx)
        
        h_orig, w_orig = img.shape[:2]

        # --- B. Création de la TARGET (Vérité terrain 3D dense) ---
        # On utilise TOUS les points pour la vérité terrain
        target_depth = project_lidar_to_depth_map(points, calib, (h_orig, w_orig))

        # --- C. Création de l'INPUT (Simulation LiDAR 2D) ---
        # On ne garde que les points dans une fine bande verticale (Z)
        mask_2d = (points[:, 2] > LIDAR_2D_Z_MIN) & (points[:, 2] < LIDAR_2D_Z_MAX)
        points_2d = points[mask_2d]
        
        input_depth = project_lidar_to_depth_map(points_2d, calib, (h_orig, w_orig))

        # --- D. Redimensionnement (Resize) ---
        img_resized = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
        
        input_d_resized = cv2.resize(input_depth, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
        target_d_resized = cv2.resize(target_depth, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)

        # --- E. Normalisation / Tensor ---
        # Image: (3, H, W) float 0-1
        x_img = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
        
        # Depth Input: (1, H, W) float normalisé 
        x_depth = torch.from_numpy(input_d_resized).unsqueeze(0).float() / MAX_DEPTH
        
        # Target: (H, W)
        y_depth = torch.from_numpy(target_d_resized).float() / MAX_DEPTH

        # Concatenation Input: RGB (3) + Sparse Depth (1) = 4 canaux
        x_input = torch.cat([x_img, x_depth], dim=0)

        return x_input, y_depth

# ============================================================
# 3) MODÈLE (UNet simple modif pour 4 canaux input)
# ============================================================
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class UNetDepth(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 4 canaux (R, G, B, Sparse_Depth) -> Output: 1 canal (Dense_Depth)
        self.enc1 = DoubleConv(4, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)
        
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = DoubleConv(256 + 128, 128)
        self.dec1 = DoubleConv(128 + 64, 64)
        
        self.out = nn.Conv2d(64, 1, kernel_size=1) # Sortie 1 canal (Depth)
        self.sigmoid = nn.Sigmoid() # Car on a normalisé entre 0 et 1

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        d2 = self.up(e3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        return self.sigmoid(self.out(d1))

# ============================================================
# 4) ENTRAINEMENT
# ============================================================
def train():
    # Liste des IDs
    all_files = sorted([f for f in os.listdir(LIDAR_DIR) if f.endswith('.bin')])
    indices = [int(f.split('.')[0]) for f in all_files]
    
    # Split
    split = int(0.9 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]
    
    train_ds = KittiDepthDataset(train_idx)
    val_ds = KittiDepthDataset(val_idx)
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=4)
    
    model = UNetDepth().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Loss: MSE mais uniquement sur les pixels valides de la target (Vérité terrain semi-dense)
    def masked_mse_loss(pred, target):
        mask = target > 0
        if mask.sum() == 0: return torch.tensor(0.0, device=device, requires_grad=True)
        diff = pred[mask] - target[mask]
        return (diff ** 2).mean()

    EPOCHS = 10
    print("Début de l'entraînement Depth Completion (Simulated 2D)...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(x).squeeze(1) # (B, H, W)
            
            loss = masked_mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
                
        print(f"=== Epoch {epoch+1} Mean Loss: {train_loss/len(train_loader):.4f} ===")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).squeeze(1)
                val_loss += masked_mse_loss(pred, y).item()
        print(f"=== Validation Loss: {val_loss/len(val_loader):.4f} ===\n")

    # Sauvegarde
    os.makedirs("checkpoints_depth", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints_depth/unet_depth_2d_sim.pth")
    print("Modèle sauvegardé.")

if __name__ == "__main__":
    train()