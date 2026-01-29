import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib as mpl

# ============================================================
# 0) CONFIGURATION (Doit être identique au train)
# ============================================================
DATA_ROOT = "/home/ilyes/Documents/COURS/IA_embarquee/LIDAR/archive"
TEST_ROOT = os.path.join(DATA_ROOT, "training")  # On utilise training pour diviser en val
IMG_DIR   = os.path.join(TEST_ROOT, "image_2")
LIDAR_DIR = os.path.join(TEST_ROOT, "velodyne")
CALIB_DIR = os.path.join(TEST_ROOT, "calib")
CKPT_PATH = "checkpoints_depth/unet_depth_2d_sim.pth"

IMG_H, IMG_W = 256, 832
MAX_DEPTH = 80.0
LIDAR_2D_Z_MIN = -0.5
LIDAR_2D_Z_MAX = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1) REDEFINITION DU MODELE (Pour charger les poids)
# ============================================================
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class UNetDepth(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(4, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = DoubleConv(256 + 128, 128)
        self.dec1 = DoubleConv(128 + 64, 64)
        self.out = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

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
# 2) UTILITAIRES DE CHARGEMENT
# ============================================================
def load_kitti_calib(idx: int):
    path = os.path.join(CALIB_DIR, f"{idx:06d}.txt")
    calib = {}
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            key, val = line.split(":", 1)
            calib[key] = np.array([float(x) for x in val.strip().split()], dtype=np.float32)
    return calib

def project_lidar(lidar_points, calib, img_shape):
    P2 = calib["P2"].reshape(3, 4)
    Tr = calib["Tr_velo_to_cam"].reshape(3, 4)

    pts_h = np.hstack((lidar_points[:, :3], np.ones((lidar_points.shape[0], 1), dtype=np.float32)))
    pts_cam = (Tr @ pts_h.T).T

    # Garder uniquement points devant la caméra
    pts_cam = pts_cam[pts_cam[:, 2] > 0]
    if pts_cam.shape[0] == 0:
        h, w = img_shape
        return np.zeros((h, w), dtype=np.float32)

    pts_img = (P2 @ np.hstack((pts_cam, np.ones((pts_cam.shape[0], 1), dtype=np.float32))).T).T
    u = (pts_img[:, 0] / pts_img[:, 2]).astype(np.int32)
    v = (pts_img[:, 1] / pts_img[:, 2]).astype(np.int32)
    depths = pts_cam[:, 2]

    h, w = img_shape
    mask = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (depths < MAX_DEPTH)

    dm = np.zeros((h, w), dtype=np.float32)
    dm[v[mask], u[mask]] = depths[mask]
    return dm

def get_sample(idx):
    # Charger image
    img_path = os.path.join(IMG_DIR, f"{idx:06d}.png")
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image introuvable: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = img.shape[:2]

    # Charger Lidar
    pts_path = os.path.join(LIDAR_DIR, f"{idx:06d}.bin")
    if not os.path.exists(pts_path):
        raise FileNotFoundError(f"LiDAR introuvable: {pts_path}")
    pts = np.fromfile(pts_path, dtype=np.float32).reshape(-1, 4)

    calib = load_kitti_calib(idx)

    # Target (Full)
    target = project_lidar(pts, calib, (h_orig, w_orig))

    # Input (Sparse 2D)
    mask_2d = (pts[:, 2] > LIDAR_2D_Z_MIN) & (pts[:, 2] < LIDAR_2D_Z_MAX)
    inp = project_lidar(pts[mask_2d], calib, (h_orig, w_orig))

    # Resize
    img_r = cv2.resize(img, (IMG_W, IMG_H))
    inp_r = cv2.resize(inp, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
    tar_r = cv2.resize(target, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)

    # Tensors
    x_img = torch.from_numpy(img_r).float().permute(2, 0, 1) / 255.0
    x_depth = torch.from_numpy(inp_r).unsqueeze(0).float() / MAX_DEPTH
    x_in = torch.cat([x_img, x_depth], dim=0).unsqueeze(0)  # Batch size 1

    return x_in, img_r, inp_r, tar_r

# ============================================================
# 3) VISUALISATION (plus "parlante" pour SPARSE)
#    - Colormap contrastée (turbo)
#    - Pixels 0 (vides) => noir
#    - Non-linéaire (log) pour mieux distinguer proche/loin
# ============================================================
def _depth_to_vis01(depth_m, dmax_vis, mode="log", invert=True, gamma=0.8):
    """
    Transforme depth (mètres) -> [0,1] pour affichage.
    mode="log" donne un dégradé plus visible sur les faibles distances.
    invert=True : proche -> valeur haute (donc couleur "chaude"/claire)
    gamma < 1 : booste encore le proche
    """
    d = depth_m.astype(np.float32)
    valid = d > 0.0

    out = np.zeros_like(d, dtype=np.float32)
    if not np.any(valid):
        return out, valid

    dmax = float(dmax_vis)

    # clamp pour ne pas écraser le contraste
    dv = np.clip(d[valid], 0.0, dmax)

    if mode == "log":
        # 0..dmax -> 0..1 avec log (meilleur contraste proche)
        vis = np.log1p(dv) / np.log1p(dmax)
    elif mode == "sqrt":
        vis = np.sqrt(dv / dmax)
    else:
        vis = dv / dmax

    if invert:
        vis = 1.0 - vis  # proche -> 1, loin -> 0

    # gamma
    vis = np.clip(vis, 0.0, 1.0)
    vis = np.power(vis, gamma)

    out[valid] = vis
    return out, valid

def show_depth(ax, depth_m, title,
               max_depth=80.0,
               dmax_vis=40.0,
               mode="log",
               invert=True,
               gamma=0.75,
               thicken=True,
               thick_kernel=3,
               add_colorbar=False):
    """
    Affichage optimisé pour SPARSE + dégradé visible.
    - dmax_vis: limite d'affichage (ex: 30/40) => contraste ++
    - mode="log": met en évidence les variations proches
    - thicken: dilate les points pour les rendre visibles
    """
    ax.set_title(title)
    ax.axis("off")

    dmax = float(dmax_vis if dmax_vis is not None else max_depth)
    vis01, valid = _depth_to_vis01(depth_m, dmax, mode=mode, invert=invert, gamma=gamma)

    if thicken:
        # On "épaissit" uniquement la visualisation, sans toucher aux données.
        # Ici on fait un max-filter via dilatation pour propager une valeur visible.
        k = int(thick_kernel)
        k = max(1, k)
        kernel = np.ones((k, k), dtype=np.uint8)

        # dilate les valeurs (max local)
        vis01 = cv2.dilate(vis01, kernel, iterations=1)

        valid_d = cv2.dilate(valid.astype(np.uint8), kernel, iterations=1).astype(bool)
    else:
        valid_d = valid

    # masquer les invalides => noir
    m = np.ma.array(vis01, mask=~valid_d)

    # colormap très contrastée
    cmap = mpl.cm.get_cmap("turbo").copy()
    cmap.set_bad(color="black")

    im = ax.imshow(m, cmap=cmap, vmin=0.0, vmax=1.0)

    # colorbar 
    if add_colorbar:
        def meters_to_vis(meters):
            tmp = np.zeros((1, 1), dtype=np.float32)
            tmp[0, 0] = meters
            v, _ = _depth_to_vis01(tmp, dmax, mode=mode, invert=invert, gamma=gamma)
            return float(v[0, 0])

        ticks_m = [0, 5, 10, 20, 30, 40]
        ticks_m = [t for t in ticks_m if t <= dmax]
        ticks_v = [meters_to_vis(t) for t in ticks_m]

        cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, ticks=ticks_v)
        cbar.ax.set_yticklabels([f"{t}m" for t in ticks_m])
        cbar.set_label("Proche ↔ Loin", rotation=90)

def main():
    print(f"Chargement de {CKPT_PATH}...")
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint introuvable: {CKPT_PATH}")

    model = UNetDepth().to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    # indices validation
    all_files = sorted([f for f in os.listdir(LIDAR_DIR) if f.endswith(".bin")])
    indices = [int(f.split(".")[0]) for f in all_files]
    split = int(0.9 * len(indices))
    val_indices = indices[split:split + 5]

    # Réglages VISU 
    DMAX_VIS_INPUT = 40.0   # plage affichée (m)
    DMAX_VIS_PRED  = 60.0   # parfois la prédiction varie plus loin
    GAMMA = 0.70            # plus petit => proche plus clair
    THICK_KERNEL = 3        # 3 ou 5 pour points plus gros

    for idx in val_indices:
        x_in, img_rgb, input_sparse, target_dense = get_sample(idx)
        x_in = x_in.to(device)

        with torch.no_grad():
            pred = model(x_in)  # (1,1,H,W)

        pred_np = pred.squeeze().cpu().numpy() * MAX_DEPTH

        plt.figure(figsize=(15, 10))

        # 1) RGB
        ax1 = plt.subplot(4, 1, 1)
        ax1.set_title(f"Image RGB - Sample {idx}")
        ax1.imshow(img_rgb)
        ax1.axis("off")

        # 2) Input sparse: turbo + log + épaississement => dégradé visible
        ax2 = plt.subplot(4, 1, 2)
        show_depth(
            ax2, input_sparse, "Input: Simulation LiDAR 2D (Ta ligne)",
            max_depth=MAX_DEPTH,
            dmax_vis=DMAX_VIS_INPUT,
            mode="log",
            invert=False,        # proche = couleurs chaudes, loin = froid
            gamma=GAMMA,
            thicken=True,
            thick_kernel=THICK_KERNEL,
            add_colorbar=False
        )

        # 3) Prediction
        ax3 = plt.subplot(4, 1, 3)
        show_depth(
            ax3, pred_np, "Prédiction: Depth Map générée",
            max_depth=MAX_DEPTH,
            dmax_vis=DMAX_VIS_PRED,
            mode="log",
            invert=False,
            gamma=GAMMA,
            thicken=False,       
            thick_kernel=THICK_KERNEL,
            add_colorbar=False
        )

        # 4) Ground Truth (sparse)
        ax4 = plt.subplot(4, 1, 4)
        show_depth(
            ax4, target_dense, "Vérité Terrain (Ground Truth KITTI)",
            max_depth=MAX_DEPTH,
            dmax_vis=DMAX_VIS_INPUT,
            mode="log",
            invert=False,
            gamma=GAMMA,
            thicken=True,
            thick_kernel=THICK_KERNEL,
            add_colorbar=False
        )

        plt.tight_layout()
        plt.show()
        print(f"Affichage sample {idx}. Ferme la fenêtre pour voir le suivant.")

if __name__ == "__main__":
    main()
