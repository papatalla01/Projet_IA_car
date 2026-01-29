import os
import torch
import torch.nn as nn

# =========================
# 1) Modèle
# =========================
IMG_H, IMG_W = 256, 832  

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

        return self.sigmoid(self.out(d1))  # (B,1,H,W)

# =========================
# 2) Export ONNX
# =========================
def export_onnx(
    ckpt_path="checkpoints_depth/unet_depth_2d_sim.pth",
    onnx_path="checkpoints_depth/unet_depth_2d_sim.onnx",
    opset=17,
    dynamic_batch=True,
):
    device = torch.device("cpu")  # export en CPU (recommandé)
    model = UNetDepth().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    dummy = torch.randn(1, 4, IMG_H, IMG_W, device=device)

    # Axes dynamiques 
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch"},
            "output": {0: "batch"},
        }

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    print(f"✅ Export ONNX OK: {onnx_path}")

    # vérification ONNX
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX checker: modèle valide")
    except Exception as e:
        print("⚠️ ONNX checker non exécuté (onnx pas installé ?) ou erreur:")
        print(e)

if __name__ == "__main__":
    export_onnx()
