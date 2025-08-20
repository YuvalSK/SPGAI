import os, random
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import tifffile as tiff
from pytorch_msssim import ssim

DATA_ROOT     = r"C:\Users\User\Projects\Epigenetics\Danenberg2022\02_processed\images\published"
CHANNELS      = 39
FIXED_CHANNEL = 32                  # ERalpha
RESIZE        = (256, 256)
SAMPLE_FRAC   = 0.20                # 20% PoC
EPOCHS        = 15
BATCH_SIZE    = 2
LR            = 1e-4
PRETRAINED    = True
SAVE_PATH     = "best_leave_one_out_tiff.pth"

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# (helps on some Windows setups; safe to remove if you don't need it)

# ---- utils ----
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def to_float01(arr):
    if arr.dtype == np.uint16: return arr.astype(np.float32) / 65535.0
    if arr.dtype == np.uint8:  return arr.astype(np.float32) / 255.0
    return arr.astype(np.float32)

def load_tiff_chw(path):
    arr = to_float01(tiff.imread(path))  # handles multi-page
    if arr.ndim == 2:            # (H,W) -> (1,H,W)
        arr = arr[None, ...]
    elif arr.ndim == 3:
        # guess (H,W,C) vs (C,H,W)
        if arr.shape[0] <= 4 and arr.shape[2] > 4:
            arr = np.transpose(arr, (2,0,1))
        elif arr.shape[2] <= 4:
            arr = np.transpose(arr, (2,0,1))
        # else assume already (C,H,W)
    elif arr.ndim == 4:          # (pages,H,W,C) -> (pages*C,H,W)
        if arr.shape[-1] == 1: arr = arr[...,0]
        else:
            arr = arr.transpose(0,3,1,2).reshape(arr.shape[0]*arr.shape[-1], arr.shape[1], arr.shape[2])
    return arr  # (C,H,W)

# ---- dataset ----
class TiffFolder(Dataset):
    def __init__(self, files, size=RESIZE, expected_c=CHANNELS):
        self.files = files; self.size = size; self.expected_c = expected_c
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        x = torch.from_numpy(load_tiff_chw(self.files[i]))  # (C,H,W) float32 in [0,1]
        if x.shape[0] != self.expected_c:
            raise ValueError(f"{self.files[i]} has {x.shape[0]} channels (expected {self.expected_c})")
        # per-image min-max normalize each channel (simple & short)
        cmin = x.amin(dim=(1,2), keepdim=True); cmax = x.amax(dim=(1,2), keepdim=True)
        x = (x - cmin) / (cmax - cmin + 1e-6)
        x = F.interpolate(x.unsqueeze(0), size=self.size, mode="bilinear", align_corners=False).squeeze(0)
        return x  # (C,H,W)

# ---- model (ResNet18 encoder + light UNet decoder) ----
class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        r = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        self.stem   = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4
    def forward(self, x):
        x = self.stem(x)   # /4
        f1 = self.l1(x)    # /4, 64
        f2 = self.l2(f1)   # /8, 128
        f3 = self.l3(f2)   # /16,256
        f4 = self.l4(f3)   # /32,512
        return f1, f2, f3, f4

class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_ch, out_ch, 1)
        self.block = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv1x1(x)
        if x.shape[-2:] != skip.shape[-2:]:
            skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], 1))

class LeaveOneOutUNet(nn.Module):
    def __init__(self, total_c=CHANNELS, pretrained=PRETRAINED):
        super().__init__()
        self.total_c = total_c
        self.proj = nn.Conv2d(total_c - 1, 3, 1, bias=False)     # (C-1)->3 for ResNet input
        self.enc  = ResNet18Backbone(pretrained)
        self.dec4 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(inplace=True))
        self.up3  = Up(256, 256, 128)
        self.up2  = Up(128, 128, 64)
        self.up1  = Up(64,  64,  64)
        self.out  = nn.Sequential(                                  # to full res
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )
    def forward(self, x, k):
        B,C,H,W = x.shape
        left, right = x[:,:k], x[:,k+1:]
        rgb = self.proj(torch.cat([left, right], 1))
        f1,f2,f3,f4 = self.enc(rgb)
        d  = self.dec4(f4)
        d  = self.up3(d, f3)
        d  = self.up2(d, f2)
        d  = self.up1(d, f1)
        y  = self.out(d)
        if y.shape[-2:] != (H,W): y = F.interpolate(y, size=(H,W), mode="bilinear", align_corners=False)
        return y

# ---- loss (compact hybrid): Charbonnier + SSIM + gradient ----
class Charbonnier(nn.Module):
    def __init__(self, eps=1e-3): super().__init__(); self.eps = eps
    def forward(self, x, y): return torch.mean(torch.sqrt((x-y)**2 + self.eps**2))
def grad_loss(p, t):
    def g(z): return z[...,1:,:]-z[...,:-1,:], z[...,:,1:]-z[...,:,:-1]
    px,py = g(p); tx,ty = g(t)
    return F.l1_loss(px,tx) + F.l1_loss(py,ty)
charb = Charbonnier()
def recon_loss(pred, target):
    return 0.6*charb(pred, target) + 0.25*grad_loss(pred, target) + 0.15*(1 - ssim(pred, target, data_range=1.0, size_average=True))

# ---- train/eval ----
def train_one_epoch(model, loader, opt, device, epoch):
    model.train(); tot = 0.0
    for step, x in enumerate(loader, 1):
        x = x.to(device)
        y = x[:, FIXED_CHANNEL:FIXED_CHANNEL+1]
        p = model(x, FIXED_CHANNEL).clamp(0,1)
        loss = recon_loss(p, y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        tot += loss.item()
        if step % 50 == 0: print(f"Epoch {epoch} Step {step}/{len(loader)} loss={loss.item():.4f}")
    return tot / max(1, len(loader))

@torch.no_grad()
def evaluate(model, loader, device, n_batches=10):
    model.eval(); L=[]
    for i, x in enumerate(loader):
        if i >= n_batches: break
        x = x.to(device)
        y = x[:, FIXED_CHANNEL:FIXED_CHANNEL+1]
        p = model(x, FIXED_CHANNEL).clamp(0,1)
        L.append(recon_loss(p, y).item())
    return float(np.mean(L)) if L else float("nan")

# ---- main ----
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    files = list(Path(DATA_ROOT).glob("*.tif")) + list(Path(DATA_ROOT).glob("*.tiff"))
    if not files: raise FileNotFoundError("No TIFFs found")
    n = max(1, int(round(len(files)*SAMPLE_FRAC)))
    files = random.sample(files, n)
    print(f"Sampling {n}/{len(list(Path(DATA_ROOT).glob('*.tif')))+len(list(Path(DATA_ROOT).glob('*.tiff')))} files")

    ds = TiffFolder([str(f) for f in files])
    dl_tr = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    dl_va = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    model = LeaveOneOutUNet(total_c=CHANNELS, pretrained=PRETRAINED).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best = 1e9
    for epoch in range(1, EPOCHS+1):
        tr = train_one_epoch(model, dl_tr, opt, device, epoch)
        va = evaluate(model, dl_va, device)
        print(f"Epoch {epoch} - train: {tr:.4f}  val: {va:.4f}")
        if va < best:
            best = va
            torch.save({"model": model.state_dict()}, SAVE_PATH)
            print("Saved:", SAVE_PATH)
    print("Done.")

if __name__ == "__main__":
    main()
