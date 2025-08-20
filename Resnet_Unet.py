import os
import math
import argparse
import random
from pathlib import Path
from typing import Optional, Tuple, List
import tifffile as tiff
from PIL import Image, ImageSequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_float32(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / 65535.0
    elif arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
    return arr

def load_tiff_as_chw(path: str) -> np.ndarray:
    """
    Loads a multi-channel TIFF into (C,H,W) float32 in [0,1].
    Handles multi-page (pages,H,W) or (H,W,C).
    """
    if tiff is not None:
        with tiff.TiffFile(path) as tif:
            arr = tif.asarray()
            arr = _to_float32(arr)
            if arr.ndim == 2:
                arr = arr[None, ...]
            elif arr.ndim == 3:
                # Try to guess (H,W,C) vs (C,H,W)
                if arr.shape[0] <= 4 and arr.shape[2] > 4:
                    arr = np.transpose(arr, (2,0,1))
                elif arr.shape[2] <= 4:
                    arr = np.transpose(arr, (2,0,1))
                else:
                    pass  # assume already (C,H,W)
            elif arr.ndim == 4:
                if arr.shape[-1] == 1:
                    arr = arr[...,0]
                else:
                    c = arr.shape[0]*arr.shape[-1]
                    arr = arr.transpose(0,3,1,2).reshape(c, arr.shape[1], arr.shape[2])
            return arr

    if Image is None:
        raise ImportError("Install `tifffile` or `Pillow` to read TIFFs.")

    img = Image.open(path)
    frames = []
    try:
        for frame in ImageSequence.Iterator(img):
            frames.append(np.array(frame))
    finally:
        img.close()

    if len(frames) == 0:
        raise ValueError(f"Could not read any frames from TIFF: {path}")
    arr = np.stack(frames, axis=0)
    arr = _to_float32(arr)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[...,0]
    elif arr.ndim == 4 and arr.shape[-1] > 1:
        pages, H, W, C = arr.shape
        arr = arr.transpose(0,3,1,2).reshape(pages*C, H, W)
    return arr  # (C,H,W)

# -------------------------
# Dataset
# -------------------------
class MultiChannelTiffDataset(Dataset):
    """
    Loads multi-channel TIFF files as (C,H,W) float32 in [0,1], normalizes per-channel,
    and resizes to a fixed (H,W).
    """
    def __init__(
        self,
        root: str,
        size: Optional[Tuple[int, int]] = (256, 256),
        file_pattern: str = "*.tif",
        alt_patterns: Optional[List[str]] = None,
        normalize: bool = True,
        expected_channels: Optional[int] = 39,
        files: Optional[List[str]] = None,
    ):
        if files is not None:
            self.files = sorted(set(map(str, files)))
        else:
            patterns = [file_pattern] + (alt_patterns or [])
            paths = []
            for pat in patterns:
                paths.extend(Path(root).glob(pat))
            self.files = sorted(set(map(str, paths)))
        if not self.files:
            raise FileNotFoundError(f"No TIFF files found under {root}.")
        self.size = size
        self.normalize = normalize
        self.expected_channels = expected_channels

    def __len__(self):
        return len(self.files)

    def _resize_tensor(self, t: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        t = t.unsqueeze(0)
        t = F.interpolate(t, size=size, mode="bilinear", align_corners=False)
        return t.squeeze(0)

    def __getitem__(self, idx: int) -> torch.Tensor:
        arr = load_tiff_as_chw(self.files[idx])  # (C,H,W)
        if self.expected_channels is not None and arr.shape[0] != self.expected_channels:
            raise ValueError(f"{self.files[idx]} has {arr.shape[0]} channels; expected {self.expected_channels}.")
        t = torch.from_numpy(arr)
        if self.normalize:
            cmins = t.amin(dim=(1,2), keepdim=True)
            cmaxs = t.amax(dim=(1,2), keepdim=True)
            denom = (cmaxs - cmins).clamp_min(1e-6)
            t = (t - cmins) / denom
        if self.size is not None:
            t = self._resize_tensor(t, self.size)
        return t

# -------------------------
# Model: ResNet50 encoder + U-Net decoder
# -------------------------
class ResNet50Backbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
    def forward(self, x: torch.Tensor):
        f0 = x
        x = self.stem(x)     # /4
        f1 = x               # 64
        x = self.layer1(x)   # /4
        f2 = x               # 256
        x = self.layer2(x)   # /8
        f3 = x               # 512
        x = self.layer3(x)   # /16
        f4 = x               # 1024
        x = self.layer4(x)   # /32
        f5 = x               # 2048
        return f0, f1, f2, f3, f4, f5

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)
    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.conv1x1(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class LeaveOneOutResNetUNet(nn.Module):
    def __init__(self, total_channels: int = 39, pretrained_backbone: bool = True):
        super().__init__()
        assert total_channels >= 2, "Need at least 2 channels to leave-one-out."
        self.total_channels = total_channels
        self.proj_to_rgb = nn.Conv2d(total_channels - 1, 3, kernel_size=1, bias=False)
        self.encoder = ResNet50Backbone(pretrained=pretrained_backbone)
        self.dec5 = ConvBlock(2048, 512)
        self.up4  = UpBlock(512, 1024, 256)
        self.up3  = UpBlock(256, 512, 128)
        self.up2  = UpBlock(128, 256, 96)
        self.up1  = UpBlock(96, 64, 64)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )
    def forward(self, x: torch.Tensor, leave_out_index: Optional[int] = None) -> torch.Tensor:
        B, C, H, W = x.shape
        if C != self.total_channels:
            raise ValueError(f"Expected {self.total_channels} channels but got {C}")
        if leave_out_index is None:
            leave_out_index = np.random.randint(0, C)
        if not (0 <= leave_out_index < C):
            raise ValueError("leave_out_index out of range.")
        left = x[:, :leave_out_index, :, :]
        right = x[:, leave_out_index+1:, :, :]
        rem = torch.cat([left, right], dim=1)
        rgb = self.proj_to_rgb(rem)
        f0, f1, f2, f3, f4, f5 = self.encoder(rgb)
        d5 = self.dec5(f5)
        d4 = self.up4(d5, f4)
        d3 = self.up3(d4, f3)
        d2 = self.up2(d3, f2)
        d1 = self.up1(d2, f1)
        out = self.final_up(d1)
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out


def train_one_epoch(model, loader, optimizer, device, epoch, grad_clip: Optional[float] = None):
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        
        fixed_channel = 6 #cd45ra
        target = batch[:, fixed_channel:fixed_channel+1, :, :]
        pred = model(batch, leave_out_index=fixed_channel)

        
        loss = F.l1_loss(pred, target) + 0.2 * F.mse_loss(pred, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        if (step + 1) % 50 == 0:
            print(f"Epoch {epoch} Step {step+1}/{len(loader)} loss={loss.item():.4f}")
    return total_loss / max(1, len(loader))

@torch.no_grad()
def evaluate(model, loader, device, num_batches: int = 10):
    model.eval()
    losses = []
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        batch = batch.to(device)
        k = np.random.randint(0, batch.shape[1])
        target = batch[:, k:k+1, :, :]
        pred = model(batch, leave_out_index=k)
        loss = F.l1_loss(pred, target) + 0.2 * F.mse_loss(pred, target)
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")

def main():
    # Hard-coded config
    data_root = r"C:\Users\User\Projects\Epigenetics\Danenberg2022\02_processed\images\published"
    sample_frac = 0.15
    sample_seed = 42
    resize = (256, 256)
    channels = 39
    batch_size = 2
    epochs = 5
    lr = 1e-4
    pretrained = True

    # Set seed and device
    set_seed(sample_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Collect and sample files
    import random
    from pathlib import Path
    all_files = list(Path(data_root).glob("*.tif")) + list(Path(data_root).glob("*.tiff"))
    if not all_files:
        raise FileNotFoundError(f"No TIFF files found under {data_root}")
    n_keep = max(1, int(round(len(all_files) * sample_frac)))
    sampled_files = random.sample(all_files, n_keep)
    print(f"Sampling {n_keep}/{len(all_files)} files ({sample_frac*100:.1f}%).")

    # Dataset + loaders
    train_set = MultiChannelTiffDataset(
        root=data_root,
        size=resize,
        expected_channels=channels,
        files=sampled_files,
    )
    val_set = MultiChannelTiffDataset(
        root=data_root,
        size=resize,
        expected_channels=channels,
        files=sampled_files,
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # Model + optimizer
    model = LeaveOneOutResNetUNet(total_channels=channels, pretrained_backbone=pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Train loop
    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, grad_clip=1.0)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch} - train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model": model.state_dict(),
                "sampled_files": [str(f) for f in sampled_files],
            }, "best_leave_one_out_tiff.pth")

    print("Done.")
main()
