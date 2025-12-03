import os
import glob
import shutil
import pickle
import json
import random
import time
import gc
import tempfile
import numpy as np
import tifffile
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import models
from sklearn.linear_model import SGDRegressor
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import KFold
from pytorch_msssim import ms_ssim
from cleanfid import fid
import optuna
from tqdm import tqdm

# ==========================================
# 1. Configuration & Utils
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def to_hwc(img):
    if img.ndim == 2:
        img = img[..., None]
    if img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
        img = np.transpose(img, (1, 2, 0))
    return img.astype(np.float32, copy=False)

def pad_collate(batch):
    """Pads tensors to the largest size in the batch."""
    X_list, Y_list, original_sizes = zip(*batch)
    max_H = max([x.shape[1] for x in X_list])
    max_W = max([x.shape[2] for x in X_list])
    
    padded_X, padded_Y = [], []
    for x, y in zip(X_list, Y_list):
        pad_H, pad_W = max_H - x.shape[1], max_W - x.shape[2]
        padded_X.append(F.pad(x, (0, pad_W, 0, pad_H)))
        padded_Y.append(F.pad(y, (0, pad_W, 0, pad_H)))
        
    return torch.stack(padded_X), torch.stack(padded_Y), original_sizes

# ==========================================
# 2. Model Architecture with Attention Gates
# ==========================================
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: gating signal (upsampled from lower layer)
        # x: skip connection from encoder
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        psi = self.relu(g1 + x1)
        return x * self.psi(psi)

class ResNet50Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1, self.relu, self.maxpool = backbone.bn1, backbone.relu, backbone.maxpool
        self.layer1, self.layer2 = backbone.layer1, backbone.layer2
        self.layer3, self.layer4 = backbone.layer3, backbone.layer4
        
        # Freezing early layers
        for param in backbone.parameters(): param.requires_grad = False
        for layer in [self.layer2, self.layer3, self.layer4]:
            for param in layer.parameters(): param.requires_grad = True

    def forward(self, x):
        f0 = self.relu(self.bn1(self.conv1(x)))
        f0_pool = self.maxpool(f0)
        f1 = self.layer1(f0_pool)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return f0, f1, f2, f3, f4

class UNetDecoder(nn.Module):
    def __init__(self, out_channels=1, p=0.2):
        super().__init__()
        # Attention Blocks: F_g (gate), F_l (skip), F_int (intermediate)
        self.attn4 = AttentionBlock(F_g=1024, F_l=1024, F_int=512)
        self.conv4 = self._block(2048, 1024, p) # 1024 (gate) + 1024 (skip) = 2048 in

        self.attn3 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.conv3 = self._block(1024, 512, p)

        self.attn2 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.conv2 = self._block(512, 256, p)

        self.attn1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.conv1 = self._block(256+64, 64, p) # 256(up) + 64(skip) -> 64

        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up0 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU())
        self.final = nn.Conv2d(32, out_channels, 1)

    def _block(self, in_ch, out_ch, p):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(), nn.Dropout2d(p),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU()
        )

    def forward(self, skips):
        f0, f1, f2, f3, f4 = skips 
        
        # Up 4
        x = F.interpolate(f4, scale_factor=2, mode='bilinear', align_corners=False)
        if x.shape[2:] != f3.shape[2:]: f3 = F.interpolate(f3, size=x.shape[2:])
        f3 = self.attn4(g=x, x=f3) # Apply Attention
        x = self.conv4(torch.cat([x, f3], dim=1))

        # Up 3
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if x.shape[2:] != f2.shape[2:]: f2 = F.interpolate(f2, size=x.shape[2:])
        f2 = self.attn3(g=x, x=f2)
        x = self.conv3(torch.cat([x, f2], dim=1))

        # Up 2
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if x.shape[2:] != f1.shape[2:]: f1 = F.interpolate(f1, size=x.shape[2:])
        f1 = self.attn2(g=x, x=f1)
        x = self.conv2(torch.cat([x, f1], dim=1))

        # Up 1
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if x.shape[2:] != f0.shape[2:]: f0 = F.interpolate(f0, size=x.shape[2:])
        f0 = self.attn1(g=x, x=f0)
        x = self.conv1(torch.cat([x, f0], dim=1))

        # Head
        x = self.final_up(x)
        x = self.up0(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return torch.sigmoid(self.final(x))

class ResNetUNet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.encoder = ResNet50Encoder(in_channels=in_channels)
        self.decoder = UNetDecoder()
    def forward(self, x):
        return self.decoder(self.encoder(x))

# ==========================================
# 3. Offline Preprocessing & Caching
# ==========================================
def fit_preprocessing_stats(files, target_idx, control_idx, n_components=3, cofactor=5.0):
    """Fits Regression, PCA, and finds GLOBAL Min/Max for targets."""
    print("[Stats] Fitting regression, PCA, and calculating global stats...")
    
    # Init
    img0 = to_hwc(tifffile.imread(files[0]))
    H, W, C = img0.shape
    input_channels = [c for c in range(C) if c != target_idx and c not in control_idx]
    
    reg_models = {j: SGDRegressor(max_iter=1000, tol=1e-3) for j in input_channels}
    pca = IncrementalPCA(n_components=n_components)
    
    # Global Target Stats
    global_min, global_max = float('inf'), float('-inf')

    # 1. Fit Regression & Global Min/Max
    for f in tqdm(files, desc="Step 1/3: Regress & MinMax"):
        img = to_hwc(tifffile.imread(f))
        
        # Update global target stats
        y_raw = img[..., target_idx]
        y_arc = np.arcsinh(y_raw / cofactor)
        global_min = min(global_min, y_arc.min())
        global_max = max(global_max, y_arc.max())
        
        # Partial fit regression
        Xc = img[..., control_idx].reshape(-1, len(control_idx))
        for j in input_channels:
            y_j = img[..., j].reshape(-1)
            reg_models[j].partial_fit(Xc, y_j)
            
    # 2. Fit PCA on Residuals
    for f in tqdm(files, desc="Step 2/3: PCA Fit"):
        img = to_hwc(tifffile.imread(f))
        Xc = img[..., control_idx].reshape(-1, len(control_idx))
        R_list = []
        for j in input_channels:
            y_j = img[..., j].reshape(-1)
            y_pred = reg_models[j].predict(Xc)
            R_list.append(np.arcsinh((y_j - y_pred)/cofactor))
        R = np.stack(R_list, axis=1) # (N, K)
        pca.partial_fit(R[::10]) # Subsample for speed

    # 3. PCA Normalization Stats
    sum_z, sumsq_z, total = np.zeros(n_components), np.zeros(n_components), 0
    for f in tqdm(files, desc="Step 3/3: PCA Norm"):
        img = to_hwc(tifffile.imread(f))
        Xc = img[..., control_idx].reshape(-1, len(control_idx))
        R_list = []
        for j in input_channels:
            y_j = img[..., j].reshape(-1)
            y_pred = reg_models[j].predict(Xc)
            R_list.append(np.arcsinh((y_j - y_pred)/cofactor))
        R = np.stack(R_list, axis=1)
        Z = pca.transform(R)
        sum_z += Z.sum(axis=0)
        sumsq_z += (Z**2).sum(axis=0)
        total += Z.shape[0]
        
    pca_mean = (sum_z / total).astype(np.float32)
    pca_std = (np.sqrt((sumsq_z/total) - pca_mean**2) + 1e-8).astype(np.float32)
    
    return reg_models, pca, pca_mean, pca_std, input_channels, global_min, global_max

def preprocess_and_cache(files, save_dir, stats, target_idx, control_idx, n_components=3, cofactor=5.0):
    """Applies stats and saves tensors to disk."""
    reg_models, pca, pca_mean, pca_std, inp_ch, g_min, g_max = stats
    os.makedirs(save_dir, exist_ok=True)
    saved_paths = []

    B_coef = np.stack([reg_models[j].coef_ for j in inp_ch], axis=-1).astype(np.float32)
    b_intr = np.array([reg_models[j].intercept_ for j in inp_ch], dtype=np.float32)

    for f_path in tqdm(files, desc=f"Caching to {save_dir}"):
        filename = os.path.basename(f_path)
        save_path = os.path.join(save_dir, filename.replace('.tif', '.pt').replace('.tiff', '.pt'))
        
        # If exists, skip (resume capability)
        if os.path.exists(save_path):
            saved_paths.append(save_path)
            continue
            
        img = to_hwc(tifffile.imread(f_path))
        H, W, _ = img.shape
        
        # 1. Target (Global Norm)
        y_arc = np.arcsinh(img[..., target_idx] / cofactor)
        y_norm = (y_arc - g_min) / (g_max - g_min + 1e-8)
        y_tensor = torch.from_numpy(y_norm[None, ...].astype(np.float32))

        # 2. Input (Vectorized Regression + PCA)
        Xc = img[..., control_idx].reshape(-1, len(control_idx))
        Yin = np.stack([img[..., j].reshape(-1) for j in inp_ch], 1)
        
        Yhat = Xc @ B_coef + b_intr
        R = np.arcsinh((Yin - Yhat) / cofactor)
        
        # PCA in chunks
        chunk = 100000
        Z_list = []
        for i in range(0, R.shape[0], chunk):
            Z_part = pca.transform(R[i:i+chunk])
            Z_list.append(Z_part)
        Z = np.vstack(Z_list)
        
        Z = (Z - pca_mean) / pca_std
        Z = Z.reshape(H, W, n_components)
        x_tensor = torch.from_numpy(Z).permute(2, 0, 1).float()
        
        torch.save((x_tensor, y_tensor), save_path)
        saved_paths.append(save_path)
        
    return saved_paths

class CachedDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
    def __len__(self):
        return len(self.file_paths)
    def __getitem__(self, idx):
        # FAST LOAD
        x, y = torch.load(self.file_paths[idx])
        return x, y, (x.shape[1], x.shape[2])

# ==========================================
# 4. Training & Optimization
# ==========================================
def train_epoch(model, loader, opt, scaler, alpha, device):
    model.train()
    running_loss = 0.0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        with autocast(enabled=True):
            y_pred = model(x)
            
            # SSIM + Huber
            # y and y_pred are already [0,1] due to Global Norm and Sigmoid
            loss_huber = F.smooth_l1_loss(y_pred, y)
            loss_ssim = 1 - ms_ssim(y_pred, y, data_range=1.0, win_size=3) # small win for small patches
            loss = alpha * loss_huber + (1 - alpha) * loss_ssim
            
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        running_loss += loss.item() * x.size(0)
    return running_loss / len(loader.dataset)

def validate(model, loader, alpha, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x).clamp(0, 1)
            loss_huber = F.smooth_l1_loss(y_pred, y)
            loss_ssim = 1 - ms_ssim(y_pred, y, data_range=1.0, win_size=3)
            loss = alpha * loss_huber + (1 - alpha) * loss_ssim
            running_loss += loss.item() * x.size(0)
    return running_loss / len(loader.dataset)

def optimize_hyperparams(train_paths, n_splits=3, n_trials=10, device='cuda'):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    # Fixed folds for offline data
    folds = list(kf.split(train_paths))
    
    def objective(trial):
        lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
        alpha = trial.suggest_float('alpha', 0.5, 0.95)
        
        fold_scores = []
        for tr_idx, va_idx in folds:
            tr_p = [train_paths[i] for i in tr_idx]
            va_p = [train_paths[i] for i in va_idx]
            
            model = ResNetUNet(in_channels=3).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            scaler = GradScaler()
            
            tr_loader = DataLoader(CachedDataset(tr_p), batch_size=4, shuffle=True, collate_fn=pad_collate)
            va_loader = DataLoader(CachedDataset(va_p), batch_size=4, shuffle=False, collate_fn=pad_collate)
            
            # Quick training (5 epochs for optuna speed)
            best_fold_val = float('inf')
            for _ in range(5): 
                train_epoch(model, tr_loader, opt, scaler, alpha, device)
                val_loss = validate(model, va_loader, alpha, device)
                best_fold_val = min(best_fold_val, val_loss)
            fold_scores.append(best_fold_val)
            
        return np.mean(fold_scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

# ==========================================
# 5. Main Pipeline
# ==========================================
def run_pipeline(data_path, cache_root, marker_tasks):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_files = sorted(glob.glob(os.path.join(data_path, "*.tif*")))
    
    # 80/20 Split
    random.shuffle(all_files)
    split = int(0.8 * len(all_files))
    train_files_raw = all_files[:split]
    test_files_raw = all_files[split:]
    
    print(f"Total files: {len(all_files)} | Train: {len(train_files_raw)} | Test: {len(test_files_raw)}")

    for task_name, target_idx, control_idx in marker_tasks:
        print(f"\n{'='*40}\nProcessing Marker: {task_name}\n{'='*40}")
        
        # A. Calculate Global Stats & PCA (ONCE using Train set)
        stats = fit_preprocessing_stats(train_files_raw, target_idx, control_idx)
        
        # B. Offline Cache
        train_cache_dir = os.path.join(cache_root, task_name, "train")
        test_cache_dir = os.path.join(cache_root, task_name, "test")
        
        train_paths = preprocess_and_cache(train_files_raw, train_cache_dir, stats, target_idx, control_idx)
        test_paths = preprocess_and_cache(test_files_raw, test_cache_dir, stats, target_idx, control_idx)
        
        # C. Hyperparameter Optimization (CV on Cached Train)
        print("Running Optuna...")
        best_params = optimize_hyperparams(train_paths, n_trials=10, device=device)
        print(f"Best Params: {best_params}")
        
        # D. Final Training
        print("Training Final Model...")
        model = ResNetUNet(in_channels=3).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
        scaler = GradScaler()
        
        # Use 90/10 split of full train for final training
        tr_len = int(0.9 * len(train_paths))
        final_tr_paths = train_paths[:tr_len]
        final_va_paths = train_paths[tr_len:]
        
        tr_loader = DataLoader(CachedDataset(final_tr_paths), batch_size=4, shuffle=True, collate_fn=pad_collate)
        va_loader = DataLoader(CachedDataset(final_va_paths), batch_size=4, shuffle=False, collate_fn=pad_collate)
        
        best_loss = float('inf')
        patience = 5
        no_improve = 0
        
        for epoch in range(20): # Train for 20 epochs
            tl = train_epoch(model, tr_loader, opt, scaler, best_params['alpha'], device)
            vl = validate(model, va_loader, best_params['alpha'], device)
            print(f"Epoch {epoch+1} | Train: {tl:.4f} | Val: {vl:.4f}")
            
            if vl < best_loss:
                best_loss = vl
                torch.save(model.state_dict(), f"{task_name}_best.pth")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print("Early stopping.")
                    break
        
        # E. Evaluation on Held-out Test
        print("Evaluating on Test Set...")
        model.load_state_dict(torch.load(f"{task_name}_best.pth"))
        model.eval()
        
        # (Optional: Add your custom evaluation metrics / visualization here)
        # Using simple Loss for now
        test_loader = DataLoader(CachedDataset(test_paths), batch_size=4, collate_fn=pad_collate)
        test_loss = validate(model, test_loader, best_params['alpha'], device)
        print(f"Final Test Loss for {task_name}: {test_loss:.4f}")

# ==========================================
# 6. Execution
# ==========================================
if __name__ == "__main__":
    DATA_PATH = "/kaggle/input/jackson2020" # Update this path
    CACHE_ROOT = "/kaggle/working/cache"
    
    # Define your tasks: (Name, Target Index, Control Indices)
    # Example: target 1 is H3K27me3, controls 0 (H3), 33 (DNA1), 34 (DNA2)
    controls = [0, 33, 34] 
    tasks = [
        ("histone_h3_trimethylate", 1, controls),
        ("histone_h3_phospho", 14, controls)
    ]
    
    set_seed(42)
    run_pipeline(DATA_PATH, CACHE_ROOT, tasks)