# -*- coding: utf-8 -*-
"""
CycleGAN on CIFAR-10 (32x32) with three loss types:
- Adversarial (LSGAN / MSE)
- Cycle-consistency (L1)
- Identity (L1)
Domains are two CIFAR-10 classes (unpaired), default: A=cat(3), B=dog(5)
"""
import os, itertools
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, utils as vutils

A_cls, B_cls = 3, 5        # CIFAR-10 indices: 3=cat, 5=dog
ngf = 64                   # generator channels
ndf = 64                   # discriminator channels
nc  = 3                    # channels 
batch_size = 32
epochs = 5
device = "cuda" if torch.cuda.is_available() else "cpu"

lambda_cyc = 10.0
lambda_idt = 5.0           # usually 0.5 * lambda_cyc
lr = 2e-4
betas = (0.5, 0.999)
save_dir = "samples_cyclegan"
os.makedirs(save_dir, exist_ok=True)

class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, 1, 0, bias=False),
            nn.InstanceNorm2d(dim), nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, 1, 0, bias=False),
            nn.InstanceNorm2d(dim),
        )
    def forward(self, x): return x + self.block(x)

class ResnetGenerator32(nn.Module):
    # c7s1-64, d128, [Res]*6, u64, c7s1-3 -> Tanh
    def __init__(self, in_c=3, out_c=3, ngf=64, n_blocks=6):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_c, ngf, 7, 1, 0, bias=False),
            nn.InstanceNorm2d(ngf), nn.ReLU(True),
            # Down: 32->16
            nn.Conv2d(ngf, ngf*2, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(ngf*2), nn.ReLU(True),
        ]
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf*2)]
        # Up: 16->32
        model += [
            nn.ConvTranspose2d(ngf*2, ngf, 3, 2, 1, output_padding=1, bias=False),
            nn.InstanceNorm2d(ngf), nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_c, 7, 1, 0),
            nn.Tanh(),
        ]
        self.net = nn.Sequential(*model)
    def forward(self, x): return self.net(x)

class NLayerDiscriminator32(nn.Module):
    # PatchGAN for 32x32: 32->16->8 (stride2), then stride1; outputs HxW logits
    def __init__(self, in_c=3, ndf=64):
        super().__init__()
        kw, p = 4, 1
        self.net = nn.Sequential(
            nn.Conv2d(in_c, ndf, kw, 2, p), nn.LeakyReLU(0.2, True),       # 32->16
            nn.Conv2d(ndf, ndf*2, kw, 2, p, bias=False),
            nn.InstanceNorm2d(ndf*2), nn.LeakyReLU(0.2, True),             # 16->8
            nn.Conv2d(ndf*2, ndf*4, kw, 1, p, bias=False),
            nn.InstanceNorm2d(ndf*4), nn.LeakyReLU(0.2, True),             # 8->8
            nn.Conv2d(ndf*4, 1, kw, 1, p),                                 # logits map
        )
    def forward(self, x): return self.net(x)

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None: nn.init.constant_(m.bias, 0.0)
    if isinstance(m, nn.InstanceNorm2d):
        if m.weight is not None: nn.init.normal_(m.weight, 1.0, 0.02)
        if m.bias is not None: nn.init.constant_(m.bias, 0.0)

# ---------------- Data: two unpaired domains from CIFAR-10 ----------------
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
trainset = datasets.CIFAR10("./data", train=True, download=True, transform=transform)

targets = trainset.targets if isinstance(trainset.targets, list) else trainset.targets.tolist()
idx_A = [i for i, t in enumerate(targets) if t == A_cls]
idx_B = [i for i, t in enumerate(targets) if t == B_cls]
subset_A = Subset(trainset, idx_A)
subset_B = Subset(trainset, idx_B)

loader_A = DataLoader(subset_A, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
loader_B = DataLoader(subset_B, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

G_AB = ResnetGenerator32(nc, nc, ngf=ngf).to(device)   # A -> B
G_BA = ResnetGenerator32(nc, nc, ngf=ngf).to(device)   # B -> A
D_A  = NLayerDiscriminator32(nc, ndf=ndf).to(device)   # real A ?
D_B  = NLayerDiscriminator32(nc, ndf=ndf).to(device)   # real B ?

G_AB.apply(init_weights); G_BA.apply(init_weights); D_A.apply(init_weights); D_B.apply(init_weights)

# Three loss types
adv = nn.MSELoss()   # LSGAN adversarial loss
l1  = nn.L1Loss()    # for cycle-consistency and identity

opt_G   = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=betas)
opt_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=betas)
opt_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=betas)

for epoch in range(1, epochs+1):
    # iterate for the shorter of the two loaders per epoch
    it_A, it_B = iter(loader_A), iter(loader_B)
    steps = min(len(loader_A), len(loader_B))
    for _ in range(steps):
        real_A = next(it_A)[0].to(device)
        real_B = next(it_B)[0].to(device)

        # ----- Train Generators (G_AB, G_BA) -----
        opt_G.zero_grad()

        # Forward translations
        fake_B = G_AB(real_A)
        fake_A = G_BA(real_B)

        # Reconstructions (cycle)
        rec_A  = G_BA(fake_B)
        rec_B  = G_AB(fake_A)

        # Identity (feed target-domain images through corresponding generator)
        idt_A = G_BA(real_A)
        idt_B = G_AB(real_B)

        # Adversarial targets match discriminator output shapes
        pred_fake_B = D_B(fake_B)
        pred_fake_A = D_A(fake_A)
        loss_gan_AB = adv(pred_fake_B, torch.ones_like(pred_fake_B))
        loss_gan_BA = adv(pred_fake_A, torch.ones_like(pred_fake_A))

        # Cycle-consistency
        loss_cyc_A = l1(rec_A, real_A) * lambda_cyc
        loss_cyc_B = l1(rec_B, real_B) * lambda_cyc

        # Identity
        loss_idt_A = l1(idt_A, real_A) * lambda_idt
        loss_idt_B = l1(idt_B, real_B) * lambda_idt

        loss_G = loss_gan_AB + loss_gan_BA + loss_cyc_A + loss_cyc_B + loss_idt_A + loss_idt_B
        loss_G.backward()
        opt_G.step()

        # ----- Train D_A -----
        opt_D_A.zero_grad()
        loss_D_A_real = adv(D_A(real_A), torch.ones_like(pred_fake_A))
        loss_D_A_fake = adv(D_A(fake_A.detach()), torch.zeros_like(pred_fake_A))
        loss_D_A = 0.5 * (loss_D_A_real + loss_D_A_fake)
        loss_D_A.backward()
        opt_D_A.step()

        # ----- Train D_B -----
        opt_D_B.zero_grad()
        loss_D_B_real = adv(D_B(real_B), torch.ones_like(pred_fake_B))
        loss_D_B_fake = adv(D_B(fake_B.detach()), torch.zeros_like(pred_fake_B))
        loss_D_B = 0.5 * (loss_D_B_real + loss_D_B_fake)
        loss_D_B.backward()
        opt_D_B.step()

    # ----- Save sample grids -----
    with torch.no_grad():
        A_vis = next(iter(loader_A))[0][:8].to(device)
        B_vis = next(iter(loader_B))[0][:8].to(device)
        AB = G_AB(A_vis).cpu(); A_rec = G_BA(AB.to(device)).cpu()
        BA = G_BA(B_vis).cpu(); B_rec = G_AB(BA.to(device)).cpu()
        grid = torch.cat([A_vis.cpu(), AB, A_rec, B_vis.cpu(), BA, B_rec], dim=0)
        vutils.save_image(grid, f"{save_dir}/epoch_{epoch:02d}.png", nrow=8, normalize=True, value_range=(-1,1))

    print(f"Epoch {epoch:02d} | "
          f"D_A {loss_D_A.item():.3f} D_B {loss_D_B.item():.3f} | "
          f"G {loss_G.item():.3f} "
          f"(GAN_AB {loss_gan_AB.item():.3f}, GAN_BA {loss_gan_BA.item():.3f}, "
          f"cycA {loss_cyc_A.item():.3f}, cycB {loss_cyc_B.item():.3f}, "
          f"idtA {loss_idt_A.item():.3f}, idtB {loss_idt_B.item():.3f})")
