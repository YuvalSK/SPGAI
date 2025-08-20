# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 10:42:18 2025

@author: User
"""
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils
import os

nz = 64      # latent dim
ngf = 64     # generator channels
ndf = 64     # discriminator channels
nc = 3       # RGB
batch_size = 32
epochs = 10
device = "cuda" if torch.cuda.is_available() else "cpu"

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8), nn.ReLU(True),

            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4), nn.ReLU(True),

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2), nn.ReLU(True),

            nn.ConvTranspose2d(ngf*2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, z): return self.main(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*4, 1, 4, 1, 0, bias=False)
        )
    def forward(self, x):
        return self.main(x).reshape(-1)

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # normalize RGB
])
dataset = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

G, D = Generator().to(device), Discriminator().to(device)
optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
criterion = nn.BCEWithLogitsLoss()

fixed_z = torch.randn(64, nz, 1, 1, device=device)

# ---- Training ----
for epoch in range(epochs):
    for real, _ in loader:
        real = real.to(device)
        b = real.size(0)

        # Train D
        z = torch.randn(b, nz, 1, 1, device=device)
        fake = G(z).detach()
        lossD = criterion(D(real), torch.ones(b, device=device)) + \
                criterion(D(fake), torch.zeros(b, device=device))
        optD.zero_grad(); lossD.backward(); optD.step()

        # Train G
        z = torch.randn(b, nz, 1, 1, device=device)
        fake = G(z)
        lossG = criterion(D(fake), torch.ones(b, device=device))
        optG.zero_grad(); lossG.backward(); optG.step()

    # Save samples
    with torch.no_grad():
        fakes = G(fixed_z).cpu()
    os.makedirs("samples", exist_ok=True)
    vutils.save_image(fakes, f"samples/epoch_{epoch+1}.png", nrow=8, normalize=True)
    print(f"Epoch {epoch+1}: D_loss={lossD.item():.3f}, G_loss={lossG.item():.3f}")
