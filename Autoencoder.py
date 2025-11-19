import os
import json
import math
import random
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm

IMAGE_FOLDER = "blueOnly"
BATCH_SIZE = 32
EPOCHS = 600
LEARNING_RATE = 1e-4
LATENT_DIM = 14
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_VAL = 120
NUM_TEST = 120
RNG_SEED = 42

class ImageOnlyDataset(Dataset):
    def __init__(self, excel_file, augment=False):
        df = pd.read_excel(excel_file, header=1).dropna()
        df['exp_id'] = df['Exp.No.'].str.extract(r'(\d+)').astype(int)
        df['image_filename'] = df['exp_id'].apply(lambda x: f"Exp{x:03d}_blueOnly.bmp")
        self.filenames = df['image_filename'].tolist()

        base = [transforms.ToTensor()]
        if augment:
            base += [
                transforms.Lambda(lambda x: x + 0.02 * torch.randn_like(x)),
                transforms.Lambda(lambda x: torch.clamp(x, 0., 1.))
            ]
        self.transform = transforms.Compose(base)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img = Image.open(os.path.join(IMAGE_FOLDER, fname)).convert("L")
        return self.transform(img), fname

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(64, 128, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(128, 256, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(256, 256, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(256, 256, 5, 2, 2), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, LATENT_DIM)
        )
    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(LATENT_DIM, 256 * 14 * 14),
            nn.ReLU()
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 5, 2, 2, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 5, 2, 2, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 5, 2, 2, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, 2, 2, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 5, 2, 2, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 6, 2, 2, output_padding=1),
            nn.Sigmoid()
        )
    def forward(self, z):
        z = self.fc(z)
        z = z.view(-1, 256, 14, 14)
        out = self.deconv(z)
        return out[:, :, :875, :875]  
    
def save_comparison(real, fake, filename, folder="comparisons"):
    diff = torch.abs(real - fake).squeeze().cpu().numpy()
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(real.squeeze().cpu(), cmap='gray'); axs[0].set_title("Real")
    axs[1].imshow(fake.squeeze().detach().cpu(), cmap='gray'); axs[1].set_title("Reconstructed")
    axs[2].imshow(diff, cmap='hot'); axs[2].set_title("Difference")
    for ax in axs: ax.axis("off")
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, filename.replace('.bmp', '_cmp.png')))
    plt.close()

def save_loss_curves(train_losses, val_losses, filename):
    plt.figure()
    plt.semilogy(train_losses, label="train")
    plt.semilogy(val_losses, label="val")
    plt.title("Autoencoder Reconstruction Loss (BCE)")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(); plt.savefig(filename, bbox_inches="tight"); plt.close()

# ============== Early Stopping ==============
class EarlyStopping:
    def __init__(self, patience=30, min_delta=1e-5, path_prefix="es_best"):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.num_bad = 0
        self.path_enc = f"{path_prefix}_encoder.pth"
        self.path_dec = f"{path_prefix}_decoder.pth"
    def step(self, val_metric, encoder, decoder):
        if (self.best - val_metric) > self.min_delta:
            self.best = val_metric
            self.num_bad = 0
            torch.save(encoder.state_dict(), self.path_enc)
            torch.save(decoder.state_dict(), self.path_dec)
        else:
            self.num_bad += 1
        return self.num_bad >= self.patience
    def restore_best(self, encoder, decoder, device):
        if os.path.exists(self.path_enc) and os.path.exists(self.path_dec):
            encoder.load_state_dict(torch.load(self.path_enc, map_location=device))
            decoder.load_state_dict(torch.load(self.path_dec, map_location=device))

def _gaussian_window(window_size=11, sigma=1.5, channels=1, device="cpu"):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = (g / g.sum()).unsqueeze(0)  
    window = (g.t() @ g).unsqueeze(0).unsqueeze(0)  
    window = window.repeat(channels, 1, 1, 1)
    return window

def ssim_torch(x, y, window=None, C1=0.01**2, C2=0.03**2):
    B, C, H, W = x.shape
    if window is None:
        window = _gaussian_window(11, 1.5, C, x.device)
    padding = 5
    mu_x = F.conv2d(x, window, padding=padding, groups=C)
    mu_y = F.conv2d(y, window, padding=padding, groups=C)
    mu_x2, mu_y2, mu_xy = mu_x*mu_x, mu_y*mu_y, mu_x*mu_y
    sigma_x2 = F.conv2d(x*x, window, padding=padding, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y*y, window, padding=padding, groups=C) - mu_y2
    sigma_xy = F.conv2d(x*y, window, padding=padding, groups=C) - mu_xy
    ssim_map = ((2*mu_xy + C1)*(2*sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1)*(sigma_x2 + sigma_y2 + C2) + 1e-12)
    return ssim_map.mean()

def psnr_torch(x, y):
    mse = F.mse_loss(x, y, reduction='none').mean(dim=[1,2,3])  
    psnr = -10.0 * torch.log10(mse + 1e-12)
    return psnr.mean()

@torch.no_grad()
def evaluate_metrics(encoder, decoder, loader, loss_fn):
    encoder.eval(); decoder.eval()
    total_bce, n_px = 0.0, 0
    psnrs, ssims = [], []
    win = None
    for y, _ in loader:
        y = y.to(DEVICE)
        recon = decoder(encoder(y))
        bce = loss_fn(recon, y)
        total_bce += bce.item() * y.size(0)
        recon = torch.clamp(recon, 0, 1)
        if win is None:
            win = _gaussian_window(11, 1.5, y.size(1), y.device)
        psnrs.append(psnr_torch(recon, y).item())
        ssims.append(ssim_torch(recon, y, window=win).item())
        n_px += y.size(0)
    mean_bce = total_bce / max(n_px, 1)
    mean_psnr = float(np.mean(psnrs)) if psnrs else 0.0
    mean_ssim = float(np.mean(ssims)) if ssims else 0.0
    return mean_bce, mean_psnr, mean_ssim

@torch.no_grad()
def evaluate_loss(encoder, decoder, loader, loss_fn):
    encoder.eval(); decoder.eval()
    total, n = 0.0, 0
    for y, _ in loader:
        y = y.to(DEVICE)
        loss = loss_fn(decoder(encoder(y)), y)
        total += loss.item() * y.size(0)
        n += y.size(0)
    return total / max(n, 1)

def save_split_indices_json(base_dataset, idx_train, idx_val, idx_test, path="split_indices.json"):
    def to_list(obj):
        return list(obj.indices) if hasattr(obj, "indices") else list(obj)
    split = {
        "train": [base_dataset.filenames[i] for i in to_list(idx_train)],
        "val":   [base_dataset.filenames[i] for i in to_list(idx_val)],
        "test":  [base_dataset.filenames[i] for i in to_list(idx_test)],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)

def save_model_size(encoder, decoder, path="model_size.txt"):
    def count_params(m):
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return total, trainable
    e_total, e_train = count_params(encoder)
    d_total, d_train = count_params(decoder)
    total = e_total + d_total
    trainable = e_train + d_train
    with open(path, "w") as f:
        f.write(f"Encoder params (total/trainable): {e_total}/{e_train}\n")
        f.write(f"Decoder params (total/trainable): {d_total}/{d_train}\n")
        f.write(f"Total params (total/trainable): {total}/{trainable}\n")

def train_autoencoder():
    base_ds = ImageOnlyDataset("data.xlsx", augment=False)
    N = len(base_ds)
    n_train = N - NUM_VAL - NUM_TEST

    g = torch.Generator().manual_seed(RNG_SEED)
    idx_train, idx_val, idx_test = random_split(range(N), [n_train, NUM_VAL, NUM_TEST], generator=g)
    save_split_indices_json(base_ds, idx_train, idx_val, idx_test, path="split_indices.json")

    ds_train = ImageOnlyDataset("data.xlsx", augment=True)
    ds_val   = ImageOnlyDataset("data.xlsx", augment=False)
    ds_test  = ImageOnlyDataset("data.xlsx", augment=False)

    train_dataset = torch.utils.data.Subset(ds_train, idx_train.indices if hasattr(idx_train, "indices") else idx_train)
    val_dataset   = torch.utils.data.Subset(ds_val,   idx_val.indices   if hasattr(idx_val, "indices")   else idx_val)
    test_dataset  = torch.utils.data.Subset(ds_test,  idx_test.indices  if hasattr(idx_test, "indices")  else idx_test)

    print(f"[Split] train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    encoder, decoder = Encoder().to(DEVICE), Decoder().to(DEVICE)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE)
    loss_fn = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(EPOCHS // 2, 1), gamma=0.1)

    save_model_size(encoder, decoder, path="model_size.txt")
    vis_indices = random.sample(range(N), min(5, N))
    torch.save(vis_indices, "ae_vis_indices.pt")

    train_losses, val_losses = [], []
    early = EarlyStopping(patience=30, min_delta=1e-5, path_prefix="es_best")

    for epoch in range(1, EPOCHS + 1):
        encoder.train(); decoder.train()
        run, n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch}/{EPOCHS}]")
        for y, _ in pbar:
            y = y.to(DEVICE)
            recon = decoder(encoder(y))
            loss = loss_fn(recon, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            run += loss.item() * y.size(0); n += y.size(0)
            pbar.set_postfix(loss=loss.item())
        train_loss = run / max(n, 1)
        train_losses.append(train_loss)

        val_loss = evaluate_loss(encoder, decoder, val_loader, loss_fn)
        val_losses.append(val_loss)
        print(f"Epoch {epoch:03d} | train BCE: {train_loss:.6f} | val BCE: {val_loss:.6f}")

        save_loss_curves(train_losses, val_losses, "autoencoder_loss_train_val.png")

        if early.step(val_loss, encoder, decoder):
            print(f"Early stopping at epoch {epoch} (best val={early.best:.6f}).")
            break

        scheduler.step()

        VIS_INTERVAL = 50 

        if epoch % VIS_INTERVAL == 0 or epoch == 1:  
           with torch.no_grad():
             for i in vis_indices:
                 y, fname = base_ds[i]
                 y = y.unsqueeze(0).to(DEVICE)
                 save_comparison(y, decoder(encoder(y)), fname, folder="ae_reconstruction")


    early.restore_best(encoder, decoder, DEVICE)

    os.makedirs("ae_val_reconstruction", exist_ok=True)
    with torch.no_grad():
        for k, (y, fname) in enumerate(val_dataset):
            if k >= 5: break
            yb = y.unsqueeze(0).to(DEVICE)
            save_comparison(yb, decoder(encoder(yb)), fname, folder="ae_val_reconstruction")
        os.makedirs("ae_test_reconstruction", exist_ok=True)
        for k, (y, fname) in enumerate(test_dataset):
            if k >= 5: break
            yb = y.unsqueeze(0).to(DEVICE)
            save_comparison(yb, decoder(encoder(yb)), fname, folder="ae_test_reconstruction")

    val_bce, val_psnr, val_ssim = evaluate_metrics(encoder, decoder, val_loader, loss_fn)
    test_bce, test_psnr, test_ssim = evaluate_metrics(encoder, decoder, test_loader, loss_fn)
    print(f"[Val ] BCE: {val_bce:.6f} | PSNR: {val_psnr:.3f} | SSIM: {val_ssim:.4f}")
    print(f"[Test] BCE: {test_bce:.6f} | PSNR: {test_psnr:.3f} | SSIM: {test_ssim:.4f}")

    metrics_df = pd.DataFrame([
        {"split": "val",  "BCE": val_bce,  "PSNR": val_psnr,  "SSIM": val_ssim},
        {"split": "test", "BCE": test_bce, "PSNR": test_psnr, "SSIM": test_ssim},
    ])
    metrics_df.to_csv("metrics.csv", index=False)

if __name__ == '__main__':
    train_autoencoder()
