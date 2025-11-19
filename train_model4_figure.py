import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from Autoencoder import Encoder, Decoder  
from sklearn.metrics import r2_score
import re
import matplotlib as mpl
import matplotlib.font_manager as font_manager
from matplotlib.ticker import FuncFormatter, NullLocator

mpl.rcParams['font.family'] = 'serif'
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
mpl.rcParams['font.serif'] = cmfont.get_name()
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.size'] = 16

IMAGE_FOLDER = "blueOnly"
EXCEL_FILE = "data.xlsx"
ENCODER_PATH = "es_best_encoder.pth"
DECODER_PATH = "es_best_decoder.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 3000
BATCH_SIZE = 32
LR = 1e-3
OUTPUT_FOLDER = "train_monitor"
LOSS_PLOT_FILE = "loss-model-4.png"
FIXED_MONITOR_SAMPLES = 5
SPLIT_JSON = "split_indices.json"  

FEATURE_COLUMNS = ['Nozzle inner diameter (mm)\n', 'Viscosity (mPa.s)', 'Flow rate (μl/min)',
                   'Density (kg/m^3)', 'Surface tension (mN/m)']

# ====== 随机性控制 ======
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(1)

def _read_excel_smart(excel_file):
    raw = pd.read_excel(excel_file, header=None)
    header_idx = None
    for i in range(min(20, len(raw))):
        row = raw.iloc[i].astype(str).str.strip().str.lower()
        if row.str.contains(r"\bexp\.?\s*no\b").any() or row.str.contains(r"\bexp\b").any():
            header_idx = i
            break
    if header_idx is None:
        header_idx = 0
    return pd.read_excel(excel_file, header=header_idx)

def _normalize(name: str):
    s = str(name).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("．", ".").replace("，", ",")
    return s

def _build_col_map(cols):
    norm2orig = {_normalize(c): c for c in cols}
    def pick(*patterns, required=True):
        for pat in patterns:
            rx = re.compile(pat)
            for nc, orig in norm2orig.items():
                if rx.search(nc):
                    return orig
        for pat in patterns:
            for nc, orig in norm2orig.items():
                if pat in nc:
                    return orig
        return None
    return pick

class LatentDataset(torch.utils.data.Dataset):
    def __init__(self, excel_file, encoder, image_folder,
                 latent_file="latents.npy", scaler_file="latent_scaler.npz"):
        self.encoder = encoder
        self.image_folder = image_folder
        self.transform = transforms.ToTensor()
        self.device = next(encoder.parameters()).device

        df = _read_excel_smart(excel_file).dropna(how="all")
        pick = _build_col_map(df.columns)

        col_exp   = pick(r"\bexp\.?\s*no\b", r"\bexp\b")
        col_d_in  = pick(r"nozzle.*inner.*diameter.*\(mm\)", "nozzle inner diameter")
        col_mu    = pick(r"viscosity.*\(mpa\.?s\)", "viscosity")
        col_flow  = pick(r"flow.*(μl|ul)\/?min", "flow rate")
        col_rho   = pick(r"density.*\(kg\/?m\^?3\)", "density")
        col_sigma = pick(r"surface.*tension.*\(m?n\/?m\)", "surface tension")

        FEATURE_COLUMNS_RESOLVED = [col_d_in, col_mu, col_flow, col_rho, col_sigma]

        df = df.dropna(subset=[col_exp])
        exp_str = df[col_exp].astype(str)
        df['exp_id'] = exp_str.str.extract(r'(\d+)').astype(int)
        df['image_filename'] = df['exp_id'].apply(lambda x: f"Exp{x:03d}_blueOnly.bmp")
        self.filenames = df['image_filename'].tolist()
        self.features = torch.tensor(df[FEATURE_COLUMNS_RESOLVED].values, dtype=torch.float32)

        if os.path.exists(latent_file) and os.path.exists(scaler_file):
            print(f"Loading latent from {latent_file} and scaler from {scaler_file}")
            self.latents = torch.tensor(np.load(latent_file), dtype=torch.float32)
            scaler_data = np.load(scaler_file)
            self.latent_mean = torch.tensor(scaler_data["mean"], dtype=torch.float32)
            self.latent_std  = torch.tensor(scaler_data["std"],  dtype=torch.float32)
        else:
            print("Computing latent representations and normalizing...")
            latents = []
            self.encoder.eval()
            with torch.no_grad():
                for fname in self.filenames:
                    img = Image.open(os.path.join(self.image_folder, fname)).convert("L")
                    img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                    z = self.encoder(img_tensor).squeeze(0).cpu()
                    latents.append(z)
            latents = torch.stack(latents)
            self.latent_mean = latents.mean(dim=0)
            self.latent_std  = latents.std(dim=0)
            self.latents     = (latents - self.latent_mean) / self.latent_std
            np.save(latent_file, self.latents.numpy())
            np.savez(scaler_file, mean=self.latent_mean.numpy(), std=self.latent_std.numpy())
            print(f"Latents saved to {latent_file} | Scaler saved to {scaler_file}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        return self.features[idx], self.latents[idx], self.filenames[idx]

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPRegressor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 100), nn.ReLU(),
            nn.Linear(100, 200), nn.ReLU(),
            nn.Linear(200, 400), nn.ReLU(),
            nn.Linear(400, 800), nn.ReLU(),
            nn.Linear(800, 400), nn.ReLU(),
            nn.Linear(400, 200), nn.ReLU(),
            nn.Linear(200, 100), nn.ReLU(),
            nn.Linear(100, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class EarlyStopping:
    def __init__(self, patience=30, min_delta=1e-5, path="es_best_model4.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.num_bad = 0
        self.path = path
    def step(self, val_metric, model):
        improved = (self.best - val_metric) > self.min_delta
        if improved:
            self.best = val_metric
            self.num_bad = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.num_bad += 1
        return self.num_bad >= self.patience
    def restore_best(self, model, device):
        if os.path.exists(self.path):
            model.load_state_dict(torch.load(self.path, map_location=device))

def save_comparison(real_img, pred_img, filename, folder="train_monitor", epoch=0):
    diff = torch.abs(real_img - pred_img).squeeze().cpu().numpy()
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(real_img.squeeze().cpu(), cmap='gray'); axs[0].set_title("Original")
    axs[1].imshow(pred_img.squeeze().detach().cpu(), cmap='gray'); axs[1].set_title("Predicted")
    axs[2].imshow(diff, cmap='hot'); axs[2].set_title("Difference")
    for ax in axs: ax.axis("off")
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, f"{filename.replace('.bmp', f'_epoch{epoch}.png')}"))
    plt.close()

def plot_loss(train_losses, val_losses, filename="loss_curve-model4.png"):
    fig, ax = plt.subplots()
    ax.semilogy(train_losses, label="Train Loss")
    ax.semilogy(val_losses,   label="Val Loss")
    ticks = [5e-1, 1e-1]
    ax.set_yticks(ticks)
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(True, which="both", linestyle="--", alpha=0.5)
    fig.savefig(filename); plt.close(fig)

def clean_filename(name): return re.sub(r'[^a-zA-Z0-9_]', '_', name)

def save_latent_parity_csv(preds, trues, names, epoch, split="val", folder="train_monitor"):
    os.makedirs(folder, exist_ok=True)
    P = np.asarray(preds); T = np.asarray(trues); names = list(names)
    D = P.shape[1]
    mse = np.mean((P - T) ** 2, axis=1)
    data = {"filename": names, "mse": mse}
    for j in range(D):
        data[f"true_z{j}"] = T[:, j]
    for j in range(D):
        data[f"pred_z{j}"] = P[:, j]
    df = pd.DataFrame(data)
    out_csv = os.path.join(folder, f"{split}_latent_parity_epoch{epoch}.csv")
    df.to_csv(out_csv, index=False)
    print(f"[Export] Saved latent parity CSV: {out_csv}")

def plot_latent_parity(preds, trues, epoch, split="val", folder="train_monitor", max_dims=4):
    os.makedirs(folder, exist_ok=True)
    P = np.asarray(preds); T = np.asarray(trues)
    D = min(max_dims, P.shape[1])
    for j in range(D):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(T[:, j], P[:, j], alpha=0.6)
        xymin = min(np.min(T[:, j]), np.min(P[:, j])); xymax = max(np.max(T[:, j]), np.max(P[:, j]))
        if not np.isfinite(xymin) or not np.isfinite(xymax) or xymin == xymax:
            xymin, xymax = -1.0, 1.0
        ax.plot([xymin, xymax], [xymin, xymax], 'r--', linewidth=1.2)
        ax.set_xlabel(f"True z{j}"); ax.set_ylabel(f"Pred z{j}")
        ax.grid(True, ls=":")
        plt.tight_layout()
        out_png = os.path.join(folder, f"latent_parity_{split}_epoch{epoch}_z{j}.png")
        plt.savefig(out_png, dpi=300); plt.close(fig)
        print(f"[Export] Saved latent parity plot: {out_png}")

def load_ae_split_indices(dataset, split_json=SPLIT_JSON):
    with open(split_json, "r", encoding="utf-8") as f:
        split = json.load(f)
    name_to_idx = {fname: i for i, fname in enumerate(dataset.filenames)}
    def map_names(names):
        idxs, missing = [], []
        for nm in names:
            if nm in name_to_idx: idxs.append(name_to_idx[nm])
            else: missing.append(nm)
        return idxs
    idx_train = map_names(split.get("train", []))
    idx_val   = map_names(split.get("val",   []))
    idx_test  = map_names(split.get("test",  []))
    print(f"[Split from AE] train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")
    return idx_train, idx_val, idx_test

def train():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    encoder = Encoder().to(DEVICE)
    decoder = Decoder().to(DEVICE)
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location='cpu'))
    decoder.load_state_dict(torch.load(DECODER_PATH, map_location='cpu'))
    encoder.eval(); decoder.eval()

    full_dataset = LatentDataset(EXCEL_FILE, encoder, IMAGE_FOLDER)
    latent_mean, latent_std = full_dataset.latent_mean.to(DEVICE), full_dataset.latent_std.to(DEVICE)

    train_idx, val_idx, test_idx = load_ae_split_indices(full_dataset, SPLIT_JSON)
    train_subset = torch.utils.data.Subset(full_dataset, train_idx)
    val_subset   = torch.utils.data.Subset(full_dataset, val_idx)
    test_subset  = torch.utils.data.Subset(full_dataset, test_idx)

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_subset,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(test_subset,  batch_size=BATCH_SIZE, shuffle=False)

    k = min(FIXED_MONITOR_SAMPLES, len(test_subset))
    fixed_monitor_indices = random.sample(range(len(test_subset)), k)
    fixed_monitor_samples = [test_subset[i] for i in fixed_monitor_indices]

    input_dim = len(FEATURE_COLUMNS)
    output_dim = full_dataset.latents.shape[1]
    model = MLPRegressor(input_dim, output_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
    criterion = nn.MSELoss()

    early = EarlyStopping(patience=100, min_delta=1e-5, path=os.path.join(OUTPUT_FOLDER, "es_best_model4.pth"))

    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS+1):
        model.train()
        total_train_loss = 0
        for features, latents, _ in train_loader:
            features, latents = features.to(DEVICE), latents.to(DEVICE)
            optimizer.zero_grad()
            pred_latents = model(features)
            loss = criterion(pred_latents, latents)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * features.size(0)
        avg_train_loss = total_train_loss / max(len(train_subset), 1)

        model.eval()
        total_val_loss = 0
        all_val_preds, all_val_trues, val_names = [], [], []
        with torch.no_grad():
            for features, latents, names in val_loader:
                features, latents = features.to(DEVICE), latents.to(DEVICE)
                pred_latents = model(features)
                loss = criterion(pred_latents, latents)
                total_val_loss += loss.item() * features.size(0)
                all_val_preds.append(pred_latents.cpu())
                all_val_trues.append(latents.cpu())
                val_names.extend(list(names))
        avg_val_loss = total_val_loss / max(len(val_subset), 1)

        y_true = torch.cat(all_val_trues, dim=0).numpy() if all_val_trues else np.zeros((0, output_dim))
        y_pred = torch.cat(all_val_preds, dim=0).numpy() if all_val_preds else np.zeros((0, output_dim))
        r2_val = r2_score(y_true, y_pred) if len(y_true) else float("nan")

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch}/{EPOCHS} - Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
              f"Val R²: {r2_val:.4f} | LR: {scheduler.get_last_lr()[0]:.6e}")

        if early.step(avg_val_loss, model):
            print(f"[EarlyStopping] Stop at epoch {epoch} (best val={early.best:.6f}).")
            break

        scheduler.step()
        if epoch % 500 == 0 or epoch == EPOCHS:
            save_path = os.path.join(OUTPUT_FOLDER, f"shape_prediction_epoch{epoch}-model4.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved model at {save_path}")

        if epoch % 100 == 0 or epoch == EPOCHS:
            plot_loss(train_losses, val_losses, filename=LOSS_PLOT_FILE)

            with torch.no_grad():
                for feature, true_latent_norm, fname in fixed_monitor_samples:
                    feature = feature.unsqueeze(0).to(DEVICE)
                    pred_latent_norm = model(feature)
                    pred_latent = pred_latent_norm * latent_std + latent_mean
                    true_latent = true_latent_norm * latent_std + latent_mean
                    pred_img = decoder(pred_latent).squeeze(0)
                    true_img = decoder(true_latent).squeeze(0)

    early.restore_best(model, DEVICE)
    model.eval()
    total_test_loss = 0
    all_preds, all_targets, test_names = [], [], []
    with torch.no_grad():
        for features, latents, names in test_loader:
            features, latents = features.to(DEVICE), latents.to(DEVICE)
            pred_latents = model(features)
            loss = criterion(pred_latents, latents)
            total_test_loss += loss.item() * features.size(0)
            all_preds.append(pred_latents.cpu()); all_targets.append(latents.cpu())
            test_names.extend(list(names))
    avg_test_loss = total_test_loss / max(len(test_subset), 1)
    with torch.no_grad():
        for feature, true_latent_norm, fname in fixed_monitor_samples:
            feature = feature.unsqueeze(0).to(DEVICE)
            pred_latent_norm = model(feature)
            pred_latent = pred_latent_norm * latent_std + latent_mean
            true_latent = true_latent_norm * latent_std + latent_mean
            pred_img = decoder(pred_latent).squeeze(0)
            true_img = decoder(true_latent).squeeze(0)
            save_comparison(true_img, pred_img, fname, folder=OUTPUT_FOLDER, epoch=epoch)

    y_true = torch.cat(all_targets, dim=0).numpy() if all_targets else np.zeros((0, output_dim))
    y_pred = torch.cat(all_preds,   dim=0).numpy() if all_preds   else np.zeros((0, output_dim))
    r2_test = r2_score(y_true, y_pred) if len(y_true) else float("nan")

    print(f"[Test] Loss: {avg_test_loss:.6f} | R²: {r2_test:.4f}")
 
    if len(y_true) > 0:
        save_latent_parity_csv(y_pred, y_true, test_names, epoch="BEST",
                               split="test", folder=OUTPUT_FOLDER)
        plot_latent_parity(y_pred, y_true, epoch="BEST", split="test",
                           folder=OUTPUT_FOLDER, max_dims=4)

if __name__ == "__main__":
    train()
