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
from sklearn.metrics import r2_score
import re
from Autoencoder import Encoder
import matplotlib as mpl
import matplotlib.font_manager as font_manager
from matplotlib.ticker import FixedLocator, ScalarFormatter

mpl.rcParams['font.family'] = 'serif'
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
mpl.rcParams['font.serif'] = cmfont.get_name()
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.size'] = 16

NAME = 'Viscosity&Tension'
IMAGE_FOLDER = "blueOnly"
EXCEL_FILE = "data.xlsx"
ENCODER_PATH = "es_best_encoder.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 3000
BATCH_SIZE = 32
LR = 1e-3
OUTPUT_FOLDER = "train_monitor"
LOSS_PLOT_FILE = "loss-model-3.png"
PHYSICS_SCALER_FILE = "physics_scaler.npz"
FIXED_MONITOR_SAMPLES = 5
SPLIT_JSON = "split_indices.json"  

INPUT_COLUMNS = ['Nozzle inner diameter (mm)\n', 'Flow rate (μl/min)', 'Density (kg/m^3)']
OUTPUT_COLUMNS = ['Surface tension (mN/m)', 'Viscosity (mPa.s)']  

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
            header_idx = i; break
    if header_idx is None: header_idx = 0
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

class ImageToPhysicsDataset(torch.utils.data.Dataset):
    def __init__(self, excel_file, encoder, image_folder,
                 scaler_file=PHYSICS_SCALER_FILE, latent_file="latents.npy", scaler_file2="latent_scaler.npz"):
        self.encoder = encoder; self.encoder.eval()
        self.image_folder = image_folder
        self.transform = transforms.ToTensor()
        self.device = next(encoder.parameters()).device

        df = _read_excel_smart(excel_file).dropna(how="all")
        pick = _build_col_map(df.columns)

        col_exp   = pick(r"\bexp\.?\s*no\b", r"\bexp\b")
        col_d_in  = pick(r"nozzle.*inner.*diameter.*\(mm\)", "nozzle inner diameter")
        col_flow  = pick(r"flow.*(μl|ul)\/?min", "flow rate")
        col_rho   = pick(r"density.*\(kg\/?m\^?3\)", "density")
        col_sigma = pick(r"surface.*tension.*\(m?n\/?m\)", "surface tension")
        col_mu    = pick(r"viscosity.*\(mpa\.?s\)", "viscosity")

        INPUT_COLUMNS_RESOLVED  = [col_d_in, col_flow, col_rho]
        OUTPUT_COLUMNS_RESOLVED = [col_sigma, col_mu]  

        df = df.dropna(subset=[col_exp])
        exp_str = df[col_exp].astype(str)
        df['exp_id'] = exp_str.str.extract(r'(\d+)').astype(int)
        df['image_filename'] = df['exp_id'].apply(lambda x: f"Exp{x:03d}_blueOnly.bmp")
        self.filenames = df['image_filename'].tolist()

        self.input_features = torch.tensor(df[INPUT_COLUMNS_RESOLVED].values, dtype=torch.float32)

        targets_raw = torch.tensor(df[OUTPUT_COLUMNS_RESOLVED].values, dtype=torch.float32)
        self.targets_log = torch.log(targets_raw + 1e-8)

        if os.path.exists(scaler_file):
            scaler_data = np.load(scaler_file)
            self.target_mean = torch.tensor(scaler_data["mean"], dtype=torch.float32)
            self.target_std  = torch.tensor(scaler_data["std"],  dtype=torch.float32)
        else:
            self.target_mean = self.targets_log.mean(dim=0)
            self.target_std  = self.targets_log.std(dim=0)
            np.savez(scaler_file, mean=self.target_mean.numpy(), std=self.target_std.numpy())

        self.targets     = (self.targets_log - self.target_mean) / self.target_std
        self.targets_raw = targets_raw

        if os.path.exists(latent_file) and os.path.exists(scaler_file2):
            self.latents = torch.tensor(np.load(latent_file), dtype=torch.float32)
            scaler_data = np.load(scaler_file2)
            self.latent_mean = torch.tensor(scaler_data["mean"], dtype=torch.float32)
            self.latent_std  = torch.tensor(scaler_data["std"],  dtype=torch.float32)
        else:
            latents = []
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
            np.savez(scaler_file2, mean=self.latent_mean.numpy(), std=self.latent_std.numpy())

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        combined_input = torch.cat([self.latents[idx], self.input_features[idx]], dim=0)
        return combined_input, self.targets[idx], self.targets_raw[idx], self.filenames[idx]

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
    def __init__(self, patience=30, min_delta=1e-5, path="es_best_model3.pth"):
        self.patience = patience; self.min_delta = min_delta
        self.best = float("inf"); self.num_bad = 0
        self.path = path
    def step(self, val_metric, model):
        improved = (self.best - val_metric) > self.min_delta
        if improved:
            self.best = val_metric; self.num_bad = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.num_bad += 1
        return self.num_bad >= self.patience
    def restore_best(self, model, device):
        if os.path.exists(self.path):
            model.load_state_dict(torch.load(self.path, map_location=device))

def plot_loss(train_losses, test_losses, filename=f"loss_curve_{NAME}-model3.png"):
    plt.figure()
    plt.semilogy(train_losses, label="Train Loss")
    plt.semilogy(test_losses, label="Val Loss")  
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.grid(); plt.savefig(filename); plt.close()

def clean_filename(name): return re.sub(r'[^a-zA-Z0-9_]', '_', name)

def save_parity_csv(preds_abs, trues_abs, names, epoch, split="val", folder="train_monitor"):
    os.makedirs(folder, exist_ok=True)
    preds = np.array(preds_abs)
    trues = np.array(trues_abs)
    names = list(names)
    num_targets = preds.shape[1] if preds.ndim > 1 else 1
    data = {"filename": names}
    for j in range(num_targets):
        tname = clean_filename(OUTPUT_COLUMNS[j] if j < len(OUTPUT_COLUMNS) else f"target{j}")
        data[f"true_{tname}"] = trues[:, j]
        data[f"pred_{tname}"] = preds[:, j]
    df = pd.DataFrame(data)
    out_csv = os.path.join(folder, f"{split}_parity_epoch{epoch}.csv")
    df.to_csv(out_csv, index=False)
    print(f"[Export] Saved parity CSV: {out_csv}")

def plot_scatter(preds, trues, epoch, folder="train_monitor", split="val", filename_suffix="-model3"):
    os.makedirs(folder, exist_ok=True)
    preds = np.array(preds); trues = np.array(trues)
    num_targets = preds.shape[1]
    for i in range(num_targets):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(trues[:, i], preds[:, i], alpha=0.7)
        for spine in ax.spines.values(): spine.set_linewidth(2.2)
        ax.tick_params(width=2.2, length=7)
        if i == 0:
            ticks = [20, 40, 60, 80]
            ax.set_xscale('linear'); ax.set_yscale('linear')
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.yaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
            ax.ticklabel_format(style='plain', useOffset=False, axis='both')
            ax.set_xlabel(r'True $\sigma$ (mN/m)'); ax.set_ylabel(r'Predicted $\sigma$ (mN/m)')
        else:
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xlabel(r'True $\mu$ (mPa$\cdot$s)'); ax.set_ylabel(r'Predicted $\mu$ (mPa$\cdot$s)')
        xymin = min(ax.get_xlim()[0], ax.get_ylim()[0]); xymax = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([xymin, xymax], [xymin, xymax], 'r--', linewidth=1.5)
        plt.tight_layout()
        safe_name = clean_filename(OUTPUT_COLUMNS[i] if i < len(OUTPUT_COLUMNS) else f"target{i}")
        out_png = os.path.join(folder, f"scatter_{split}_epoch{epoch}_{safe_name}{filename_suffix}.png")
        plt.savefig(out_png, dpi=300)
        plt.close(fig)
        print(f"[Export] Saved parity plot: {out_png}")

def load_ae_split_indices(dataset, split_json=SPLIT_JSON):
    with open(split_json, "r", encoding="utf-8") as f: split = json.load(f)
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
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location='cpu'))
    encoder.eval()

    dataset = ImageToPhysicsDataset(EXCEL_FILE, encoder, IMAGE_FOLDER)
    target_mean, target_std = dataset.target_mean.to(DEVICE), dataset.target_std.to(DEVICE)

    train_idx, val_idx, test_idx = load_ae_split_indices(dataset, SPLIT_JSON)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, train_idx),
                                               batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, val_idx),
                                             batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, test_idx),
                                              batch_size=BATCH_SIZE, shuffle=False)

    input_dim = dataset.latents.shape[1] + len(INPUT_COLUMNS)
    output_dim = dataset.targets.shape[1]
    model = MLPRegressor(input_dim, output_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
    criterion = nn.MSELoss()

    early = EarlyStopping(patience=100, min_delta=1e-5, path=os.path.join(OUTPUT_FOLDER, "es_best_model3.pth"))

    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS+1):
        model.train()
        total_train_loss = 0
        all_train_preds, all_train_trues = [], []
        for inputs, targets_norm, targets_raw, _ in train_loader:
            inputs, targets_norm = inputs.to(DEVICE), targets_norm.to(DEVICE)
            optimizer.zero_grad()
            preds_norm = model(inputs)
            loss = criterion(preds_norm, targets_norm)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * inputs.size(0)

            preds_log = preds_norm * target_std + target_mean
            preds = torch.exp(preds_log)
            all_train_preds.append(preds.detach().cpu().numpy())
            all_train_trues.append(targets_raw.numpy())
        avg_train_loss = total_train_loss / max(len(train_idx), 1)

        model.eval()
        total_val_loss = 0
        all_val_preds, all_val_trues, val_names = [], [], []
        with torch.no_grad():
            for inputs, targets_norm, targets_raw, names in val_loader:
                inputs, targets_norm = inputs.to(DEVICE), targets_norm.to(DEVICE)
                preds_norm = model(inputs)
                loss = criterion(preds_norm, targets_norm)
                total_val_loss += loss.item() * inputs.size(0)

                preds_log = preds_norm * target_std + target_mean
                preds = torch.exp(preds_log)
                all_val_preds.append(preds.cpu().numpy())
                all_val_trues.append(targets_raw.numpy())
                val_names.extend(list(names))
        avg_val_loss = total_val_loss / max(len(val_idx), 1)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        all_train_preds = np.concatenate(all_train_preds, axis=0) if all_train_preds else np.zeros((0, output_dim))
        all_train_trues = np.concatenate(all_train_trues, axis=0) if all_train_trues else np.zeros((0, output_dim))
        all_val_preds   = np.concatenate(all_val_preds,   axis=0) if all_val_preds   else np.zeros((0, output_dim))
        all_val_trues   = np.concatenate(all_val_trues,   axis=0) if all_val_trues   else np.zeros((0, output_dim))

        r2_train_each = [r2_score(all_train_trues[:, i], all_train_preds[:, i]) if len(all_train_trues) else float("nan")
                         for i in range(output_dim)]
        r2_val_each   = [r2_score(all_val_trues[:, i],   all_val_preds[:, i])   if len(all_val_trues)   else float("nan")
                         for i in range(output_dim)]

        print(f"Epoch {epoch}/{EPOCHS} - "
              f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
              f"R² Train: {[f'{r:.4f}' for r in r2_train_each]} | R² Val: {[f'{r:.4f}' for r in r2_val_each]} | "
              f"LR: {scheduler.get_last_lr()[0]:.6e}")

        if epoch % 100 == 0 or epoch == EPOCHS:
            plot_loss(train_losses, val_losses, filename=LOSS_PLOT_FILE)

        if early.step(avg_val_loss, model):
            print(f"[EarlyStopping] Stop at epoch {epoch} (best val={early.best:.6f}).")
            break

        scheduler.step()
        if epoch % 500 == 0 or epoch == EPOCHS:
            save_path = os.path.join(OUTPUT_FOLDER, f"{NAME}_prediction_epoch{epoch}-model3.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved model at {save_path}")

    early.restore_best(model, DEVICE)
    model.eval()
    total_test_loss = 0
    all_test_preds, all_test_trues, test_names = [], [], []
    with torch.no_grad():
        for inputs, targets_norm, targets_raw, names in test_loader:
            inputs, targets_norm = inputs.to(DEVICE), targets_norm.to(DEVICE)
            preds_norm = model(inputs)
            loss = criterion(preds_norm, targets_norm)
            total_test_loss += loss.item() * inputs.size(0)
            preds_log = preds_norm * target_std + target_mean
            preds = torch.exp(preds_log)
            all_test_preds.append(preds.cpu().numpy())
            all_test_trues.append(targets_raw.numpy())
            test_names.extend(list(names))
    avg_test_loss = total_test_loss / max(len(test_idx), 1)

    all_test_preds = np.concatenate(all_test_preds, axis=0) if all_test_preds else np.zeros((0, output_dim))
    all_test_trues = np.concatenate(all_test_trues, axis=0) if all_test_trues else np.zeros((0, output_dim))
    r2_test_each = [r2_score(all_test_trues[:, i], all_test_preds[:, i]) if len(all_test_trues) else float("nan")
                    for i in range(output_dim)]

    print(f"[Test] Loss: {avg_test_loss:.6f} | R² Test: {[f'{r:.4f}' for r in r2_test_each]}")
    
    if len(all_test_preds) > 0:
        save_parity_csv(all_test_preds, all_test_trues, test_names, epoch="BEST",
                        split="test", folder=OUTPUT_FOLDER)
        plot_scatter(all_test_preds, all_test_trues, epoch="BEST", folder=OUTPUT_FOLDER,
                     split="test", filename_suffix="-model3")

if __name__ == "__main__":
    train()
