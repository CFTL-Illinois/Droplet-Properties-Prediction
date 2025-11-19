import os
import json
import torch
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
from matplotlib.ticker import FuncFormatter, NullLocator, FixedLocator, ScalarFormatter
import xgboost as xgb

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
OUTPUT_FOLDER = "train_monitor"
LOSS_PLOT_FILE = "loss-model-4-xgb.png"
FIXED_MONITOR_SAMPLES = 5
SPLIT_JSON = "split_indices.json"  

FEATURE_COLUMNS = ['Nozzle inner diameter (mm)\n', 'Viscosity (mPa.s)',
                   'Flow rate (μl/min)', 'Density (kg/m^3)', 'Surface tension (mN/m)']

XGB_PARAMS = dict(
    learning_rate=0.02,
    max_depth=3,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    reg_alpha=0.0,
    objective='reg:squarederror',
    seed=1,
    tree_method='hist',
    eval_metric='rmse',
)
NUM_BOOST_ROUND = 10000
EARLY_STOPPING_ROUNDS = 100

def set_seed(seed=1):
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
            print(f"[Latent] Loading from {latent_file} & {scaler_file}")
            self.latents = torch.tensor(np.load(latent_file), dtype=torch.float32)
            scaler_data = np.load(scaler_file)
            self.latent_mean = torch.tensor(scaler_data["mean"], dtype=torch.float32)
            self.latent_std  = torch.tensor(scaler_data["std"],  dtype=torch.float32)
        else:
            print("[Latent] Computing with encoder & normalizing...")
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
            print(f"[Latent] Saved to {latent_file} | Scaler to {scaler_file}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        return self.features[idx], self.latents[idx], self.filenames[idx]

def save_comparison(real_img, pred_img, filename, folder="train_monitor", epoch="BEST"):
    diff = torch.abs(real_img - pred_img).squeeze().detach().cpu().numpy()
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(real_img.squeeze().detach().cpu(), cmap='gray'); axs[0].set_title("Original")
    axs[1].imshow(pred_img.squeeze().detach().cpu(), cmap='gray'); axs[1].set_title("Predicted")
    axs[2].imshow(diff, cmap='hot'); axs[2].set_title("Difference")
    for ax in axs: ax.axis("off")
    os.makedirs(folder, exist_ok=True)
    safe = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
    plt.savefig(os.path.join(folder, f"{safe.replace('.bmp', f'_epoch{epoch}.png')}"))
    plt.close()

def plot_loss(train_losses, val_losses, filename="loss_curve-model4-xgb.png"):
    fig, ax = plt.subplots()
    ax.semilogy(train_losses, label="Train RMSE (avg)")
    ax.semilogy(val_losses,   label="Val RMSE (avg)")
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.set_xlabel("Iteration"); ax.set_ylabel("RMSE")
    ax.legend(); ax.grid(True, which="both", linestyle="--", alpha=0.5)
    fig.savefig(filename, dpi=200); plt.close(fig)

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
        tmin, tmax = np.min(T[:, j]), np.max(T[:, j])
        pmin, pmax = np.min(P[:, j]), np.max(P[:, j])
        if not np.isfinite(tmin) or not np.isfinite(tmax) or not np.isfinite(pmin) or not np.isfinite(pmax):
            xymin, xymax = -1.0, 1.0
        else:
            xymin = min(tmin, pmin); xymax = max(tmax, pmax)
            if xymin == xymax:
                xymin, xymax = xymin - 1.0, xymax + 1.0
        ax.plot([xymin, xymax], [xymin, xymax], 'r--', linewidth=1.2)
        ax.set_xlabel(f"True z{j} (normalized)"); ax.set_ylabel(f"Pred z{j} (normalized)")
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

def train_xgb():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    encoder = Encoder().to(DEVICE)
    decoder = Decoder().to(DEVICE)
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location='cpu'))
    decoder.load_state_dict(torch.load(DECODER_PATH, map_location='cpu'))
    encoder.eval(); decoder.eval()

    full_dataset = LatentDataset(EXCEL_FILE, encoder, IMAGE_FOLDER)
    latent_mean = full_dataset.latent_mean.numpy()
    latent_std  = full_dataset.latent_std.numpy()

    train_idx, val_idx, test_idx = load_ae_split_indices(full_dataset, SPLIT_JSON)

    X, Z_norm, names = [], [], []
    for i in range(len(full_dataset)):
        feats, z_norm, nm = full_dataset[i]
        X.append(feats.numpy())
        Z_norm.append(z_norm.numpy())
        names.append(nm)
    X = np.stack(X, axis=0)
    Z_norm = np.stack(Z_norm, axis=0)

    X_train, Ztr = X[train_idx], Z_norm[train_idx]
    X_val,   Zva = X[val_idx],   Z_norm[val_idx]
    X_test,  Zte = X[test_idx],  Z_norm[test_idx]

    Dz = Z_norm.shape[1]
    models = []
    train_metrics = []
    val_metrics = []

    for j in range(Dz):
        dtrain = xgb.DMatrix(X_train, label=Ztr[:, j])
        dval   = xgb.DMatrix(X_val,   label=Zva[:, j])

        evals_result = {}
        booster = xgb.train(
            params=XGB_PARAMS,
            dtrain=dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            evals_result=evals_result,
            verbose_eval=False
        )
        models.append(booster)
        train_metrics.append(evals_result["train"]["rmse"])
        val_metrics.append(evals_result["val"]["rmse"])

    def predict_latent(models, X_):
        dtest = xgb.DMatrix(X_)
        pred_norm_list = []
        for m in models:
            pred = m.predict(dtest, iteration_range=(0, m.best_iteration + 1))
            pred_norm_list.append(pred)
        Z_pred_norm = np.column_stack(pred_norm_list)  
        Z_pred = Z_pred_norm * latent_std + latent_mean
        return Z_pred_norm, Z_pred

    Zva_pred_norm, Zva_pred = predict_latent(models, X_val)
    r2_val = r2_score(Zva, Zva_pred_norm) if len(Zva) > 0 else float("nan")
    print(f"[Val] Latent R² (overall, normalized): {r2_val:.4f}")

    if len(Zva) > 0:
        val_names = [names[i] for i in val_idx]
        save_latent_parity_csv(Zva_pred_norm, Zva, val_names, epoch="BEST",
                               split="val", folder=OUTPUT_FOLDER)
        plot_latent_parity(Zva_pred_norm, Zva, epoch="BEST",
                           split="val", folder=OUTPUT_FOLDER, max_dims=4)

    Zte_pred_norm, Zte_pred = predict_latent(models, X_test)
    r2_test = r2_score(Zte, Zte_pred_norm) if len(Zte) > 0 else float("nan")
    print(f"[Test] Latent R² (overall, normalized): {r2_test:.4f}")

    if len(Zte) > 0:
        test_names = [names[i] for i in test_idx]
        save_latent_parity_csv(Zte_pred_norm, Zte, test_names, epoch="BEST",
                               split="test", folder=OUTPUT_FOLDER)
        plot_latent_parity(Zte_pred_norm, Zte, epoch="BEST",
                           split="test", folder=OUTPUT_FOLDER, max_dims=4)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    k = min(FIXED_MONITOR_SAMPLES, len(test_idx))
    fixed_ids = random.sample(test_idx, k)
    with torch.no_grad():
        for idx in fixed_ids:
            feats, z_true_norm_t, fname = full_dataset[idx]
            feats_np = feats.unsqueeze(0).numpy()

            z_pred_norm = np.array([
                m.predict(xgb.DMatrix(feats_np), iteration_range=(0, m.best_iteration + 1))[0]
                for m in models
            ], dtype=np.float32)
            z_pred = z_pred_norm * latent_std + latent_mean

            z_true = z_true_norm_t.numpy() * latent_std + latent_mean

            z_pred_t = torch.from_numpy(z_pred).unsqueeze(0).to(DEVICE)
            z_true_t = torch.from_numpy(z_true).unsqueeze(0).to(DEVICE)
            pred_img = decoder(z_pred_t).squeeze(0)
            true_img = decoder(z_true_t).squeeze(0)

            save_comparison(true_img, pred_img, fname, folder=OUTPUT_FOLDER, epoch="BEST")

    if train_metrics and val_metrics:
        L = min(min(len(a) for a in train_metrics),
                min(len(b) for b in val_metrics))
        tr_avg = np.mean([a[:L] for a in train_metrics], axis=0)
        va_avg = np.mean([b[:L] for b in val_metrics], axis=0)
        plot_loss(tr_avg, va_avg, filename=LOSS_PLOT_FILE)

if __name__ == "__main__":
    train_xgb()
