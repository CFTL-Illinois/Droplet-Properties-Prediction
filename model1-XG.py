import os
import json
import torch
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
import xgboost as xgb

mpl.rcParams['font.family'] = 'serif'
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
mpl.rcParams['font.serif'] = cmfont.get_name()
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.size'] = 16

NAME = 'Viscosity'
IMAGE_FOLDER = "blueOnly"
EXCEL_FILE = "data.xlsx"
ENCODER_PATH = "es_best_encoder.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_FOLDER = "train_monitor"
LOSS_PLOT_FILE = "loss-model-1-xgb.png"
PHYSICS_SCALER_FILE = "physics_scaler.npz"
SPLIT_JSON = "split_indices.json"

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
NUM_BOOST_ROUND = 5000
EARLY_STOPPING_ROUNDS = 100

INPUT_COLUMNS = ['Nozzle inner diameter (mm)\n', 'Flow rate (μl/min)', 'Density (kg/m^3)', 'Surface tension (mN/m)' ]
OUTPUT_COLUMNS = ['Viscosity (mPa.s)']  

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
                 scaler_file=PHYSICS_SCALER_FILE,
                 latent_file="latents.npy",
                 scaler_file2="latent_scaler.npz"):
        self.encoder = encoder
        self.encoder.eval()
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

        INPUT_COLUMNS_RESOLVED  = [col_d_in, col_flow, col_rho, col_sigma]
        OUTPUT_COLUMNS_RESOLVED = [col_mu]

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

        # latent
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

def clean_filename(name): 
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)

def save_parity_csv(preds_abs, trues_abs, names, epoch, split="val", folder="train_monitor"):
    os.makedirs(folder, exist_ok=True)
    preds = np.array(preds_abs)
    trues = np.array(trues_abs)
    names = list(names)
    data = {"filename": names}
    for j in range(preds.shape[1]):
        tname = clean_filename(OUTPUT_COLUMNS[j] if j < len(OUTPUT_COLUMNS) else f"target{j}")
        data[f"true_{tname}"] = trues[:, j]
        data[f"pred_{tname}"] = preds[:, j]
    df = pd.DataFrame(data)
    out_csv = os.path.join(folder, f"{split}_parity_epoch{epoch}.csv")
    df.to_csv(out_csv, index=False)
    print(f"[Export] {out_csv}")

def plot_scatter(preds, trues, epoch, folder="train_monitor", split="val", filename_suffix="-xgb"):
    os.makedirs(folder, exist_ok=True)
    preds = np.array(preds); trues = np.array(trues)
    for i in range(preds.shape[1]):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(trues[:, i], preds[:, i], alpha=0.7)
        for spine in ax.spines.values(): spine.set_linewidth(2.0)
        ax.tick_params(width=2.0, length=6)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(r'True $\mu$ (mPa$\cdot$s)'); ax.set_ylabel(r'Pred $\mu$ (mPa$\cdot$s)')
        xymin = min(ax.get_xlim()[0], ax.get_ylim()[0]); xymax = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([xymin, xymax], [xymin, xymax], 'r--', linewidth=1.2)
        plt.tight_layout()
        safe_name = clean_filename(OUTPUT_COLUMNS[i])
        out_png = os.path.join(folder, f"scatter_{split}_epoch{epoch}_{safe_name}{filename_suffix}.png")
        plt.savefig(out_png, dpi=300); plt.close(fig)
        print(f"[Export] {out_png}")

def load_ae_split_indices(dataset, split_json=SPLIT_JSON):
    with open(split_json, "r", encoding="utf-8") as f:
        split = json.load(f)
    name_to_idx = {fname: i for i, fname in enumerate(dataset.filenames)}
    def map_names(names):
        idxs, miss = [], []
        for nm in names:
            if nm in name_to_idx: idxs.append(name_to_idx[nm])
            else: miss.append(nm)
        return idxs
    idx_train = map_names(split.get("train", []))
    idx_val   = map_names(split.get("val",   []))
    idx_test  = map_names(split.get("test",  []))
    print(f"[Split] train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")
    return idx_train, idx_val, idx_test

def train_xgb_multiout():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    encoder = Encoder().to(DEVICE)
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location='cpu'))
    encoder.eval()

    dataset = ImageToPhysicsDataset(EXCEL_FILE, encoder, IMAGE_FOLDER)
    target_mean = dataset.target_mean.numpy()   
    target_std  = dataset.target_std.numpy()    

    train_idx, val_idx, test_idx = load_ae_split_indices(dataset, SPLIT_JSON)

    X, y_norm, y_abs, names = [], [], [], []
    for i in range(len(dataset)):
        xi, ti_norm, ti_raw, nm = dataset[i]
        X.append(xi.numpy())
        y_norm.append(ti_norm.numpy())
        y_abs.append(ti_raw.numpy())
        names.append(nm)
    X      = np.stack(X, axis=0)
    y_norm = np.stack(y_norm, axis=0)
    y_abs  = np.stack(y_abs, axis=0)

    X_train, y_train = X[train_idx], y_norm[train_idx]
    X_val,   y_val   = X[val_idx],   y_norm[val_idx]
    X_test,  y_test  = X[test_idx],  y_norm[test_idx]

    dtrain_full = [xgb.DMatrix(X_train, label=y_train[:, j]) for j in range(1)]
    dval_full   = [xgb.DMatrix(X_val,   label=y_val[:, j])   for j in range(1)]

    models = []
    train_metrics = []
    val_metrics = []

    for j in range(1):
        evals_result = {}
        booster = xgb.train(
            params=XGB_PARAMS,
            dtrain=dtrain_full[j],
            num_boost_round=NUM_BOOST_ROUND,
            evals=[(dtrain_full[j], "train"), (dval_full[j], "val")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            evals_result=evals_result,
            verbose_eval=False
        )
        models.append(booster)
        train_metrics.append(evals_result["train"]["rmse"])
        val_metrics.append(evals_result["val"]["rmse"])

    def predict_abs(models, X_):
        dtest = xgb.DMatrix(X_)
        preds_norm_list = []
        for m in models:
            pred = m.predict(dtest, iteration_range=(0, m.best_iteration + 1))
            preds_norm_list.append(pred)
        y_pred_norm = np.column_stack(preds_norm_list)   
        y_pred_log  = y_pred_norm * target_std + target_mean
        return np.exp(y_pred_log)

    val_preds_abs = predict_abs(models, X_val)
    val_trues_abs = np.exp(y_val * target_std + target_mean)
    r2_val_each = [r2_score(val_trues_abs[:, j], val_preds_abs[:, j]) for j in range(1)]
    print(f"[Val] R² σ, μ:", [f"{r:.4f}" for r in r2_val_each])

    if len(val_preds_abs) > 0:
        val_names = [names[i] for i in val_idx]
        save_parity_csv(val_preds_abs, val_trues_abs, val_names,
                        epoch="BEST", split="val", folder=OUTPUT_FOLDER)
        plot_scatter(val_preds_abs, val_trues_abs, epoch="BEST",
                     folder=OUTPUT_FOLDER, split="val", filename_suffix="-xgb")

    test_preds_abs = predict_abs(models, X_test)
    test_trues_abs = np.exp(y_test * target_std + target_mean)
    r2_test_each = [r2_score(test_trues_abs[:, j], test_preds_abs[:, j]) for j in range(1)]
    print(f"[Test] R² σ, μ:", [f"{r:.4f}" for r in r2_test_each])

    if len(test_preds_abs) > 0:
        test_names = [names[i] for i in test_idx]
        save_parity_csv(test_preds_abs, test_trues_abs, test_names,
                        epoch="BEST", split="test", folder=OUTPUT_FOLDER)
        plot_scatter(test_preds_abs, test_trues_abs, epoch="BEST",
                     folder=OUTPUT_FOLDER, split="test", filename_suffix="-xgb")

    if train_metrics and val_metrics:
        L = min(min(len(a) for a in train_metrics),
                min(len(b) for b in val_metrics))
        tr_curve = np.mean([a[:L] for a in train_metrics], axis=0)
        va_curve = np.mean([b[:L] for b in val_metrics], axis=0)
        plt.figure()
        plt.semilogy(tr_curve, label="Train RMSE (avg)")
        plt.semilogy(va_curve, label="Val RMSE (avg)")
        plt.xlabel("Iteration"); plt.ylabel("RMSE")
        plt.legend(); plt.grid(); plt.tight_layout()
        plt.savefig(LOSS_PLOT_FILE, dpi=200)
        plt.close()

if __name__ == "__main__":
    train_xgb_multiout()
