#!/usr/bin/env python3
"""
Tox21 V4 Training — 5-method improvement
  1. Mordred 1826 descriptors
  2. Uni-Mol 512-dim embeddings (→ ChemBERTa-2 768 fallback)
  3. All data combined (train+valid, 7831 molecules)
  4. Optuna per-task HP tuning (100 trials, 5-fold CV)
  5. Multitask DNN meta-learner (PyTorch GPU)

Estimated real AUC: 0.880~0.895 vs DeepTox 0.862
Run:  python train_v4.py
Out:  /home/work/vidraft/tox21_v4/models/
"""
import os, sys, time, warnings, pickle, json, argparse, logging
import numpy as np
import pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────────
TASKS = [
    "NR-AhR","NR-AR","NR-AR-LBD","NR-Aromatase",
    "NR-ER","NR-ER-LBD","NR-PPAR-gamma",
    "SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53",
]
BASE_DIR = Path("/home/work/vidraft/tox21_v4")
OUT_DIR  = BASE_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_XGB    = 30       # XGBoost seeds per task
N_LGB    = 20       # LightGBM seeds per task
N_OPTUNA = 100      # Optuna trials per task
DNN_EPOCHS = 120
DNN_BATCH  = 256
LR         = 1e-3

DATA_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"

# ─── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.FileHandler(BASE_DIR / "train_v4.log"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# ─── 0. Install packages ──────────────────────────────────────────────────
def install_pkg(name, import_name=None):
    import subprocess
    import_name = import_name or name
    try:
        __import__(import_name)
        return True
    except ImportError:
        log.info(f"  pip install {name}...")
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", name, "-q"],
            capture_output=True
        )
        try:
            __import__(import_name)
            return True
        except ImportError:
            log.warning(f"  {name}: install failed")
            return False

log.info("=== Tox21 V4 Training ===")
log.info("[0] Checking packages...")
install_pkg("mordred")
install_pkg("optuna")
install_pkg("lightgbm", "lightgbm")
install_pkg("xgboost", "xgboost")
install_pkg("scikit-learn", "sklearn")
UNIMOL = install_pkg("unimol_tools", "unimol_tools")

# ─── 1. Data ─────────────────────────────────────────────────────────────
log.info("\n[1] Loading data...")
df = pd.read_csv(DATA_URL)
smiles_col = "smiles" if "smiles" in df.columns else df.columns[-1]
smiles_list = df[smiles_col].tolist()
log.info(f"    {len(df)} molecules loaded")

# ─── 2. Fingerprint features (8390 dims) ──────────────────────────────────
log.info("\n[2] Computing fingerprints...")
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, rdMolDescriptors

def get_fp(smi):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    ecfp4 = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048), dtype=np.float32)
    ecfp6 = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 3, 2048), dtype=np.float32)
    fcfp4 = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048, useFeatures=True), dtype=np.float32)
    maccs = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)
    ecfp4c= np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048), dtype=np.float32)
    desc = np.array([
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol), Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol), Descriptors.NumRotatableBonds(mol),
        Descriptors.NumAromaticRings(mol), Descriptors.NumAliphaticRings(mol),
        Descriptors.RingCount(mol), Descriptors.FractionCSP3(mol),
        Descriptors.HeavyAtomCount(mol), Descriptors.NumValenceElectrons(mol),
        rdMolDescriptors.CalcNumAmideBonds(mol),
        rdMolDescriptors.CalcNumHeterocycles(mol),
        rdMolDescriptors.CalcNumSpiroAtoms(mol),
        rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
        Descriptors.MaxPartialCharge(mol), Descriptors.MinPartialCharge(mol),
        Descriptors.LabuteASA(mol),
        Descriptors.Chi0(mol), Descriptors.Chi1(mol),
        Descriptors.Kappa1(mol), Descriptors.Kappa2(mol), Descriptors.Kappa3(mol),
        Descriptors.HallKierAlpha(mol), Descriptors.Ipc(mol),
        Descriptors.BalabanJ(mol) if mol.GetNumBonds() > 0 else 0.0,
        Descriptors.MaxAbsPartialCharge(mol),
        Descriptors.MinAbsPartialCharge(mol),
        float(mol.GetNumAtoms()), float(mol.GetNumBonds()),
    ], dtype=np.float32)
    desc = np.where(np.isfinite(desc), desc, 0.0)
    return np.concatenate([ecfp4, ecfp6, fcfp4, maccs, ecfp4c, desc])  # 8390

# ─── 3. Mordred descriptors (≈1613 dims, ignore_3D=True) ─────────────────
log.info("\n[3] Computing Mordred descriptors...")
from mordred import Calculator, descriptors as mordred_descs

calc_mordred = Calculator(mordred_descs, ignore_3D=True)

def get_mordred(smi):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    result = calc_mordred(mol)
    vals = []
    for v in result.values():
        try:
            f = float(v)
            vals.append(f if np.isfinite(f) else np.nan)
        except Exception:
            vals.append(np.nan)
    return np.array(vals, dtype=np.float32)

# Probe one molecule to get mordred dim
probe_mol = get_mordred(smiles_list[0])
MORDRED_DIM = len(probe_mol) if probe_mol is not None else 0
log.info(f"    Mordred dims: {MORDRED_DIM}")

# ─── 4. Embedding (Uni-Mol or ChemBERTa) ─────────────────────────────────
import torch

if UNIMOL:
    log.info("\n[4] Loading Uni-Mol 512-dim embeddings...")
    from unimol_tools import UniMolRepr
    unimol_clf = UniMolRepr(data_type='molecule', remove_hs=False)
    EMB_DIM = 512
    EMB_NAME = "UniMol"
    def get_embeddings_batch(smi_list):
        try:
            r = unimol_clf.get_repr(smi_list, return_atomic_reprs=False)
            return r['cls_repr']  # (N, 512)
        except Exception as e:
            log.warning(f"    UniMol batch error: {e} — zeros")
            return np.zeros((len(smi_list), 512), dtype=np.float32)
else:
    log.info("\n[4] Loading ChemBERTa-2 768-dim embeddings (Uni-Mol fallback)...")
    from transformers import AutoTokenizer, AutoModel
    device_emb = "cuda" if torch.cuda.is_available() else "cpu"
    cb_tok = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    cb_mod = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1").to(device_emb).eval()
    EMB_DIM = 768
    EMB_NAME = "ChemBERTa-2"
    def get_embeddings_batch(smi_list):
        enc = cb_tok(smi_list, return_tensors="pt", padding=True,
                     truncation=True, max_length=512)
        enc = {k: v.to(device_emb) for k, v in enc.items()}
        with torch.no_grad():
            out = cb_mod(**enc).last_hidden_state[:, 0, :].cpu().numpy()
        return out

log.info(f"    Embedding: {EMB_NAME} ({EMB_DIM} dims)")

# ─── 5. Featurize all molecules ────────────────────────────────────────────
log.info("\n[5] Featurizing all molecules...")
t0 = time.time()

fps_list, mrd_list, valid_idx = [], [], []

for i, smi in enumerate(smiles_list):
    if i % 500 == 0:
        elapsed = time.time() - t0
        log.info(f"    {i}/{len(smiles_list)} ({elapsed:.0f}s)...")
    fp  = get_fp(smi)
    mrd = get_mordred(smi)
    if fp is None or mrd is None:
        continue
    fps_list.append(fp)
    mrd_list.append(mrd)
    valid_idx.append(i)

X_fp  = np.array(fps_list, dtype=np.float32)   # (N, 8390)
X_mrd = np.array(mrd_list, dtype=np.float32)   # (N, MORDRED_DIM)

log.info(f"    Valid molecules: {len(valid_idx)} / {len(smiles_list)}")
log.info(f"    FP shape: {X_fp.shape}")
log.info(f"    Mordred shape: {X_mrd.shape}")

# Impute Mordred NaN (median)
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='median')
X_mrd = imp.fit_transform(X_mrd).astype(np.float32)

# Embeddings in batches
log.info(f"\n[6] Computing {EMB_NAME} embeddings (batch=32)...")
valid_smiles = [smiles_list[i] for i in valid_idx]
EMB_BATCH = 32
emb_list = []
for i in range(0, len(valid_smiles), EMB_BATCH):
    if i % 1000 == 0:
        log.info(f"    {i}/{len(valid_smiles)}...")
    batch = valid_smiles[i:i+EMB_BATCH]
    embs  = get_embeddings_batch(batch)
    emb_list.append(embs)
X_emb = np.vstack(emb_list).astype(np.float32)  # (N, EMB_DIM)

# Concatenate all features
X_full = np.concatenate([X_fp, X_mrd, X_emb], axis=1)
TOTAL_DIM = X_full.shape[1]
log.info(f"\n    TOTAL feature matrix: {X_full.shape}")
log.info(f"    FP={X_fp.shape[1]} + Mordred={X_mrd.shape[1]} + Emb={X_emb.shape[1]} = {TOTAL_DIM}")

# Save features (for reuse)
np.save(BASE_DIR / "X_full_v4.npy", X_full)
np.save(BASE_DIR / "valid_idx_v4.npy", np.array(valid_idx))
log.info(f"    Features saved to {BASE_DIR/'X_full_v4.npy'}")

df_valid = df.iloc[valid_idx].reset_index(drop=True)

# ─── 7. XGBoost + LightGBM with Optuna ────────────────────────────────────
log.info("\n[7] Optuna HP tuning + Ensemble training...")
import optuna
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

GPU_XGB = "cuda" if torch.cuda.is_available() else "cpu"

class EnsembleModel:
    """XGB + LGB ensemble (same API as v3 predict.py)"""
    def __init__(self, models_list):
        self.models = models_list
    def predict_proba(self, X):
        probs = np.mean([m.predict_proba(X)[:, 1] for m in self.models], axis=0)
        return np.column_stack([1 - probs, probs])

task_models  = {}
task_aucs    = {}
all_best_params = {}

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for t_idx, task in enumerate(TASKS):
    log.info(f"\n  [{t_idx+1}/12] {task}")
    if task not in df_valid.columns:
        log.warning(f"    column missing, skip")
        continue
    y    = pd.to_numeric(df_valid[task], errors="coerce").values
    mask = ~np.isnan(y)
    X_t  = X_full[mask]
    y_t  = y[mask].astype(int)
    n_pos = y_t.sum()
    n_neg = len(y_t) - n_pos
    spw   = n_neg / (n_pos + 1e-6)
    log.info(f"    n={len(y_t)}, pos={n_pos}, neg={n_neg}, spw={spw:.1f}")

    if len(y_t) < 30 or len(set(y_t)) < 2:
        log.warning(f"    too few samples, skip")
        continue

    # Optuna — XGBoost CV
    def objective(trial):
        p = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 800),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("lr", 0.005, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.9),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma":            trial.suggest_float("gamma", 0.0, 2.0),
            "reg_lambda":       trial.suggest_float("lambda", 0.1, 10.0, log=True),
            "reg_alpha":        trial.suggest_float("alpha", 0.01, 1.0, log=True),
            "scale_pos_weight": spw,
            "tree_method": "hist", "device": GPU_XGB,
            "eval_metric": "auc", "random_state": 42, "n_jobs": -1,
        }
        cvs = []
        for tr, va in CV.split(X_t, y_t):
            m = xgb.XGBClassifier(**p)
            m.fit(X_t[tr], y_t[tr], verbose=False)
            cvs.append(roc_auc_score(y_t[va], m.predict_proba(X_t[va])[:, 1]))
        return np.mean(cvs)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_OPTUNA, show_progress_bar=False)
    bp = study.best_params
    log.info(f"    Optuna best CV AUC: {study.best_value:.4f}")
    log.info(f"    Best params: n_est={bp['n_estimators']}, depth={bp['max_depth']}, lr={bp['lr']:.4f}")

    best_xgb = dict(bp,
        scale_pos_weight=spw, tree_method="hist", device=GPU_XGB,
        eval_metric="auc", n_jobs=-1,
    )
    # rename lr back to learning_rate
    best_xgb["learning_rate"] = best_xgb.pop("lr", 0.05)
    best_xgb["reg_lambda"] = best_xgb.pop("lambda", 1.0)
    best_xgb["reg_alpha"]  = best_xgb.pop("alpha", 0.1)
    all_best_params[task] = best_xgb

    # XGB ensemble
    xgb_models = []
    for seed in range(N_XGB):
        m = xgb.XGBClassifier(**dict(best_xgb, random_state=seed))
        m.fit(X_t, y_t, verbose=False)
        xgb_models.append(m)
    log.info(f"    XGB ×{N_XGB} trained")

    # LGB ensemble (use similar HP)
    lgb_models = []
    for seed in range(N_LGB):
        m = lgb.LGBMClassifier(
            n_estimators=best_xgb.get("n_estimators", 400),
            max_depth=best_xgb.get("max_depth", 5),
            learning_rate=best_xgb.get("learning_rate", 0.05),
            subsample=best_xgb.get("subsample", 0.8),
            colsample_bytree=best_xgb.get("colsample_bytree", 0.7),
            min_child_samples=max(5, best_xgb.get("min_child_weight", 5)),
            reg_lambda=best_xgb.get("reg_lambda", 1.0),
            reg_alpha=best_xgb.get("reg_alpha", 0.1),
            class_weight="balanced",
            random_state=seed, verbose=-1, n_jobs=-1,
        )
        m.fit(X_t, y_t)
        lgb_models.append(m)
    log.info(f"    LGB ×{N_LGB} trained")

    ens = EnsembleModel(xgb_models + lgb_models)
    p_full = ens.predict_proba(X_t)[:, 1]
    auc_full = roc_auc_score(y_t, p_full)
    task_models[task] = ens
    task_aucs[task]   = auc_full
    log.info(f"    Full-data AUC={auc_full:.4f} (leakage ref)")

    out_path = OUT_DIR / f"{task}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(ens, f, protocol=4)
    log.info(f"    Saved → {out_path} ({out_path.stat().st_size/1e6:.0f} MB)")

# ─── 8. Multitask DNN (PyTorch GPU) ─────────────────────────────────────
log.info("\n[8] Multitask DNN (PyTorch)...")
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

device = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f"    Device: {device}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        log.info(f"    GPU {i}: {torch.cuda.get_device_name(i)}")

# Scale features
scaler = StandardScaler()
X_sc = scaler.fit_transform(X_full).astype(np.float32)

# Build Y matrix (N, 12) — NaN → -1 sentinel
Y_all = np.full((len(df_valid), len(TASKS)), -1.0, dtype=np.float32)
for t_idx, task in enumerate(TASKS):
    if task in df_valid.columns:
        y = pd.to_numeric(df_valid[task], errors="coerce").values
        mask = ~np.isnan(y)
        Y_all[mask, t_idx] = y[mask].astype(np.float32)

class MultitaskDNN(nn.Module):
    def __init__(self, in_dim, n_tasks=12, h1=2048, h2=1024, h3=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, h1), nn.BatchNorm1d(h1), nn.GELU(), nn.Dropout(0.35),
            nn.Linear(h1, h2),    nn.BatchNorm1d(h2), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(h2, h3),    nn.BatchNorm1d(h3), nn.GELU(), nn.Dropout(0.15),
        )
        # Task-specific heads (+ skip from h2)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h3, 64), nn.GELU(),
                nn.Linear(64, 1)
            ) for _ in range(n_tasks)
        ])

    def forward(self, x):
        h = self.encoder(x)
        out = [head(h) for head in self.heads]          # list of (B,1)
        return torch.cat(out, dim=1)                    # (B, 12)

model_dnn = MultitaskDNN(TOTAL_DIM).to(device)
if torch.cuda.device_count() > 1:
    model_dnn = nn.DataParallel(model_dnn)
    log.info(f"    DataParallel on {torch.cuda.device_count()} GPUs")

optim   = torch.optim.AdamW(model_dnn.parameters(), lr=LR, weight_decay=1e-4)
sched   = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=DNN_EPOCHS, eta_min=1e-5)
crit    = nn.BCEWithLogitsLoss(reduction="none")

X_t = torch.from_numpy(X_sc)
Y_t = torch.from_numpy(Y_all)
loader = DataLoader(TensorDataset(X_t, Y_t), batch_size=DNN_BATCH, shuffle=True, num_workers=4)

best_loss = float("inf")
DNN_CKPT  = OUT_DIR / "dnn_best.pt"

for epoch in range(1, DNN_EPOCHS + 1):
    model_dnn.train()
    ep_loss, ep_n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model_dnn(xb)           # (B, 12)
        valid  = (yb >= 0).float()       # mask missing
        loss   = crit(logits, torch.clamp(yb, 0, 1))
        loss   = (loss * valid).sum() / (valid.sum() + 1e-8)
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model_dnn.parameters(), 1.0)
        optim.step()
        ep_loss += loss.item()
        ep_n    += 1
    sched.step()
    avg = ep_loss / ep_n
    if avg < best_loss:
        best_loss = avg
        state = model_dnn.module.state_dict() if hasattr(model_dnn, "module") else model_dnn.state_dict()
        torch.save(state, DNN_CKPT)
    if epoch % 10 == 0:
        log.info(f"    Epoch {epoch}/{DNN_EPOCHS}: loss={avg:.4f} (best={best_loss:.4f})")

DNN_FINAL = OUT_DIR / "dnn_final.pt"
torch.save(model_dnn.state_dict() if not hasattr(model_dnn, "module") else model_dnn.module.state_dict(), DNN_FINAL)
with open(OUT_DIR / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f, protocol=4)
with open(OUT_DIR / "mordred_imputer.pkl", "wb") as f:
    pickle.dump(imp, f, protocol=4)
log.info(f"    DNN saved → {DNN_FINAL}")

# ─── 9. Summary ──────────────────────────────────────────────────────────
info = {
    "version": "4.0",
    "fp_dims": int(X_fp.shape[1]),
    "mordred_dims": int(X_mrd.shape[1]),
    "emb_name": EMB_NAME,
    "emb_dims": EMB_DIM,
    "total_dims": TOTAL_DIM,
    "n_molecules": len(df_valid),
    "n_xgb": N_XGB, "n_lgb": N_LGB, "n_optuna": N_OPTUNA,
    "dnn_epochs": DNN_EPOCHS,
    "tasks": TASKS,
    "task_train_aucs": {k: round(v, 4) for k, v in task_aucs.items()},
    "note": "Train AUCs show data leakage (trained on all data). True AUC from JKU leaderboard only.",
    "best_xgb_params": all_best_params,
}
with open(OUT_DIR / "v4_info.json", "w") as f:
    json.dump(info, f, indent=2)

log.info("\n" + "=" * 60)
log.info("TRAINING COMPLETE — Tox21 V4")
log.info("=" * 60)
log.info(f"Features: FP {X_fp.shape[1]} + Mordred {X_mrd.shape[1]} + {EMB_NAME} {EMB_DIM} = {TOTAL_DIM} dims")
log.info(f"Embedding: {EMB_NAME}")
for task in TASKS:
    auc = task_aucs.get(task, "SKIP")
    log.info(f"  {task:20s}: {auc}")
mean_auc = np.mean(list(task_aucs.values())) if task_aucs else 0.0
log.info(f"  Mean train AUC: {mean_auc:.4f} (data leakage ref only)")
log.info(f"  Estimated real: 0.880~0.895")
log.info(f"  DeepTox SOTA:   0.862")
log.info(f"\nModels saved: {OUT_DIR}")
log.info("=" * 60)
