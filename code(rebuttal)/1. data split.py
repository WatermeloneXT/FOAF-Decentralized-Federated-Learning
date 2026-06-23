"""
Decentralized RS — Train/Test Split Ratio Experiment (ML-100K)
==============================================================
Compares three split strategies:  90/10  |  80/20  |  70/30
Val set is always 20% of the training portion (proportional).
Metrics tracked per ratio:
  • Test RMSE
  • Convergence speed (epochs to best val loss)
  • Communication cost (total commute × parameter bytes)

Drop in project root alongside src/ and dataset/.
Run:  python split_ratio_experiment.py
"""

from pathlib import Path
import os

new_path = Path("/Users/haowen/Documents/Decentral RS/fed-learning-main")

if new_path.exists():
    os.chdir(new_path)
    print(f"Working directory changed to: {Path.cwd()}")
else:
    print("Path does not exist.")


import copy, json, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.optim import SGD
from tqdm import tqdm

from src.models.MatrixFactorization import UMF
from src.graphs import random_k_out_graph
from src.users import User
from src.data_utils import create_dataloader
from src.training.decentralized import (
    decentralized_train_n_epochs,
    decentralized_validate_loop,
)

warnings.filterwarnings("ignore")
torch.manual_seed(0)
np.random.seed(42)


# ──────────────────────────────────────────────────────────────────────────────
# Hyper-parameters  (mirrors your notebook exactly)
# ──────────────────────────────────────────────────────────────────────────────
HP = dict(
    n_factors    = 30,
    sparse       = False,
    batch_size   = 10,
    lr           = 0.03871364416669273,
    weight_decay = 0.14214480688557163,
    mom          = 0.001,
    graph_seed   = 1,
    n_epochs     = 50,
    loader_type  = "rs",
    # DP-SGD
    use_dp       = True,
    dp_clip_norm = 1.0,
    dp_noise_std = 0.01,
)

# Split ratios to benchmark: (train_frac, label)
SPLITS = [
    (0.90, "90/10"),
    (0.80, "80/20"),
    (0.70, "70/30"),
]

# Val is always 20 % of the training portion (proportional)
VAL_FRAC = 0.20


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def make_train_test_split(full_df: pd.DataFrame, train_frac: float):
    """Split full interaction data into train / test by train_frac."""
    return train_test_split(full_df, train_size=train_frac, random_state=42)


def make_val_split(train_df: pd.DataFrame, val_frac: float = VAL_FRAC):
    """Carve val out of train proportionally."""
    return train_test_split(train_df, test_size=val_frac, random_state=0)


def build_users(n_users: int, n_items: int, hp: dict) -> dict:
    users = {}
    for i in tqdm(range(n_users), desc="  Init users", leave=False):
        model = UMF(n_items, n_factors=hp["n_factors"], sparse=hp["sparse"])
        opt   = SGD(model.parameters(), lr=hp["lr"],
                    weight_decay=hp["weight_decay"], momentum=hp["mom"])
        users[i] = User(id=i, model=model, optimizer=opt, model_name="umf")
    return users


def dp_epsilon(sigma, n_steps, n_train, batch_size, delta=1e-5):
    q = batch_size / n_train
    return np.sqrt(2 * n_steps * np.log(1 / delta)) * q / sigma


# ──────────────────────────────────────────────────────────────────────────────
# One experiment
# ──────────────────────────────────────────────────────────────────────────────
def run_experiment(label: str, train_df: pd.DataFrame,
                   val_df: pd.DataFrame, test_df: pd.DataFrame,
                   n_items: int, hp: dict) -> dict:

    print(f"\n{'─'*60}")
    print(f"  Ratio {label}  |  train={len(train_df)}  val={len(val_df)}"
          f"  test={len(test_df)}")
    print(f"{'─'*60}")

    n_users = train_df["user_id"].nunique()

    train_loader = create_dataloader(df=train_df, dl_type=hp["loader_type"],
                                     batch_size=hp["batch_size"])
    val_loader   = create_dataloader(df=val_df,  dl_type=hp["loader_type"])
    test_loader  = create_dataloader(df=test_df, dl_type=hp["loader_type"])

    users = build_users(n_users, n_items, hp)
    graph = random_k_out_graph(n=n_users, k=2, seed=hp["graph_seed"])

    torch.manual_seed(0)
    t0 = time.time()
    train_losses, val_losses, time_per_epoch, commutes = decentralized_train_n_epochs(
        user_models=users,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=hp["n_epochs"],
        graph=graph,
    )
    elapsed = time.time() - t0

    test_rmse         = float(decentralized_validate_loop(users, test_loader))
    best_val          = float(min(val_losses))
    best_epoch        = int(np.argmin(val_losses)) + 1   # 1-indexed
    epochs_run        = len(train_losses)

    # Communication cost: commute × n_factors × 4 bytes (float32)
    param_bytes        = hp["n_factors"] * 4
    total_commute      = int(sum(commutes))
    comm_cost_mb       = round(total_commute * param_bytes / 1e6, 3)
    avg_commute_epoch  = round(total_commute / max(epochs_run, 1), 1)

    # Privacy budget at current noise level
    eps = dp_epsilon(hp["dp_noise_std"], epochs_run * len(train_loader),
                     len(train_df), hp["batch_size"])

    result = dict(
        label             = label,
        n_train           = len(train_df),
        n_val             = len(val_df),
        n_test            = len(test_df),
        n_users           = n_users,
        n_items           = n_items,
        test_rmse         = round(test_rmse, 6),
        best_val_loss     = round(best_val, 6),
        best_epoch        = best_epoch,
        epochs_run        = epochs_run,
        train_losses      = [round(x, 6) for x in train_losses],
        val_losses        = [round(x, 6) for x in val_losses],
        time_per_epoch    = [round(x, 3) for x in time_per_epoch],
        commutes          = commutes,
        total_commute     = total_commute,
        comm_cost_mb      = comm_cost_mb,
        avg_commute_epoch = avg_commute_epoch,
        elapsed_s         = round(elapsed, 2),
        dp_epsilon        = round(eps, 4),
        dp_noise_std      = hp["dp_noise_std"],
    )

    print(f"  ✓  Test RMSE: {test_rmse:.4f}  |  Best Val @ epoch {best_epoch}"
          f"  |  Comm: {comm_cost_mb} MB  |  ε={eps:.2f}  |  {elapsed:.1f}s")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    full_df  = pd.concat([
        pd.read_csv("dataset/ml100k_train.csv"),
        pd.read_csv("dataset/ml100k_test.csv"),
    ]).reset_index(drop=True)

    n_items = full_df["item_id"].nunique()
    print(f"ML-100K  |  Total interactions: {len(full_df)}  |  Items: {n_items}")

    all_results = []
    for train_frac, label in SPLITS:
        train_df, test_df = make_train_test_split(full_df, train_frac)
        train_df, val_df  = make_val_split(train_df)           # proportional val
        res = run_experiment(label, train_df, val_df, test_df, n_items, HP)
        all_results.append(res)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═"*80)
    print(f"{'Ratio':<8} {'TrainN':>7} {'TestN':>7} {'TestRMSE':>10}"
          f" {'BestEpoch':>10} {'CommMB':>9} {'ε':>7}")
    print("═"*80)
    for r in all_results:
        print(f"{r['label']:<8} {r['n_train']:>7} {r['n_test']:>7}"
              f" {r['test_rmse']:>10.4f} {r['best_epoch']:>10}"
              f" {r['comm_cost_mb']:>9.2f} {r['dp_epsilon']:>7.2f}")
    print("═"*80)

    out = Path("split_ratio_results.json")
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved → {out}")
    return all_results


if __name__ == "__main__":
    main()