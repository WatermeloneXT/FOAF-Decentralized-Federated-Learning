
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import mlflow
import mlflow.pytorch

# ------------------------------
# Matrix Factorization Model
# ------------------------------
class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, n_factors=32):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.tensor([0.0]))

        nn.init.normal_(self.user_factors.weight, std=0.1)
        nn.init.normal_(self.item_factors.weight, std=0.1)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, users, items):
        dot = (self.user_factors(users) * self.item_factors(items)).sum(dim=1, keepdim=True)
        preds = dot + self.user_bias(users) + self.item_bias(items) + self.global_bias
        return preds.squeeze()

# ------------------------------
# Dataset Class
# ------------------------------
class RatingsDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df["user_id"].values, dtype=torch.long)
        self.items = torch.tensor(df["item_id"].values, dtype=torch.long)
        self.ratings = torch.tensor(df["rating"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

# ------------------------------
# Training and Evaluation
# ------------------------------
def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for users, items, ratings in tqdm(train_loader):
        # Communication cost: count transmitted parameters per batch
        # Assume full model sent/received once per epoch
        if epoch == 1 and idx == 0:
            param_bytes = sum(p.numel() for p in model.parameters()) * 4  # float32 = 4 bytes
            mlflow.log_metric("communication_bytes_per_epoch", param_bytes)
    
        users, items, ratings = users.to(device), items.to(device), ratings.to(device)
        optimizer.zero_grad()
        preds = model(users, items)
        loss = criterion(preds, ratings)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(ratings)
    return np.sqrt(total_loss / len(train_loader.dataset))

def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for users, items, ratings in val_loader:
            users, items, ratings = users.to(device), items.to(device), ratings.to(device)
            preds = model(users, items)
            loss = criterion(preds, ratings)
            total_loss += loss.item() * len(ratings)
    return np.sqrt(total_loss / len(val_loader.dataset))

# ------------------------------
# Main Execution with MLflow
# ------------------------------
def main():
    df = pd.read_csv("ml100k.csv")  # ensure columns: user_id, item_id, rating
    df["user_id"] -= df["user_id"].min()
    df["item_id"] -= df["item_id"].min()

    n_users = df["user_id"].nunique()
    n_items = df["item_id"].nunique()
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_loader = DataLoader(RatingsDataset(train_df), batch_size=1024, shuffle=True)
    val_loader = DataLoader(RatingsDataset(val_df), batch_size=1024)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MatrixFactorization(n_users, n_items).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    criterion = nn.MSELoss()

    mlflow.start_run()
    mlflow.log_params({
        "model": "MatrixFactorization",
        "n_factors": 32,
        "optimizer": "Adam",
        "lr": 0.01,
        "weight_decay": 1e-5,
        "batch_size": 1024
    })

    for epoch in range(1, 21):
        train_rmse = train_model(model, train_loader, optimizer, criterion, device)
        val_rmse = evaluate_model(model, val_loader, criterion, device)
        mlflow.log_metric("train_rmse", train_rmse, step=epoch)
        mlflow.log_metric("val_rmse", val_rmse, step=epoch)
        print(f"Epoch {epoch}: Train RMSE = {train_rmse:.4f}, Val RMSE = {val_rmse:.4f}")

    mlflow.pytorch.log_model(model, "model")
    mlflow.end_run()

if __name__ == "__main__":
    main()
