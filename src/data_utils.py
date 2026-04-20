from enum import Enum, auto
from math import ceil
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from dotenv import dotenv_values
from torch.utils.data import (DataLoader, Dataset, IterableDataset,
                              TensorDataset)

#DATA_DIR = Path(dotenv_values(".env").get("DATA_DIR"))

DL_TYPES = ("oaat", "aaat", "rs", "centralized", "urs", "userprop")

# class DataloaderTypes(Enum):
#     OAAT = "oaat"
#     AAAT = "aaat"
#     RS = "rs"


class DataName(Enum):
    HM = auto()
    HM_SUBSET = auto()
    HM_2881 = auto()
    HM_4000 = auto()
    ML100K = auto()
    ML100K_100 = auto()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def double_chunks(X, y, n):
    for i in range(0, X.shape[0], n):
        yield X[i : i + n], y[i : i + n]


class UserStratifiedDataset(Dataset):
    """Randomly select data from a given user."""

    def __init__(self, x: np.ndarray, y: np.ndarray, batch_size: int | None = None):
        self.x = torch.tensor(x)
        self.y = torch.tensor(y).unsqueeze(-1)
        self.batch_size = batch_size

    def __len__(self):
        # return 
        return len(self.x[:, 0].unique())

    def __getitem__(self, user):
        user_idx = self.x[:, 0] == user
        x = self.x[user_idx, :]
        y = self.y[user_idx]
        if self.batch_size is None:
            return x, y
        else:
            n = x.shape[0]
            idx = torch.randint(low=0, high=n, size=(self.batch_size,))
            return x[idx, :], y[idx]


class UserProportionDataset(Dataset):
    """Randomly select a proportion of a user's items."""

    def __init__(self, x: np.ndarray, y: np.ndarray, p: float) -> None:
        self.x = torch.tensor(x)
        self.y = torch.tensor(y).unsqueeze(-1)
        self.p = p

    def __len__(self):
        return len(self.x[:, 0].unique())

    def __getitem__(self, user):
        user_idx = self.x[:, 0] == user
        x = self.x[user_idx, :]
        y = self.y[user_idx]
        n_batch = max(1, round(x.shape[0] * self.p))
        idx = torch.randint(low=0, high=x.shape[0], size=(n_batch,))
        return x[idx, :], y[idx]


class MyIterableDataset(IterableDataset):
    def __init__(self, X, y, batch_size=3, shuffle=True):
        super().__init__()
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return ceil(self.X.shape[0] / self.batch_size)

    def __iter__(self):
        n = self.X.shape[0]
        if self.shuffle:
            idx = torch.randperm(n)
        else:
            idx = range(n)
        return double_chunks(
            self.X[idx], self.y[idx], self.batch_size
        )  # , chunks(self.y, self.batch_size)


class ChainDatasetShuffle(IterableDataset):
    def __init__(self, datasets, shuffle=True):
        super().__init__()
        self.datasets = datasets
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            idx = torch.randperm(len(self.datasets))
            for i in idx:
                yield from self.datasets[i]
            # for d in self.datasets[idx]:
            #     yield from d
        else:
            for d in self.datasets:
                yield from d

    def __len__(self):
        total = 0
        for d in self.datasets:
            assert isinstance(d, IterableDataset), "ChainDataset only supports IterableDataset"
            total += len(d)  # type: ignore[arg-type]
        return total


class UserRandomSamplingDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, batch_size: int = 10):
        self.x = torch.tensor(x)
        self.y = torch.tensor(y).unsqueeze(-1)
        self.batch_size = batch_size

    def __len__(self):
        # return len(self.x[:, 0].unique())
        return self.x.shape[0]

    def __getitem__(self, idx):
        user = self.x[idx, 0]
        user_idx = self.x[:, 0] == user
        x = self.x[user_idx, :]
        y = self.y[user_idx]
        batch_size = min(x.shape[0], self.batch_size)
        selected_indices = torch.randint(x.shape[0], size=(batch_size,))
        return x[selected_indices, :], y[selected_indices]


def create_batched_dataloaders(train_df, batch_size=5):
    datasets = []
    for g, _df in train_df.groupby("user_id"):
        x = _df[["user_id", "item_id"]].to_numpy()
        y = _df["rating"].to_numpy()
        datasets.append(MyIterableDataset(torch.tensor(x), torch.tensor(y), batch_size=batch_size))
    cdataset = ChainDatasetShuffle(datasets)
    return DataLoader(cdataset)


def read_in_data(data_name) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Option in `DataName`"""
    if data_name == "HM":
        train_df = pd.read_csv(DATA_DIR / "hm_train_df.csv")
        val_df = pd.read_csv(DATA_DIR / "hm_val_df.csv")
        train_df.columns = ["user_id", "item_id", "bought"]
        val_df.columns = ["user_id", "item_id", "bought"]
    elif data_name == "HM_4000":
        train_df = pd.read_csv(DATA_DIR / "hm_4000_item_subset_train.csv")
        val_df = pd.read_csv(DATA_DIR / "hm_4000_item_subset_test.csv")
    elif data_name == "HM_2881":
        train_df = pd.read_csv(DATA_DIR / "hm_2881_item_subset_train.csv")
        val_df = pd.read_csv(DATA_DIR / "hm_2881_item_subset_test.csv")
    elif data_name == "HM_SUBSET":
        train_df = pd.read_csv(DATA_DIR / "hm_subset_train.csv")
        val_df = pd.read_csv(DATA_DIR / "hm_subset_test.csv")
    elif data_name == "ML100K":
        train_df = pd.read_csv(DATA_DIR / "ml100k_train.csv")
        val_df = pd.read_csv(DATA_DIR / "ml100k_test.csv")
    elif data_name == "ML100K_100":
        train_df = pd.read_csv(DATA_DIR / "ml100k_100user_train.csv")
        val_df = pd.read_csv(DATA_DIR / "ml100k_100user_test.csv")
    else:
        raise NotImplementedError
    return train_df, val_df


def create_dataloader(df, dl_type, batch_size=1, p=None) -> DataLoader:
    assert dl_type in DL_TYPES

    X, y = df.iloc[:, :2].to_numpy(), df.iloc[:, 2].to_numpy()
    if dl_type == "oaat":
        return DataLoader(
            TensorDataset(torch.tensor(X), torch.tensor(y).unsqueeze(-1)),
            batch_size=1,
            shuffle=True,
        )
    elif dl_type == "aaat":
        return DataLoader(UserStratifiedDataset(X, y), shuffle=True)
    elif dl_type == "rs":
        return DataLoader(
            UserRandomSamplingDataset(X, y, batch_size=batch_size),
            batch_size=1,
            shuffle=True,
        )
    elif dl_type == "centralized":
        return DataLoader(
            TensorDataset(torch.tensor(X), torch.tensor(y).unsqueeze(-1)),
            batch_size=batch_size,
            shuffle=True,
        )
    elif dl_type == "urs":
        return DataLoader(
            UserStratifiedDataset(X, y, batch_size=batch_size),
            batch_size=1,
            shuffle=True,
        )
    elif dl_type == "userprop":
        assert p is not None
        return DataLoader(
            UserProportionDataset(X, y, p),
            batch_size=1,
            shuffle=True,
        )
    else:
        raise NotImplementedError
