import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np


class OldDataset(Dataset):
    def __init__(self, x, y):
        self.x = x.reset_index(drop=True)
        self.y = y.reset_index(drop=True)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        row = self.x.iloc[index]
        user = torch.tensor(row.user_id).int()
        movie = torch.tensor(row.item_id).int()
        target = torch.tensor(self.y.iloc[index]).float()

        return user, movie, target


def read_data(data):
    return tuple(d for d in data[:-1]), data[-1]


def validate(model_dict, val_loader):
    tbar = val_loader
    criterion = nn.MSELoss()
    loss_list = []
    for i in range(len(model_dict)):
        model_dict["model{0}".format(i)].eval()

    with torch.no_grad():
        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)
            logits = model_dict["model{0}".format(int(inputs[0]))](inputs)
            loss = criterion(logits, target)

            loss_list.append(loss.detach().cpu().item())
        avg_loss = np.sqrt(np.mean(loss_list))

    return avg_loss
