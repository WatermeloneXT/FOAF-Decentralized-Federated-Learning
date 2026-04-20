import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from src.users import IndependentUser


class TripletDataset(Dataset):
    """Selects u, i, j triplets."""

    def __init__(self, data: np.ndarray, cutoff=1):
        """
        data: array of (user_id, item_id, rating)
        cutoff: x is positive if cutoff <= x
        """
        data = torch.tensor(data)
        self.pos_ui_pairs = data[data[:, 2] >= cutoff, :2]
        self.neg_item_map = {}
        n_items = len(torch.unique(data[:, 1]))
        self.all_items = torch.arange(start=0, end=n_items)

        neg_mask = data[:, 2] < cutoff
        for user_id in torch.unique(data[:, 0]):
            neg_items = data[(data[:, 0] == user_id) & neg_mask, 1]
            if neg_items.numel() > 0:
                self.neg_item_map[int(user_id)] = data[(data[:, 0] == user_id) & neg_mask, 1]
            else:
                self.neg_item_map[int(user_id)] = self.all_items

    def __len__(self):
        return len(self.pos_ui_pairs)

    def __getitem__(self, idx):
        user_id = int(self.pos_ui_pairs[idx, 0])
        sample_idx = torch.randint(len(self.neg_item_map[user_id]), (1,))
        return (
            self.pos_ui_pairs[idx, 0],
            self.pos_ui_pairs[idx, 1],
            self.neg_item_map[user_id][sample_idx].squeeze(-1),
        )


class DecValidationDataset(Dataset):
    def __init__(self, train_data, val_data, cutoff, n_items) -> None:
        train_data = set(train_data[:, 1].tolist())
        self.items_to_rank = torch.as_tensor(list(set(range(n_items)) - train_data), dtype=int)
        self.true_pos = val_data[val_data[:, 2] >= cutoff, 1]

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx):
        return (self.items_to_rank, self.true_pos)


class ValidationDatasetIgnoreTraining(Dataset):
    def __init__(self, train_data, val_data, cutoff, n_items) -> None:
        self.val_data = torch.tensor(val_data)
        self.cutoff = cutoff
        self.all_items = set(range(n_items))
        self.users_train_data = {}

        train_data = torch.tensor(train_data)
        for user_id in torch.unique(self.val_data[:, 0]):
            self.users_train_data[user_id.item()] = set(
                train_data[train_data[:, 0] == user_id, 1].tolist()
            )

    def __len__(self) -> int:
        """Number of users."""
        return len(torch.unique(self.val_data[:, 0]))

    def __getitem__(self, idx):
        true_pos = self.val_data[
            (self.val_data[:, 0] == idx) & (self.val_data[:, 2] >= self.cutoff), 1
        ]
        users_items_to_rank = self.all_items - self.users_train_data[idx]
        return (
            torch.ones(len(users_items_to_rank), dtype=int) * idx,
            torch.as_tensor(list(users_items_to_rank), dtype=int),
            true_pos,
        )


class ValidationDataset(Dataset):
    """Return (user_id, item_id's to rank, item_id's of true pos."""

    def __init__(self, data, cutoff, n_items) -> None:
        self.data = torch.tensor(data)
        self.cutoff = cutoff
        self.items_to_rank = torch.arange(n_items)

    def __len__(self) -> int:
        """Number of users."""
        return len(torch.unique(self.data[:, 0]))

    def __getitem__(self, idx):
        true_pos = self.data[(self.data[:, 0] == idx) & (self.data[:, 2] >= self.cutoff), 1]
        return torch.ones_like(self.items_to_rank) * idx, self.items_to_rank, true_pos


def bpr_loss_fn(pos, neg):
    return -(pos - neg).sigmoid().log().sum()


def centralized_train_loop(model, train_loader, optimizer):
    model.train()
    total_n_obs = 0
    total_sum_loss = 0
    # tbar = tqdm(train_loader)
    ten_percent = len(train_loader) // 10
    for idx, (u, i, j) in enumerate(train_loader):
        n_obs = u.numel()
        optimizer.zero_grad()
        pos_score, neg_score = model(u, i, j)
        loss = bpr_loss_fn(pos_score, neg_score)
        loss.backward()  # Calculate Gradients
        optimizer.step()  # Current user's update

        total_n_obs += n_obs
        total_sum_loss += loss.detach().numpy() * n_obs
        avg_loss = total_sum_loss / total_n_obs
        if idx % ten_percent == 0:
            if np.isnan(avg_loss):
                print("Training Loss is NaN")
                raise ValueError
            # if idx % 1000 == 0:
            # tbar.set_description(f"Training Loss: {avg_loss:.05f} ")
    return total_sum_loss / total_n_obs


def centralized_validate_loop(model, val_loader, top_k):
    """Calculates average recall and ndcg at k."""
    all_hr, all_ndcg = [], []
    model.eval()
    with torch.no_grad():
        for idx, (user_id, items_to_rank, true_pos) in enumerate(val_loader):
            # if inputs.ndim == 3:
            #     inputs = inputs.squeeze(0)
            #     target = target.squeeze(0)
            # print(f"{user_id.shape=}, {items_to_rank.shape=}, {true_pos.shape=}")
            # print(f"True Positive: {true_pos}")
            pos_score = model.predict(user_id, items_to_rank)

            # print(pos_score.shape)
            _, indices = torch.topk(pos_score, top_k)
            # print(torch.take(items_to_rank, indices).shape)
            topk_item = torch.take(items_to_rank, indices).squeeze(0).numpy().tolist()
            # print(topk_item)
            # print(true_pos.squeeze(0).numpy().tolist())
            for gt_item in true_pos.squeeze(0).numpy().tolist():
                # print(gt_item)
                all_ndcg.append(ndcg(gt_item, topk_item))
                all_hr.append(hit(gt_item, topk_item))

    return np.mean(all_hr), np.mean(all_ndcg)


def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0


def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index + 2))
    return 0


def metrics(model, test_loader, top_k):
    HR, NDCG = [], []

    for user, item_i, item_j in test_loader:
        # user = user.cuda()
        # item_i = item_i.cuda()
        # item_j = item_j.cuda()  # not useful when testing

        prediction_i, _ = model(user, item_i, item_j)
        _, indices = torch.topk(prediction_i, top_k)
        recommends = torch.take(item_i, indices).numpy().tolist()

        gt_item = item_i[0].item()
        HR.append(hit(gt_item, recommends))
        NDCG.append(ndcg(gt_item, recommends))

    return np.mean(HR), np.mean(NDCG)


def share_gradient(user_id: int, users: dict[int, IndependentUser]):
    user = users[user_id]
    item_parameter_names = ["item_factors.weight", "item_bias.weight"]

    user_gradients = {name: user.model.get_parameter(name).grad for name in item_parameter_names}
    # print(f"{user_gradients[item_parameter_names[0]].sum()=}")

    # print(f"{user_id=}, {user.neighbors=}")

    for neighbor_id in user.neighbors:
        neighbor = users[neighbor_id]
        neighbor_model = neighbor.model
        neighbor_optimizer = neighbor.optimizer
        neighbor_model.zero_grad()  # required - diverges without this line
        for name in item_parameter_names:
            param = neighbor_model.get_parameter(name)
            param.grad = user_gradients[name].clone()
        neighbor_optimizer.step()


def dec_validate_single_user(model, val_loader, top_k):
    all_hr, all_ndcg = [], []
    model.eval()
    with torch.no_grad():
        for idx, (items_to_rank, true_pos) in enumerate(val_loader):
            pos_score = model.predict(items_to_rank)
            _, indices = torch.topk(pos_score, top_k)
            topk_item = torch.take(items_to_rank, indices).squeeze(0).numpy().tolist()
            for gt_item in true_pos.squeeze(0).numpy().tolist():
                all_ndcg.append(ndcg(gt_item, topk_item))
                all_hr.append(hit(gt_item, topk_item))
    return np.mean(all_hr), np.mean(all_ndcg)


def decentralized_validate_loop(users: dict[int, IndependentUser], topk: int, is_test=False):
    n_users = len(users)
    all_hr, all_ndcg = np.zeros(n_users), np.zeros(n_users)
    with torch.no_grad():
        for user_id in users:
            user = users[user_id]
            if is_test:
                u_hr, u_ndcg = dec_validate_single_user(user.model, user.test_dl, top_k=topk)
            else:
                u_hr, u_ndcg = dec_validate_single_user(user.model, user.val_dl, top_k=topk)
            all_hr[user_id] = u_hr
            all_ndcg[user_id] = u_ndcg
    return all_hr, all_ndcg


def decentralized_training_loop(users: dict[int, IndependentUser]):
    """Loop through each user once per epoch."""
    n_users = len(users)
    user_dl = DataLoader(
        TensorDataset(torch.arange(n_users, dtype=int)), batch_size=1, shuffle=True
    )
    # shuffled_user_ids = torch.randperm(n_users)
    for idx, user_id in enumerate(user_dl):
        user_id = user_id[0].item()
        user = users[user_id]
        optimizer = user.optimizer
        optimizer.zero_grad()

        # Manually iterate through each user's dataloader so all items are seen.
        try:
            _, i, j = next(user.train_iter)
            # print(f"Next: {_}, {i}, {j}")
            pos, neg = user.model.forward(i, j)
            # print(f"{pos.shape=}, {neg.shape=}")
            loss = bpr_loss_fn(pos, neg)
            # print(f"{loss=}")
            loss.backward()
            share_gradient(user_id, users)
            optimizer.step()
        except StopIteration:
            user.reset_train_iter()
