from torch import nn, tensor


class BPR(nn.Module):
    def __init__(self, n_users, n_items, n_factors, sparse=True):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors, sparse=sparse)
        self.item_factors = nn.Embedding(n_items, n_factors, sparse=sparse)
        self.item_biases = nn.Embedding(n_items, 1, sparse=sparse)

    def forward(self, user, pos_idx, neg_idx):
        pos_pred = (self.user_factors(user) * self.item_factors(pos_idx)).sum(-1).unsqueeze(
            -1
        ) + self.item_biases(pos_idx)
        neg_pred = (self.user_factors(user) * self.item_factors(neg_idx)).sum(-1).unsqueeze(
            -1
        ) + self.item_biases(neg_idx)
        return pos_pred, neg_pred

    def predict(self, user, idx):
        pred = (self.user_factors(user) * self.item_factors(idx)).sum(-1).unsqueeze(
            -1
        ) + self.item_biases(idx)
        return pred.squeeze(-1)

class DecBPR(nn.Module):
    def __init__(self, n_items, n_factors, sparse=False):
        super().__init__()
        self.user_factors = nn.Embedding(1, n_factors, sparse=sparse)
        self.item_factors = nn.Embedding(n_items, n_factors, sparse=sparse)
        self.item_bias = nn.Embedding(n_items, 1, sparse=sparse)

    def forward(self, pos_idx, neg_idx):
        user = tensor(0)
        pos_pred = (self.user_factors(user) * self.item_factors(pos_idx)).sum(-1).unsqueeze(
            -1
        ) + self.item_bias(pos_idx)
        neg_pred = (self.user_factors(user) * self.item_factors(neg_idx)).sum(-1).unsqueeze(
            -1
        ) + self.item_bias(neg_idx)
        return pos_pred, neg_pred

    def predict(self, idx):
        user = tensor(0)
        pred = (self.user_factors(user) * self.item_factors(idx)).sum(-1).unsqueeze(
            -1
        ) + self.item_bias(idx)
        return pred.squeeze(-1)
