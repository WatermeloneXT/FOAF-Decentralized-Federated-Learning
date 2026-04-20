from torch import nn, tensor

# SVD model architecture


class HMModel(nn.Module):
    def __init__(self, n_users, n_items):
        super().__init__()

        self.user_emb = nn.Embedding(n_users, embedding_dim=30)
        self.article_emb = nn.Embedding(n_items, embedding_dim=30)
        self.user_bias = nn.Embedding(n_users, embedding_dim=1)
        self.article_bias = nn.Embedding(n_items, embedding_dim=1)

    def forward(self, inputs):
        user_index, item_index = inputs[:, 0], inputs[:, 1]
        u_e = self.user_emb(user_index)
        i_e = self.article_emb(item_index)
        u_b = self.user_bias(user_index)
        i_b = self.article_bias(item_index)

        x = (u_e * i_e).sum(1) + u_b + i_b

        return x


class MF(nn.Module):
    def __init__(self, n_users: int, n_items: int, n_factors: int = 30, sparse: bool = False):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors, sparse=sparse)
        self.item_factors = nn.Embedding(n_items, n_factors, sparse=sparse)
        self.user_bias = nn.Embedding(n_users, 1, sparse=sparse)
        self.item_bias = nn.Embedding(n_items, 1, sparse=sparse)

    def forward(self, users, items):
        preds = (self.user_factors(users) * self.item_factors(items)).sum(dim=1, keepdims=True)
        preds += self.user_bias(users)
        preds += self.item_bias(items)
        return preds

    def predict(self, users, items):
        return self.forward(users, items)


class UMF(nn.Module):
    def __init__(self, n_items, n_factors=30, sparse=False):
        super().__init__()
        self.user_factors = nn.Embedding(1, n_factors, sparse=sparse)
        self.item_factors = nn.Embedding(n_items, n_factors, sparse=sparse)
        self.user_bias = nn.Embedding(1, 1, sparse=sparse)
        self.item_bias = nn.Embedding(n_items, 1, sparse=sparse)

    def forward(self, users, items):
        user = tensor(0)
        preds = (self.user_factors(user) * self.item_factors(items)).sum(dim=1, keepdims=True)
        preds += self.user_bias(user)
        preds += self.item_bias(items)
        return preds

    def predict(self, users, items):
        return self.forward(users, items)
