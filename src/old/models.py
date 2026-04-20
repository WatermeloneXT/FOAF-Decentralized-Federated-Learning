import torch
from torch import nn

# SVD model architecture


class OldMF(nn.Module):
    def __init__(self, n_users, n_items):
        super(OldMF, self).__init__()
        torch.manual_seed(1)

        self.user_emb = nn.Embedding(n_users, embedding_dim=30).float()
        self.article_emb = nn.Embedding(n_items, embedding_dim=30).float()
        self.user_bias = nn.Embedding(n_users, embedding_dim=1).float()
        self.article_bias = nn.Embedding(n_items, embedding_dim=1).float()

    def forward(self, inputs):
        user_index, item_index = inputs[0], inputs[1]
        u_e = self.user_emb(user_index)
        i_e = self.article_emb(item_index)
        u_b = self.user_bias(user_index)
        i_b = self.article_bias(item_index)

        x = (u_e * i_e).sum(1) + u_b + i_b

        return x


# Generalized SVD model architecture


class OldGMF(nn.Module):
    def __init__(self, n_users, n_items):
        super(OldGMF, self).__init__()
        torch.manual_seed(1)

        self.user_emb = nn.Embedding(n_users, embedding_dim=30).float()
        self.article_emb = nn.Embedding(n_items, embedding_dim=30).float()
        self.user_bias = nn.Embedding(n_users, embedding_dim=1).float()
        self.article_bias = nn.Embedding(n_items, embedding_dim=1).float()

        self.user_layer_dict = nn.ModuleDict()
        for i in range(n_users):
            self.user_layer_dict["layer{0}".format(i)] = nn.Linear(30, 10)
        self.article_layer_dict = nn.ModuleDict()
        for i in range(n_items):
            self.article_layer_dict["layer{0}".format(i)] = nn.Linear(30, 10)

    def forward(self, inputs):
        user_index, item_index = inputs[0], inputs[1]
        u_e = self.user_emb(user_index)
        i_e = self.article_emb(item_index)
        u_e = self.user_layer_dict["layer{0}".format(int(user_index))](u_e)
        i_e = self.article_layer_dict["layer{0}".format(int(item_index))](i_e)

        u_b = self.user_bias(user_index)
        i_b = self.article_bias(item_index)
        x = (u_e * i_e).sum(1) + u_b + i_b

        return x
