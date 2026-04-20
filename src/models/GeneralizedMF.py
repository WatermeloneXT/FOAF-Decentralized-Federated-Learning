# Generalized SVD model architecture
from torch import nn, tensor


class GeneralizedMFSharedItemLayer(nn.Module):
    def __init__(self, n_items, n_factors=30, layer_size=10):
        super().__init__()

        self.user_factors = nn.Embedding(1, embedding_dim=n_factors).float()
        self.item_factors = nn.Embedding(n_items, embedding_dim=n_factors).float()
        self.user_bias = nn.Embedding(1, embedding_dim=1).float()
        self.item_bias = nn.Embedding(n_items, embedding_dim=1).float()

        self.user_layers = nn.Linear(n_factors, layer_size)

        # self.user_layer_dict = nn.ModuleDict()
        # self.user_layer_dict[f"layer{0}"] = nn.Linear(n_factors, 10)
        self.item_layers = nn.Linear(n_factors, layer_size)

    def forward(self, users, items):
        user = tensor(0)
        u_e = self.user_factors(user)
        i_e = self.item_factors(items)
        u_e = self.user_layers(u_e)
        i_e = self.item_layers(i_e)

        u_b = self.user_bias(user)
        i_b = self.item_bias(items)
        x = (u_e * i_e).sum(1) + u_b + i_b
        return x


class GeneralizedMF(nn.Module):
    def __init__(self, n_items, n_factors=30, layer_size=10):
        super().__init__()

        self.user_factors = nn.Embedding(1, embedding_dim=n_factors).float()
        self.item_factors = nn.Embedding(n_items, embedding_dim=n_factors).float()
        self.user_bias = nn.Embedding(1, embedding_dim=1).float()
        self.item_bias = nn.Embedding(n_items, embedding_dim=1).float()

        self.user_layers = nn.Linear(n_factors, layer_size)

        # self.user_layer_dict = nn.ModuleDict()
        # self.user_layer_dict[f"layer{0}"] = nn.Linear(n_factors, 10)
        self.item_layer_dict = nn.ModuleDict()
        for i in range(n_items):
            self.item_layer_dict["item{0}".format(i)] = nn.Linear(n_factors, layer_size)

    def forward(self, users, items):
        user = tensor(0)
        u_e = self.user_factors(user)
        i_e = self.item_factors(items)
        u_e = self.user_layers(u_e)
        i_e = self.item_layer_dict["item{0}".format(int(items))](i_e)

        u_b = self.user_bias(user)
        i_b = self.item_bias(items)
        x = (u_e * i_e).sum(1) + u_b + i_b
        return x


class GeneralizedDeepMF(nn.Module):
    def __init__(self, n_items, n_factors=7, layer_size=5, n_layers=2, sparse=False):
        super().__init__()
        self.user_factors = nn.Embedding(1, embedding_dim=n_factors).float()
        self.item_factors = nn.Embedding(n_items, embedding_dim=n_factors).float()
        self.user_bias = nn.Embedding(1, embedding_dim=1).float()
        self.item_bias = nn.Embedding(n_items, embedding_dim=1).float()

        user_layers = [nn.Linear(n_factors, layer_size)]
        for i in range(n_layers-1):
            user_layers.append(nn.Linear(layer_size, layer_size))
        self.user_layers = nn.Sequential(
            *user_layers
        )

        self.item_layer_dict = nn.ModuleDict()
        for i in range(n_items):
            item_layers = [nn.Linear(n_factors, layer_size)]
            for _ in range(n_layers-1):
                item_layers.append(nn.Linear(layer_size, layer_size))
            self.item_layer_dict["item{0}".format(i)] = nn.Sequential(
                *item_layers
            )

    def forward(self, users, items):
        user = tensor(0)
        u_e = self.user_factors(user)
        i_e = self.item_factors(items)
        u_e = self.user_layers(u_e)
        i_e = self.item_layer_dict["item{0}".format(int(items))](i_e)

        u_b = self.user_bias(user)
        i_b = self.item_bias(items)
        x = (u_e * i_e).sum(1) + u_b + i_b
        return x


class NonlinearGMF(nn.Module):
    def __init__(self, n_items, n_factors=7, layer_size=5, n_layers=2, sparse=False):
        super().__init__()
        self.user_factors = nn.Embedding(1, embedding_dim=n_factors).float()
        self.item_factors = nn.Embedding(n_items, embedding_dim=n_factors).float()
        self.user_bias = nn.Embedding(1, embedding_dim=1).float()
        self.item_bias = nn.Embedding(n_items, embedding_dim=1).float()

        user_layers = [nn.Linear(n_factors, layer_size)]
        for i in range(n_layers-1):
            user_layers.append(nn.ReLU())
            user_layers.append(nn.Linear(layer_size, layer_size))
        self.user_layers = nn.Sequential(
            *user_layers
        )

        self.item_layer_dict = nn.ModuleDict()
        for i in range(n_items):
            item_layers = [nn.Linear(n_factors, layer_size)]
            for _ in range(n_layers-1):
                item_layers.append(nn.ReLU())
                item_layers.append(nn.Linear(layer_size, layer_size))
            self.item_layer_dict["item{0}".format(i)] = nn.Sequential(
                *item_layers
            )

    def forward(self, users, items):
        user = tensor(0)
        u_e = self.user_factors(user)
        i_e = self.item_factors(items)
        u_e = self.user_layers(u_e)
        i_e = self.item_layer_dict["item{0}".format(int(items))](i_e)

        u_b = self.user_bias(user)
        i_b = self.item_bias(items)
        x = (u_e * i_e).sum(1) + u_b + i_b
        return x
