from collections import Counter

import networkx as nx
import numpy as np
from networkx.utils import weighted_choice

GRAPH_TYPES = [f"random_{x}_out" for x in range(0, 6)] + ["scale_free", "cycle", "fully_connected"]


def random_k_out_graph(n, k, alpha=50, self_loops=False, seed=None):
    rng = np.random.default_rng(seed)
    if alpha < 0:
        raise ValueError("alpha must be positive")
    G = nx.empty_graph(n, create_using=nx.MultiDiGraph)
    weights = Counter({v: alpha for v in G})
    for i in range(k * n):
        u = rng.choice([v for v, d in G.out_degree() if d < k])

        if not self_loops:
            adjustment = Counter({u: weights[u]})
        else:
            adjustment = Counter()
        v = weighted_choice(weights - adjustment, seed=rng)
        G.add_edge(u, v)
        weights[v] += 1
    return G


def create_random_5_out_graph(n_users, seed=1):
    return random_k_out_graph(n=n_users, k=5, alpha=50, seed=seed)


def create_random_2_out_graph(n_users, seed=1):
    return random_k_out_graph(n=n_users, k=2, alpha=50, seed=seed)


def create_scale_free_graph(n_users, seed=1):
    return nx.scale_free_graph(
        n_users,
        alpha=0.50,
        beta=0.25,
        gamma=0.25,
        delta_in=0.2,
        delta_out=0,
        seed=seed,
    )


def create_cycle_graph(n_users):
    return nx.cycle_graph(n_users).to_directed()


def add_order(graph: nx.Graph):
    new_graph = graph.copy()
    for i in graph:
        new_nbrs = set()
        first_nbrs = list(graph.neighbors(i))
        second_nbrs = []
        for nbr in first_nbrs:
            second_nbrs.extend(list(graph.neighbors(nbr)))
        for sec in second_nbrs:
            if i != sec:
                new_nbrs.add(sec)
        # print(f"Node {i} | 1st: {list(first_nbrs)} | 2nd: {second_nbrs} | all: {new_nbrs} | {len(new_nbrs)}")
        for new_nbr in new_nbrs:
            new_graph.add_edge(i, new_nbr)
    return new_graph


def create_graph(graph_type: str, n_users: int, seed: int = 1, order=1):
    if graph_type in [f"random_{k}_out" for k in range(0, 6)]:
        k = int(graph_type.split("_")[1])
        graph = random_k_out_graph(n=n_users, k=k, seed=seed)
    elif graph_type == "scale_free":
        graph = create_scale_free_graph(n_users, seed)
    elif graph_type == "cycle":
        graph = create_cycle_graph(n_users)
    elif graph_type == "fully_connected":
        graph = nx.complete_graph(n=n_users)
    else:
        raise NotImplementedError
    for _ in range(order - 1):
        graph = add_order(graph)
    return graph
