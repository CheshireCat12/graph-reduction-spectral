from collections import defaultdict
from typing import List

import networkx as nx
import numpy as np
from tqdm import tqdm


def _merge_nodes_by_summing(subgraphs: List[nx.Graph]) -> List[nx.Graph]:
    """

    Args:
        subgraphs:

    Returns:

    """
    reduced_subgraphs = []

    for idx_subgraph, subgraph in enumerate(subgraphs):
        np_node_values = np.array([node_val['x']
                                   for node_idx, node_val
                                   in subgraph.nodes(data=True)])
        H = nx.Graph()
        idx_first_node = list(subgraph.nodes)[0]
        H.add_node(idx_first_node, x=np.sum(np_node_values, axis=0))

        reduced_subgraphs.append(H)

    return reduced_subgraphs


def create_adj_avg(adj_cur):
    """
    create adjacency
    Source:
    https://github.com/BorgwardtLab/WWL/blob/master/experiments/utilities.py

    Args:
        adj_cur:

    Returns:

    """
    deg = np.sum(adj_cur, axis=1)
    deg = np.asarray(deg).reshape(-1)

    deg[deg != 1] -= 1

    deg = 1 / deg
    deg_mat = np.diag(deg)
    adj_cur = adj_cur.dot(deg_mat.T).T

    return adj_cur


def continuous_weisfeiler_lehman(subgraph: nx.Graph, n_iter: int) -> np.ndarray:
    """
    Adapted from:
    https://github.com/BorgwardtLab/WWL/blob/master/experiments/utilities.py
    
    Args:
        subgraph:
        n_iter:

    Returns:

    """
    node_features = np.array([val for _, val in subgraph.nodes(data='x')])

    graph_feat = [node_features]

    for i in range(n_iter + 1):
        adj_mat = nx.to_numpy_array(subgraph)
        adj_cur = adj_mat + np.identity(adj_mat.shape[0])
        adj_cur = create_adj_avg(adj_cur)

        np.fill_diagonal(adj_cur, 0)
        graph_feat_cur = 0.5 * np.dot(adj_cur, graph_feat[i - 1]) + graph_feat[i - 1]
        graph_feat.append(graph_feat_cur)

    labels_sequence = np.sum(np.sum(graph_feat, axis=0), axis=0)

    return labels_sequence


def _merge_nodes_by_hashing(subgraphs: List[nx.Graph]) -> List[nx.Graph]:
    """

    Args:
        subgraphs:

    Returns:

    """
    reduced_subgraphs = []

    for idx_subgraph, subgraph in tqdm(enumerate(subgraphs),
                                       total=len(subgraphs),
                                       desc='Compute node hash'):
        idx_first_node = list(subgraph.nodes)[0]

        node_hash = continuous_weisfeiler_lehman(subgraph, n_iter=10)

        H = nx.Graph()
        H.add_node(idx_first_node, x=node_hash)

        reduced_subgraphs.append(H)

    return reduced_subgraphs


MERGING_METHODS = {'sum': _merge_nodes_by_summing,
                   'hash': _merge_nodes_by_hashing}


def merge_nodes(subgraphs: List[nx.Graph],
                method: str) -> List[nx.Graph]:
    """
    Inplace reduction of the subgraphs

    Args:
        subgraphs:
        method:

    Returns:

    """
    assert method in MERGING_METHODS, f'The merging method: {method} is not available!'

    reduced_subgraphs = MERGING_METHODS[method](subgraphs)

    return reduced_subgraphs
