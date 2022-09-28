from typing import List

import networkx as nx
import numpy as np


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


def _merge_nodes_by_hashing(subgraphs: List[nx.Graph]) -> List[nx.Graph]:
    """

    Args:
        subgraphs:

    Returns:

    """
    pass


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
