import random as rnd
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from tqdm import tqdm

from src.utils import Lookup
from src.utils import plot_graph_nx


def _get_nodes_per_cluster(clusters: List[int]) -> Dict[int, List[int]]:
    """
    Assign each node to its corresponding cluster

    Args:
        clusters:

    Returns:
        Dict of cluster idx with their nodes
    """
    nodes_per_cluster = defaultdict(list)

    for idx_node, idx_cluster in enumerate(clusters):
        nodes_per_cluster[idx_cluster].append(idx_node)

    return nodes_per_cluster


def _get_edges_inter_clusters(graph: nx.Graph,
                              nodes_per_cluster: Dict[int, List[int]]) -> np.ndarray:
    """
    Compute the number of edges that may exist between the different clusters of a graph

    Args:
        graph:
        nodes_per_cluster:

    Returns:

    """
    n_clusters = len(nodes_per_cluster)
    edges_inter_clusters = np.zeros((n_clusters, n_clusters), dtype=np.int32)

    np_graph = nx.to_numpy_array(graph)
    for i, j in combinations(range(n_clusters), 2):
        c_i = nodes_per_cluster[i]
        c_j = nodes_per_cluster[j]

        # Compute the number of edges between cluster c_i and c_j
        n_edges_between_cluster = np.sum(np_graph[np.ix_(c_i, c_j)])
        edges_inter_clusters[i][j] = n_edges_between_cluster

    return edges_inter_clusters


def subgraph_splitting(graphs: List[nx.Graph],
                       node_clusterings: List[List[int]]) -> Tuple[List[nx.Graph], Lookup]:
    """
    Split the graphs according to the node clustering

    Args:
        graphs:
        node_clusterings:

    Returns:

    """
    current_idx_graph = 0
    subgraphs = []
    lookup = Lookup()

    # Zip graph with its corresponding node clustering
    for (idx_graph, graph), clusters in tqdm(zip(enumerate(graphs), node_clusterings),
                                             total=len(graphs),
                                             desc='Subgraph Creation'):
        # nodes_per_cluster = {idx_cluster (int): [idx_node, ...]}
        nodes_per_cluster = _get_nodes_per_cluster(clusters)

        # print(graph)
        # print(nodes_per_cluster[0])
        # plot_graph_nx(graph, [0] * len(graph.nodes))
        # plot_graph_nx(nx.subgraph(graph, nodes_per_cluster[0]), [0] * len(nx.subgraph(graph, nodes_per_cluster[0]).nodes))
        new_subgraphs = [nx.subgraph(graph, nodes_per_cluster[idx_sub])
                         for idx_sub in sorted(nodes_per_cluster.keys())]
        subgraphs.extend(new_subgraphs)

        # for subgraph in new_subgraphs:
        #     plot_graph_nx(subgraph, [0] * len(subgraph.nodes))

        subgraph_indices = [idx_subgraph + current_idx_graph
                            for idx_subgraph, _ in enumerate(new_subgraphs)]
        current_idx_graph += len(subgraph_indices)

        edges_inter_clusters = _get_edges_inter_clusters(graph, nodes_per_cluster)

        lookup.append(subgraph_indices,
                      edges_inter_clusters)

    return subgraphs, lookup


###################################
#       Rebuild graph             #
###################################

def _add_nodes_from_subgraphs(graph: nx.Graph,
                              subgraphs: List[nx.Graph]) -> nx.Graph:
    """
    Add the nodes and induced edges of the subgraphs to the rebuild graph.

    Args:
        graph: Graph to add the nodes
        subgraphs: List of subgraphs of the graph

    Returns:
        A graph containing all the subgraphs

    """
    for subgraph in subgraphs:
        graph = nx.union(graph, subgraph)

    return graph


def _add_inter_subgraphs_edges(graph: nx.Graph,
                               subgraphs: List[nx.Graph],
                               mat_edges_inter_clusters: np.ndarray) -> None:
    """

    Args:
        graph:
        subgraphs:
        mat_edges_inter_clusters:

    Returns:

    """
    for i, j in zip(*np.where(mat_edges_inter_clusters > 0)):
        n_edges = mat_edges_inter_clusters[i][j]
        edges_i = rnd.choices(list(subgraphs[i].nodes), k=n_edges)
        edges_j = rnd.choices(list(subgraphs[j].nodes), k=n_edges)

        for edge_i, edge_j in zip(edges_i, edges_j):
            graph.add_edge(edge_i, edge_j)


def rebuild_graphs(subgraphs: List[nx.Graph],
                   lookup: Lookup) -> List[nx.Graph]:
    """

    Args:
        subgraphs:
        lookup:

    Returns:

    """
    rnd.seed(0)

    graphs = []

    for idx, subgraph_indices, mat_edges_inter_clusters in tqdm(lookup, desc='Rebuild graphs'):
        current_subgraphs = [subgraphs[sub_idx]
                             for sub_idx in subgraph_indices]

        new_graph = nx.Graph()
        new_graph = _add_nodes_from_subgraphs(new_graph, current_subgraphs)

        _add_inter_subgraphs_edges(new_graph,
                                   current_subgraphs,
                                   mat_edges_inter_clusters)

        graphs.append(new_graph)

        # current_clustering = [idx_
        #                       for idx_, subgraph in enumerate(current_subgraphs)
        #                       for i in range(len(subgraph.nodes))]
        # plot_graph_nx(new_graph, current_clustering,
        #               name=f'./test_reconstruct2/{idx}_reconstruct.png')

    return graphs
