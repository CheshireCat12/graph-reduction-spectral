import math
import warnings
from typing import List

import networkx as nx
import numpy as np
import scipy as sp
import scipy.sparse  # call as sp.sparse
import scipy.sparse.linalg  # call as sp.sparse.linalg
from sklearn.cluster import KMeans, AgglomerativeClustering
from tqdm import tqdm


def spectral_clustering(graphs: List[nx.Graph],
                        dim_embedding: int,
                        n_nodes_per_cluster: int,
                        clustering_method: str) -> List[List[int]]:
    """
    Apply the spectral graph clustering method on the graph dataset.
    First, the graph embedding is computed using the spectral embedding.
    Second, the nodes are partitioned using the kmeans clustering algorithm.

    Args:
        graphs : List of NetworkX graphs
        dim_embedding: Number of dimensions of the embedding space
        n_nodes_per_cluster: Dynamically compute the number of clusters
            per graph with `n_clusters = n_nodes / n_nodes_per_cluster`
        clustering_method:

    Returns:
        List of list containing the clusters for each node,
    """

    embedded_graphs = spectral_embedding(graphs, dim_embedding)

    assert clustering_method in CLUSTERING_METHODS,\
        f'Clustering method: {clustering_method} not implemented!'
    clustered_graphs = CLUSTERING_METHODS[clustering_method](embedded_graphs,
                                                             n_nodes_per_cluster)

    return clustered_graphs


def spectral_embedding(graphs: List[nx.Graph],
                       dim_embedding: int) -> List[np.ndarray]:
    """
    Embed all the graphs in a `dim_embedding` vector space by using
    the `dim_embedding`-smallest eigenvalues and their corresponding
    eigenvectors of the Laplacian matrix.

    Use either a dense or a sparse Laplacian matrix depending on the size
    of the graphs to speed up the eigenvalues/eigenvectors computation.

    Args:
        graphs: List of Networkx graphs to embed
        dim_embedding: Number of dimensions of the embedding space

    Returns:
        List of embedded graphs
    """
    embedded_graphs = []
    for graph in tqdm(graphs, desc='Spectral Graph Embedding'):
        n_nodes = len(graph.nodes)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            laplacian = nx.laplacian_matrix(graph)

        if n_nodes < 500:
            eigenvalues, eigenvectors = sp.linalg.eigh(laplacian.todense())
        else:
            k = dim_embedding + 1
            ncv = max(2 * k + 1, int(np.sqrt(n_nodes)))
            eigenvalues, eigenvectors = sp.sparse.linalg.eigsh(laplacian.astype('d'),
                                                               k=k,
                                                               which='SM',
                                                               ncv=ncv)
        # Select the `dim_embedding`-smallest eigenvectors and discarding the first one
        index = np.argsort(eigenvalues)[1:dim_embedding + 1]
        eigenvectors = eigenvectors[:, index]

        embedded_graphs.append(eigenvectors)

    return embedded_graphs


def _get_n_clusters(graph_embedding: np.ndarray,
                    n_nodes_per_cluster: int) -> int:
    """Compute the number of cluster per graph."""
    n_nodes, _ = graph_embedding.shape

    n = n_nodes / n_nodes_per_cluster
    n_clusters = math.ceil(n) if (n % 1) > .5 else math.floor(n)

    return n_clusters or 1


def agglomerative_clustering(embedded_graphs: List[np.ndarray],
                            n_nodes_per_cluster: int = 10) -> List[List[int]]:
    """
    Run agglomerative clustering algorithm on the embedded graphs

    Args:
        embedded_graphs:
        n_nodes_per_cluster:

    Returns:
        List[List[int]]: A list containing the cluster of each node
    """
    clusters_per_graph = []

    for graph_embedding in tqdm(embedded_graphs, desc='Node Clustering - Agglomerative'):
        n_clusters = _get_n_clusters(graph_embedding, n_nodes_per_cluster)
        agglomerative = AgglomerativeClustering(n_clusters=n_clusters)

        kmeans_clusters = agglomerative.fit_predict(graph_embedding)
        clusters_per_graph.append(kmeans_clusters)

    return clusters_per_graph


def kmeans_clustering(embedded_graphs: List[np.ndarray],
                      n_nodes_per_cluster: int = 10) -> List[List[int]]:
    """
    Run KMeans algorithm on the embedded graphs

    Args:
        embedded_graphs (List[np.ndarray]): List of the embedded graphs
        n_nodes_per_cluster (int): Dinamically compute the number of clusters
            per graph with `n_clusters = n_nodes / n_nodes_per_cluster`

    Returns:
        List[List[int]]: A list containing the cluster of each node
    """
    clusters_per_graph = []

    for graph_embedding in tqdm(embedded_graphs, desc='Node Clustering - Kmeans'):
        n_clusters = _get_n_clusters(graph_embedding, n_nodes_per_cluster)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)

        kmeans_clusters = kmeans.fit_predict(graph_embedding)
        clusters_per_graph.append(kmeans_clusters)

    return clusters_per_graph


CLUSTERING_METHODS = {
    'kmeans': kmeans_clustering,
    'agglomerative': agglomerative_clustering,
}
