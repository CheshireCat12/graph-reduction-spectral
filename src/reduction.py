from functools import partialmethod

from tqdm import tqdm

from src.graph_splitting import subgraph_splitting, rebuild_graphs
from src.mergin_nodes import merge_nodes
from src.spectral_clustering import spectral_clustering
from src.utils import load_graphs_from_TUDataset, save_graphs
from src.utils import split_graphs_by_cc, reassemble_cc_graphs
from src.utils import set_global_verbose


def reduce_graphs(dataset: str,
                  root_dataset: str,
                  split_by_cc: bool,
                  embedding_algorithm: str,
                  dim_embedding: int,
                  clustering_algorithm: str,
                  rho: float,
                  node_merging_method: str,
                  folder_results: str,
                  verbose: bool):

    set_global_verbose(verbose)

    # Load graph dataset
    graphs, classes = load_graphs_from_TUDataset(root_dataset,
                                                 dataset)

    # Split the graph with more than one connected component
    if split_by_cc:
        graphs, cc_to_graph = split_graphs_by_cc(graphs)

    # Find the node clustering for each graph
    n_nodes_per_cluster = int(1 / rho)
    node_clustering = spectral_clustering(graphs,
                                          dim_embedding,
                                          n_nodes_per_cluster=n_nodes_per_cluster,
                                          clustering_method=clustering_algorithm)

    # Split the graphs according to the clustering
    subgraphs, lookup = subgraph_splitting(graphs, node_clustering)

    reduced_subgraphs = merge_nodes(subgraphs, node_merging_method)

    reduced_graphs = rebuild_graphs(reduced_subgraphs, lookup)

    if split_by_cc:
        reduced_graphs = reassemble_cc_graphs(reduced_graphs,
                                              cc_to_graph)

    save_graphs(folder_results, reduced_graphs, classes)
