import logging
from pathlib import Path

from src.graph_splitting import subgraph_splitting, rebuild_graphs
from src.merging_nodes import merge_nodes
from src.spectral_clustering import spectral_clustering
from src.utils import load_graphs_from_TUDataset, save_graphs
from src.utils import set_global_verbose, save_parameters
from src.utils import split_graphs_by_cc, reassemble_cc_graphs

FILE_PARAMETERS = 'parameters.json'


def reduce_graphs(dataset: str,
                  root_dataset: str,
                  split_by_cc: bool,
                  embedding_algorithm: str,
                  dim_embedding: int,
                  clustering_algorithm: str,
                  reduction_factor: int,
                  node_merging_method: str,
                  folder_results: str,
                  verbose: bool,
                  args):
    """

    Args:
        dataset:
        root_dataset:
        split_by_cc:
        embedding_algorithm:
        dim_embedding:
        clustering_algorithm:
        reduction_factor:
        node_merging_method:
        folder_results:
        verbose:
        args:

    Returns:

    """
    set_global_verbose(verbose)
    
    logging.info(f'Run Spectral Graph Reduction - {dataset}')

    Path(folder_results).mkdir(parents=True, exist_ok=True)

    save_parameters(folder_results, FILE_PARAMETERS, vars(args))

    # Load graph dataset
    graphs, classes = load_graphs_from_TUDataset(root_dataset,
                                                 dataset)

    # Split the graph with more than one connected component
    if split_by_cc:
        graphs, cc_to_graph = split_graphs_by_cc(graphs)

    # Find the node clustering for each graph
    n_nodes_per_cluster = reduction_factor
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
