from functools import partialmethod

from tqdm import tqdm

from src.graph_splitting import subgraph_splitting, rebuild_graphs
from src.mergin_nodes import merge_nodes
from src.spectral_clustering import spectral_clustering
from src.utils import load_graphs_from_TUDataset
from src.utils import split_graphs_by_cc, reassemble_cc_graphs, save_graphs


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
    # Globally silent tqdm if verbose is False
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=not verbose)

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
    # for idx, graph in enumerate(graphs):
    #     plot_graph_nx(graph, node_clustering[idx], name=f'./test_reconstruct2/{idx}.png')

    # Split the graphs according to the clustering
    subgraphs, lookup = subgraph_splitting(graphs, node_clustering)

    reduced_subgraphs = merge_nodes(subgraphs, node_merging_method)

    reduced_graphs = rebuild_graphs(reduced_subgraphs, lookup)

    if split_by_cc:
        reduced_graphs = reassemble_cc_graphs(reduced_graphs,
                                              cc_to_graph)

    # plot_graph_nx(reduced_graphs[0], [0] * len(reduced_graphs[0]))

    save_graphs(folder_results, reduced_graphs, classes)
