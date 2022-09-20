import csv
import json
import os
import pathlib
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch_geometric.utils as tutils
from torch_geometric.datasets import TUDataset
from tqdm import tqdm


############################################
#                  Loader                  #
############################################

def load_graphs_from_TUDataset(root: str,
                               name_dataset: str) -> Tuple[List[nx.Graph], np.ndarray]:
    """
    Use the Pytorch Geometric (PyG) loader to download the graphs from the TUDataset.
    The raw graphs from PyG are saved in `root`.

    The created NetworkX graphs have a node attribute `x` that is an `np.array`.
    The corresponding class of each graph is also retrieved from the TUDataset graphs.

    Args:
        root: Path where to save the raw graph dataset
        name_dataset: Name of the graph dataset to load

    Returns:
        List of the loaded NetworkX graphs and `np.array` of the corresponding class of each graph
    """
    dataset = TUDataset(root=root, name=name_dataset)

    node_attr = 'x'

    # Convert the PyG graphs into NetworkX graphs
    nx_graphs = [tutils.to_networkx(graph, node_attrs=[node_attr], to_undirected=True)
                 for graph in dataset]

    # Cast the node attribute x from list into np.array
    for nx_graph in nx_graphs:
        for idx_node, data_node in nx_graph.nodes(data=True):
            nx_graph.nodes[idx_node][node_attr] = np.array(data_node[node_attr])

    graph_cls = np.array([int(graph.y) for graph in dataset])

    return nx_graphs, graph_cls


############################################
#              graph writer                #
############################################

def _modify_node_type_to_str(graph: nx.Graph) -> nx.Graph:
    """
    Modify the type of the node attribute.
    A shallow copy of the graph is created to modify the node attribute's type
    Change the `np.ndarray` or `list` node attribute `x` into `str` attribute

    Args:
        graph: Graph to modify the type of the nodes' attribute

    Returns:
        Modified copy of the graph
    """
    # TODO: Handle the np.ndarray node attribute
    # TODO: Should the attr_node 'x' be a function parameter?
    node_attr = 'x'
    new_graph = graph.copy()

    for idx_node, data_node in graph.nodes(data=True):
        new_graph.nodes[idx_node][node_attr] = str(data_node[node_attr])

    return new_graph


def _write_classes(graph_cls: np.ndarray,
                   filename: str) -> None:
    """
    Save the class of each graph in a tuple (graph_name, graph_cls).

    Args:
        graph_cls: List of graph classes. The idx in the array of
                   the class must correspond to the graph idx to which it belongs.
        filename: Filename where to save the graph classes

    Returns:

    """
    with open(filename, mode='w') as csv_file:
        fieldnames = ['graph_file', 'class']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for idx_graph, cls in enumerate(graph_cls):
            writer.writerow({'graph_file': f'gr_{idx_graph}.graphml',
                             'class': str(cls)})


def save_graphs(path: str,
                graphs: List[nx.Graph],
                graph_cls: Optional[np.ndarray] = None) -> None:
    """
    Save the given graphs in `path` under `.graphml` format.
    The saved graphs are named according to their position in the list
    (e.g., the first graph in the list is named `graph_0.graphml`).

    The `np.ndarray` node attribute `x` is modified into `str` attribute

    Args:
        path: Path to the folder where to save the graphs.
            If the folder doesn't exist it is created.
        graphs: List of graphs to save
        graph_cls:
    """
    # Make sure that the path to the folder exist, if not create it.
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    for idx_graph, graph in tqdm(enumerate(graphs),
                                 total=len(graphs),
                                 desc='Save Graphs'):
        # Change the np.ndarray or list node attribute to str (graph copy)
        copied_graph = _modify_node_type_to_str(graph)

        filename = f'graph_{idx_graph}.graphml'
        path_to_graph = os.path.join(path, filename)
        nx.write_graphml_lxml(copied_graph, path_to_graph, prettyprint=True)

    if graph_cls is not None:
        filename_cls = os.path.join(path, 'graph_classes.csv')
        _write_classes(graph_cls, filename_cls)


############################################
#             save parameters              #
############################################

def save_parameters(folder: str, filename: str, parameters: dict):
    """
    Save the parameters in `folder/filename`

    Args:
        folder:
        filename: File in which to save the parameters.
        parameters: Parameters to save

    Returns:

    """
    filename = os.path.join(folder, filename)
    with open(filename, 'w') as file:
        json.dump(parameters, file, indent=4)


############################################
#             Graph splitter               #
############################################

def split_graphs_by_cc(graphs: List[nx.Graph]) -> Tuple[List[nx.Graph], Dict[int, List]]:
    """
    Split the graphs by connected components and discard single nodes.
    The cc_to_graph dict keeps track of the CCs to be able to recreate the original graphs.
    Relabel the nodes if there is more than 1 cc.

    Args:
        graphs: List of graphs to split by connected components

    Returns:
        List of graphs that contains only one connected components and that has more than one node
    """
    current_reduced_graph = 0
    cc_graphs = []
    cc_to_graph = defaultdict(list)  # {0: [0, 1, 2], 1: [3, 4], ...}

    for idx_graph, graph in enumerate(graphs):
        for cc in nx.connected_components(graph):
            H = nx.subgraph(graph, cc)

            # Check if the subgraph contains more than one node
            if len(H.nodes) <= 1:
                continue

            # Relabel the nodes so that they are in increasing order for all the newly created CCs
            idx_map = {old_idx: new_idx
                       for new_idx, old_idx in enumerate(sorted(H.nodes))}
            H = nx.relabel_nodes(H, mapping=idx_map)

            cc_graphs.append(H)
            cc_to_graph[idx_graph].append(current_reduced_graph)

            current_reduced_graph += 1

    return cc_graphs, cc_to_graph


def reassemble_cc_graphs(graphs: List[nx.Graph],
                         cc_to_graph: Dict[int, List]) -> List[nx.Graph]:
    """
    Reassemble the connected components into as single graph.

    Args:
        graphs:
        cc_to_graph:

    Returns:
        List of the graphs with the original connected components
    """
    new_graphs = []
    for indices_graph in cc_to_graph.values():
        new_graph = nx.Graph()

        for idx_graph in indices_graph:
            cc_graph = graphs[idx_graph]
            new_graph = nx.disjoint_union(new_graph, cc_graph)

        new_graphs.append(new_graph)

    return new_graphs


############################################
#             Lookup class                 #
############################################

class Lookup:
    """
    This class is used to keep track of the edges and the different components of the reduced graphs.

    Attributes:
        subgraphs_per_graph (List)
        edges_inter_clusters (List)
    """

    def __init__(self):
        self.subgraphs_per_graph = []
        self.edges_inter_clusters = []

    def append(self,
               idx_subgraphs: List[int],
               mat_n_edges: np.ndarray) -> None:
        self.subgraphs_per_graph.append(idx_subgraphs)
        self.edges_inter_clusters.append(mat_n_edges)

    def __iter__(self):
        """test"""
        for idx, (subgraph_indices, mat_edges_inter_clusters) in (
                enumerate(zip(self.subgraphs_per_graph,
                              self.edges_inter_clusters))):
            yield idx, subgraph_indices, mat_edges_inter_clusters

    def __len__(self):
        return len(self.subgraphs_per_graph)


############################################
#          graph visualization             #
############################################

def plot_graph_nx(graph: nx.Graph,
                  node_color: List = None,
                  figsize: Tuple[int, int] = (8, 8),
                  name: str = None,
                  ) -> None:
    """

    Args:
        graph:
        node_color:
        figsize:
        name:

    Returns:

    """
    dim_attr = len(set(node_color))

    f = plt.figure(figsize=figsize)
    nx.draw(graph,
            ax=f.add_subplot(111),
            pos=nx.kamada_kawai_layout(graph),
            node_size=50,
            node_color=node_color,
            font_size=15,
            with_labels=True,
            vmin=0,
            vmax=dim_attr + 1,
            cmap=plt.cm.get_cmap('Set1'))

    if name is not None:
        f.savefig(f'{name}')
    else:
        plt.show()


############################################
#              Verbose                     #
############################################

def set_global_verbose(verbose: bool = False) -> None:
    """
    Set the global verbose.
    Activate the logging module (use `logging.info('Hello world!')`)
    Activate the tqdm loading bar.

    Args:
        verbose: If `True` activate the global verbose

    Returns:

    """
    import logging
    from functools import partialmethod
    from tqdm import tqdm

    if verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=not verbose)
