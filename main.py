import argparse

from src.reduction import reduce_graphs


def main(args):
    reduce_graphs(dataset=args.dataset,
                  root_dataset=args.root_dataset,
                  split_by_cc=args.split_by_cc,
                  embedding_algorithm=args.embedding_algorithm,
                  dim_embedding=args.dim_embedding,
                  clustering_algorithm=args.clustering_algorithm,
                  rho=args.rho,
                  node_merging_method=args.node_merging_method,
                  folder_results=args.folder_results,
                  verbose=args.verbose)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(description='Graph reduction by coarsening')
    subparser = args_parser.add_subparsers()

    args_parser.add_argument('--dataset',
                             type=str,
                             required=True,
                             help='Graph dataset to reduce (the dataset has to be in TUDataset)')
    args_parser.add_argument('--root_dataset',
                             type=str,
                             default='/tmp/data',
                             help='Root of the dataset')

    args_parser.add_argument('--split_by_cc',
                             action='store_true',
                             help='If set to True the algorithm will split the graph with connected components')

    args_parser.add_argument('--embedding_algorithm',
                             choices=['spectral'],
                             required=True,
                             help='Select the embedding algorithm to embed the graphs')
    args_parser.add_argument('--dim_embedding',
                             type=int,
                             default=3,
                             help='Set the number of initial partitions')

    args_parser.add_argument('--clustering_algorithm',
                             choices=['kmeans', 'agglomerative'],
                             required=True,
                             help='Select the clustering algorithm to use with the embedded graphs')
    args_parser.add_argument('--rho',
                             type=float,
                             default=0.5,
                             help='Choose the reduction factor (i.e. the percentage of node remaining in the reduced '
                                  'graph)')

    args_parser.add_argument('--node_merging_method',
                             choices=['sum', 'hash'],
                             required=True,
                             help='Select the node merging method to use to concatenate the nodes together')

    args_parser.add_argument('--folder_results',
                             type=str,
                             required=True,
                             help='Folder where to save the reduced graphs')

    args_parser.add_argument('-v',
                             '--verbose',
                             action='store_true',
                             help='Activate verbose print')

    parse_args = args_parser.parse_args()

    main(parse_args)
