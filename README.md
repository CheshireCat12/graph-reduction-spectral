# Graph reduction

### Hyperparameters to test

- Number of dimensions [2, 3, 4, 5, 8]
- Reduction levels [0.5, 0.25, 0.125]
- clustering algorithms ['kmeans', 'agglomerative']
- node merging method ['sum', 'hash']
- With hash merging we can try w/ and w/o using the labels during the hashing process
- (with hash merging it could be interesting to try hierarchical reduction. First reduce to 50%, then 25%, and so on)

### Dataset to test

Small molecules:
- Mutagenicity   4337    2    30.32    30.77 (standard)
- NCI1           4110    2    29.87    32.30 (standard)
- NCI109         4127    2    29.68    32.13 (standard)
- NCI-H23H      40353    2    46.67    48.69 (big number of graphs)

Bioinformatics:
- DD             1178    2    284.32   715.66 (standard)
- ENZYMES         600    6    32.63    62.14 (standard)
- PROTEINS       1113    2    39.06    72.82 (standard)

Computer vision:
- FIRSTMM_DB       41    11   1377.27  3074.10 (big graphs)
- MSRC_9          221    8    40.58    97.94 (big number of classes)
- MSRC_21         563    20   77.52    198.32 (big number of classes)

Social networks:
- COLLAB              5000	3	74.49   2457.78 (standard)
- github_stargazers   12725	2	113.79	234.64 (big number of graphs)
- REDDIT-BINARY       2000	2	429.63	497.75 (big graphs and big number of graphs)
- REDDIT-MULTI-5K     4999	5	508.52	594.87 (big graphs and big number of graphs)
- REDDIT-MULTI-12K    11929	11	391.41	456.89 (big number of graphs)

Synthetic:
- COLORS-3       10500	11	61.31	91.03 (big number of graphs)
- SYNTHETIC      300	2	100.00	196.00 (standard)
