"""
================================== PYTORCH GEOMETRIC - INTRODUCTION ================================

Pytorch Geometric provides following main features:
1. Data handling of graphs
2. Common benchmark datasets
3. Mini-Batches
4. Data transforms
5. Learning methods on graphs
"""

# ================================= DATA HANDLING OF GRAPHS ========================================

# A single graph in PyTorch Geometric is decribed by torch_geometric.data.Data.
# This Data object holds all information needed to define an arbitrary graph.
# There are already some predefined attributes:
#   data.x - Node feature matrix with shape [num_nodes, num_node_features]
#   data.edge_index - graph connectivity in COO format with shape [2, num_edges] and type torch.long
#   data.edge_attr - Edge feature matrix with shape [num_edges, num_edge_features]
#   data.y - target to train against (may have arbitrary shape)
#   data.pos - Node position matrix with shape [num_nodes, num_dimensions]

# None of these attributes is required. In fact, the Data object is not even restricted to these
# attributes. We can, e.g., extend it by data.face to save the connectivity of triangles from
# a 3D mesh in a tensor with shape [3, num_faces] and type torch.long.


# We show a simple example of an unweighted and undirected graph with three nodes and four edges.
# Each node is assigned exactly one feature:


import torch
from torch_geometric.data import Data

# tensor which describe connection between nodes: 0, 1, 2
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

# create tensor with features (values of each node.)
# value for node 0 = -1, value for node 1 = 0, value for node 3 = 1
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)
print(data)


# Besides of being a plain old python object, torch_geometric.data.Data provides a number of
# utility functions, e.g.:

# shows name of the tensor with nodes and name of tensor with features
print(data.keys)

# present values of the tensor with nodes: 0, 1, 2
print(data['x'])

# shows that x and edge_index are in the data.
for key, item in data:
    print('{} found in data')

# boolen output: False (because we have only two attributes now in the model: data, which are
# nodes and edge_indexes.)
print('edge_attr' in data)

# print how many nodes, edges and features we have in the model
print(data.num_nodes)
print(data.num_edges)
print(data.num_features)

# print boolean output weather there are:
print(data.contains_isolated_nodes())
print(data.contains_self_loops())
print(data.is_directed())


# ================================= COMMON BENCHMARK DATASETS ======================================

# PyTorch Geometric contains a large number of common benchmark datasets, e.g. all Planetoid
# datasets (Cora, Citeseer, Pubmed), all graph classification datasets from
# http://graphkernels.cs.tu-dortmund.de/, the QM9 dataset, and a handful of 3D mesh/point cloud
# datasets (FAUST, ModelNet10/40, ShapeNet).

# Initializing a dataset is straightforward. The dataset will be automatically downloaded and
# process the graphs to the previously decribed Data format. E.g., to load the ENZYMES dataset
# (consisting of 600 graphs within 6 classes), type:


from torch_geometric.datasets import TUDataset

# Donwload all 600 graphs
dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')

print(dataset)  # ENZYMES(600)
print(len(dataset))  # 600
print(dataset.num_classes)  # 6
print(dataset.num_features)  # 21

# Now have access to all 600 graphs in the dataset:
data = dataset[0]
print(data)  # Data(edge_index=[2, 168], x=[37, 21], y=[1])
print(data.is_undirected())  # True
# first graph in the dataset contains 37 nodes, each one having 21 features. There are 168/2 = 84
# undirected edges and the graph is assigned to exactly one class.

# We can even use slice, long or byte tensors to split the dataset. E.g. to create a 90/10
# train/test split, type:
train_dataset = dataset[:540]
print(train_dataset)  # ENZYMES(540)
test_dataset = dataset[540:]
print(test_dataset)  # ENZYMES(60)

# to be sure that data was shuffeled before the split, use:
dataset = dataset.shuffle()

# this is equivalent of doing:
perm = torch.randperm(len(dataset))
dataset = dataset[perm]


# download CORA - the standard benchmark dataset for semi-supervised graph node classificiation:
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/tmp/Cora', name='Cora')
print(len(dataset))  # 1
print(dataset.num_classes)  # 7
print(dataset.num_features)  # 1433

# Here, the dataset contains only a single, undirected citation graph:
data = dataset[0]  # Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708],
#                           val_mask=[2708], test_mask=[2708])
print(data.is_undirected())  # True
print(data.train_mask.sum())  # 140
print(data.val_mask.sum())  # 500
print(data.test_mask.sum())  # 1000

# This time, the Data objects holds additional attributes: train_mask, val_mask and test_mask:
#   train_mask - denotes against which nodes to train (140 nodes)
#   val_mask - denotes which nodes to use for validation, e.g. to perform early stopping (500 nodes)
#   test_mask - denotes against which nodes to test (1000 nodes)


# ====================================== MINI-BATCHES ==============================================
