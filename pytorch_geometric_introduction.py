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

# Neural Networks are generally trained in a batch-wise fashion. PyTorch Geometric achieves
# parallelization over a mini-batch by creating sparse block diagonal adjacency matrices (defined
# by edge_index and edge_attr) and concatenating feature and target matrices in the node dimension.
# This composition allows differing number of nodes and edges over examples in one batch.

# PyTorch Geometric consists its own torch_geometric.data.DataLoader, which already takes care of
# this concatenation process. Let’s learn about it in an example:

from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    batch  # Batch(x=[1082, 21], edge_index=[2, 4066], y=[32], batch=[1082])
    batch.num_graphs  # 32

# torch_geometric.data.Batch inherits from torch_geometric.data.Data and contains an additional
# attribute: batch
# batch is a column vector of graph identifiers for all nodes of all graphs in the batch:
# batch = [0 ... 0 1 ... n-2 n-1 ... n-1].T

# You can use it to, e.g., average node features in the node dimension for each graph individually:
from torch_scatter import scatter_mean
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for data in loader:
    data  # Batch(x=[1082, 21], edge_index=[2, 4066], y=[32], batch=[1082])
    data.num_graphs  # 32
    x = scatter_mean(data.x, data.batch, dim=0)
    x.size()  # torch.Size([32, 21])


# ================================== DATA TRANSFORMS ===============================================

# ransforms are a common way in torchvision to transform images and perform augmentation. PyTorch
# Geometric comes with its own transforms, which expect a Data object as input and return a new
# transformed Data object. Transforms can be chained together using
# torch_geometric.transforms.Compose and are applied before saving a processed dataset
# (pre_transform) on disk or before accessing a graph in a dataset (transform).

# Let’s look at an example, where we apply transforms on the ShapeNet dataset (containing 17,000
# 3D shape point clouds and per point labels from 16 shape categories).

from torch_geometric.datasets import ShapeNet

dataset = ShapeNet(root='/tmp/ShapeNet', category='Airplane')
print(data[0])  # Data(pos=[2518, 3], y=[2518])

# We can convert the point cloud dataset into a graph dataset by generating nearest neighbor graphs
# from the point clouds via transforms:

import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet

dataset = ShapeNet(root='/tmp/ShapeNet', category='Airplane', pre_transform=T.NNGraph(k=6))
print(data[0])  # Data(edge_index=[2, 17768], pos=[2518, 3], y=[2518])

# In addition, we can use the transform argument to randomly augment a Data object,
# e.g. translating each node position by a small number:

import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet

dataset = ShapeNet(root='/tmp/ShapeNet', category='Airplane', pre_transform=T.NNGraph(k=6),
                   transform=T.RandomTranslate(0.01))
print(data[0])  # Data(edge_index=[2, 17768], pos=[2518, 3], y=[2518])


# =============================== LEARNING METHODS ON GRAPHS =======================================

# it’s time to implement our first graph neural network!

# We will use a simple GCN layer and replicate the experiments on the Cora citation dataset.
# For a high-level explanation on GCN, have a look at its blog post:
# http://tkipf.github.io/graph-convolutional-networks/

# First we need to load the Cora dataset:
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]
print(dataset)

# Implementation of the Two-layer GRAPH CONVOLUTION Network
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(data.num_features, 16)
        self.conv2 = GCNConv(16, data.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

# We use ReLU as our non-linearity acitivation function and output a softmax distribution over
# the number of classes. Let’s train this model on the train nodes for 200 epochs.


# perform calculations on GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define neural network
model = Net().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# perform training
model.train()

for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

# Finally we can evaluate our model on the test nodes:
model.eval()
_, pred = model(data).max(dim=1)
correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
acc = correct / data.test_mask.sum().item()
print('Accuracy: {:,.4f}'.format(acc))
