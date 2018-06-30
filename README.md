# Pytorch Geometric
Geometric Deep Learning Extension Library for PyTorch

## General
PyTorch Geometric is a geometric deep learning extension library for PyTorch.

It consists of various methods for deep learning on graphs and other irregular structures, also known as geometric deep learning, from a variety of published papers. In addition, it consists of an easy-to-use mini-batch loader, a large number of common benchmark datasets (based on simple interfaces to create your own), and helpful transforms, both for learning on arbitrary graphs as well as on 3D meshes or point clouds.

To install Pytorch Geometric go to [Instalation Guide](https://rusty1s.github.io/pytorch_geometric/build/html/notes/installation.html)


## Fundamentals

PyTorch Geometric through self-contained examples. At its core, PyTorch Geometric provides the following main features:

- [Data handling of graphs](https://rusty1s.github.io/pytorch_geometric/build/html/notes/introduction.html#data-handling-of-graphs)
- [Common benchmark datasets](https://rusty1s.github.io/pytorch_geometric/build/html/notes/introduction.html#common-benchmark-datasets)
- [Mini-Batches](https://rusty1s.github.io/pytorch_geometric/build/html/notes/introduction.html#mini-batches)
- [Data transforms](https://rusty1s.github.io/pytorch_geometric/build/html/notes/introduction.html#data-transforms)
- [Learning methods on graphs](https://rusty1s.github.io/pytorch_geometric/build/html/notes/introduction.html#learning-methods-on-graphs)


In detail, the following methods are currently implemented:
* **[SplineConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.SplineConv)** from Fey *et al.*: [SplineCNN: Fast Geometric Deep Learning with Continuous B-Spline Kernels](https://arxiv.org/abs/1711.08920) (CVPR 2018)
* **[GCNConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.GCNConv)** from Kipf and Welling: [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) (ICLR 2017)
* **[ChebConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.ChebConv)** from Defferrard *et al.*: [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375) (NIPS 2017)
* **[NNConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.NNConv)** adapted from Gilmer *et al.*: [Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212) (ICML 2017)
* **[GATConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.GATConv)** from Veličković *et al.*: [Graph Attention Networks](https://arxiv.org/abs/1710.10903) (ICLR 2018)
* **[AGNNProp](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.prop.AGNNProp)** from Kiran *et al.*: [Attention-based Graph Neural Network for Semi-Supervised Learning](https://arxiv.org/abs/1803.03735)
* **[SAGEConv](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.conv.SAGEConv)** from Hamilton *et al.*: [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) (NIPS 2017)
* **[Graclus Pooling](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.pool.graclus)** from Dhillon *et al.*: [Weighted Graph Cuts without Eigenvectors: A Multilevel Approach](http://www.cs.utexas.edu/users/inderjit/public_papers/multilevel_pami.pdf) (PAMI 2007)
* **[Voxel Grid Pooling](https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#torch_geometric.nn.pool.voxel_grid)**
