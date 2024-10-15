#  An Explicit Framework for 3D Point Cloud Normalization

Justin Baker*, Shih-Hsin Wang*, Tommaso de Fernex, and Bao Wang

This repository contains the official implementation for  ["An Explicit Framework for 3D Point Cloud Normalization"](https://icml.cc/virtual/2024/poster/34002)
(ICML 2024).

*In Progress*: more flexible data handling and distributed normalization.



# Usage

`PyOrbit` is a library for normalizing 3D point clouds. It is designed to be used in conjunction with `NumPy` or `PyTorch` point cloud data.

Currently two types of normalization are supported: `PointCloud` and `CategoricalPointCloud`. The `Frame` and `CatFrame` classes can be used to normalized the point cloud by calling `.get_frame(point_cloud)` or `.get_frame(point_cloud, categorical_data)`.


Several useful examples can be found in the `examples` directory.


Training requires additional installation and can be performed by running the following command:

```bash
python3 ./training/ae_qm9.py
```

# Installation

## Datasets

**ModelNet40** can be downloaded by running

```bash
python3 ./datasets/modelnet40.py
```

and then processed by running the jupyter notebook `./datasets/modelnet40.ipynb`.

**QM9** will be downloaded automatically by the `torch_geometric` library.

## Requirements

Install PyOrbit as a library:

```bash
git clone https://github.com/Bayer-Group/alignment
cd alignment
pip3 install -e .

```

Training the autoencoder requires the additional library:

```bash
git clone https://github.com/Bayer-Group/giae
cd giae
pip3 install -e .
```


# Citation


If you find our work useful in your research, please consider citing:

```
@inproceedings{
baker2024explicit,
title={An Explicit Frame Construction for Normalizing 3{D} Point Clouds},
author={Baker, Justin and Wang, Shih-Hsin and De Fernex, Tommaso and Wang, Bao},
booktitle={Proceedings of the 41st International Conference on Machine Learning},
pages={2456--2473},
year={2024},
editor={Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
volume={235},
series={Proceedings of Machine Learning Research},
month={21--27 Jul},
publisher={PMLR},
pdf={https://raw.githubusercontent.com/mlresearch/v235/main/assets/baker24a/baker24a.pdf},
url={https://proceedings.mlr.press/v235/baker24a.html},
}

```

```
@inproceedings{
wang2024rethinking,
title={Rethinking the Benefits of Steerable Features in 3D Equivariant Graph Neural Networks},
author={Shih-Hsin Wang and Yung-Chang Hsu and Justin Baker and Andrea L. Bertozzi and Jack Xin and Bao Wang},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=mGHJAyR8w0}
}
```

# Acknowledgements


Our implementation is based on [NumPy](https://numpy.org/), [PyTorch](https://pytorch.org/), and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/).
