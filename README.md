This is the code of the preprint "*Improved learning theory for kernel distribution regression with two-stage sampling*"
of François Bachoc, Louis Béthune, Alberto González-Sanz, Jean-Michel Loubes.  

You may cite it as:
```
@article{bachoc2023improved,
  title={Improved learning theory for kernel distribution regression with two-stage sampling},
  author={Bachoc, Fran{\c{c}}ois and B{\'e}thune, Louis and Gonz{\'a}lez-Sanz, Alberto and Loubes, Jean-Michel},
  journal={arXiv preprint arXiv:2308.14335},
  year={2023}
}
```

## Pummeler Software and Contributions

The **Pummeler** software, used for experiments, was written by [Flaxman et al. (2015)](https://github.com/djsutherland/pummeler) and distributed under the **MIT License**. The initial files remain unchanged, except for `pummeler/cli.py`, which now includes relevant help messages for new arguments. You can view these messages by invoking:

```
pummeler featurize --help
```

#### Embedding Options:
```
  --skip-linear         Skip linear embedding (original baseline).
  --sinkhorn-mu         Use Sinkhorn Mu kernel features.
  --pad-clouds          Pad clouds to the same size for Sinkhorn Mu kernel features.
  --mu-size MU_SIZE     Size of the support of reference measure (default: 32).
  --sliced-wasserstein  Use sliced-Wasserstein features.
  --n_projections N_PROJECTIONS
                        Number of random projection directions.
  --n_discrete N_DISCRETE
                        Number of ticks along the integration direction.
```

### Code Contributions

The goal was to benchmark Sliced Wasserstein Kernel (Kolouri, 2016) and Mu-Sinkhorn (Bachoc, 2023) against the ones of the original's project (MMD kernel, Random Fourier Features).  

Our contributions include the following new files:

- `pummeler/mu_sinkhorn.py`: General implementation of the Sinkhorn kernel.
- `pummeler/mu_sinkhorn_featurize.py`: Embedding associated with the Sinkhorn kernel, compliant with the **Pummeler** API.
- `pummeler/sliced_wasserstein_featurize.py`: Embedding associated with the Sliced Wasserstein kernel, compliant with the **Pummeler** API.

We have also improved the documentation by adding detailed docstrings, including descriptions of arguments, return values, class attributes, and typing, for all classes and functions.  

### Additional Notebooks

The following notebooks are also documented:

- `notebooks/toy_experiments.ipynb`: Contains Figures 3 and 4.
- `notebooks/radon_transform_plot.ipynb`: Contains Figure 2.
- `notebooks/us_data_processing.ipynb`: Contains results of Table 1.
- `notebooks/geomap_plot.ipynb`: Contains results of Figure 5.
