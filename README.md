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

## Pummeler package

This project started as a fork of the pummeler package, that can be found here: https://github.com/djsutherland/pummeler

## New files

The following files were added:
* mu_sinkhorn.py
* mu_sinkhorn_featurize.py
* sliced_wasserstein_featurize.py

The goal was to benchmark Sliced Wasserstein Kernel (Kolouri, 2016) and Mu-Sinkhorn (Bachoc, 2023) against the ones of the original's project (MMD kernel, Random Fourier Features).  
