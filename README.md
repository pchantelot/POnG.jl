# POnG
**P**rogramme pour les **On**des **G**uidées

[//]: # "[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://pchantelot.github.io/POnG.jl/stable/)"

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://pchantelot.github.io/POnG.jl/dev/)
[![Build Status](https://github.com/pchantelot/POnG.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/pchantelot/POnG.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19069061.svg)](https://doi.org/10.5281/zenodo.19069061)

Compute the dispersion relations of guided elastic waves in plates, tubes and strips for elastic and almost incompressible viscoelastic media subjected to an homogeneous finite deformation. 

This work extends the codes developed in:
- [GEW dispersion script](https://github.com/dakiefer/GEW_dispersion_script), doi: [10.5281/zenodo.7010603](https://doi.org/10.5281/zenodo.7010603) 
- [GEW soft strip](https://github.com/dakiefer/GEW_soft_strip), doi: [10.5281/zenodo.10372576](http://doi.org/10.5281/zenodo.10372576)

## Installation
```julia
using Pkg
Pkg.add(url="https://github.com/pchantelot/POnG.jl")
using POnG
```
Note that to run the included example scripts, the packages `LinearAlgebra` and `Kronecker` must be added. Plots are generated using `CairoMakie`.

## Dependencies
The functions `chebdif.jl` and `chebpoints.jl` are bundled from [DMSuite.jl](https://github.com/l90lpa/DMSuite.jl) with their license file.

The functions `lpoly.jl`, `lgrpointsleft.jl`, and `lgrdiffleft.jl` are adapted from the book:
> J. Shen, T. Tang and L. Wang, Spectral Methods: Algorithms, Analysis and Applications, Springer Science & Business Media (2011) [https://doi.org/10.1007/978-3-540-71041-7](https://doi.org/10.1007/978-3-540-71041-7)

The package [NonlinearEigenProblems.jl](https://github.com/nep-pack/NonlinearEigenproblems.jl) is developed by E. Jarlebring, M. Bennedich, G. Mele, E. Ringh and  P. Upadhyaya:
> E. Jarlebring, M. Bennedich, G. Mele, E. Ringh and  P. Upadhyaya, NEP-PACK: A Julia package for nonlinear eigenproblems, *[arXiv:1811.09592](https://arxiv.org/abs/1811.09592)* (2018)

## Citing POnG
If you use this code, please cite the zenodo repository:

- P. Chantelot, POnG.jl, [https://doi.org/10.5281/zenodo.19069061](https://doi.org/10.5281/zenodo.19069061) ([https://github.com/pchantelot/POnG.jl](https://github.com/pchantelot/POnG.jl)).

If relevant, please cite the following publications:

- [1] A. Delory, D. A. Kiefer, M. Lanoy, A. Eddi, C. Prada, and F. Lemoult, Viscoelastic dynamics of a soft strip subject to a large deformation, *Soft Matter*, (2024), doi: [10.1039/D3SM01485A](https://doi.org/10.1039/D3SM01485A).

- [2] P. Chantelot, A. Delory, C. Prada, and F. Lemoult, Wave propagation in a model artery, [10.48550/arXiv.2507.17698](https://doi.org/10.48550/arXiv.2507.17698) (2025).

- [3] P. Chantelot, S. Croquette and F. Lemoult, Guided elastic waves informed material modelling of soft incompressible media, [10.48550/arXiv.2603.18839](https://doi.org/10.48550/arXiv.2603.18839) (2026).


## Authors
Code created in 2024-2026 by

P. Chantelot, Institut Langevin, ESPCI Paris, Université PSL - [Google Scholar](https://scholar.google.fr/citations?user=BQWXUKYAAAAJ&hl=fr&oi=ao) 

D. A. Kiefer, Institut Langevin, ESPCI Paris, Université PSL - [Google Scholar](https://scholar.google.fr/citations?hl=fr&user=odSy3v4AAAAJ)