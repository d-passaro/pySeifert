# pySeifert
A [Sage](https://www.sagemath.org/) module for the computation of $\hat{Z}$ invariants of Seifert manifolds with three and four singular fibers. PySeifert was developed as a companion to "3 Manifolds and VOA characters" [ArXiv:2201.04640](https://www.arxiv.org/abs/2201.04640), to aide in the computation of topological invariants for Seifert manifolds and characters of certain vertex operator algebras.

## Autors:
    Miranda C. N. Cheng,
    Sungbong Chun,
    Boris Feigin,
    Francesca Ferrari,
    Sergei Gukov,
    Sarah M. Harrison,
    Davide Passaro (2022-05-06): initial version


## Key Features

PySeifert implements a Seifert class to represent Seifert manifolds with three or four sigular fibers with functions to compute:
  - Plumbing matrix
  - Lattice dilation factor
  - $m$, $\vec s$, $\vec k$ and Spin$^c$ structures
  - $\hat{Z}$ invariant integrand contributions, including, in the three singular fiber case when Wilson operators are attached to central, mid or end nodes.
  - $\hat{Z}$ invariants
  - Other quantities needed for the computation of the above, described in ArXiv:2201.04640.

PySeifert also includes functions for the computation of q-series associated to characters of Log-V(p) and Log-V(p,p') VOAs which were used to compare to the $\hat{Z}$ invariants and provide examples for the ArXiv:2201.04640.


## How To Use

Clone the repository and load the PySeifert module in Sage using the [load](https://doc.sagemath.org/html/en/reference/repl/sage/repl/load.html#sage.repl.load.load) function from Sage:
```
load("path/to/pySeifert.sage")   
```
Functions are documented in code, and running:
```
   help(function)
```
will print some basic information and usage examples.


## Requirements

This module has been tested using Sage version 9.6 interfaced through Python version 3.10.4. PySeifert requires numpy and itertools.
