MutInf
------
Copyright 2010 Christopher McClendon and Gregory Friedland

MutInf is an analysis package written in Python, inline C, and R that analyzes data from Molecular Dynamics Simulations to identify statistically significant correlated motions and calculate residue-by-residue conformational entropies.

All code written by the above authors. I (Stefan Doerr) only updated the code to work with python 2.7 and packaged it for easy installation with pip.

For more information on the code, method and usage see:
http://www.jacobsonlab.org/mutinf_manual/


Installation
------------
Download and install miniconda from: https://docs.conda.io/en/latest/miniconda.html

```
git clone https://github.com/stefdoerr/mutinf
conda create -n mutinf python=2.7
cd mutinf
pip install .
```

Or directly install it from git

```
pip install git+https://github.com/stefdoerr/mutinf
```