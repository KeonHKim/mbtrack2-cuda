MBTRACK2-CUDA
========

MBTRACK2-CUDA is a CUDA-based GPU implementation of MBTRACK2, a Python tracking code for evaluating collective effects in circular accelerators.

The main purpose of MBTRACK2-CUDA is to enable single- and multi-bunch tracking simulations on a single PC equipped with a single GPU. The entire tracking procedure is executed on the GPU, allowing all bunches and macro-particles to be treated simultaneously without relying on multi-CPU clusters.

The code is written in Python and uses Numba-CUDA for GPU execution. MBTRACK2-CUDA is developed as an independent CUDA port of MBTRACK2, focusing on GPU-resident computations. While it does not yet include all features of the original MBTRACK2, it implements the core tracking models required for studies of collective effects.

The original MBTRACK2 code can be found at the following link:
```
https://gitlab.synchrotron-soleil.fr/PA/collective-effects/mbtrack2
```

References
----------
K. H. Kim, and E.-S. Kim. "CUDA-based GPU-only computation for efficient tracking simulation of single and multi-bunch collective effects. Nucl. Sci. Tech. 37, 6 (2026).

A. Gamelin, W. Foosang, and R. Nagaoka. “mbtrack2, a collective effect library in Python.” IPAC'21, (2021).

X. Gu and A. Gamelin. MBTRACK2: A Longitudinal RF Beam Loading Tracking Code. Technical Report BNL-224788-2023-TECH. Brookhaven National Laboratory, (2023).

H. S. Xu, U. Locans, A. Adelmann, and L. Stingelin. "Calculation of longitudinal collective instabilities with mbtrack-cuda." Nucl. Instrum. Methods Phys. Res. A, 922:345-351, (2019).
