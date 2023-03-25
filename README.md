mbtrack2
========

mbtrack2 is a coherent object-oriented framework written in python to work on collective effects in synchrotrons.

mbtrack2 is composed of different modules allowing to easily write scripts for single bunch or multi-bunch tracking using MPI parallelization in a transparent way. The base of the tracking model of mbtrack2 is inspired by mbtrack, a C multi-bunch tracking code initially developed at SOLEIL.

Installation
------------
Clone the mbtrack2 repo and enter the repo:
```
git clone https://gitlab.synchrotron-soleil.fr/PA/collective-effects/mbtrack2.git
cd mbtrack2
```

### Using conda

To create a new conda environment for mbtrack2 run:

```
conda env create -f mbtrack2.yml
conda activate mbtrack2
```

Or to update your current conda environment to be able to run mbtrack2:

```
conda env update --file mbtrack2.yml
```

To test your installation run:
```
from mbtrack2 import *
```

### Using pip

Run:

```
pip install -r requirements.txt
```
To test your installation run:
```
from mbtrack2 import *
```

Examples
--------
Jupyter notebooks demonstrating mbtrack2 features are available in the example folder and can be opened online using google colab:
+ mbtrack2 base features [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GamelinAl/mbtrack2_examples/blob/main/mbtrack2_demo.ipynb)
+ dealing with RF cavities and longitudinal beam dynamics [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GamelinAl/mbtrack2_examples/blob/main/mbtrack2_cavity_resonator.ipynb)

References
----------
A. Gamelin, W. Foosang, and R. Nagaoka, “mbtrack2, a Collective Effect Library in Python”, presented at the 12th Int. Particle Accelerator Conf. (IPAC'21), Campinas, Brazil, May 2021, paper MOPAB070.

Yamamoto, Naoto, Alexis Gamelin, and Ryutaro Nagaoka. "Investigation of Longitudinal Beam Dynamics With Harmonic Cavities by Using the Code Mbtrack." Proc. 10th International Partile Accelerator Conference (IPAC’19), Melbourne, Australia, 19-24 May 2019. 2019.
