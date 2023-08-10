# BOhoneycomb
This repository contains the Python scripts for the Bayesian shape optimisation of hexagonal honeycomb metamaterial using finite element software Abaqus (Dessault Systemes).

The scripts were developed by Igor Kuszczak (github.com/IgorKuszczak/bayes-opt-for-abaqus) as part of the research work entitled "Bayesian shape optimisation of hexagonal honeycomb metamaterial" by I. Kuszczak, F. I Azam, M.A. Bessa, P.J. Tan, and F. Bosi. The work was inspired by the Data-driven Design Framework for Materials and Structures developed at Miguel A. Bessa's research group (github.com/bessagroup/f3dasm)

# Installation
To run this repository, first install the Ax platform:
```
conda install pytorch torchvision -c pytorch  # OSX only
pip3 install ax-platform
```
and then install the remaining dependencies by running:
```
pip install -r requirements.txt
```
A working abaqus installation is required for running the Abaqus examples used
in the paper. In the code, Abaqus is accessed via the Command Window, and thus
it is a good idea to check that the 'abaqus' command works in the cmd terminal.
Additionally, the examples require a working installation of CATIA V5 to open
and modify the geometries.
