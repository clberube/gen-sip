# gen-sip

This repository contains all the scripts necessary to reproduce the experimental results in the paper

- Bérubé, C.L., & Baron, F. (2022). Bayesian inference of petrophysical properties with generative spectral induced polarization models. https://doi.org/10.48550/arXiv.2208.00957

## Dependencies
- Python 3.8+
- Matplotlib
- NumPy
- Pandas
- PyTorch
- SALib
- Seaborn

## Usage
Simply run the scripts in order to generate the results.

> **Note**  
> Do not to skip steps because results from a previous scripts may be used by the following ones.

0. Creates the complex conductivity data by sampling the PPIP model parameter space with the LHS method.
1. Trains the CVAE on the complexe conductivity data.
2. Runs the dimension reduction experiment.
3. Runs the sensitivity analysis.
4. (a) Runs the unconstrained parameter estimation experiment. (b) Runs the conditional parameter estimation experiment.
5. (a) Samples and plots the unconstrained PPIP model parameter space. (b) Samples and plots the conditional parameter space.

> **Warning**  
> Each script exports results in the folders specified at the top of that script.
Please create empty folders with the specified names if any "No Such File or Directory" errors are raised.
