# Running the statistical model in PyMC
We run the statistical model in PyMC, a python library for running Bayesian models. We chose to run the model in Python using NUTS and the JAX library that can run the simulation faster using GPU. We were inspired by https://www.pymc-labs.io/blog-posts/pymc-stan-benchmark/
## Setting the conda environment
We created a conda environment `pymc_env_jax`:
```
conda create -c conda-forge -n pymc_env_jax "pymc>=4" # see here: https://www.pymc.io/projects/docs/en/stable/installation.html
conda activate pymc_env_jax
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html # see here for more explanation: https://github.com/google/jax#pip-installation-gpu-cuda
pip install numpyro
# if a warning about ptxas having a wrong version is printed, need to install the right version for the conda environment - also see https://github.com/google/jax/discussions/10327
which ptxas # here you can see the current version used
conda install cuda -c nvidia # install nvidia  - this will install the right version of ptxas
which ptxas #check ptxas again - should point to the conda environment
```
## Running the model
To run the model, activate the conda environment and simply run the python code:
```
conda activate pymc_env_jax
python ar1_model.py
```
