# qttools
Quantum Transport Algorithms Toolbox

## How to install
```
1. Create a conda environment
    $ conda env create -f environment.yml

2. Install mpi4py
    $ conda install -c conda-forge mpi4py mpich

3. Install CuPy
    $ conda install -c conda-forge cupy cuda-version=XX.X

4. Install Qttools
    $ cd path/to/qttools
    $ pip install --no-dependencies -e .
```

Notes:
- You might want to install a different mpi backend depending on the available 
libraries in your system.
- To find out which version of cuda is running on your machine use the command
`nvcc --version` or `nvidia-smi`.


