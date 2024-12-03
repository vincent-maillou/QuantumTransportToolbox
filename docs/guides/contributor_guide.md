## Setting up the development environment

First, clone the repository:

```bash
git clone git@github.com:vincent-maillou/qttools.git
cd qttools
```

After having obtained the repository, create a new conda environment
from the provided development environment `environment-dev.yml` file and
activate it:

```bash
conda env create -f environment-dev.yml
conda activate quatrex-dev
```

Unlike the regular environment (`environment.yml`), the development
environment already includes some development tools (`black`, `isort`,
`ruff`, `pytest`, ...) as well as `mpi4py` built against `mpich` from
the `conda-forge` channel. This is to ensure consistency with what
happens in the CI.

If you have a CUDA-compatible GPU and want to leverage it, you will need
to install `cupy` via the `conda-forge` channel or pip:

=== "conda"

    ```bash
    conda install -c conda-forge cupy cuda-version=XX.X
    ```

    !!! tip

        To determine which version of cuda is running on your machine use
        `nvcc --version` or `nvidia-smi`.


=== "pip"

    ```bash
    pip install cupy
    ```


Finally, install `qttools` in editable mode:

```bash
pip install --no-dependencies --editable .
```

## Running the Test Suite

Now you should be able to run the tests and the linters using the `just`
command runner:

```bash
just test
just format
just lint
```

To also build the documentation, you will need to install the
documentation dependencies:

```bash
pip install --editable '.[docs]'
```

## Building the Documentation

You should then be able to serve the documentation locally:

```bash
mkdocs serve
```
