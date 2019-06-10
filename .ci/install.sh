# Install miniconda or load cached version

# Path to the conda distribution
export PATH="$HOME/miniconda/bin:$PATH"

# Check if the conda command exists, and if not,
# download and install miniconda
if ! command -v conda > /dev/null; then
      wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
      bash miniconda.sh -b -p $HOME/miniconda -u;
      conda config --add channels conda-forge;
      conda config --set always_yes yes;
      conda update --all;
      conda create --yes -n test python=$PYTHON_VERSION;
      conda activate test;
      conda install tectonic;
      conda install -c conda-forge numpy=$NUMPY_VERSION scipy matplotlib setuptools pytest pytest-cov pip sympy;
fi

# Display some debugging info
conda info -a
