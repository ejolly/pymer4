# Installation

!!! warning "**Do NOT install** using `pip`"
    Due to the cross-language nature of `pymer4`, installing via `pip` is **not supported**.  
    You **must** install `pymer4` using `conda` as outlined below

!!! Info "Windows Users"
    Unfortunately, Windows it **not officially support** as package installation can be unreliable. We recommend using the [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install) and setting up a conda install through there.

## Install using `conda` (recommended)

!!! note "Note"

    It's important to take note of `-c conda-forge` in the the `conda` commands below. This ensures that additional dependencies for `pymer4` are installed via the `conda-forge` channel which contains the relevant R packages, rather than Anaconda `defaults` which does not.

If you don't already have Anaconda/Miniconda setup, follow first follow the instructions [here](https://www.anaconda.com/docs/getting-started/miniconda/install)

If you have an existing `conda` environment you can install `pymer4` into it using:

```bash
conda install -c ejolly -c conda-forge pymer4
```

Otherwise you can create a new environment, which will also install a few other scientific Python libraries like `numpy` and `seaborn`; in the example below we name the new environment `pymer4`:

```bash
conda create --n pymer4 -c ejolly -c conda-forge pymer4
```

## Install development version

This is release is synchronized to the latest state of the `main` branch on Github. It may contain upcoming fixes, but undiscovered bugs as well.

```bash
conda install -c ejolly/label/pre-release -c conda-forge pymer4
```

## Making sure the install worked

You can test the installation by running the following command in a
terminal

``` bash
python -c "from pymer4.test_install import test_install; test_install()"
```

## Installation Issues

If you accidentally installed using `pip` you will have an unsupported version that is highly unlikely to work. Please install using one of the alternative methods above. Otherwise, the following solutions may help.

### Kernel Crashes in Jupyter Notebooks/Lab

Sometimes using `pymer4` interactively can cause the Python kernel to
crash. This is more likely to happen if you have multiple interactive
sessions running simulatenously. One way around this is to put this at
the top of your notebook/code:

``` python
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
```

Or set the following environment variable prior to launching your
interactive sessions:

``` bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

### Compiler Issues on intel macOS

Some of the more cryptic error messages you might encounter on macOS are
due to compiler issues that give `rpy2` (a package dependency of
`pymer4`) some issues during install. Here\'s a fix that should work for
that:

1.  Install [homebrew](https://brew.sh/) if you don\'t have it already
    by running the command at the link (it\'s a great pacakage manager
    for macOS). To check if you already have it, do `which brew` in your
    Terminal. If nothing pops up you don\'t have it.

2.  Fix brew permissions: `sudo chown -R $(whoami) $(brew --prefix)/*`
    (this is **necessary** on macOS Sierra or higher (\>= macOS 10.12))

3.  Update homebrew `brew update`

4.  Install the xz uitility `brew install xz`

5.  At this point you can try to re-install `pymer4` and re-test the
    installation. If it still doesn\'t work follow the next few steps
    below

6.  Install an updated compiler: `brew install gcc`, or if you have
    homebrew already, `brew upgrade gcc`

7.  Enable the new compiler for use:

    ``` bash
    export CC="$(find `brew info gcc | grep usr | sed 's/(.*//' | awk '{printf $1"/bin"}'` -name 'x86*gcc-?')"
    export CFLAGS="-W"
    ```

8.  If the above results in any error output (it should return nothing)
    you might need to manually find out where the new compiler is
    installed. To do so use `brew info gcc` and `cd` into the directory
    that begins with `/usr` in the output of that command. From there
    `cd` into `bin` and look for a file that begins with `x86` and ends
    with `gcc-7`. It\'s possible that the directory ends with `gcc-8` or
    a higher number based on how recently you installed from homebrew.
    In that case, just use the latest version. Copy the *full path* to
    that file and run the following:

    ``` bash
    export CC= pathYouCopiedInQuotes
    export CFLAGS="-W"
    ```

9.  Finally install `rpy2` using the new compiler you just installed:
    `pip install rpy2` if you have R/RStudio or
    `conda install -c conda-forge rpy2` if you don\'t.

10. Now you should be able to `pip install pymer4` :)
