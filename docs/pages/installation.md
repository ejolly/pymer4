# Installation

:::{admonition} Windows Users
:class: info, dropdown
*Unfortunately, Windows it not officially supported as package installation can be unreliable. We recommend using the [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install) and setting up a conda install through there. Alternatively you can try to follow [this guide](https://joonro.github.io/blog/posts/install-rpy2-windows-10/) to setup an R installation with the `rpy2` Python library.*
:::

:::{admonition} **Do NOT use `pip`**
:class: danger
Due to the cross-language nature of `pymer4` installing via `pip` is **not officially supported**.  
Please use one of the two options below.
:::

## 1. Using Anaconda

*If you don't already have Anaconda/Miniconda setup, follow first follow the instructions [here](https://www.anaconda.com/docs/getting-started/miniconda/install)*


To install into existing environment use

```bash
conda install -c ejolly -c conda-forge pymer4
```

To create a new environment with additional scientific Python libraries

```bash
conda create --n pymer4 -c ejolly -c conda-forge pymer4
```

You can test the installation by running the following command in a terminal within the environment you installed `pymer4`

```bash
python -c "from pymer4 import test_install; test_install()"
```

:::{note}
The `-c conda-forge` in the the `conda` commands above ensures that additional dependencies for `pymer4` are installed via the `conda-forge` channel which contains the relevant R packages, rather than Anaconda `defaults` which does not.
:::

## 2. Using Google Collab

If you are having trouble or don't want to install `pymer4` locally, you can use it in a Google Colab notebook by following the directions below. Or you can just copy the [Example Notebook](https://colab.research.google.com/drive/19D15LAid9GgqSm9kU_TXy9ERUM7mBvnN?usp=sharing) that we've setup. 


### Setting up `conda` on Collab 

In the first cell of your note book add the following code and run it. This will cause your notebook kernel to appear to "crash" and restart - **this is expected**

```bash
!pip install -q condacolab
import condacolab
condacolab.install()
```

### Installing `pymer4`

Once the kernel restarts, you'll have the `!mamba` command available that you can use to install `pymer4`. You don't need to rerun the previous cell. Just create and run a new cell below it with the following code:

```bash
!mamba install -q pymer4 -c ejolly -c conda-forge
```

Then you should be all set!

## Using the development version of `pymer4`

To support easy installation of the latest "bleeding-edge" version of `pymer4` on the `main` branch of Github you can use the command below to install from the `pre-release` channel on anaconda.org. The `main` branch may often contain upcoming fixes and features slated for a new release.

```bash
conda install -c ejolly/label/pre-release -c conda-forge pymer4
```

## Installation issues on intel macOS

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

4.  Install the xz utility `brew install xz`

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
