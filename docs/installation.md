# Installation

```{note}
`pymer4 >= 0.9.0` is only available for installation using `conda`  Installation using `pip` is not possible due to the cross-language design (R <-> Python)
```

## Latest stable release

This is the recommended installation option.

If you have an existing `conda` environment you can install `pymer4` into it using:

```bash
conda install -c ejolly -c conda-forge pymer4
```

Otherwise you can make a brand new environment called `pymer4` (which will also install a few other scientific Python libraries like `numpy` and `seaborn`) using:

```bash
conda create --n pymer4 -c ejolly -c conda-forge pymer4
```

## Latest development release

This is release is synchronized to the latest state of the `main` branch on Github. It may contain upcoming fixes, but undiscovered bugs as well.

```bash
conda install -c ejolly/label/pre-release -c conda-forge pymer4
```

```{info}
Both commands above pull dependencies from conda-forge *first* rather
than the default Anaconda channel. It's good practice to maintain this
channel priority if you add additional packages to your environment. So
be mindful of adding a `-c conda-forge` flag if you install any additional packages into your environment.
```

## Making sure the install worked

You can test the installation by running the following command in a
terminal

> ``` bash
> python -c "from pymer4.test_install import test_install; test_install()"
> ```

## Speed Ups on Intel CPUs

If you are installing on an Intel CPU, you can additionally request the
highly optimized Intel Math Kernel Library (MKL) which uses optimized
math libraries for Basic Linear Algebra Subprograms (BLAS) computations
and can provide substantial speed ups for `pymer4` as well as`numpy`.

> ``` bash
> conda install -c ejolly -c conda-forge -c defaults pymer4 "blas=*=mkl*"
> ```

This isn\'t recommended for other CPUs (e.g. AMD) as MKL will actually
*slow down* computations. Instead you can request OpenBLAS, which is the
default when installing `pymer4` from conda-forge. If you want to
install this explicitly the following command will work:

> ``` bash
> conda install -c ejolly -c conda-forge -c defaults pymer4 "blas=*=openblas*"
> ```

## Installation Issues

If you have installed via `pip` it\'s recommended you try the `conda`
method described above prior to raising an issue on github. Otherwise
the following solutions may help.

### Kernel Crashes in Jupyter Notebooks/Lab

Sometimes using `pymer4` interactively can cause the Python kernel to
crash. This is more likely to happen if you have multiple interactive
sessions running simulatenously. One way around this is to put this at
the top of your notebook/code:

> ``` python
> import os
> os.environ['KMP_DUPLICATE_LIB_OK']='True'
> ```

Or set the following environment variable prior to launching your
interactive sessions:

> ``` bash
> export KMP_DUPLICATE_LIB_OK=TRUE
> ```

### Compiler Issues on macOS

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

    > ``` bash
    > export CC="$(find `brew info gcc | grep usr | sed 's/(.*//' | awk '{printf $1"/bin"}'` -name 'x86*gcc-?')"
    > export CFLAGS="-W"
    > ```

8.  If the above results in any error output (it should return nothing)
    you might need to manually find out where the new compiler is
    installed. To do so use `brew info gcc` and `cd` into the directory
    that begins with `/usr` in the output of that command. From there
    `cd` into `bin` and look for a file that begins with `x86` and ends
    with `gcc-7`. It\'s possible that the directory ends with `gcc-8` or
    a higher number based on how recently you installed from homebrew.
    In that case, just use the latest version. Copy the *full path* to
    that file and run the following:

    > ``` bash
    > export CC= pathYouCopiedInQuotes
    > export CFLAGS="-W"
    > ```

9.  Finally install `rpy2` using the new compiler you just installed:
    `pip install rpy2` if you have R/RStudio or
    `conda install -c conda-forge rpy2` if you don\'t.

10. Now you should be able to `pip install pymer4` :)
