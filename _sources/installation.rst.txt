Installation
============
.. note::
    :code:`pymer4` since version 0.6.0 is only compatible with Python 3. Versions 0.5.0 and lower will work with Python 2, but will not contain any new features.

Using Anaconda (recommended)
----------------------------

For the latest stable release (recommended)
+++++++++++++++++++++++++++++++++++++++++++

:code:`pymer4` has some dependecies that can only be resolved using `conda-forge
<https://conda-forge.org/>`_ (e.g. recent versions of :code:`R, lme4, rpy2` etc). For
this reason I recommend preferring :code:`conda-forge` for all packages in the
environment you use for :code:`pymer4`. If you prefer to use the defaults Anaconda
channel you can switch the order of the channel priorities in any of the commands below,
i.e. :code:`-c conda-forge -c defaults` to :code:`-c defaults -c conda-forge`. 

Creating and installing into a new environment
##############################################

    .. code-block:: bash

        conda create --name pymer4 -c ejolly -c conda-forge -c defaults pymer4
        conda activate pymer4 

Installing into an existing environment
#######################################

    .. code-block:: bash

        conda install -c ejolly -c conda-forge -c defaults pymer4

.. note::
    Both commands above pull dependencies from conda-forge *first* rather than the
    default Anaconda channel. It's good practice to maintain this channel priority if
    you add additional packages to your environment. So be mindful of adding a :code:`-c
    conda-forge` flag if you install any additional packages into your environment. You
    can avoid this if you deliberately reversed the order of channels when installing
    :code:`pymer4` as noted above.
    
For the latest development release
++++++++++++++++++++++++++++++++++

Simply use either command above and substitute :code:`ejolly` for :code:`ejolly/label/pre-release`, i.e.

    .. code-block:: bash

        conda install -c ejolly/label/pre-release -c conda-forge -c defaults pymer4

Speed Ups on Intel CPUs
+++++++++++++++++++++++

If you are installing on an Intel CPU, you can additionally request the highly optimized
Intel Math Kernel Library (MKL) which uses optimized math libraries for Basic Linear Algebra Subprograms (BLAS) computations and can provide substantial speed ups for :code:`pymer4` as well as:code:`numpy`. 

    .. code-block:: bash

        conda install -c ejolly -c conda-forge -c defaults pymer4 "blas=*=mkl*"

This isn't recommended for other CPUs (e.g. AMD) as MKL will actually *slow down* computations. Instead you can request OpenBLAS, which is the default when installing :code:`pymer4` from conda-forge. If you want to install this explicitly the following command will work:

    .. code-block:: bash

        conda install -c ejolly -c conda-forge -c defaults pymer4 "blas=*=openblas*"


Using pip
---------

.. warning::
    It's strongly advised to use the conda install method above because of how notoriously finicky it can be to install :code:`rpy2` on various platforms. I recommend only following the directions below if you're comfortable with :code:`pip` and the command line or prefer not to use Anaconda.

Prerequisites
+++++++++++++
:code:`pymer4` requires a working R installation along with three R packages: :code:`lme4`, :code:`lmerTest`, and :code:`emmeans`. Follow either option below to make sure these are installed.

1. If you already have R/RStudio installed
##########################################
Make sure you also have the 3 required R packages which can be installed from within R/RStudio using: 

    .. code-block:: R

        install.packages(c('lme4','lmerTest','emmeans'))

2. If you don't have R/RStudio installed
########################################
The `Anaconda Python distribution <https://www.anaconda.com/distribution/>`_ can also install and maintain R and R-packages for you. To install R and the the required packages through Anaconda:

 .. code-block:: bash

        conda install -c conda-forge r r-base r-lmertest r-emmeans rpy2

For the latest stable release
+++++++++++++++++++++++++++++
After either option you can pip install :code:`pymer4`

    .. code-block:: bash

        pip install pymer4

For the latest development release
++++++++++++++++++++++++++++++++++
Install via github:

    .. code-block:: bash

        pip install git+https://github.com/ejolly/pymer4.git


Making sure the install worked
------------------------------
You can test the installation by running the following command in a terminal

    .. code-block:: bash

        python -c "from pymer4.test_install import test_install; test_install()"

Installation Issues
-------------------

If you have installed via :code:`pip` it's recommended you try the :code:`conda` method described above prior to raising an issue on github. Otherwise the following solutions may help. 

Missing R packages
++++++++++++++++++

If you follow step 2 in the prerequisites above (i.e. let Anaconda install R for you), some users have reported that the ``conda install`` command above sometimes doesn't install everything you need; for example the `matrix <https://cran.r-project.org/web/packages/Matrix/index.html>`_ package. You can fix this by either installing any missing packages from within R directly by first launching R at a terminal using ``R``, then adding the package with ``install.packages("Matrix")`` or by using Anaconda and prepending ``r-`` infront of the *lowercase* name of the package: ``conda install -c conda-forge r-matrix``. 


Compiler Issues on macOS
++++++++++++++++++++++++
Some of the more cryptic error messages you might encounter on macOS are due to compiler issues that give ``rpy2`` (a package dependency of ``pymer4``) some issues during install. Here's a fix that should work for that:

1. Install `homebrew <https://brew.sh/>`_ if you don't have it already by running the command at the link (it's a great pacakage manager for macOS). To check if you already have it, do ``which brew`` in your Terminal. If nothing pops up you don't have it.
2. Fix brew permissions: ``sudo chown -R $(whoami) $(brew --prefix)/*`` (this is **necessary** on macOS Sierra or higher (>= macOS 10.12))
3. Update homebrew ``brew update``
4. Install the xz uitility ``brew install xz``
5. At this point you can try to re-install ``pymer4`` and re-test the installation. If it still doesn't work follow the next few steps below
6. Install an updated compiler: ``brew install gcc``, or if you have homebrew already, ``brew upgrade gcc``
7. Enable the new compiler for use:

    .. code-block:: bash

        export CC="$(find `brew info gcc | grep usr | sed 's/(.*//' | awk '{printf $1"/bin"}'` -name 'x86*gcc-?')"
        export CFLAGS="-W"

8. If the above results in any error output (it should return nothing) you might need to manually find out where the new compiler is installed. To do so use ``brew info gcc`` and ``cd`` into the directory that begins with ``/usr`` in the output of that command. From there ``cd`` into ``bin`` and look for a file that begins with ``x86`` and ends with ``gcc-7``. It's possible that the directory ends with ``gcc-8`` or a higher number based on how recently you installed from homebrew. In that case, just use the latest version. Copy the *full path* to that file and run the following:

    .. code-block:: bash

        export CC= pathYouCopiedInQuotes
        export CFLAGS="-W"

9. Finally install ``rpy2`` using the new compiler you just installed: ``pip install rpy2`` if you have R/RStudio or ``conda install -c conda-forge rpy2`` if you don't.
10. Now you should be able to ``pip install pymer4`` :)

Kernel Crashes in Jupyter Notebooks/Lab
---------------------------------------
Sometimes using ``pymer4`` interactively can cause the Python kernel to crash. This is more likely to happen if you have multiple interactive sessions running simulatenously. One way around this is to put this at the top of your notebook/code:

    .. code-block:: python

        import os
        os.environ['KMP_DUPLICATE_LIB_OK']='True'

Or set the following environment variable prior to launching your interactive sessions:

    .. code-block:: bash

        export KMP_DUPLICATE_LIB_OK=TRUE
