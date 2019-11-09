Installation
============
.. note::
    :code:`pymer4` since version 0.6.0 is only compatible with Python 3. Versions 0.5.0 and lower will work with Python 2, but will not contain any new features.

Prerequisites
-------------
:code:`pymer4` requires a working R installation along with three R packages: :code:`lme4`, :code:`lmerTest`, and :code:`emmeans`. Follow either option below to make sure these are installed.

1. If you already have R/RStudio installed
+++++++++++++++++++++++++++++++++++++++++++++
Make sure you also have the 3 required R packages which can be installed from within R/RStudio using: 

    .. code-block:: R

        install.packages(c('lme4','lmerTest','emmeans'))

2. If you don't have R/RStudio installed
+++++++++++++++++++++++++++++++++++++++++++
It's highly recommended that you install and use the `Anaconda Python distribution <https://www.anaconda.com/distribution/>`_ to manage your Python packages/environment. Anaconda can also install and maintain R and R-packages for you. To install R and the the required packages through Anaconda:

 .. code-block:: bash

        conda install -c conda-forge r r-base r-lmertest r-emmeans rpy2

Package Installation
--------------------
After either option you can pip install :code:`pymer4`

    .. code-block:: bash

        pip install pymer4

You can also test the installation by running the following command in a terminal

    .. code-block:: bash

        python -c "from pymer4.test_install import test_install; test_install()"

Installation Issues
-------------------

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
