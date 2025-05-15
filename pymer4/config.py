from rpy2.situation import get_r_home
from rpy2.robjects.packages import importr
from .models import lm
from .io import load_dataset

installed_packages = importr("utils").installed_packages

__all__ = ["Rhome", "check_rlibs", "check_modelfit", "test_install"]


def Rhome():
    return get_r_home()


def check_rlibs():
    required = (
        "lmerTest",
        "emmeans",
        "tidyverse",
        "broom",
        "broom.mixed",
        "arrow",
    )
    libs = [lib for lib in installed_packages() if isinstance(lib, str)]
    missing_libs = []
    for required_lib in required:
        if required_lib not in libs:
            missing_libs.append(required_lib)
    if missing_libs:
        raise ModuleNotFoundError(
            f"The following R packages were not found among your installed packages and are required by pymer4.\nIf you installed pymer4 using pip because you already have R/RStudio installed, then you should manually install this package in R and try using pymer4 again.\nIf you installed pymer4 using conda/pixi then you can try installing any missing packages from conda-forge by prefixing their names with 'r-', e.g. conda install 'r-emmeans' -c conda-forge\nMissing: {missing_libs}"
        )
    else:
        print(f"All required R libraries found:\n{required}")


def check_modelfit():
    try:
        lm("Reaction ~ Days", data=load_dataset("sleep")).fit()
        print("Installation working successfully!")
    except Exception as e:
        print("Error! {}".format(e))


def test_install():
    Rhome()
    check_rlibs()
    check_modelfit()
