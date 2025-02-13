from rpy2.situation import get_r_home
from rpy2.robjects.packages import importr

installed_packages = importr("utils").installed_packages

__all__ = ["Rhome", "check"]


def Rhome():
    return get_r_home()


def check():
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
