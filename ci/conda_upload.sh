# Anaconda Cloud package uploader.
# 
#  * This assumes TravisCI ran conda build and the linux-64 package tar.bz
#    is in the default location /home/travis/miniconda3/conda-bld/linux-64
# 
#  * The upload only fires if the package version from meta.yaml is
#    (exactly) Major.Minor.Patch and $TRAVIS_BRANCH is master or
#    vMajor.Minor.Patch.
#
#  * Routine commits to master are uploaded to the pre-release label.
#
#  * Tagged releases vMajor.Minor.Patch are uploaded to the main label.
#
#  * Version strings other than Major.Minor.Patch disable uploading
#   entirely.

# some guarding ...
if [[ -z ${CONDA_DEFAULT_ENV} ]]; then
    echo "activate a conda env before running conda_upload.sh"
    exit -1
fi

# intended for TravisCI deploy but can be tricked into running locally
if [[ "$TRAVIS" != "true" || -z "$TRAVIS_BRANCH" || -z "${PACKAGE_NAME}" ]]; then
    echo "conda_upload.sh is meant to run on TravisCI"
    exit -2
fi

# set the parent of conda-bld or use $CONDA_PREFIX for local testing
# bld_prefix=${HOME}/miniconda3
bld_prefix="/home/travis/miniconda"  # from the .travis.yml

# on travis there should be a single linux-64 package tarball. insist
tarball=`/bin/ls -1 ${bld_prefix}/conda-bld/linux-64/${PACKAGE_NAME}-*-*.tar.bz2`
n_tarballs=`echo "${tarball}" | wc -w`
if (( $n_tarballs != 1 )); then
    echo "found $n_tarballs package tarballs there must be exactly 1"
    echo "$tarball"
    exit -3
fi

# the tarball name is PATH/PKG-VERSION-ABBREV_COMMIT.tar.bz2 
# pull out the version and git short commit hash
splitr=".*${PACKAGE_NAME}-\(.\+\)-\(.\+\)\.tar\.bz2"

# entire version string which may or may not be Major.Minor.Patch
version=$(echo $tarball | sed -n "s/${splitr}/\1/p")

# git short hash with g or r=release prefix
abbrev_commit=$(echo $tarball | sed -n "s/${splitr}/\2/p")

# just the numeric Major.Minor.Patch portion of version, possibly empty
mmp=$(echo $version | sed -n "s/\(\([0-9]\+\.\)\{1,2\}[0-9]\+\).*/\1/p")

# assume this is a dry run, unless version is M.N.P exactly and the
# TRAVIS_BRANCH is master or a tagged vM.N.P release, in which case,
# set the destination label appropriately.
label="dry-run"  
if [[ "${version}" = "$mmp" ]]; then

    # commit to master uploads to pre-release
    if [[ $TRAVIS_BRANCH = "master" ]]; then
	label="pre-release"
    fi

    # github release tagged vM.N.P uploads to main w/ r prefix to the commit short hash
    if [[ $TRAVIS_BRANCH = v"$mmp" ]]; then
	label="main"
    fi
fi


# build package binaries for selected platforms
mkdir -p ${bld_prefix}/conda-convert/linux-64
cp ${tarball} ${bld_prefix}/conda-convert/linux-64
cd ${bld_prefix}/conda-convert
conda convert -p linux-64 -p osx-64 -p win-64  linux-64/${PACKAGE_NAME}*tar.bz2

# POSIX trick sets $ANACONDA_TOKEN if unset or empty string 
ANACONDA_TOKEN=${ANACONDA_TOKEN:-[not_set]}
conda_cmd="anaconda --token $ANACONDA_TOKEN upload ./**/${PACKAGE_NAME}*.tar.bz2 --label ${label} --skip-existing"

# echo values to the TravisCI log for general info/debugging
echo "package name: $PACKAGE_NAME"
echo "conda meta.yaml version: $version"
echo "github short hash: $abbrev_commit"
echo "travis branch: $TRAVIS_BRANCH"
echo "travis tag: $TRAVIS_TAG"
echo "conda-bld: ${bld_prefix}/conda-bld/linux-64"
echo "tarball: $tarball"
echo "conda label: ${label}"
echo "conda upload command: ${conda_cmd}"
echo "platforms:"
echo "$(ls ./**/${PACKAGE_NAME}*.tar.bz2)"

# trigger the upload and destination or 
if [[ $ANACONDA_TOKEN != "[not_set]" && ( $label = "main" || $label = "pre-release" ) ]]; then

    conda install anaconda-client

    echo "uploading to Anconda Cloud: $PACKAGE_NAME$ $version ..."
    if ${conda_cmd}; then
    	echo "OK"
    else
    	echo "Failed"
    	exit -5
    fi
else
    echo "$PACKAGE_NAME $TRAVIS_BRANCH $version conda_upload.sh dry run ... OK"
fi
exit 0
