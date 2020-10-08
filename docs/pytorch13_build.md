# Deprecated PyTorch 1.3 based build

These instructions are retained for the transition. Refer to top-level README for up to date instructions.

### PyTorch 1.3 - ATen pseudo-device type dispatch

The currently functional approach to PyTorch integration uses an ATen pseudo
device for program capture. It is activated by including the PyTorch cmake
path and settind `-DNPCOMP_ENABLE_TORCH_TYPE_DISPATCH=ON`. This approach has a
very fragile dependency on a specific PyTorch revisions in the ~1.3 era and
currently must be built via the docker image in `docker/pytorch-1.3`.

We are migrating to newer approaches that build with more recent PyTorch
versions, but these are not yet functional (see below).

Docker container setup:

```shell
# One of the maintainers does periodically push new images. To use one of these,
# skip the build step and use:
#   BUILD_IMAGE_TAG="stellaraccident/npcomp:build-pytorch-1.3"
# Since we are not planning to support this branch long term, this process is
# entirely ad-hoc at present and geared for project maintainers and build bots
# to be able to make progress.
# See https://hub.docker.com/repository/docker/stellaraccident/npcomp
BUILD_IMAGE_TAG="local/npcomp:build-pytorch-1.3"

# Build the docker image (rebuilds PyTorch, so takes quite some time).
docker build docker/pytorch-1.3 --tag $BUILD_IMAGE_TAG

# Docker workflow (or use your own preferences).
# Create a volume for npcomp build artifacts.
docker volume create npcomp-pytorch-1.3-build

# Run the container, mounting /npcomp to the source directory and the volume
# above to the /build directory. The source directory is mounted read-only to
# avoid the container putting root owned files there.
# Replace `$HOME/src/mlir-npcomp` with an appropriate path to where the project
# is checked out.
docker run \
  --mount type=bind,source=$HOME/src/mlir-npcomp,target=/npcomp,readonly \
  --mount source=npcomp-pytorch-1.3-build,target=/build \
  --rm -it $BUILD_IMAGE_TAG /bin/bash
```

```shell
# From within the docker image.
# Install MLIR and configure project.
cd /npcomp
BUILD_DIR=/build ./build_tools/install_mlir.sh
BUILD_DIR=/build ./build_tools/cmake_configure.sh \
  -DCMAKE_PREFIX_PATH=/opt/conda/lib/python3.6/site-packages/torch/share/cmake \
  -DNPCOMP_ENABLE_TORCH_TYPE_DISPATCH=ON

# Build.
cd /build
ninja
ninja check-npcomp
ninja check-frontends-pytorch
```
