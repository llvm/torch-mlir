# Build the docker images for npcomp:
#   npcomp:build-pytorch-1.6
#   me/npcomp:build-pytorch-1.6  (additional dev packages and current user)
function npcomp_docker_build() {
  if ! [ -f "docker/pytorch-1.6/Dockerfile" ]; then
    echo "Please run out of mlir-npcomp/ source directory..."
    return 1
  fi
  echo "Building out of $(pwd)..."
  docker build docker/pytorch-1.6 --tag npcomp:build-pytorch-1.6
  npcomp_docker_build_for_me npcomp:build-pytorch-1.6
}

# Start a container named "npcomp" in the background with the current-user
# dev image built above.
function npcomp_docker_start() {
  local host_src_dir="${1-$HOME/src/mlir-npcomp}"
  if ! [ -d "$host_src_dir" ]; then
    echo "mlir-npcomp source directory not found:"
    echo "Pass path to host source directory as argument (default=$host_src_dir)."
    return 1
  fi
  docker volume create npcomp-build
  docker run -d --rm --name "$container" \
    --mount source=npcomp-build,target=/build \
    --mount type=bind,source=$host_src_dir,target=/src/mlir-npcomp \
    me/npcomp:build-pytorch-1.6 tail -f /dev/null
}

# Stop the container named "npcomp".
function npcomp_docker_stop() {
  docker stop npcomp
}

# Get an interactive bash shell to the "npcomp" container.
function npcomp_docker_login() {
  docker_execme -it npcomp /bin/bash
}

### Implementation helpers below.
# From a root image, build an image just for me, hard-coded with a user
# matching the host user and a home directory that mirrors that on the host.
function npcomp_docker_build_for_me() {
  local root_image="$1"
  echo "
    FROM $root_image

    USER root
    RUN apt install -y sudo byobu git procps lsb-release
    RUN addgroup --gid $(id -g $USER) $USER
    RUN mkdir -p $(dirname $HOME) && useradd -m -d $HOME --gid $(id -g $USER) --uid $(id -u $USER) $USER
    RUN echo '$USER ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers
    USER $USER
  " | docker build --tag me/${root_image} -
}
