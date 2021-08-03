FROM ubuntu:20.04
LABEL org.opencontainers.image.authors="Stella Laurenzo <stellaraccident@gmail.com>, Sean Silva <silvasean@google.com>"

USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update

# Python and build system.
RUN apt-get install -y python-is-python3 python3-pip
RUN apt-get install -y git cmake ccache ninja-build
RUN apt-get install -y clang lld
RUN ln -s /usr/bin/llvm-symbolizer-10 /usr/bin/llvm-symbolizer

# Install PyTorch
# Installs under: /usr/local/lib/python3.8/dist-packages/torch
RUN pip3 install numpy
RUN pip3 install pybind11
RUN pip3 install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
RUN ln -s /usr/local/lib/python3.8/dist-packages/torch /pytorch

# Build configuration
RUN mkdir /build && chmod a+rw /build
ENV CC clang
ENV CXX clang++
ENV LDFLAGS "-fuse-ld=/usr/bin/ld.lld"
ENV CMAKE_GENERATOR "Ninja"
ENV CMAKE_C_COMPILER_LAUNCHER ccache
ENV CMAKE_CXX_COMPILER_LAUNCHER ccache
ENV CMAKE_PREFIX_PATH /pytorch/share/cmake
ENV LLVM_BUILD_DIR /build/llvm-build
ENV LLVM_INSTALL_DIR /build/llvm-install
ENV NPCOMP_BUILD_DIR /build/npcomp
