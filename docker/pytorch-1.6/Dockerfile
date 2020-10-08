FROM ubuntu:20.04
MAINTAINER Stella Laurenzo <stellaraccident@gmail.com>

USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update

# Python and build system.
RUN apt install -y python3 python-is-python3
RUN apt install -y python3-pip pybind11-dev python3-pybind11
RUN apt install -y git cmake ccache ninja-build
RUN apt install -y clang lld
RUN ln -s /usr/bin/llvm-symbolizer-10 /usr/bin/llvm-symbolizer

# Install PyTorch
# Installs under: /usr/local/lib/python3.8/dist-packages/torch
RUN pip3 install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN ln -s /usr/local/lib/python3.8/dist-packages/torch /pytorch

# Build configuration
RUN mkdir /ccache && ln -s /usr/bin/ccache /ccache/clang && ln -s /usr/bin/ccache /ccache/clang++
RUN mkdir /build && chmod a+rw /build /ccache
ENV PATH "/ccache:${PATH}"
ENV CC clang
ENV CXX clang++
# Binary distributions of torch force CXX11 ABI 0 :(
ENV CXXFLAGS "-D_GLIBCXX_USE_CXX11_ABI=0"
ENV LDFLAGS "-fuse-ld=/usr/bin/ld.lld"
ENV CMAKE_PREFIX_PATH /pytorch/share/cmake
ENV LLVM_BUILD_DIR /build/llvm-build
ENV LLVM_INSTALL_DIR /build/llvm-install
ENV NPCOMP_BUILD_DIR /build/npcomp
