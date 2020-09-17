FROM nvcr.io/nvidia/pytorch:19.10-py3
MAINTAINER Stephen Neuendorffer <stephenn@xilinx.com>

#
# get the basics
#
USER root
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install software-properties-common --assume-yes
RUN apt-get install wget curl unzip libxml2-dev --assume-yes
RUN apt-get install autoconf libtool g++ g++-multilib --assume-yes
RUN apt-get install build-essential python3 cmake git gitk --assume-yes
RUN apt-get install clang-8 lld-8 ninja-build --assume-yes
RUN apt-get install libncurses5-dev --assume-yes

RUN /opt/conda/bin/conda install matplotlib pybind11
#torchvision

ENV LD_LIBRARY_PATH "${LD_LIBRARY_PATH}:/opt/conda/lib"
# Rebuild pytorch

WORKDIR /opt/pytorch/pytorch

# this is the recommended rebuild command from NVIDIA
# with the cleanup of the build area omitted.
RUN TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5+PTX" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    NCCL_INCLUDE_DIR="/usr/include/" \
    NCCL_LIB_DIR="/usr/lib/" \
    python setup.py install

WORKDIR /workspace

# Additional packages for building npcomp
RUN apt-get install clang-10 lld-10 --assume-yes
RUN conda install -c gaiar nnpack

# Make it possible to symbolize stack traces in crashes.
RUN ln -s /usr/bin/llvm-symbolizer-10 /usr/bin/llvm-symbolizer

# Additional env for building npcomp and running tests.
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda-10.1/compat/lib.real"
ENV CC=clang-10
ENV CXX=clang++-10
ENV CXXFLAGS "-I/opt/conda/include"
ENV LDFLAGS "-fuse-ld=/usr/bin/ld.lld-10 -L/opt/conda/lib"
