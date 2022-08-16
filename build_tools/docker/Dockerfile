ARG BASE_IMG=ubuntu:18.04
FROM ${BASE_IMG} as dev-base

ARG ARCH="x86_64"
ARG TARGETARCH="amd64"
ARG BAZEL_VERSION=4.2.1

# Install basic packages
RUN apt-get update                                              && \
    apt-get install -y                                             \
    python3.8                                                      \
    python3.8-dev                                                  \
    cmake                                                          \
    ccache                                                         \
    ninja-build                                                    \
    git                                                            \
    python3-pip                                                    \
    wget                                                           \
    clang-10                                                       \
    automake                                                       \
    libtool                                                        \
    curl                                                           \
    make                                                           \
    unzip

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 10

# Install bazel
RUN wget -q https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-linux-${ARCH} -O /usr/bin/bazel \
    && chmod a+x /usr/bin/bazel

COPY externals/llvm-project/mlir/python/requirements.txt /opt/app/mlir-requirements.txt
COPY requirements.txt /opt/app/torch-mlir-requirements.txt
WORKDIR /opt/app
RUN python -m pip install --upgrade pip
RUN python -m pip install -r mlir-requirements.txt
RUN python -m pip install -r torch-mlir-requirements.txt

WORKDIR /opt/src/torch-mlir
