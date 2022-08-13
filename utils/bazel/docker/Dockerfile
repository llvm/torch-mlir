ARG BASE_IMG=ubuntu:18.04
FROM ${BASE_IMG} as dev-base

ARG ARCH="x86_64"
ARG BAZEL_VERSION=5.2.0

# Install basic packages
RUN apt-get update                                              && \
    apt-get install -y                                             \
    clang-10                                                       \
    curl                                                           \
    git                                                            \
    python3-pip                                                    \
    python3.8                                                      \
    python3.8-dev                                                  \
    wget                                                           \
    unzip

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 10

RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-10 10
RUN update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-10 10

# Install bazel
RUN wget -q https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-linux-${ARCH} -O /usr/bin/bazel \
    && chmod a+x /usr/bin/bazel

COPY requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app
RUN python -m pip install --upgrade pip
RUN python -m pip install --ignore-installed -r requirements.txt

WORKDIR /opt/src/torch-mlir
