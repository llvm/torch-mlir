ARG BASE_IMG=ubuntu:22.04
FROM ${BASE_IMG} as dev-base

# Install basic packages
RUN apt-get update && \
    apt-get install -y \
    wget \
    curl \
    git \
    python3-pip \
    python3.10 \
    python3.10-dev \
    unzip \
    libomp-dev

# Install clang
ARG REPO_NAME="deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-16 main"
RUN echo $REPO_NAME >> /etc/apt/sources.list.d/llvm.list && \
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
    apt-get update && \
    apt-get install -y \
    clang-16

# Install bazel
ARG ARCH="x86_64"
ARG BAZEL_VERSION=6.4.0
RUN wget -q https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-linux-${ARCH} -O /usr/bin/bazel \
    && chmod a+x /usr/bin/bazel

# Install torch-mlir requirements
COPY requirements.txt /opt/app/requirements.txt
COPY build-requirements.txt /opt/app/build-requirements.txt
COPY test-requirements.txt  /opt/app/test-requirements.txt
COPY torchvision-requirements.txt  /opt/app/torchvision-requirements.txt
COPY pytorch-requirements.txt /opt/app/pytorch-requirements.txt
WORKDIR /opt/app
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade --ignore-installed -r requirements.txt

# Clean up
RUN apt-get clean \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/src/torch-mlir/utils/bazel
