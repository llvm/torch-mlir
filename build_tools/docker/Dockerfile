ARG BASE_IMG=ubuntu:22.04
FROM ${BASE_IMG} as dev-base

# Disable apt-key parse waring. If someone knows how to do whatever the "proper"
# thing is then feel free. The warning complains about parsing apt-key output,
# which we're not even doing.
ARG APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1

ARG ARCH="x86_64"
ARG REPO_NAME="deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy main"
RUN apt-get update && \
    apt-get install -y \
    ca-certificates \
    software-properties-common \
    wget \
    apt-transport-https \
    ccache \
    curl \
    cmake \
    ninja-build \
    git \
    gnupg \
    lsb-release \
    python3-pip \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    unzip && \
    echo $REPO_NAME >> /etc/apt/sources.list.d/llvm.list && \
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key| apt-key add - && \
    apt-get update && \
    apt-get install -y \
    clang \
    lld

######## Bazel ########
WORKDIR /install-bazel
ARG BAZEL_VERSION=5.2.0

# https://bazel.build/install/ubuntu
RUN curl -fsSL https://bazel.build/bazel-release.pub.gpg \
  | gpg --dearmor >bazel-archive-keyring.gpg \
  && mv bazel-archive-keyring.gpg /usr/share/keyrings \
  && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" \
  | tee /etc/apt/sources.list.d/bazel.list \
  && apt-get update \
  && apt-get install -y "bazel=${BAZEL_VERSION?}" \
  && rm -rf /install-bazel

### Clean up
RUN apt-get clean \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /main_checkout/torch-mlir
