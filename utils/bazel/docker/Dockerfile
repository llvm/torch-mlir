ARG BASE_IMG=ubuntu:22.04
FROM ${BASE_IMG} as dev-base

# Install basic packages. Bazel supplies its Python test dependencies, so
# an explicit installation of project requirements isn't needed in this image.
RUN apt-get update && \
    apt-get install -y \
    wget \
    curl \
    git \
    gnupg \
    python3 \
    unzip

# Install clang
ARG REPO_NAME="deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-16 main"
RUN echo $REPO_NAME >> /etc/apt/sources.list.d/llvm.list && \
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
    apt-get update && \
    apt-get install -y \
    clang-16

# Install bazel
ARG TARGETARCH
ARG BAZEL_VERSION=8.7.0
RUN ARCH=${TARGETARCH} \
    && if [ ${TARGETARCH} = "amd64" ]; then ARCH="x86_64"; fi \
    && wget -q https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-linux-${ARCH} -O /usr/bin/bazel \
    && chmod a+x /usr/bin/bazel

RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-16 100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-16 100

# Clean up
RUN apt-get clean \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/src/torch-mlir/utils/bazel
