FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    golang-1.21 \
    golang-1.21-go \
    git \
    curl \
    wget \
    build-essential \
    libssl-dev \
    libgmp-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/lib/go-1.21/bin/go /usr/local/bin/go
RUN ln -s /usr/lib/go-1.21/bin/gofmt /usr/local/bin/gofmt
RUN ln -s /usr/lib/go-1.21/bin/goimports /usr/local/bin/goimports

RUN useradd -m -s /bin/bash vscode
USER vscode
WORKDIR /workspace

ENV PATH="/home/vscode/.local/bin:$PATH"
ENV PYTHONPATH="/workspace:$PYTHONPATH"
ENV GOROOT="/usr/lib/go-1.21"
ENV GOPATH="/home/vscode/go"
ENV PATH="$GOROOT/bin:$GOPATH/bin:$PATH" 