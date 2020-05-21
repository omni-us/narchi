FROM ubuntu:20.04

MAINTAINER Mauricio Villegas <mauricio@omnius.com>

ENV \
 DEBIAN_FRONTEND=noninteractive \
 LANG=en_US.UTF-8 \
 LC_ALL=C.UTF-8

RUN apt-get update --fix-missing \
 && apt-get dist-upgrade -y \
 && apt-get install -y --no-install-recommends \
      curl \
      less \
      nano \
      build-essential \
      software-properties-common \
      graphviz \
      libgraphviz-dev \
 && add-apt-repository -y -u ppa:deadsnakes/ppa \
 && apt-get install -y --no-install-recommends \
      python3.6 \
      python3.7 \
      python3.8 \
      python3.6-dev \
      python3.7-dev \
      python3.8-dev \
      python3.8-distutils \
 && curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
 && for PY in 3.6 3.7 3.8; do \
      python$PY /tmp/get-pip.py; \
    done \
 && apt-get purge -y \
      software-properties-common \
 && apt-get autoremove -y \
 && apt-get purge -y $(dpkg -l | awk '{if($1=="rc")print $2}') \
 && apt-get clean \
 && rm -rf /tmp/* /var/lib/apt/lists/*

COPY requirements.txt /tmp/

RUN for PY in 3.6 3.7 3.8; do \
      pip$PY install pygraphviz \
        --install-option="--include-path=/usr/include/graphviz" \
        --install-option="--library-path=/usr/lib/x86_64-linux-gnu/graphviz/"; \
      pip$PY install -r /tmp/requirements.txt; \
    done

COPY torch-*.whl /tmp/

RUN for PY in 3.6 3.7 3.8; do \
      pip$PY install /tmp/torch-*-cp$(echo $PY | tr -d .)-*.whl; \
    done \
 && rm /tmp/*

#RUN apt-get update --fix-missing \
# && apt-get install -y --no-install-recommends \
#      cmake \
#      git #\
# && git clone --branch v1.5.0 --depth 1 --shallow-submodules --recursive https://github.com/pytorch/pytorch.git /tmp/pytorch \
# && for PY in 3.6 3.7 3.8; do \
#       { git ls-files --others --ignored --exclude-standard --directory;
#         git ls-files --others --directory;
#       } | xargs --no-run-if-empty rm -fr;
#      export MAX_JOBS=4; \
#      export USE_CUDA=0; \
#      export USE_CUDNN=0; \
#      export USE_FBGEMM=0; \
#      export USE_MKLDNN=0; \
#      export USE_QNNPACK=0; \
#      export USE_DISTRIBUTED=0; \
#      export BUILD_CAFFE2_OPS=0; \
#      python$PY setup.py bdist_wheel; \
#    done
