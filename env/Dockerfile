ARG UBUNTU_VERSION=20.04
ARG CUDA_VERSION=12.4.0
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION}
# An ARG declared before a FROM is outside of a build stage,
# so it can’t be used in any instruction after a FROM
ARG USER=gabriel
ARG PASSWORD=${USER}123$
ARG PYTHON_VERSION=3.12.0
# To use the default value of an ARG declared before the first FROM,
# use an ARG instruction without a value inside of a build stage:
ARG CUDA_VERSION

# Install ubuntu packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        ca-certificates \
        sudo \
        locales \
        openssh-server \
        libgl1-mesa-dev \
        libglib2.0-0 \
        vim && \
    # Remove the effect of `apt-get update`
    rm -rf /var/lib/apt/lists/* && \
    # Make the "en_US.UTF-8" locale
    localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
ENV LANG en_US.utf8

# Setup timezone
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

####################################################################################
# START USER SPECIFIC COMMANDS
####################################################################################

# Create an user for the app.
RUN useradd --create-home --shell /bin/bash --groups sudo ${USER}
RUN echo ${USER}:${PASSWORD} | chpasswd
USER ${USER}
ENV HOME /home/${USER}
WORKDIR $HOME

# Install miniconda (python)
# Referenced PyTorch's Dockerfile:
#   https://github.com/pytorch/pytorch/blob/master/docker/pytorch/Dockerfile
RUN curl -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 
RUN chmod +x miniconda.sh 
RUN ./miniconda.sh -b -p conda 
RUN rm miniconda.sh 
RUN conda/bin/conda install -y python=$PYTHON_VERSION jupyter jupyterlab 
RUN conda/bin/conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia 
COPY environment.yml .
RUN conda/bin/conda env create -f environment.yml
ENV PATH $HOME/conda/bin:$PATH
RUN touch $HOME/.bashrc && \
    echo "export PATH=$HOME/conda/bin:$PATH" >> $HOME/.bashrc
RUN conda init
# Expose port 8888 for JupyterLab
EXPOSE 22 8888

# Start openssh server
USER root
RUN mkdir /run/sshd
CMD ["/usr/sbin/sshd","-D"]