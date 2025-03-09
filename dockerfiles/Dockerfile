ARG base_img=nvidia/cuda:11.6.1-devel-ubuntu20.04
FROM ${base_img} AS byogg

LABEL maintainer="Giuseppe Stracquadanio (giuseppe dot stracquadanio [you know which symbol] visinf dot tu-darmstadt dot de)"
# Non-interactive installation mode
ENV DEBIAN_FRONTEND=noninteractive
ENV DOCKER_UPDATED_ON=20_06_2023

SHELL ["/bin/bash", "-c"]

# Install CMake
# Add Kitware APT repository for CMake
# Add Kitware repository and ignore signature verification
RUN apt-get update && apt-get install -y software-properties-common lsb-release wget gnupg && \
    wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc | apt-key add - && \
    apt-add-repository "deb [trusted=yes] https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" && \
    apt-get update && apt-get install -y cmake && \
    rm -rf /var/lib/apt/lists/*

RUN echo "deb [arch=amd64] http://www.icub.org/ubuntu jammy contrib/science" | tee /etc/apt/sources.list.d/icub.list
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 57A5ACB6110576A6 

# Install essentials
RUN apt-get update && apt-get install -y apt-utils sudo psmisc protobuf-compiler libatlas-base-dev \
    nano vim curl build-essential git gitk cmake-curses-gui autoconf xserver-xorg-video-dummy xserver-xorg-legacy \
    net-tools libjpeg-dev ffmpeg apt-transport-https ca-certificates gnupg libace-dev ycm-cmake-modules locales \
    python3-setuptools python3-pip iproute2 python3-tornado swig lsof iftop htop iputils-ping gdb bash-completion mlocate 
    #&& rm -rf /var/lib/apt/lists/*

RUN pip install numpy bpytop

RUN sed -i 's/allowed_users=console/allowed_users=anybody/' /etc/X11/Xwrapper.config
#COPY ./common/xorg.conf /etc/X11/xorg.conf

# Install yarp dependencies
RUN echo "deb http://www.icub.org/ubuntu `lsb_release -cs` contrib/science" > /etc/apt/sources.list.d/icub.list
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 57A5ACB6110576A6
RUN apt-get update && apt-get install -y icub-common libopencv-dev && sudo rm -rf /var/lib/apt/lists/*

USER root
RUN mkdir /home/robotology
RUN mkdir /home/repos 
ENV robotology_install_folder=/home/robotology
ENV repos_install_folder=/home/repos
ENV robotology_install_user=root

# Build ycm
USER $robotology_install_user
WORKDIR $robotology_install_folder
RUN git clone https://github.com/robotology/ycm.git -b master
RUN cd ycm && mkdir build && cd build && \
    cmake .. \
    -DCMAKE_BUILD_TYPE=Release && \
    make -j11
# ENV YCM_DIR=/home/user1/robotology/ycm/build
ENV YCM_DIR=/home/robotology/ycm/build

# Build YARP
USER $robotology_install_user
WORKDIR $robotology_install_folder
RUN git clone https://github.com/robotology/yarp.git -b yarp-3.10
RUN cd yarp && mkdir build && cd build && \
    cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DYARP_COMPILE_libYARP_math=ON \
    -DYARP_COMPILE_GUIS=ON \
    -DYARP_COMPILE_DEVICE_PLUGINS=ON \
    -DENABLE_yarpcar_mjpeg=ON \
    -DENABLE_yarpcar_human=ON \
    -DENABLE_yarpcar_segmentationimage=ON \
    -DENABLE_yarpcar_portmonitor=ON \
#    -DENABLE_yarpmod_portaudio=ON \
#    -DENABLE_yarpmod_portaudioPlayer=ON \
#    -DENABLE_yarpmod_portaudioRecorder=ON \
#    -DENABLE_yarpmod_fakeBattery=ON  \
#    -DENABLE_yarpmod_controlboardwrapper2=ON \
#    -DENABLE_yarpmod_laserFromDepth=ON \
#    -DENABLE_yarpmod_laserFromExternalPort=ON \
#    -DENABLE_yarpmod_laserFromDepth=ON \
#    -DENABLE_yarpmod_laserFromPointCloud=ON \
#    -DENABLE_yarpmod_laserFromRosTopic=ON \
#    -DENABLE_yarppm_soundfilter_resample=ON \
    -DENABLE_yarppm_bottle_compression_zlib=ON \
    -DENABLE_yarppm_depthimage_compression_zlib=ON \
    -DENABLE_yarppm_image_compression_ffmpeg=ON \
    -DENABLE_yarppm_depthimage_to_mono=ON \
    -DENABLE_yarppm_depthimage_to_rgb=ON && \
    make -j11
ENV YARP_ROOT=$robotology_install_folder/yarp
ENV YARP_DIR=$robotology_install_folder/yarp/build

# YARP bindings
USER $robotology_install_user
WORKDIR $robotology_install_folder
RUN cd yarp && cd bindings &&  mkdir build && cd build && cmake .. -DCREATE_PYTHON=ON && make -j11

# Install YARP completion
RUN sudo ln -s /usr/local/share/bash-completion/completions/yarp /usr/share/bash-completion/completions

# Set environmental variables
USER $robotology_install_user
# RUN echo "PS1='\[\e]0;\u \w\a\]\[\033[01;32m\]\u\[\033[00m\] \[\033[01;34m\]\w\[\033[00m\] \$ '" >> /home/$robotology_install_user/.bashrc
ENV PATH=$PATH:$YARP_DIR/bin
ENV DISPLAY=:1
ENV YARP_DATA_DIRS=$YARP_DIR/share/yarp
ENV LD_LIBRARY_PATH=$robotology_install_folder/yarp/build/lib/yarp/
ENV YARP_COLORED_OUTPUT=1
ENV QT_X11_NO_MITSHM=1
ENV PYTHONPATH=/home/robotology/yarp/bindings/build/lib/python3/

# Clone byogg repo
RUN git clone https://github.com/hsp-iit/byogg.git -b main /repos/byogg

ENV PYTHONPATH=$PYTHONPATH:~/repos/byogg/src
ENV CUDA_HOME=/usr/local/cuda

ENV workdir=$repos_install_folder/byogg
WORKDIR $workdir

RUN mkdir -p ~/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
    rm ~/miniconda3/miniconda.sh && \
    source ~/miniconda3/bin/activate && \
    conda init --all && \
    conda env create -f environment.yaml

RUN chmod +x install.sh
RUN chmod +x get-weights.sh
RUN pip install gdown