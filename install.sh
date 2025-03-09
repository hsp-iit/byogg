# Set env variables here.
BYOGG_PATH=/home/repos/byogg
CONDA_ROOT=/root/miniconda3/condabin/conda
CONDA_ENV=byogg

# check if GLIBCXX_3.4.29 is available
check_glibcxx() {
    strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX_3.4.29 > /dev/null
    return $?  # Returns 0 if found, non-zero if not found
}

read -p "Are you sure you want to continue with the installation? (y/n): " answer
if [[ "$answer" == [Yy] ]]; then
    echo "Installing..."

    echo "export LD_LIBRARY_PATH=${CONDA_ROOT}/envs/${CONDA_ENV}/lib:/usr/lib/x86_64-linux-gnu:${YARP_DIR}/lib/yarp/:${YARP_ROOT}/bindings/build/lib/python3:$LD_LIBRARY_PATH" >> ~/.bashrc
    echo "export PYTHONPATH=${BYOGG_PATH}/src:${YARP_ROOT}/bindings/build/lib/python3:${BYOGG_PATH}/src/grasping/contact_graspnet/pointnet2/utils:$PYTHONPATH" >> ~/.bashrc 
    echo "export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7:$LD_PRELOAD" >> ~/.bashrc
    source ~/.bashrc

    source /root/miniconda3/etc/profile.d/conda.sh 
    conda activate ${CONDA_ENV}

    # Install dpvo package
    cd ${BYOGG_PATH}/src/odometry/dpvo
    pip install .

    # Compile pointnet custom CUDA operators
    cd ${BYOGG_PATH}/src/grasping/contact_graspnet
    sh compile_pointnet_tfops.sh

    # Check if a yarp installation is supposedly present in this way, as we later also use those variables in the script
    if [[ "${YARP_DIR}" != "${YARP_ROOT}/build" ]]; then
        echo "Assertion failed: Yarp is probably not installed. If yes, please set YARP_DIR and YARP_ROOT." >&2
        exit 1
    fi

    # Reinstall yarp python bindings, just to make sure that they are built with the correct target Python version
    cd ${YARP_ROOT}/bindings &&  mkdir build && cd build && cmake .. -DCREATE_PYTHON=ON && make -j11 
    cd ${BYOGG_PATH}

    # Check if GLIBCXX_3.4.29 is available
    if ! check_glibcxx; then
        echo "GLIBCXX_3.4.29 not found. Upgrading libstdc++6..."
        # Add the PPA repository
        sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
        # Update package lists
        sudo apt-get update
        # Upgrade libstdc++6
        sudo apt-get upgrade -y libstdc++6
    else
        echo "GLIBCXX_3.4.29 symbol has been found. You have a good version of libstdc++6. No action needed."
    fi

elif [[ "$answer" == [Nn] ]]; then
    echo "Aborted."
else
    echo "Invalid response. Please answer y or n."
fi


