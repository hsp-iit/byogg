<h1 align="center">Bring Your Own Grasp Generator: Leveraging Robot Grasp Generation for Prosthetic Grasping
  </h1> 
<p align="center">
  Giuseppe Stracquadanio
  ·
  Federico Vasile
  ·
  Elisa Maiettini
   ·
  Nicolò Boccardo
   ·
  Lorenzo Natale
</p>
<p align="center">
IIT
  ·
University of Genoa
</p>

<p align="center">
  <a href='https://arxiv.org/abs/2503.00466'>
    <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red' alt='Paper PDF'>
  </a>
  <a href='https://hsp-iit.github.io/byogg/'>
    <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
  </a>
</p>

<h2 align="center"> ICRA 2025</h3>

## Overview

We present a novel eye-in-hand prosthetic grasping system that follows the shared-autonomy principles. Our system initiates the approach-to-grasp action based on user’s command and automatically configures the DoFs of a prosthetic hand. First, it reconstructs
the 3D geometry of the target object without the need of a depth camera. Then, it tracks the hand motion during the approach-to-grasp action and finally selects a candidate grasp
configuration according to user’s intentions.

![Main Image](assets/repo-main.png)

In this repository, we provide the code to reproduce the prediction of a candidate grasp configuration on pre-recorded sequences, as well as new grasping sequences recorded with a camera. 

## TODO:
- [ ] Provide some example data (e.g., captured with Hannes camera) for testing the code.

## Installation (with Docker)

> [!IMPORTANT] 
> Installation with Docker is the only recommended way. You wil need ~22GB to build the image.

1. Build a docker image from the provided Dockerfile. It will automatically build dependencies (i.e., [yarp](https://github.com/robotology/yarp)) and install the conda environment needed to run the scripts.
```bash
docker build -t hsp-iit/byogg -f dockerfiles/Dockerfile . 
```
2. Run the container in detached mode and open a shell on the container.
```bash
docker run -t -d --name byogg-container --network host --pid host -v /dev:/dev -v /tmp/.X11-unix:/tmp/.X11-unix -e QT_X11_NO_MITSHM=1 -e DISPLAY=$DISPLAY --gpus all hsp-iit/byogg bash
docker exec -it byogg-container bash
```
> [!NOTE] 
> Steps 3. and 4. should be run inside the Docker container.

3. Activate the conda environment. 
```bash
conda activate byogg
```
4. Run the install.sh script.
```bash
./install.sh # to install DPVO and build custom Contact-GraspNet CUDA Kernels
```


> [!NOTE] 
> When not needed anymore, you can stop the running detached container with 
> ```bash
> docker stop byogg-container
> ```

## Get model weights
Simply run the script to download model weights (~1.6GB). We host our fine-tuned depth estimation weights and also open-source weights from [Contact-GraspNet](https://github.com/NVlabs/contact_graspnet) and [DPVO](https://github.com/princeton-vl/DPVO). For the latters, you can also download weights from the official repos.

> ```bash
> ./get-weights.sh
> ```

## Get pre-recorded sequences
Coming soon!

## Run the modules 
1. Run `xhost +` on your machine. `sudo apt install x11-xserver-utils` if needed.

> [!NOTE] 
> All next steps should be run inside the byogg container. For every point or module, run: 
> ```bash
> docker exec -it byogg-container bash
> ```
> and then activate the conda env
> ```bash
> conda activate byogg
> ```

2. Run the **yarpserver**.
```bash
yarpserver --write
```

3. Run the **yarpmanager** with the byogg yarp application and select `Applications > byogg-yarp-app`.
```bash
yarpmanager --application src/yarp-app/yarpmanager/byogg.xml
```

> [!TIP]
> [Troubleshooting] If you get an error from Qt, then you probably want to run `xhost +` on your local machine.

4. [**Optional**] From the yarpmanager GUI, you can run **yarpviews** to visualize the rgb input and the predicted depth map. Simply click on the `Run All`  button located on the vertical bar on the left.

5. Run all the needed yarp modules. It requires opening (many `:see_no_evil:`) shells on the running container. 
You will need to run the following yarp modules (i.e., the Python script associated to each of these yarp modules): 
    - **controller** `(src/yarp-app/rfmodules/controller.py)`, the controller module with the main logic
    - **depther** `(src/yarp-app/rfmodules/depther.py)`, the depth estimation module
    - **grasper** `(src/yarp-app/rfmodules/grasper.py)`, the grasp generation module
    - **odometer** `(src/yarp-app/rfmodules/odometer.py)`, the visual odometry module
    - **reader** `(src/yarp-app/rfmodules/reader.py)`, the camera or file reader
    - **visualizer** `(src/yarp-app/rfmodules/visualizer.py)`, the Open3D visualizer

    You can run the module scripts in any order. You need to run all the previous listed modules for the system to work properly.
    ```bash
    python src/yarp-app/rfmodules/<MODULE>.py
    ```

> [!TIP]
> [Troubleshooting] If you get the `undefined symbol: _ZN5torch3jit17parseSchemaOrNameERKSs` error from torch_scatter, simply `pip uninstall torch-scatter` and re-install it via `pip install torch-scatter --no-cache-dir`.

6. When all the modules are up and running, you need to connect their yarp ports to allow for data communication. You can do it from the yarpmanager GUI. Click on the `Connect All` button located on the vertical bar on the left.

7. [OPTIONAL] By default, the reader module will read frames from the camera or from a specified rgb folder. The camera mount index or the folder path can be specified under `yarp/stream` using the YAML config file `src/yap-app/configs/default.yaml`. It is possible to change the read folder at run-time by sending a message to the reader module via [RPC](https://www.yarp.it/latest/rpc_ports.html). Open new shell on the container and run

    ```bash
    yarp rpc /reader/command:i
    ```

    Then, send the command `set path/to/rgb/folder` to set a new read folder.
    When the pipeline will be activated, the module will read from the folder.
    You can use the `scripts/process_video.py` to pre-process your videos into frames of the correct resolution.
    For instance:
    ```python
    python scripts/process_video.py data/videos/example_0000.mp4 data/frames/example_0000
    ```
    and then set the path to the rgb folder in `src/yap-app/configs/default.yaml`:
    ```yaml
    stream: data/frames/example_0000 # path/to/rgb/folder
    ```

8. To manually start the pipeline (i.e., to process input frames), you can send a message to the controller module via RPC. Open new shell on the container and run

    ```bash
    yarp rpc /controller/command:i
    ```

    Then, send the command `controller ptrigger`. 
    
    ### (quite important) Brief explanation of the Controller logic.
    When you send the command `controller ptrigger` to the controller, the **reader** will start to read and output frames to all the modules. The **controller** will switch its state from `IDLE` to `ODOMETRY`. The **visualizer** will show the PCD, the predicted grasp poses (in red) and the selected one (in green). When the `ctrigger` logic is activated (i.e., the approaching stage is over), the controller will switch its state from `ODOMETRY` to `GRASPING`. The selected grasp pose is finally mapped to Hannes. It is possible to manually trigger the `GRASPING` stage by sending `controller ctrigger` via RPC. The pipeline will return to the `IDLE` stage after some seconds (you can set this time interval under `global/pipeline/times` in the config file). Look under `hannes/ctrigger` for other useful options.

## License Statement
Our work is released under [MIT](LICENSE) License. In our work we use Contact-GraspNet for grasp generation and, as such, our work can be intended as a derivative work of Contact-GraspNet. Note that Contact-GraspNet is under [NVIDIA](src/grasping/contact_graspnet/License.pdf) license. Thus, the use limitation described in Section 3.3 of [NVIDIA](src/grasping/contact_graspnet/License.pdf) license does apply to our work.

## Citation
If you found our work useful, please consider citing it.

```bibtex
@misc{stracquadanio2025bringgraspgeneratorleveraging,
  title={Bring Your Own Grasp Generator: Leveraging Robot Grasp Generation for Prosthetic Grasping}, 
  author={Giuseppe Stracquadanio and Federico Vasile and Elisa Maiettini and Nicolò Boccardo and Lorenzo Natale},
  year={2025},
  eprint={2503.00466},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2503.00466}, 
}
```