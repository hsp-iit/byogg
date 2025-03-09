# TODO: Maybe think about moving these "global" env variables in the src/yarp-app/configs/default.yaml file
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/workspace/hannes-graspvo/src
export DATA_ROOT=/docker_volume/example_images/pcd_data
export IHANNES_PCD_PRED_DATA=${DATA_ROOT}/pred_pcd
export IHANNES_PCD_RS_DATA=${DATA_ROOT}/rs_pcd


# Run the inference demo. Name of the pcd file (with .npy extension) should be given as a parameter
python scripts/contact_graspnet_inference.py  --np_path=${IHANNES_PCD_PRED_DATA}/$1