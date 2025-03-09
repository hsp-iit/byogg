# Example: ./scripts/dumping/dump_and_viz_everything.sh /docker_volume/example_images/rgb_00002 mustard_pinch_2 

FRAME_PATH=$1
FIRST_FRAME=$1/00000000.jpg
SAVE_NAME=$2

CUDA_VISIBLE_DEVICES=1 python scripts/dumping/dump_point_cloud.py \
    --rgb_path $FIRST_FRAME \
    --save_id $SAVE_NAME \
    --format contact-graspnet

CUDA_VISIBLE_DEVICES=1 python scripts/contact_graspnet_inference.py \
    --np_path /docker_volume/pcd_data/pred_pcd/pred_$2.npy 

CUDA_VISIBLE_DEVICES=0 python scripts/dpvo_demo.py \
    --imagedir $FRAME_PATH \
    --stride 1 \
    --output_name $SAVE_NAME

## Visualization with cameras (obtained through visual odometry) and grasp pruning.
python scripts/visualization/visualize_all.py \
    --pcd pred_$SAVE_NAME.npy \
    --cameras $SAVE_NAME.npy \
    --grasps predictions_pred_$SAVE_NAME.npz

## Visualization with grasps only.
python scripts/visualization/visualize_all.py \
   --pcd pred_$SAVE_NAME.npy \
   --cameras $SAVE_NAME.npy \
   --grasps predictions_pred_$SAVE_NAME.npz \
   --viz_cameras 0