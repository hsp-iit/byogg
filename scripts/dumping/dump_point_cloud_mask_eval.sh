# Dispenser/Mustard (rgb_00002, pinch)
CUDA_VISIBLE_DEVICES=1 python scripts/dumping/dump_point_cloud.py \
    --rgb_path /docker_volume/example_images_with_segmaps/dispenser/mustard/pinch/rgb_00002/00000000.jpg \
    --seg_path /docker_volume/example_images_with_segmaps/dispenser/mustard/pinch/partsmask_00002/00000000.png \
    --save_id dispenser_mustard_pinch_00002 \
    --format contact-graspnet

# Dispenser/Mustard (rgb_01000, power)
CUDA_VISIBLE_DEVICES=1 python scripts/dumping/dump_point_cloud.py \
    --rgb_path /docker_volume/example_images_with_segmaps/dispenser/mustard/power/rgb_01000/00000000.jpg \
    --seg_path /docker_volume/example_images_with_segmaps/dispenser/mustard/power/partsmask_01000/00000000.png \
    --save_id dispenser_mustard_power_01000 \
    --format contact-graspnet

# Dispenser/Pitcher (rgb_01001, pinch)
CUDA_VISIBLE_DEVICES=1 python scripts/dumping/dump_point_cloud.py \
    --rgb_path /docker_volume/example_images_with_segmaps/dispenser/pitcher/pinch/rgb_01001/00000000.jpg \
    --seg_path /docker_volume/example_images_with_segmaps/dispenser/pitcher/pinch/partsmask_01001/00000000.png \
    --save_id dispenser_pitcher_pinch_01001 \
    --format contact-graspnet

# Dispenser/Pitcher (rgb, lateral)
CUDA_VISIBLE_DEVICES=1 python scripts/dumping/dump_point_cloud.py \
    --rgb_path /docker_volume/example_images_with_segmaps/dispenser/pitcher/lateral/rgb/00000000.jpg \
    --seg_path /docker_volume/example_images_with_segmaps/dispenser/pitcher/lateral/partsmask/00000000.png \
    --save_id dispenser_pitcher_lateral \
    --format contact-graspnet

# Mug/025_mug (rgb_03023, pinch)
CUDA_VISIBLE_DEVICES=1 python scripts/dumping/dump_point_cloud.py \
    --rgb_path /docker_volume/example_images_with_segmaps/mug/025_mug/pinch/rgb_03023/00000000.jpg \
    --seg_path /docker_volume/example_images_with_segmaps/mug/025_mug/pinch/partsmask_03023/00000000.png \
    --save_id mug_025_mug_pinch_03023 \
    --format contact-graspnet

# Mug/025_mug (rgb_00018, power)
CUDA_VISIBLE_DEVICES=1 python scripts/dumping/dump_point_cloud.py \
    --rgb_path /docker_volume/example_images_with_segmaps/mug/025_mug/power/rgb_00018/00000000.jpg \
    --seg_path /docker_volume/example_images_with_segmaps/mug/025_mug/power/partsmask_00018/00000000.png \
    --save_id mug_025_mug_power_00018 \
    --format contact-graspnet

# Can/010_potted_meat_can (rgb_00025, power)
CUDA_VISIBLE_DEVICES=1 python scripts/dumping/dump_point_cloud.py \
    --rgb_path /docker_volume/example_images_with_segmaps/can/010_potted_meat_can/power/rgb_00025/00000000.jpg \
    --seg_path /docker_volume/example_images_with_segmaps/can/010_potted_meat_can/power/partsmask_00025/00000000.png \
    --save_id can_010_potted_meat_can_power_00025 \
    --format contact-graspnet

# long_fruit/011_banana (rgb_01000, pinch)
CUDA_VISIBLE_DEVICES=1 python scripts/dumping/dump_point_cloud.py \
    --rgb_path /docker_volume/example_images_with_segmaps/long_fruit/011_banana/pinch/rgb_01000/00000000.jpg \
    --seg_path /docker_volume/example_images_with_segmaps/long_fruit/011_banana/pinch/partsmask_01000/00000000.png \
    --save_id long_fruit_011_banana_pinch_01000 \
    --format contact-graspnet

# box/003_cracker_box (rgb, power)
CUDA_VISIBLE_DEVICES=1 python scripts/dumping/dump_point_cloud.py \
    --rgb_path /docker_volume/example_images_with_segmaps/box/003_cracker_box/power/rgb/00000000.jpg \
    --seg_path /docker_volume/example_images_with_segmaps/box/003_cracker_box/power/partsmask/00000000.png \
    --save_id box_003_cracker_box_power \
    --format contact-graspnet

# small_tool/037_scissors (rgb_01000, lateral)
CUDA_VISIBLE_DEVICES=1 python scripts/dumping/dump_point_cloud.py \
    --rgb_path /docker_volume/example_images_with_segmaps/small_tool/037_scissors/lateral/rgb_01000/00000000.jpg \
    --seg_path /docker_volume/example_images_with_segmaps/small_tool/037_scissors/lateral/partsmask_01000/00000000.png \
    --save_id small_tool_037_scissors_lateral_01000 \
    --format contact-graspnet

# small_tool/037_scissors (rgb_01004, lateral)
CUDA_VISIBLE_DEVICES=1 python scripts/dumping/dump_point_cloud.py \
    --rgb_path /docker_volume/example_images_with_segmaps/small_tool/037_scissors/lateral/rgb_01004/00000000.jpg \
    --seg_path /docker_volume/example_images_with_segmaps/small_tool/037_scissors/lateral/partsmask_01004/00000000.png \
    --save_id small_tool_037_scissors_lateral_01004 \
    --format contact-graspnet







