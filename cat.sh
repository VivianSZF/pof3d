python train_geod.py \
    --outdir output_folder \
    --cfg cats \
    --data /input/datasets/cats/cats_processed.zip \
    --create_label_fov 18.837 \
    --gpus 8 \
    --batch 32 \
    --gamma 5 \
    --dataset_resolution 256 \
    --neural_rendering_resolution_initial 64 \
    --neural_rendering_resolution_final 64 \
    --superres True \
    --dual_discrimination True \
    --flip_to_disd_weight 1.0 \
    --flip_type flip_both \
    --lr_multiplier 0.01 \
    --lr_multiplier_pose 0.01 \
    --gen_pose_cond True \
    --dis_pose_cond True \
    --dis_cam_weight 10.0 \
    --pose_gd_weight 10.0