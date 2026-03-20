# Example paper-faithful evaluation command
bash 3dgsVC/scripts/test.sh \
    --config 3dgsVC/configs/paper_lowmid.yaml \
    --dataset "/data/data54/wanghaobo/data/3D_Data/raw_data2/ksp.mat" \
    --weights "./results/paper_lowmid_acc8_pts200000_original_tile_cuda_seed42/checkpoints/best.pth" \
    --acceleration 8 \
    --gpu 1 \
    --slices_axial "20 40 10" \
    --slices_coronal "30 60 150" \
    --slices_sagittal "51 52 53"
