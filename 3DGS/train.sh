# Paper-faithful 8x low-mid acceleration reproduction (default paper path)
bash 3dgsVC/scripts/train.sh \
    --config 3dgsVC/configs/paper_lowmid.yaml \
    --acceleration 8 \
    --gpu 1 \
    --voxelizer tile_cuda \
    --strategy original \
    --slices_axial "20 40 10" \
    --slices_coronal "30 60 150" \
    --slices_sagittal "51 52 53"
