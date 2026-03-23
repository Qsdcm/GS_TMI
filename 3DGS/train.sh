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

cd /data0/congcong/code/haobo/GS_TMI/3DGS
conda activate pt1.10
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=3dgsVC python 3dgsVC/train.py \
  --config 3dgsVC/configs/paper_lowmid.yaml \
  --max_iterations 3000 \
  --output_dir ./3dgsVC/results \
  --slices_axial 60 120 180 \
  --slices_coronal 60 120 180 \
  --slices_sagittal 40 80 120