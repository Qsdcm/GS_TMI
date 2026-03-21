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

  cd /data/data54/wanghaobo/Reproduce/GS_TMI/3DGS
  source /opt/anaconda3/etc/profile.d/conda.sh
  conda activate pt110
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=3dgsVC python 3dgsVC/train.py \
    --config 3dgsVC/configs/paper_lowmid.yaml \
    --max_iterations 3000 \
    --output_dir ./3dgsVC/results \
    --slices_axial 57 85 114 \
    --slices_coronal 45 68 90 \
    --slices_sagittal 52 78 104