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
conda activate pt110
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=3dgsVC python 3dgsVC/train.py \
  --config 3dgsVC/configs/paper_lowmid.yaml \
  --max_iterations 3000 \
  --output_dir ./3dgsVC/results_new \
  --acceleration 10 \
  --slices_axial 60 120 180 \
  --slices_coronal 60 120 180 \
  --slices_sagittal 40 80 120
cd /data/data54/wanghaobo/Reproduce/GS_TMI/3DGS
conda activate pt110
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=3dgsVC python 3dgsVC/train.py \
  --config 3dgsVC/configs/paper_lowmid.yaml \
  --max_iterations 3000 \
  --output_dir ./3dgsVC/results_CE \
  --acceleration 10 \
  --slices_axial 60 120 180 \
  --slices_coronal 60 120 180 \
  --slices_sagittal 40 80 120