bash scripts/test.sh \
    --dataset "/data0/congcong/data/Anke_2D/Anke_3D/Gao_3D_dataread/03_0_5T_kspace_norm_nopermute.h5" \
    --weights /data0/congcong/code/haobo/V3/3DGS/3dgsVC/LOWoutputs/acc2_pts510_grad0.005_seed42/checkpoints/best.pth \
    --acceleration 2 \
    --gpu 3 \
    --slices_axial "30 29 31" \
    --slices_coronal "60 120" \
    --slices_sagittal "70 140"