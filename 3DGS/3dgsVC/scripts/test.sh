#!/bin/bash
# 3DGSMR Testing Script (Fixed)
#
# 用法:
#   bash scripts/test.sh --dataset data.h5 --weights best.pth --acceleration 8 --slices_sagittal "50 100"

set -e

# ======================= 基础配置 =======================
PROJECT_ROOT="/data0/congcong/code/haobo/V3/3DGS/3dgsVC"
CONFIG="${PROJECT_ROOT}/configs/default.yaml"
GPU=0

# 必需参数
DATASET=""
WEIGHTS=""
ACCELERATION=""
SAVE_VOLUME="true"
SAVE_SLICES="true"

# 切片参数 (初始化为空)
SLICES_AXIAL=""
SLICES_CORONAL=""
SLICES_SAGITTAL=""

# ======================= 解析参数 =======================
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --weights) WEIGHTS="$2"; shift 2 ;;
        --acceleration) ACCELERATION="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --no_save) SAVE_VOLUME="false"; SAVE_SLICES="false"; shift 1 ;;
        
        # [新增] 切片参数解析
        --slices_axial) SLICES_AXIAL="$2"; shift 2 ;;
        --slices_coronal) SLICES_CORONAL="$2"; shift 2 ;;
        --slices_sagittal) SLICES_SAGITTAL="$2"; shift 2 ;;
        
        --help)
            echo "Usage: bash scripts/test.sh --dataset PATH --weights PATH [OPTIONS]"
            echo "Options:"
            echo "  --slices_axial '50 100'     Custom slices for Axial view"
            echo "  --slices_sagittal '50 100'  Custom slices for Sagittal view"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "${DATASET}" ] || [ -z "${WEIGHTS}" ]; then
    echo "Error: --dataset and --weights are required."
    exit 1
fi

# ======================= 环境设置 =======================
export CUDA_VISIBLE_DEVICES=${GPU}
cd ${PROJECT_ROOT}

# ======================= 构建命令 =======================
CMD="python test.py --dataset ${DATASET} --weights ${WEIGHTS} --config ${CONFIG} --gpu 0"

if [ -n "${ACCELERATION}" ]; then CMD="${CMD} --acceleration ${ACCELERATION}"; fi
if [ "${SAVE_VOLUME}" = "true" ]; then CMD="${CMD} --save_volume"; fi
if [ "${SAVE_SLICES}" = "true" ]; then CMD="${CMD} --save_slices"; fi

# [新增] 传递切片参数 (注意不加引号，利用shell特性展开为多个args)
if [ -n "${SLICES_AXIAL}" ]; then CMD="${CMD} --slices_axial ${SLICES_AXIAL}"; fi
if [ -n "${SLICES_CORONAL}" ]; then CMD="${CMD} --slices_coronal ${SLICES_CORONAL}"; fi
if [ -n "${SLICES_SAGITTAL}" ]; then CMD="${CMD} --slices_sagittal ${SLICES_SAGITTAL}"; fi

# ======================= 运行 =======================
echo "Executing: ${CMD}"
echo ""

${CMD}