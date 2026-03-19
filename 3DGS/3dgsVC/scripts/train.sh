#!/bin/bash
# 3DGSMR Training Script (Updated)

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${PROJECT_ROOT}/configs/default.yaml"
GPU=0

DATA_PATH=""
OUTPUT_DIR=""
ACCELERATION=""
MAX_ITERATIONS=""
INITIAL_POINTS=""
SEED=""
RESUME=""

# [新增] 切片参数
SLICES_AXIAL=""
SLICES_CORONAL=""
SLICES_SAGITTAL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --acceleration) ACCELERATION="$2"; shift 2 ;;
        --max_iterations) MAX_ITERATIONS="$2"; shift 2 ;;
        --initial_points) INITIAL_POINTS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --resume) RESUME="$2"; shift 2 ;;
        --data_path) DATA_PATH="$2"; shift 2 ;;
        
        # [新增]
        --slices_axial) SLICES_AXIAL="$2"; shift 2 ;;
        --slices_coronal) SLICES_CORONAL="$2"; shift 2 ;;
        --slices_sagittal) SLICES_SAGITTAL="$2"; shift 2 ;;
        
        --help)
            echo "Usage: bash scripts/train.sh [OPTIONS]"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES=${GPU}
cd ${PROJECT_ROOT}

CMD="python train.py --config ${CONFIG} --gpu 0"

if [ -n "${DATA_PATH}" ]; then CMD="${CMD} --data_path ${DATA_PATH}"; fi
if [ -n "${OUTPUT_DIR}" ]; then CMD="${CMD} --output_dir ${OUTPUT_DIR}"; fi
if [ -n "${ACCELERATION}" ]; then CMD="${CMD} --acceleration ${ACCELERATION}"; fi
if [ -n "${MAX_ITERATIONS}" ]; then CMD="${CMD} --max_iterations ${MAX_ITERATIONS}"; fi
if [ -n "${INITIAL_POINTS}" ]; then CMD="${CMD} --initial_points ${INITIAL_POINTS}"; fi
if [ -n "${SEED}" ]; then CMD="${CMD} --seed ${SEED}"; fi
if [ -n "${RESUME}" ]; then CMD="${CMD} --resume ${RESUME}"; fi

# [新增] 传递切片参数
if [ -n "${SLICES_AXIAL}" ]; then CMD="${CMD} --slices_axial ${SLICES_AXIAL}"; fi
if [ -n "${SLICES_CORONAL}" ]; then CMD="${CMD} --slices_coronal ${SLICES_CORONAL}"; fi
if [ -n "${SLICES_SAGITTAL}" ]; then CMD="${CMD} --slices_sagittal ${SLICES_SAGITTAL}"; fi

echo "Executing: ${CMD}"
echo ""

${CMD}