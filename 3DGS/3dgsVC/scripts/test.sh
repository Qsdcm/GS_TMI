#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${PROJECT_ROOT}/configs/default.yaml"
GPU=0
DATASET=""
WEIGHTS=""
ACCELERATION=""
SAVE_VOLUME="true"
SAVE_SLICES="true"
SLICES_AXIAL=""
SLICES_CORONAL=""
SLICES_SAGITTAL=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --weights) WEIGHTS="$2"; shift 2 ;;
        --acceleration) ACCELERATION="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --no_save) SAVE_VOLUME="false"; SAVE_SLICES="false"; shift 1 ;;
        --slices_axial) SLICES_AXIAL="$2"; shift 2 ;;
        --slices_coronal) SLICES_CORONAL="$2"; shift 2 ;;
        --slices_sagittal) SLICES_SAGITTAL="$2"; shift 2 ;;
        --help)
            echo "Usage: bash 3dgsVC/scripts/test.sh --dataset PATH --weights PATH [OPTIONS]"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "${DATASET}" ] || [ -z "${WEIGHTS}" ]; then
    echo "Error: --dataset and --weights are required."
    exit 1
fi

export CUDA_VISIBLE_DEVICES=${GPU}
cd "${PROJECT_ROOT}"
CMD="python test.py --dataset ${DATASET} --weights ${WEIGHTS} --config ${CONFIG} --gpu 0"
if [ -n "${ACCELERATION}" ]; then CMD="${CMD} --acceleration ${ACCELERATION}"; fi
if [ -n "${OUTPUT_DIR}" ]; then CMD="${CMD} --output_dir ${OUTPUT_DIR}"; fi
if [ "${SAVE_VOLUME}" = "true" ]; then CMD="${CMD} --save_volume"; fi
if [ "${SAVE_SLICES}" = "true" ]; then CMD="${CMD} --save_slices"; fi
if [ -n "${SLICES_AXIAL}" ]; then CMD="${CMD} --slices_axial ${SLICES_AXIAL}"; fi
if [ -n "${SLICES_CORONAL}" ]; then CMD="${CMD} --slices_coronal ${SLICES_CORONAL}"; fi
if [ -n "${SLICES_SAGITTAL}" ]; then CMD="${CMD} --slices_sagittal ${SLICES_SAGITTAL}"; fi

echo "Executing: ${CMD}"
${CMD}
