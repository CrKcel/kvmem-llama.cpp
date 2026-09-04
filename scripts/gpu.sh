#!/usr/bin/env bash
# Physical GPUs on this machine (nvidia-smi PCI order):
#   GPU 0  NVIDIA GeForce RTX 5050 Laptop GPU   8151 MiB   — models < 27B
#   GPU 1  NVIDIA GeForce RTX 5090 Laptop GPU  24463 MiB   — 27B only
#
# CUDA's default device order is FASTEST_FIRST, so CUDA device 0 is the 5090
# unless CUDA_DEVICE_ORDER=PCI_BUS_ID. Bind by UUID so we cannot miss.
#
# Usage:
#   source scripts/gpu.sh small    # 5050
#   source scripts/gpu.sh 27b      # 5090

# nvidia-smi UUIDs on this machine
KVMEM_UUID_5050="GPU-14f08a8c-8d62-4338-8ae4-c669889cdb29"
KVMEM_UUID_5090="GPU-58a7c28b-e698-307f-c149-24d4ecd88bf4"

gpu_for() {
    case "$1" in
        small|lt27b|5050)
            echo "$KVMEM_UUID_5050"
            ;;
        27b|5090)
            echo "$KVMEM_UUID_5090"
            ;;
        *)
            echo "usage: source scripts/gpu.sh {small|27b}" >&2
            return 2
            ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    gpu_for "${1:-}"
    exit $?
fi

uuid="$(gpu_for "${1:-}")" || return $?
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$uuid"
if [[ "$uuid" == "$KVMEM_UUID_5050" ]]; then
    export KVMEM_GPU_INDEX=0
    export KVMEM_GPU_NAME="RTX 5050"
    export KVMEM_GPU_EXPECT="RTX 5050"
else
    export KVMEM_GPU_INDEX=1
    export KVMEM_GPU_NAME="RTX 5090"
    export KVMEM_GPU_EXPECT="RTX 5090"
fi
echo "CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ($KVMEM_GPU_NAME)"
