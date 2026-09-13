#!/usr/bin/env bash
# Physical GPUs on this machine (nvidia-smi PCI order):
#   GPU 0  NVIDIA GeForce RTX 5050 Laptop GPU   8151 MiB   — models < 27B
#   GPU 1  27B card: RTX 5090 Laptop (24463 MiB) or RTX 5060 Ti (16311 MiB)
#
# CUDA's default device order is FASTEST_FIRST unless CUDA_DEVICE_ORDER=PCI_BUS_ID.
# Bind by UUID so we cannot miss. Never put 27B on the 5050.
#
# Usage:
#   source scripts/gpu.sh small     # 5050
#   source scripts/gpu.sh 27b       # 5090 if present, else 5060 Ti
#   source scripts/gpu.sh 5060ti
#   source scripts/gpu.sh 5090

# nvidia-smi UUIDs on this machine
KVMEM_UUID_5050="GPU-14f08a8c-8d62-4338-8ae4-c669889cdb29"
KVMEM_UUID_5090="GPU-58a7c28b-e698-307f-c149-24d4ecd88bf4"
KVMEM_UUID_5060TI="GPU-5847813c-9e6e-bb43-cc5e-621aac091b6c"

gpu_present() {
    nvidia-smi -L 2>/dev/null | grep -q "$1"
}

gpu_for() {
    case "$1" in
        small|lt27b|5050)
            echo "$KVMEM_UUID_5050"
            ;;
        5060ti|5060)
            echo "$KVMEM_UUID_5060TI"
            ;;
        5090)
            echo "$KVMEM_UUID_5090"
            ;;
        27b)
            if gpu_present "$KVMEM_UUID_5090"; then
                echo "$KVMEM_UUID_5090"
            elif gpu_present "$KVMEM_UUID_5060TI"; then
                echo "$KVMEM_UUID_5060TI"
            else
                echo "no 27B GPU (5090 / 5060 Ti) visible" >&2
                return 2
            fi
            ;;
        *)
            echo "usage: source scripts/gpu.sh {small|27b|5090|5060ti}" >&2
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
elif [[ "$uuid" == "$KVMEM_UUID_5060TI" ]]; then
    export KVMEM_GPU_INDEX=1
    export KVMEM_GPU_NAME="RTX 5060 Ti"
    export KVMEM_GPU_EXPECT="RTX 5060 Ti"
else
    export KVMEM_GPU_INDEX=1
    export KVMEM_GPU_NAME="RTX 5090"
    export KVMEM_GPU_EXPECT="RTX 5090"
fi
echo "CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ($KVMEM_GPU_NAME)"
