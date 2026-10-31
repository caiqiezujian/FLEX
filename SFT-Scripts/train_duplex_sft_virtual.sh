#!/usr/bin/env bash
# SWIFT 3.12.1 / Megatron SFT. Same optimization recipe as the supplied script;
# virtual audio input and unmerged adapter saves. Run in the existing omni_ms container.
# System prompts come from the prepared JSONL; no separate --system override.
set -euo pipefail

if (( $# < 3 )); then
  echo "Usage: bash train_duplex_sft_virtual.sh MCORE_MODEL TRAIN_JSONL OUTPUT_DIR [--gpu-ids 0,1,2,6] [--compat-old-driver] [native Megatron options...]" >&2
  exit 2
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="$1"
DATASET_PATH="$2"
OUTPUT_PATH="$3"
shift 3
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,6}"
COMPAT_OLD_DRIVER=0
EXTRA_ARGS=()
while (( $# )); do
  case "$1" in
    --gpu-ids)
      if (( $# < 2 )); then echo "--gpu-ids requires a value" >&2; exit 2; fi
      GPU_LIST="$2"; shift 2 ;;
    --compat-old-driver) COMPAT_OLD_DRIVER=1; shift ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done
IFS=',' read -r -a DEVICES <<< "$GPU_LIST"
if (( ${#DEVICES[@]} != 4 )); then
  echo "This recipe uses 4 GPUs (global=16/EP=2). Set --gpu-ids to four IDs." >&2
  exit 2
fi
if [[ ! -d "$MODEL_PATH" || ! -f "$DATASET_PATH" ]]; then
  echo "Missing model directory or training JSONL. Paths must exist inside this container." >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export ENABLE_AUDIO_OUTPUT=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
ATTENTION_ARGS=(--attention_backend flash)
PACKING_ARGS=(--packing true)
MICRO_BATCH=4
if (( COMPAT_OLD_DRIVER )); then
  # These are coupled: TE unfused does not support the THD padding-free layout.
  for arg in "${EXTRA_ARGS[@]}"; do
    case "$arg" in
      --attn_impl|--attn_impl=*|--attention_backend|--attention_backend=*|--packing|--packing=*|--padding_free|--padding_free=*|--transformer_impl|--transformer_impl=*)
        echo "--compat-old-driver controls $arg; remove that override to preserve the fallback." >&2
        exit 2 ;;
    esac
  done
  export FLEX_SFT_ATTENTION_COMPAT=1
  export NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=0 NVTE_UNFUSED_ATTN=1
  export NVTE_MASKED_SOFTMAX_FUSION=0
  ATTENTION_ARGS=(--attn_impl sdpa --attention_backend unfused)
  PACKING_ARGS=(--packing false --padding_free false)
  MICRO_BATCH=1
  echo "[train] Old-driver compatibility: HF math SDPA; TE unfused; packing/padding_free=false; micro=1/global=16. Expect lower speed and different memory usage."
else
  unset FLEX_SFT_ATTENTION_COMPAT
fi

python3 "$SCRIPT_DIR/check_duplex_virtual.py" "$DATASET_PATH" --rows 2 --framework
echo "[train] Original audio is read by range; no small WAV files will be written. GPUs=$GPU_LIST"
python3 -m torch.distributed.run \
  --nproc_per_node 4 --nnodes 1 --master_port "${MASTER_PORT:-29571}" \
  --module swift.cli._megatron.sft \
  --load "$MODEL_PATH" \
  --dataset "$DATASET_PATH" \
  --external_plugins "$SCRIPT_DIR/duplex_sft_plugin.py" \
  --template flex_duplex_qwen3_omni \
  --train_type lora \
  --lora_rank 8 --lora_alpha 32 \
  --target_modules all-linear \
  --modules_to_save word_embeddings output_layer \
  --new_special_tokens "$SCRIPT_DIR/duplex_protocol_tokens.txt" \
  --loss_scale default \
  --sequence_parallel true "${PACKING_ARGS[@]}" \
  --freeze_llm false --freeze_vit true --freeze_aligner true \
  --load_from_cache_file false \
  --split_dataset_ratio 0.01 \
  --expert_model_parallel_size 2 \
  --moe_permute_fusion true --moe_grouped_gemm true \
  --moe_shared_expert_overlap true --moe_aux_loss_coeff 1e-3 \
  --micro_batch_size "$MICRO_BATCH" --global_batch_size 16 \
  --recompute_granularity full --recompute_method uniform --recompute_num_layers 1 \
  --finetune true --cross_entropy_loss_fusion false \
  --lr 1e-4 --lr_warmup_fraction 0.05 --min_lr 1e-5 \
  --max_epochs 3 \
  --save "$OUTPUT_PATH" \
  --save_safetensors true --merge_lora false \
  --eval_interval 1000000 --save_interval 3000 \
  --vit_gradient_checkpointing true --max_length 4096 \
  --num_workers 8 --dataset_num_proc 8 \
  --no_save_optim true --no_save_rng true "${ATTENTION_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
