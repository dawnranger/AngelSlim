# #!/bin/bash

DATASET_PATH=
CONFIG_DIR=angelslim/compressor/speculative/train/configs
TARGET_MODEL_NAME_OR_PATH=tencent/HunyuanOCR
DRAFT_MODEL_CONFIG_PATH=$CONFIG_DIR/hunyuan_ocr-eagle3.json
TARGET_BACKEND=hf
MAX_MODEL_LEN=8192
CHAT_TEMPLATE_TYPE=hunyuan_vl
OUTPUT_DIR=
echo $DATASET_PATH
echo $OUTPUT_DIR
torchrun --nproc_per_node=8 tools/generate_hidden_for_draft_model.py \
    --modal_type VLM \
    --dataset_path $DATASET_PATH \
    --target_model_name_or_path $TARGET_MODEL_NAME_OR_PATH \
    --draft_model_config_path  $DRAFT_MODEL_CONFIG_PATH \
    --target_backend $TARGET_BACKEND \
    --torch_dtype bfloat16 \
    --max_model_len $MAX_MODEL_LEN \
    --chat_template_type $CHAT_TEMPLATE_TYPE \
    --outdir $OUTPUT_DIR
