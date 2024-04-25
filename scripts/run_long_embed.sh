#!/bin/bash
# debug_mode="-m debugpy --listen 127.0.0.1:6679 --wait-for-client"

model_name_or_path=$1
model_type=$2

if [[ $# -ge 3 ]]; then
    encode_max_len=$3
fi

# task_list=("LEMBSummScreenFDRetrieval" "LEMBQMSumRetrieval" "LEMBWikimQARetrieval" "LEMBNarrativeQARetrieval")
# task_list=("LEMBNeedleRetrieval" "LEMBPasskeyRetrieval")
task_list=("LEMBSummScreenFDRetrieval" "LEMBQMSumRetrieval" "LEMBWikimQARetrieval" "LEMBNarrativeQARetrieval" "LEMBNeedleRetrieval" "LEMBPasskeyRetrieval")

export CHUNKING_MODE="no_chunk"
export CUDA_VISIBLE_DEVICES=2

unset RANK

if [ $model_type = "mistral_ntk" ]; then
    python ${debug_mode} src/test_long_embed.py \
        --model_name_or_path ${model_name_or_path} \
        --output_dir ./results/ \
        --window_length_list 256 512 1024 2048 4096 8192 16384 32768 \
        --batch_size 4 \
        --use_fp16 \
        --use_xformers \
        --task_list "${task_list[@]}" \
        --encode_max_length ${encode_max_len} \
        --rope_theta 100000 \

elif [ $model_type = "mistral_se" ]; then
    python ${debug_mode} src/test_long_embed.py \
        --model_name_or_path ${model_name_or_path} \
        --output_dir ./results/ \
        --window_length_list 256 512 1024 2048 4096 8192 16384 32768 \
        --batch_size 1 \
        --use_fp16 \
        --task_list "${task_list[@]}" \
        --encode_max_length ${encode_max_len} \
        --use_self_extend \
        --group_size_1 9 \
        --group_size_2 512 \

elif [ $model_type = "e5rope_ntk" ]; then

    python ${debug_mode} src/test_long_embed.py \
        --model_name_or_path ${model_name_or_path} \
        --output_dir ./results/ \
        --window_length_list 256 512 1024 2048 4096 8192 16384 32768 \
        --batch_size 16 \
        --use_fp16 \
        --use_xformers \
        --task_list "${task_list[@]}" \
        --encode_max_length ${encode_max_len} \
        --rope_theta 100000 \

elif [ $model_type = "e5rope_se" ]; then

    python ${debug_mode} src/test_long_embed.py \
        --model_name_or_path ${model_name_or_path} \
        --output_dir ./results/ \
        --window_length_list 256 512 1024 2048 4096 8192 16384 32768 \
        --batch_size 4 \
        --use_fp16 \
        --task_list "${task_list[@]}" \
        --encode_max_length ${encode_max_len} \
        --use_self_extend \
        --group_size_1 9 \
        --group_size_2 64 \

elif [ $model_type = "group" ]; then

    python ${debug_mode} src/test_long_embed.py \
        --model_name_or_path ${model_name_or_path} \
        --output_dir ./results/ \
        --window_length_list 256 512 1024 2048 4096 8192 16384 32768 \
        --batch_size 4 \
        --use_fp16 \
        --use_xformers \
        --task_list "${task_list[@]}" \
        --encode_max_length ${encode_max_len} \
        --pos_mode "group" \

elif [ $model_type = "recur" ]; then

    python ${debug_mode} src/test_long_embed.py \
        --model_name_or_path ${model_name_or_path} \
        --output_dir ./results/ \
        --window_length_list 256 512 1024 2048 4096 8192 16384 32768 \
        --batch_size 16 \
        --use_fp16 \
        --use_xformers \
        --task_list "${task_list[@]}" \
        --encode_max_length ${encode_max_len} \
        --pos_mode "recurrent" \


elif [ $model_type = "interpolate" ]; then

    python ${debug_mode} src/test_long_embed.py \
        --model_name_or_path ${model_name_or_path} \
        --output_dir ./results/ \
        --window_length_list 256 512 1024 2048 4096 8192 16384 32768 \
        --batch_size 4 \
        --use_fp16 \
        --use_xformers \
        --task_list "${task_list[@]}" \
        --encode_max_length ${encode_max_len} \
        --pos_mode "interpolate" \

elif [ $model_type = "pcw" ]; then

    export MAX_TOKEN_NUM=$4
    export CHUNKING_MODE="chunk"
    export CHUNK_SIZE=${encode_max_len}
    # in this case, encode_max_len is size of each context window, e.g., 512
    # and MAX_TOKEN_NUM is the total number of tokens to process, e.g., 4096

    python ${debug_mode} src/test_long_embed.py \
        --model_name_or_path ${model_name_or_path} \
        --output_dir ./results/ \
        --window_length_list 256 512 1024 2048 4096 8192 16384 32768 \
        --batch_size 8 \
        --use_xformers \
        --pos_mode "original" \
        --task_list "${task_list[@]}" \
        --encode_max_length ${encode_max_len} \
        --chunk_mode "token" \

# for nomic
elif [ $model_type = "nomic" ]; then

    python ${debug_mode} src/test_long_embed.py \
        --model_name_or_path ${model_name_or_path} \
        --output_dir ./results/ \
        --window_length_list 256 512 1024 2048 4096 8192 16384 32768 \
        --batch_size 8 \
        --use_fp16 \
        --task_list "${task_list[@]}" \
        --encode_max_length 8192 \
        --rotary_scaling_factor 16

# for bge-m3
elif [ $model_type = "bge-m3" ]; then

    python ${debug_mode} src/test_long_embed.py \
        --model_name_or_path ${model_name_or_path} \
        --output_dir ./results/ \
        --window_length_list 256 512 1024 2048 4096 8192 16384 32768 \
        --batch_size 4 \
        --use_fp16 \
        --use_xformers \
        --task_list "${task_list[@]}" \
        --encode_max_length 8192 \

# for e5_mistral
elif [ $model_type = "mistral" ]; then

    python ${debug_mode} src/test_long_embed.py \
        --model_name_or_path ${model_name_or_path} \
        --output_dir ./results/ \
        --window_length_list 256 512 1024 2048 4096 8192 16384 32768 \
        --batch_size 8 \
        --use_fp16 \
        --use_xformers \
        --task_list "${task_list[@]}" \
        --encode_max_length 4096 \

# for others
elif [ $model_type = "default" ]; then

    python ${debug_mode} src/test_long_embed.py \
        --model_name_or_path ${model_name_or_path} \
        --output_dir ./results/ \
        --window_length_list 256 512 1024 2048 4096 8192 16384 32768 \
        --batch_size 8 \
        --use_fp16 \
        --use_xformers \
        --task_list "${task_list[@]}" \

else
    echo "Invalid model type: ${model_type}"
fi