#!/usr/bin/env bash
# -*- coding: utf-8 -*-

set -u
# set -e  # 如果希望某条命令失败就立刻停止整个脚本，可以打开这行

# ======================
#   在这里填写你要执行的命令
# ======================
declare -a COMMANDS=(
    # "python test_0319.py \
    #     --data_jsonl ./data/kodak_p16_1-1.jsonl \
    #     --dataset k"
        # --model_id /home/vipuser/Model/QWEN_3_0.6B \

    # "python bpp_test_0319_2.py \
    #     --image ./data/kodim01.png \
        # --pixel_emb models/pixel_emb_llama-k-0319_stage1.pt \
        # --readout_head models/HeadPixel_llama-k-0319_stage1.pt \
        # --prefix models/SP_llama-k-0319_stage1.pt \
        # --intra_pos models/INP_llama-k-0319_stage1.pt \
        # --output results/kodim01_bpp.png"
        # --eval_jsonl ./data/kodak_p16_7-24.jsonl \
    # "python bpp_patch_2.py \
    #     --eval_jsonl ./data/kodak_p16_7-24.jsonl \
    #     --pixel_emb models/pixel_emb_llama-k-0319_stage1.pt \
    #     --readout_head models/HeadPixel_llama-k-0319_stage1.pt \
    #     --prefix models/SP_llama-k-0319_stage1.pt \
    #     --intra_pos models/INP_llama-k-0319_stage1.pt \
    #     --format_config configs/default_format.yaml \
    #     --sample_idx 0 \
    #     --patch_start_pixel 0 \
    #     --output patch_bpp_sample0.png\
    #     --patch_size 4 \
    #     --compare\
    #     --tag ps4"

    # --prefix models/SP_llama-k-0319_stage1.pt \
        # --sample_idx 0 \
        # --compare\


    "python bpp_patch_3.py\
        --eval_jsonl ./data/kodak_p16_7-24.jsonl \
        --pixel_emb models/pixel_emb_llama-k-0319_stage1.pt \
        --readout_head models/HeadPixel_llama-k-0319_stage1.pt \
        --intra_pos models/INP_llama-k-0319_stage1.pt \
        --format_config configs/default_format.yaml \
        --output patch_bpp_sample0.png\
        --grid_rows 32 \
        --grid_cols 32 \
        --tag kodim01_full"

        # 逐像素（最高精度）

#     python visualize_patch_bpp_ablation.py \
#   --eval_jsonl data/kodak_p16_7-24.jsonl \
#   --sample_idx 0 \
#   --patch_size 16 \
#   --compare
    # "python bpp_patch.py \
    #     --eval_jsonl ./data/kodak_p16_1-1.jsonl \
    #     --pixel_emb models/pixel_emb_llama-k-0319_stage1.pt \
    #     --readout_head models/HeadPixel_llama-k-0319_stage1.pt \
    #     --prefix models/SP_llama-k-0319_stage1.pt \
    #     --intra_pos models/INP_llama-k-0319_stage1.pt \
    #     --format_config configs/default_format.yaml \
    #     --sample_idx 0 \
    #     --patch_start_pixel 0 \
    #     --output patch_bpp_sample0.png"
    "python --version"
    "python -m pip --version"
)

# ======================
#        配置部分
# ======================
LOG_FILE="command_exec_$(date +%Y%m%d_%H%M%S).log"
{
    echo "执行开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "日志文件: $LOG_FILE"
    echo "总共要执行 ${#COMMANDS[@]} 条命令"
    echo "----------------------------------------"
} | tee -a "$LOG_FILE"

# ======================
#     执行并计时
# ======================
total_start=$(date +%s)

for i in "${!COMMANDS[@]}"; do
    cmd="${COMMANDS[$i]}"
    echo_num=$((i+1))

    {
        echo ""
        echo "[$echo_num/${#COMMANDS[@]}] $(date '+%H:%M:%S') ── $cmd"
        echo "----------------------------------------"
    } | tee -a "$LOG_FILE"

    cmd_start=$(date +%s.%N)

    # 执行命令，并同时输出到终端和日志文件
    bash -c "$cmd" 2>&1 | tee -a "$LOG_FILE"

    cmd_end=$(date +%s.%N)
    duration=$(echo "$cmd_end - $cmd_start" | bc)

    {
        printf "执行耗时: %.3f 秒\n" "$duration"
        echo "----------------------------------------"
    } | tee -a "$LOG_FILE"
done

total_end=$(date +%s)
total_duration=$((total_end - total_start))

{
    echo ""
    echo "全部执行完成"
    echo "总耗时: ${total_duration} 秒"
    echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "日志已保存至: $(pwd)/$LOG_FILE"
} | tee -a "$LOG_FILE"

# exit 0