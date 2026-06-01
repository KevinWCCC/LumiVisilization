    #!/usr/bin/env bash
    # -*- coding: utf-8 -*-

    set -u
    # set -e  # 如果希望某条命令失败就立刻停止整个脚本，可以打开这行

    # ======================
    #   在这里填写你要执行的命令
    # ======================
    declare -a COMMANDS=(
        "python bpp_test_0319.py --mode curve \
            --logs ./logs/log_train_pixel_b_qwen-k-0318_stage1_0318_1717.log \
            --out bpp_kodak_stage1.png"
        # "python test2.py --jsonl data/kodak_p16_1-1.jsonl \
        #     --idx 0 \
        #     --pixel_emb_ckpt ./pixel.pt \
        #     --save results/tsne_pe_comparison.pdf"
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