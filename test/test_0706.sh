# # python pllm_0617_01.py \

python test_0706.py\
    --model_id /home/vipuser/Model/LLAMA_3.1_B \
    --dataset koadk \
    --ckpt_dir models \
    --use_best \
    --yellow_jitter_range 5 \
    --output pixel_embedding_0706
    # --no_annotation\

# # python pllm_0617_01.py \
# python test_0703_02.py --dataset koadk

# python vis_pixel_embeeding_bar.py\
#     --model_id /home/vipuser/Model/LLAMA_3.1_B \
#     --dataset koadk \
#     --ckpt_dir models \
#     --use_best \
#     --yellow_jitter_range 5 \
#     --output pixel_tsne_bar_mar
#     # --no_annotation\
    
    