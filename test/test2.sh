
# python visualize_tsne_pixel_hidden.py \
#     --pixel_emb_ckpt model/pixel_emb_k_stage1.pt\
#     --prefix_ckpt model/SP_k_stage1.pt \
#     --intra_pos_ckpt model/INP_k_stage1.pt \
#     --num_patches 30 \
#     --output tsne_comparison_1.png

# # ========== 模式 1：完全 Baseline（无任何自定义组件） ==========
python visualize_tsne_ablation.py \
    --eval_jsonl data/kodak_p16_7-24.jsonl \
    --pixel_emb_ckpt model/pixel_emb_k_stage1.pt \
    --intra_pos_ckpt model/INP_k_stage1.pt \
    --num_patches 8 \
    --max_points 12000 \
    --perplexity 40 \
    --output tsne_pe_head_inp_1.png

# # ========== 模式 2：只使用 PE + Heading（无 INP） ==========
# python /home/workdir/artifacts/visualize_tsne_ablation.py \
#     --eval_jsonl data/kodak_p16_7-24.jsonl \
#     --pixel_emb_ckpt models/pixel_emb_k_stage1.pt \
#     --mode_name "PE+Head" \
#     --num_patches 8 \
#     --max_points 12000 \
#     --perplexity 40 \
#     --output tsne_pe_only_1.png
    # --mode_name "PE+Head+INP" \


# ========== 模式 3：PE + Heading + INP（完整配置） ==========
# python visualize_tsne_ablation.py \
#     --eval_jsonl data/kodak_p16_7-24.jsonl \
#     --pixel_emb_ckpt model/pixel_emb_k_stage1.pt \
#     --intra_pos_ckpt model/INP_k_stage1.pt \
#     --prefix_ckpt model/SP_k_stage1.pt \
#     --num_patches 8 \
#     --max_points 12000 \
#     --perplexity 40 \
#     --output tsne_pe_head_inp_1.png