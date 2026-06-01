python visualize_tsne_pixel_hidden.py \
    --pixel_emb_ckpt models/pixel_emb_k_stage1.pt\
    --prefix_ckpt models/SP_k_stage1.pt \
    --intra_pos_ckpt models/INP_k_stage1.pt \
    --num_patches 30 \
    --output tsne_comparison.png