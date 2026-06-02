python visualize_tsne_pixel_hidden.py \
    --pixel_emb_ckpt model/pixel_emb_k_stage1.pt\
    --prefix_ckpt model/SP_k_stage1.pt \
    --intra_pos_ckpt model/INP_k_stage1.pt \
    --num_patches 30 \
    --output tsne_comparison_2.png

python visualize_tsne_pixel_hidden.py \
    --pixel_emb_ckpt model/pixel_emb_k_stage1.pt\
    --prefix_ckpt model/SP_k_stage1.pt \
    --num_patches 30 \
    --output tsne_comparison_3.png