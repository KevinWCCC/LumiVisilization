# python test_0604.py \
#   --model_id /home/vipuser/Model/LLAMA_3.1_B \
#   --pixel_emb_ckpt model/pixel_emb_k_stage1.pt\
#   --output tsne_ascii_vs_pixel.png \
#   --device cuda \



#   --perplexity 30 \

python viz_pixel_embedding_demo.py \
#   --model_id /home/vipuser/Model/LLAMA_3.1_B \
#   --pixel_emb_ckpt model/pixel_emb_k_stage1.pt\
#   --output tsne_ascii_vs_pixel.png \
#   --device cuda \
#   --perplexity 30 \
#   --n_iter 1000

#   --pixel_emb_ckpt models/pixel_emb_yourdataset_stage1_best.pt \

# python visualize_tsne_embeddings.py \
#   --model_id /home/vipuser/Model/LLAMA_3.1_B \
#   --pixel_emb_ckpt models/pixel_emb_yourdataset_stage1_best.pt \
#   --output tsne_ascii_vs_pixel.png \
#   --device cuda \
#   --perplexity 30 \
#   --n_iter 1000


  