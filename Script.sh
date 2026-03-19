# test 1
# python test.py --jsonl_path "./kodak_p16_1-1.jsonl" 

# test 2
python test2.py --jsonl data/kodak_p16_1-1.jsonl \
    --idx 0 \
    --pixel_emb_ckpt models/pixel_emb_init.pt \
    --save results/tsne_pe_comparison.pdf