# test 1
# python test.py --jsonl_path "./kodak_p16_1-1.jsonl" 

# test 2
# python test2.py --jsonl data/kodak_p16_1-1.jsonl \
#     --idx 0 \
#     --pixel_emb_ckpt models/pixel_emb_init.pt \
#     --save results/tsne_pe_comparison.pdf

# python test_bpp.py --image ./data/kodim01.png --bpps 10 8 7 6

# python test_bpp_2.py --image ./data/kodim01.png --bpps 10 11 12 14

# python test_bpp_2.py --image ./data/kodim01.png --bpps 14 1.5 --window_size 15

python test_bpp_2.py --image ./data/SCI01.bmp --bpps 4 1.5 --window_size 15

# python test_bpp_2.py --image ./data/BRACS_265_N_1.png --bpps 15 1.5 --window_size 15

#  --block_size 32


# python test_sne.py

