Median train step time: 8503.63 ms My best wall time so far is 15s. Implemented with 1)
FSDP, 2) Flashattention, 3) check pointing every four block 4) fuse loss and lm head. I tune
the tile size for to be 64 * 64. And I also use mixed precision in training. I also played around
with DDP with is much faster but it can’t fit it in the memory. I also used high precision.
