### For Ligand Module
BS=35
EXPERIMENT='LM/BINARY_'
POOLING='gap'
DEVICE=2
TARGET='braf'

CUDA_VISIBLE_DEVICES=$DEVICE python ensamble.py --batch_size 30 --save $EXPERIMENT$TARGET --target $TARGET --use_gpu --conv_encode_edge --balanced_loader --binary --graph_pooling $POOLING
