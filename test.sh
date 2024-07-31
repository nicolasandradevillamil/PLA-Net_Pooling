device=1
TARGET='fgfr1'
EXPERIMENT='LMPM/BINARY_'
LM_PATH='./log/LM'
POOLING='gap'

CUDA_VISIBLE_DEVICES=$device python ensamble.py --batch_size 30 --save $EXPERIMENT$TARGET --target $TARGET --use_gpu --conv_encode_edge --balanced_loader --binary  --use_prot --LMPM --graph_pooling $POOLING
