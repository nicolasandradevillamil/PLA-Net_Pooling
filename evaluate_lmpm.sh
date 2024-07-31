device=2
TARGET=akt1
BS=8
EXPERIMENT='LMPM/BINARY_'

CUDA_VISIBLE_DEVICES=$DEVICE python ensamble.py --batch_size 30 --save $EXPERIMENT$TARGET --target $TARGET --use_gpu --conv_encode_edge --balanced_loader --binary  --use_prot --LMPM
