device=3
TARGET=thrb

### For Ligand Module
BS=35
EXPERIMENT='LM/BINARY_'
POOLING='gap'

CUDA_VISIBLE_DEVICES=$device python main.py --use_gpu --conv_encode_edge --learn_t --cross_val 1 --save $EXPERIMENT$TARGET --batch_size $BS --balanced_loader --batch_size $BS --binary --target $TARGET --graph_pooling $POOLING
CUDA_VISIBLE_DEVICES=$device python main.py --use_gpu --conv_encode_edge --learn_t --cross_val 2 --save $EXPERIMENT$TARGET --batch_size $BS --balanced_loader --batch_size $BS --binary --target $TARGET --graph_pooling $POOLING
CUDA_VISIBLE_DEVICES=$device python main.py --use_gpu --conv_encode_edge --learn_t --cross_val 3 --save $EXPERIMENT$TARGET --batch_size $BS --balanced_loader --batch_size $BS --binary --target $TARGET --graph_pooling $POOLING
CUDA_VISIBLE_DEVICES=$device python main.py --use_gpu --conv_encode_edge --learn_t --cross_val 4 --save $EXPERIMENT$TARGET --batch_size $BS --balanced_loader --batch_size $BS --binary --target $TARGET --graph_pooling $POOLING
