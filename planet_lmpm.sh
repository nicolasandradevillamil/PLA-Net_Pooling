device=2
TARGET=thrb

BS=35
EXPERIMENT='LMPM/BINARY_'
LM_PATH='/home/nandradev/Copia_PLA-Net/log/LM'
POOLING='gap'

CUDA_VISIBLE_DEVICES=$device python main.py --use_gpu --conv_encode_edge --learn_t --cross_val 1 --save $EXPERIMENT$TARGET --batch_size $BS --balanced_loader --batch_size $BS --binary --target $TARGET  --use_prot --LMPM  --freeze_molecule --model_load_init_path $LM_PATH --graph_pooling $POOLING
CUDA_VISIBLE_DEVICES=$device python main.py --use_gpu --conv_encode_edge --learn_t --cross_val 2 --save $EXPERIMENT$TARGET --batch_size $BS --balanced_loader --batch_size $BS --binary --target $TARGET  --use_prot --LMPM  --freeze_molecule --model_load_init_path $LM_PATH --graph_pooling $POOLING
CUDA_VISIBLE_DEVICES=$device python main.py --use_gpu --conv_encode_edge --learn_t --cross_val 3 --save $EXPERIMENT$TARGET --batch_size $BS --balanced_loader --batch_size $BS --binary --target $TARGET  --use_prot --LMPM  --freeze_molecule --model_load_init_path $LM_PATH --graph_pooling $POOLING
CUDA_VISIBLE_DEVICES=$device python main.py --use_gpu --conv_encode_edge --learn_t --cross_val 4 --save $EXPERIMENT$TARGET --batch_size $BS --balanced_loader --batch_size $BS --binary --target $TARGET  --use_prot --LMPM  --freeze_molecule --model_load_init_path $LM_PATH --graph_pooling $POOLING