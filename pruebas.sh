device=1
TARGET=src

BS=35
EXPERIMENT='LMPM/BINARY_'
LM_PATH='/home/jpuentes/jpuentes2/planet_original/original/PLA-Net/log/LM'

CUDA_VISIBLE_DEVICES=$device python main.py --use_gpu --conv_encode_edge --learn_t --cross_val 1 --save $EXPERIMENT$TARGET --batch_size $BS --balanced_loader --batch_size $BS --binary --target $TARGET  --use_prot --LMPM  --freeze_molecule --model_load_init_path $LM_PATH
