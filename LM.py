import subprocess

def run_experiment(target):
    device = 1
    bs = 2560
    experiment = 'LM/BINARY_temporal'
    pooling = 'gap'

    for cross_val in range(1, 5):
        command = f"CUDA_VISIBLE_DEVICES={device} python main.py --use_gpu --conv_encode_edge --learn_t --cross_val {cross_val} --save {experiment}{target} --batch_size {bs} --balanced_loader --batch_size {bs} --binary --target {target} --graph_pooling {pooling}"
        subprocess.run(command, shell=True)

targets = ['fgfr1','jak2','pa2ga','ada17','andr','ital','braf']
for target in targets:
    run_experiment(target)