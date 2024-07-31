import subprocess
import csv

def run_experiment(target, csv_writer):
    device = 0
    experiment = 'LMPM/BINARY_'
    lm_path = '/home/nandradev/GAP_PLA-Net/log/LM'
    pooling = 'gap'
    batch_sizes = [35, 20, 10]

    for cross_val in range(1, 5):
        for bs in batch_sizes:
            try:
                command = f"CUDA_VISIBLE_DEVICES={device} python main.py --use_gpu --conv_encode_edge --learn_t --cross_val {cross_val} --save {experiment}{target} --batch_size {bs} --balanced_loader --binary --target {target} --use_prot --LMPM --freeze_molecule --model_load_init_path {lm_path} --graph_pooling {pooling}"
                subprocess.run(command, shell=True, check=True)
                csv_writer.writerow([target, cross_val, bs])
                break  # Exit the loop if the command succeeds
            except subprocess.CalledProcessError:
                print(f"Error occurred with batch size {bs} for target {target} and cross_val {cross_val}. Trying next batch size.")

targets = ["fgfr1", "jak2"]
with open('batch_sizes_used.csv', mode='w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(['Target', 'Cross_Val', 'Batch_Size'])
    for target in targets:
        run_experiment(target, csv_writer)