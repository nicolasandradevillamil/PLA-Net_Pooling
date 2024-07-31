import os
from tqdm import tqdm
import argparse
import subprocess

parser = argparse.ArgumentParser(description='Run models for a list of proteins.')
parser.add_argument('--proteins', type=str, help='Comma-separated list of protein names (e.g., "cp3a4,hdac2,prgr")')

args = parser.parse_args()
proteins = args.proteins.split(',')
print("Running for proteins:", proteins)

for protein in tqdm(proteins, desc="Proteins"):
    print(f"\nProcessing protein: {protein}")
    for cross_val in tqdm(range(1, 5), desc=f"Cross-validation for {protein}", leave=False):
        batch_size = 35
        while batch_size > 0:
            command = (
                f"CUDA_VISIBLE_DEVICES=1 python main.py --use_gpu --conv_encode_edge --learn_t --cross_val {cross_val} --save LMPM/BINARY_{protein} --balanced_loader --batch_size {batch_size} --binary --target {protein} --use_prot --LMPM --freeze_molecule --model_load_init_path /home/jpuentes/jpuentes2/planet_original/original/PLA-Net/log/LM"
            )
            try:
                subprocess.check_call(command, shell=True)
                print(f"Successfully executed with batch size {batch_size}")
                break  # Exit the loop if command was successful
            except subprocess.CalledProcessError:
                print(f"Failed with batch size {batch_size}, trying with batch size {batch_size-1}")
                batch_size -= 1  # Decrement batch size and try again

'''
#RUN LM
proteins = ['andr', 'ital', 'braf', 'thb', 'grik1', 'kpcb', 'hivint', 'pyrd', 'cdk2', 'fabp4', 'rxra', 'try1', 'kith', 'cp3a4', 'hdac2', 'prgr', 'src', 'xiap', 'pgh1', 'ppara', 'fnta', 'comt', 'cxcr4', 'tysy']

for protein in tqdm(proteins, desc="Proteins"):
    print(f"\nProcessing protein: {protein}")
    for cross_val in tqdm(range(1, 5), desc=f"Cross-validation for {protein}", leave=False):
        command = (
            f"CUDA_VISIBLE_DEVICES=0 python main.py --use_gpu --conv_encode_edge --learn_t "
            f"--cross_val {cross_val} --save LM/BINARY_{protein} --batch_size 2560 "
            f"--balanced_loader --binary --target {protein} --lr 5e-5"
        )
        os.system(command)'''