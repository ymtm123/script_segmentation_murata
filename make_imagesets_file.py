import glob
import os
import random

base_dir = "./data"
src_dir = os.path.join(base_dir, "PNGImages")
dst_dir = os.path.join(base_dir, "ImageSets/Segmentation")

train_rate = 0.8

file_paths = glob.glob(f"{src_dir}/*")
file_names = [os.path.basename(p).split(".")[0] for p in file_paths]
random.shuffle(file_names)

n_files = len(file_names)
n_train_files = int(n_files * train_rate)

with open(f"{dst_dir}/train.txt", encoding='utf-8', mode='w') as f:
    for i in range(n_train_files):
        f.write(file_names[i] + "\n")

with open(f"{dst_dir}/val.txt", encoding='utf-8', mode='w') as f:
    for i in range(n_train_files, n_files):
        f.write(file_names[i] + "\n")

with open(f"{dst_dir}/trainval.txt", encoding='utf-8', mode='w') as f:
    for i in range(n_files):
        f.write(file_names[i] + "\n")

