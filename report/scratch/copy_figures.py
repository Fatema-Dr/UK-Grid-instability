import shutil
import os

src_dir = "/home/yogipatel/Documents/UK-Grid-instability/Final Report/figures/"
dst_dir = "/home/yogipatel/Documents/UK-Grid-instability/report/figures/"

if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

for item in os.listdir(src_dir):
    s = os.path.join(src_dir, item)
    d = os.path.join(dst_dir, item)
    if os.path.isfile(s):
        shutil.copy2(s, d)
        print(f"Copied {item}")
