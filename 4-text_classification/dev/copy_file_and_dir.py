"""
复制文件、文件夹
"""
#%%
import shutil
import os
from pathlib import Path

#%%
os.makedirs("dst", exist_ok=True)
exclude = ["foo.py", "use_torchtext.py", "dst", "vocab"]  # 一定要去掉dst，否则目标文件夹dst里会无限嵌套dst
current_path = Path(__file__).parent
for path in os.listdir(current_path):
    if path not in exclude:
        if os.path.isdir(path):
            shutil.copytree(path, "./dst/" + path)
        else:
            shutil.copy(path, "./dst/" + path)