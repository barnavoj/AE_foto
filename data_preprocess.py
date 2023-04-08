import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2
import os
from glob import glob


src_dir = "database\\src"
dst_dir = "database\\train"

dsize = 224

subfolders = [f.path for f in os.scandir(src_dir) if f.is_dir()]

for subfolder in subfolders:

    filenames = glob(os.path.join(subfolder ,"*.jpg"))
    if not filenames:
        filenames = glob(os.path.join(subfolder ,"*.png"))
    if not filenames:
        filenames = glob(os.path.join(subfolder ,"*.tif"))
    for i, file in enumerate(filenames):
        img = cv2.imread(file)
        img = cv2.resize(img, dsize=(dsize, dsize))
        print(file)
        cv2.imwrite(os.path.join(dst_dir, file.split("\\")[3].split(".")[0] + ".png") , img)