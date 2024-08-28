from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.viz import show_raw_pointcloud,cat_3d

from matplotlib import pyplot as pl
import os
import cv2
# os.add_dll_directory("C:\\Software\\COLMAP\\lib")
import pycolmap
import open3d as o3d
import numpy as np
from dust3r.utils.geometry import xy_grid
import torch
from PIL import Image
from pathlib import Path
import copy

device = 'cuda'
batch_size = 1
schedule = 'cosine'
lr = 0.01
niter = 300

# load images
name_list = [os.path.join('./data/9001gate',i) for i in os.listdir('./data/9001gate')]          
images = load_images(name_list, size=512, square_ok=True)

model_name = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
# you can put the path to a local checkpoint in model_name if needed
model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)


