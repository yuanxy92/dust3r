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
import viz_3d

device = 'cuda'
batch_size = 1
schedule = 'cosine'
lr = 0.01
niter = 300

# load images
dirname = './data/9001gate'
npyname = './data/9001gate.npy'
name_list = [os.path.join(dirname,i) for i in os.listdir(dirname)]          
images = load_images(name_list, size=512, square_ok=True)

model_name = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
# you can put the path to a local checkpoint in model_name if needed
model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)

pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
output = inference(pairs, model, device, batch_size=batch_size)

# at this stage, you have the raw dust3r predictions
view1, pred1 = output['view1'], output['pred1']
view2, pred2 = output['view2'], output['pred2']

scene = global_aligner(output, device=device, min_conf_thr=1.5, mode=GlobalAlignerMode.PointCloudOptimizer)
scene.preset_focal([246.8 / 400.0 * 512.0]*len(name_list))
loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
focals = scene.get_focals()
avg_focal = sum(focals)/len(focals)

# save results
viz_3d.save_dust3r_poses_and_depth(scene, npyname)

# viz_3d.draw_dust3r_scene(scene)
# viz_3d.draw_dust3r_match_ez(scene)