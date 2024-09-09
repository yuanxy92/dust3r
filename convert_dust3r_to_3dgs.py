from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.viz import show_raw_pointcloud,cat_3d
from pano.pano.stitcher import SphProj, no_blend, linear_blend, multiband_blend, no_blend
from pano.ptcloud_to_pano import convert_dust3r_to_pano, pano_stitch

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
import shutil
import argparse

device = 'cuda'
batch_size = 1
schedule = 'cosine'
lr = 0.01
niter = 300

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", type=str, help="data root dir", default='/data/xiaoyun/OV6946_Arm_6_cameras/20240829_256px_v2')
    parser.add_argument("-d", "--dataname", type=str, help="data name", default='data_0829_8')
    return parser

def list_all_files(folder_path):
    try:
        # Walk through the folder and subfolders to list all files
        all_files = []
        base_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                all_files.append(os.path.join(root, file))
                base_files.append(file)
        return all_files, base_files
    except FileNotFoundError:
        return f"Error: The folder '{folder_path}' does not exist."
    except Exception as e:
        return f"An error occurred: {str(e)}"

def main():
    # Create the parser
    parser = get_args_parser()
    # Parse the arguments
    args = parser.parse_args()

    # dataset dir
    rootdir = args.root
    datadir = os.path.join(rootdir, f'{args.dataname}')
    outdir = os.path.join(rootdir, f'{args.dataname}_recon')
    # dust3r dir name
    out_dust3r = os.path.join(outdir, 'dust3r_result.npy')
    # pano dir name
    outdir_colmap = os.path.join(outdir, 'colmap')
    os.makedirs(outdir_colmap, exist_ok=True)

    # get image names
    all_filenames, base_filenames = list_all_files(datadir)

    # apply dust3r
    images = load_images(all_filenames, size=512, square_ok=True)
    model_name = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"

    # dust3r inference
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)
    # align dust3r point clouds
    scene = global_aligner(output, device=device, min_conf_thr=1.5, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
    focals = scene.get_focals()
    avg_focal = sum(focals)/len(focals)
    # save results
    viz_3d.save_dust3r_poses_and_depth(scene, out_dust3r)

if __name__ == '__main__':
    main()