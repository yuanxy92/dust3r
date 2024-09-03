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
    parser.add_argument("-s", "--start", type=int, default=200, help="start frame index")
    parser.add_argument("-t", "--stop", type=int, default=900, help="stop frame index")
    parser.add_argument("-p", "--step", type=int, default=10, help="frame index step")
    return parser

if __name__ == '__main__':
    # Create the parser
    parser = get_args_parser()
    # Parse the arguments
    args = parser.parse_args()

    # dataset dir
    rootdir = args.root
    datadir = os.path.join(rootdir, f'{args.dataname}_undis256_sr')
    outdir = os.path.join(rootdir, f'{args.dataname}_recon')
    os.makedirs(outdir, exist_ok=True)
    # dust3r dir name
    outdir_dust3r = os.path.join(outdir, 'dust3r_npy')
    os.makedirs(outdir_dust3r, exist_ok=True)
    # pano dir name
    outdir_pano = os.path.join(outdir, 'pano')
    os.makedirs(outdir_pano, exist_ok=True)

    # iterative over all the images
    cam_indices = ['0', '1', '2', '4', '5', '6']
    for frame_idx in range(args.start, args.stop, args.step):
        # check if all the images exists
        isexist = True
        name_list = []
        for cam_idx in cam_indices:
            filename = os.path.join(datadir, f'camera_{cam_idx}_frame_{frame_idx}_corrected.png')
            isexist = isexist and os.path.isfile(filename)
            name_list.append(filename)
        if isexist == False:
            continue

        # apply dust3r
        framename = f'frame_{frame_idx}'
        dust3r_outname = os.path.join(outdir_dust3r, f'{framename}.npy')
        images = load_images(name_list, size=512, square_ok=True)
        model_name = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
        # dust3r inference
        model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, model, device, batch_size=batch_size)
        # align dust3r point clouds
        scene = global_aligner(output, device=device, min_conf_thr=1.5, mode=GlobalAlignerMode.PointCloudOptimizer)
        scene.preset_focal([246.8 / 400.0 * 512.0]*len(name_list))
        loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
        focals = scene.get_focals()
        avg_focal = sum(focals)/len(focals)
        # save results
        viz_3d.save_dust3r_poses_and_depth(scene, dust3r_outname)

        # apply pano stitching
        os.makedirs(outdir, exist_ok=True)
        with open(dust3r_outname, 'rb') as f:
            imgs = np.load(f)
            focals = np.load(f)
            poses = np.load(f)
            pts3d = np.load(f)
            confidence_masks = np.load(f)
        # generate bundle adjust image
        ba_imgs = convert_dust3r_to_pano(imgs, focals, poses, pts3d, confidence_masks)
        result = pano_stitch(ba_imgs, outdir=outdir_pano, outname=framename,
            blender=multiband_blend, equalize=False, crop=False)
