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
import shutil
import argparse

from colmapio import colmap_io_tool as ct

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
    
def convert_dust3r_cameras_to_colmap_cameras(scene, imgnames, outdir):
    # get images, focal length
    imgs = scene.imgs
    focals = scene.get_focals()
    focals = focals.squeeze().detach().cpu().numpy()
    poses = scene.get_im_poses()
    poses = poses.detach().cpu().numpy()
    pts3d = scene.get_pts3d()
    pts3d = [pts.detach().cpu().numpy() for pts in pts3d]
    # confidence_masks = scene.get_conf()
    confidence_masks = scene.get_masks()
    for i in range(len(imgs)):
        confidence_masks[i] = confidence_masks[i].cpu().numpy()

    # make dirs like COLMAP
    outdir_colmap = os.path.join(outdir, 'colmap')
    outdir_colmap_sparse = os.path.join(outdir_colmap, 'sparse/0')
    outdir_colmap_images = os.path.join(outdir_colmap, 'images')
    os.makedirs(outdir_colmap_sparse, exist_ok=True)
    os.makedirs(outdir_colmap_images, exist_ok=True)

    # compute scale in dust3r
    img_temp = cv2.imread(imgnames[0])
    scale = img_temp.shape[0] / imgs[0].shape[0]

    # convert cameras
    colmap_cameras = []
    for idx in range(len(imgs)):
        camera = ct.Camera(
            idx + 1, 
            "PINHOLE", 
            imgs[idx].shape[1] * scale, 
            imgs[idx].shape[0] * scale,
            [focals[idx] * scale, focals[idx] * scale, 
             imgs[idx].shape[1] / 2 * scale, imgs[idx].shape[0] / 2 * scale]
            )
        colmap_cameras.append(camera)
    # write cameras
    camera_pathname = os.path.join(outdir_colmap_sparse, 'cameras.txt')
    ct.write_cameras_text(colmap_cameras, camera_pathname)

    # convert images
    colmap_images = []
    for idx in range(len(imgs)):
        basename = os.path.basename(imgnames[idx])
        pose = poses[idx]
        rotation_mat = poses[idx][:3, :3]
        rotation_mat = rotation_mat.transpose()
        tvec = poses[idx][:3, 3]
        qvec = ct.rotmat2qvec(rotation_mat)
        image = ct.BaseImage(
            idx + 1,
            qvec,
            tvec,
            idx + 1,
            basename,
            [],
            []
        )
        colmap_images.append(image)
        # copy images
        shutil.copyfile(imgnames[idx], os.path.join(outdir_colmap_images, basename))
    # write images
    image_pathname = os.path.join(outdir_colmap_sparse, 'images.txt')
    ct.write_images_text(colmap_images, image_pathname)

    # generate 3d points
    pts3d = scene.get_pts3d()
    pts3d = [pts.detach().cpu().numpy() for pts in pts3d]
    pts2d_all_list, pts3d_all_list, all_rgbs = [], [], []
    for i in range(len(imgs)):
        conf_i = confidence_masks[i]
        pts2d_all_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])  # imgs[i].shape[:2] = (H, W)
        pts3d_all_list.append(pts3d[i][conf_i])
        img = imgs[i]
        H,W = img.shape[:2]
        for pos in pts2d_all_list[i]:
            x,y = pos
            all_rgbs.append(img[y,x,:])
    all_pts3d = np.concatenate(pts3d_all_list, axis=0)
    n_viz = min(80000, len(all_rgbs))
    idx_to_viz = list(np.round(np.linspace(0, len(all_rgbs)-1, n_viz)).astype(int))
    vis_rgbs = [all_rgbs[idx] for idx in idx_to_viz]
    vis_xyzs = all_pts3d[idx_to_viz]
    points3D = o3d.geometry.PointCloud()
    points3D.points = o3d.utility.Vector3dVector(vis_xyzs)
    points3D.colors = o3d.utility.Vector3dVector(vis_rgbs)
    # write 3d points
    pt3d_pathname = os.path.join(outdir_colmap_sparse, 'points3D.ply')
    o3d.io.write_point_cloud(pt3d_pathname, points3D)


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

    # save to colmap results
    convert_dust3r_cameras_to_colmap_cameras(scene, all_filenames, outdir)


if __name__ == '__main__':
    main()