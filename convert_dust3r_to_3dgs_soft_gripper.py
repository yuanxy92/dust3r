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
from plyfile import PlyData, PlyElement
from colmapio import colmap_io_tool as ct
from colmapio.colmap_tranform import apply_4x4_transform, apply_camera_transform_4x4

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

def project_3d_2d(R,T,f,cx,cy,p):
    x,y,z = R @ p + T
    u = int((f * x)/ z + cx)
    v = int((f * y) / z + cy)
    return u,v,z

def generate_depth_map(pt_map,R,T,f,cx,cy):
    depth = np.zeros_like(pt_map[:,:,2])
    for i in range(pt_map.shape[0]):
        for j in range(pt_map.shape[1]):
            p = pt_map[i,j,:]
            # if p[-1] == 1:
            #     z = 1
            u,v,z = project_3d_2d(R,T,f,cx,cy,p)
            depth[i,j] = z
    return depth

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

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
    
def convert_dust3r_cameras_to_colmap_cameras(scene, imgnames, outdir, \
    scale_low_conf_ = 5.0, mask_dir = None, transform4x4 = np.eye(4), transformscale = 1.0):
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
    outdir_colmap_depths = os.path.join(outdir_colmap, 'depths')
    os.makedirs(outdir_colmap_sparse, exist_ok=True)
    os.makedirs(outdir_colmap_images, exist_ok=True)
    os.makedirs(outdir_colmap_depths, exist_ok=True)

    # compute scale in dust3r
    img_temp = cv2.imread(imgnames[0])
    scale = img_temp.shape[0] / imgs[0].shape[0]
    orig_w = int(img_temp.shape[1])
    orig_h = int(img_temp.shape[0])
    current_w = int(imgs[0].shape[1])
    current_h = int(imgs[0].shape[0])
    scale_3dpts = 50

    # convert cameras
    colmap_cameras = []
    for idx in range(len(imgs)):
        camera = ct.Camera(
            idx + 1, 
            "PINHOLE", 
            orig_w, 
            orig_h,
            [focals[idx] * scale, focals[idx] * scale, 
             imgs[idx].shape[1] / 2 * scale, imgs[idx].shape[0] / 2 * scale]
            )
        colmap_cameras.append(camera)
    # write cameras
    camera_pathname = os.path.join(outdir_colmap_sparse, 'cameras.txt')
    ct.write_cameras_text(colmap_cameras, camera_pathname)
    ct.write_cameras_binary(colmap_cameras, os.path.join(outdir_colmap_sparse, 'cameras.bin'))

    # convert images
    colmap_images = []
    colmap_rots = []
    colmap_trans = []
    camera_positions = []
    for idx in range(len(imgs)):
        basename = os.path.basename(imgnames[idx])
        # inverse RT matrix, from cam2world to world2cam 
        rotation_mat = poses[idx][:3, :3]
        tvec = poses[idx][:3, 3]
        rotation_mat = rotation_mat.transpose()
        tvec = -rotation_mat @ tvec * transformscale * scale_3dpts

        extrinsic_orig = np.eye(4)
        extrinsic_orig[:3, :3] = rotation_mat
        extrinsic_orig[:3, 3] = tvec
        extrinsic_new = apply_camera_transform_4x4(extrinsic_orig, transform4x4)
        rotation_mat = extrinsic_new[:3, :3] 
        tvec = extrinsic_new[:3, 3] 
        
        qvec = ct.rotmat2qvec(rotation_mat)
        image = ct.BaseImage(
            idx + 1,
            qvec,
            tvec ,
            idx + 1,
            basename,
            [],
            []
        )
        colmap_rots.append(rotation_mat)
        colmap_trans.append(tvec)
        camera_position = -rotation_mat.transpose() @ tvec
        camera_positions.append(camera_position)
        colmap_images.append(image)
        # copy images
        if mask_dir is None:
            shutil.copyfile(imgnames[idx], os.path.join(outdir_colmap_images, basename))
        else:
            # generate RGBA image
            img_bgr = cv2.imread(imgnames[idx])
            img_mask = cv2.imread(os.path.join(mask_dir, basename), flags=cv2.IMREAD_UNCHANGED)
            img_mask = img_mask[:, :, 3:4]
            img_bgra = np.concatenate([img_bgr, img_mask], axis=2)
            cv2.imwrite(os.path.join(outdir_colmap_images, basename), img_bgra)

    # write images
    image_pathname = os.path.join(outdir_colmap_sparse, 'images.txt')
    ct.write_images_text(colmap_images, image_pathname)
    ct.write_images_binary(colmap_images, os.path.join(outdir_colmap_sparse, 'images.bin'))

    # generate 3d points
    pts3d = scene.get_pts3d()
    pts3d = [pts.detach().cpu().numpy() for pts in pts3d]
    for pts3d_idx in range(len(pts3d)):
        pt3d_ = pts3d[pts3d_idx]
        s0, s1, s2 = pt3d_.shape
        pt3d_ = pt3d_.reshape((s0 * s1, s2)) * transformscale * scale_3dpts
        pt3d_ = apply_4x4_transform(pt3d_, transform4x4) 
        pts3d[pts3d_idx] = pt3d_.reshape((s0, s1, s2))

    # add points in mask
    pts2d_all_list = []
    pts3d_all_list = []
    all_pts3d = []
    all_rgbs = []
    for i in range(len(imgs)):
        basename = os.path.basename(imgnames[i])
        if mask_dir is not None:
            mask = cv2.imread(os.path.join(mask_dir, basename), flags=cv2.IMREAD_UNCHANGED)
            if mask.shape[2] == 4:
                conf_i = mask[:, :, 3].astype(np.bool_)
            else:
                conf_i = np.ones((mask.shape[0], mask.shape[1])).astype(np.bool_)
        else:
            conf_i = np.ones((imgs[i].shape[0], imgs[i].shape[1])).astype(np.bool_)
        pts2d_all_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])  # imgs[i].shape[:2] = (H, W)
        pts3d_all_list.append(pts3d[i][conf_i])
        img = imgs[i]
        for pos in pts2d_all_list[i]:
            x,y = pos
            all_rgbs.append(img[y,x,:] * 255.0)
    all_pts3d = np.concatenate(pts3d_all_list, axis=0)

    # # add points with high confidence
    # pts2d_all_list, pts3d_all_list, all_rgbs_high = [], [], []
    # for i in range(len(imgs)):
    #     conf_i = confidence_masks[i]
    #     pts2d_all_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])  # imgs[i].shape[:2] = (H, W)
    #     pts3d_all_list.append(pts3d[i][conf_i])
    #     img = imgs[i]
    #     for pos in pts2d_all_list[i]:
    #         x,y = pos
    #         all_rgbs_high.append(img[y,x,:] * 255.0)
    # all_pts3d_high = np.concatenate(pts3d_all_list, axis=0)

    # # add points with low confidence
    # pts2d_all_list, pts3d_all_list, all_rgbs_low = [], [], []
    # for i in range(len(imgs)):
    #     conf_i = np.logical_not(confidence_masks[i])
    #     pts2d_all_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])  # imgs[i].shape[:2] = (H, W)
    #     pts3d_all_list.append(pts3d[i][conf_i])
    #     img = imgs[i]
    #     for pos in pts2d_all_list[i]:
    #         x,y = pos
    #         all_rgbs_low.append(img[y,x,:] * 255.0)
    #     # scale pts3d_all_list
    #     pts3d_all_list[i] = pts3d_all_list[i] - camera_position
    #     pts3d_all_list[i] = pts3d_all_list[i] * scale_low_conf_ + camera_position
    # all_pts3d_low = np.concatenate(pts3d_all_list, axis=0)
    # # concate to generate the final pts3d and rgb
    # all_pts3d = np.concatenate([all_pts3d_high, all_pts3d_low], axis=0)
    # all_rgbs = np.concatenate([all_rgbs_high, all_rgbs_low], axis=0)


    n_viz = min(160000, len(all_rgbs))
    idx_to_viz = list(np.round(np.linspace(0, len(all_rgbs)-1, n_viz)).astype(int))
    vis_rgbs = [all_rgbs[idx] for idx in idx_to_viz]
    vis_xyzs = all_pts3d[idx_to_viz].astype(np.float32)

    pt3d_pathname = os.path.join(outdir_colmap_sparse, 'points3D_o3d.ply')
    storePly(pt3d_pathname, vis_xyzs, vis_rgbs)
    pt3d_pathname = os.path.join(outdir_colmap_sparse, 'points3D.ply')
    storePly(pt3d_pathname, vis_xyzs, vis_rgbs)
    pt3d_pathname = os.path.join(outdir_colmap_sparse, 'points3d.ply')
    storePly(pt3d_pathname, vis_xyzs, vis_rgbs)

    # generate depth maps
    for idx in range(len(imgs)):
        basename = os.path.basename(imgnames[idx])[:-4]
        conf_i = confidence_masks[idx] 
        pt_map = pts3d[idx]
        # convert point clouds to depth maps
        depthimage = generate_depth_map(pt_map, colmap_rots[idx], colmap_trans[idx], 
            focals[idx], current_w / 2, current_h / 2)
        depthimage = depthimage
        depthimage[conf_i == False] = depthimage[conf_i == False] * scale_low_conf_
        depthfilename = os.path.join(outdir_colmap_depths, f'{basename}.npy')
        with open(depthfilename, 'wb') as f:
            np.save(f, depthimage)
        depthimage_visual = (depthimage / np.max(depthimage) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(outdir_colmap_depths, f'{basename}.png'), depthimage_visual)

def main():
    # Create the parser
    parser = get_args_parser()
    # Parse the arguments
    args = parser.parse_args()

    # dataset dir
    rootdir = args.root
    datadir = os.path.join(rootdir, f'{args.dataname}')
    print(f'\nData dir is {rootdir}/{datadir}\n')
    outdir = os.path.join(rootdir, f'{args.dataname}_recon')
    # dust3r dir name
    out_dust3r = os.path.join(outdir, 'dust3r_result.npy')
    # pano dir name
    outdir_colmap = os.path.join(outdir, 'colmap')
    os.makedirs(outdir_colmap, exist_ok=True)

    # get image names
    all_filenames, base_filenames = list_all_files(datadir)
    all_filenames.sort()

    # apply dust3r
    folder_or_list_mask = f'{datadir}_mask'

    if not os.path.isdir(folder_or_list_mask):
        folder_or_list_mask = None
    images, masks = load_images(all_filenames, size=512, square_ok=True, folder_or_list_mask=folder_or_list_mask)
    model_name = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"

    # dust3r inference
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    if len(masks) > 0:
        mask_pairs = make_pairs(masks, scene_graph='complete', prefilter=None, symmetrize=True)
    
    if len(masks) == 0:
        output = inference(pairs, model, device, batch_size=batch_size)
    else:
        output = inference(pairs, model, device, batch_size=batch_size, mask_pairs=mask_pairs)

    # align dust3r point clouds
    scene = global_aligner(output, device=device, min_conf_thr=2.5, mode=GlobalAlignerMode.PointCloudOptimizer)
    # scene.preset_focal([246.8 / 400.0 * 512.0]*len(images))
    scene.preset_focal([377.8] * len(images))
    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
    focals = scene.get_focals()
    avg_focal = sum(focals)/len(focals)
    for fidx, focal in enumerate(focals):
        print(f'Focal length of image {fidx} = {focal}')
    print(f'Average focal length = {avg_focal}')

    # save results
    viz_3d.save_dust3r_poses_and_depth(scene, out_dust3r)

    # save to colmap results
    

    transform4x4 = [[-0.149,	0.064,	0.0235,	1.786],
                    [-0.001,	-0.059,	0.153,	0.262],
                    [0.068,	0.139,	0.054,	1.627],
                    [0.000,	0.000,	0.000,	1.000]]

    # transform4x4 = [[0.262,	-0.182,	0.012,	0.724],
    #                 [0.094,	0.153,	0.264,	-0.430],
    #                 [-0.156,	-0.213,	0.180,	6.008],
    #                 [0.000,	0.000,	0.000,	1.000]]

    

    transform4x4 = np.array(transform4x4)
    # transform4x4 = np.eye(4)

    transformscale = np.linalg.norm(transform4x4[:3, :3], axis=0)[0]
    transform4x4[:3, :3] = transform4x4[:3, :3] / transformscale

    if len(masks) > 0:
        # convert_dust3r_cameras_to_colmap_cameras(scene, all_filenames, outdir, scale_low_conf_= 1.0)
        convert_dust3r_cameras_to_colmap_cameras(scene, all_filenames, outdir, scale_low_conf_= 1.0, mask_dir = folder_or_list_mask, transform4x4=transform4x4, transformscale=transformscale)
    else:
        convert_dust3r_cameras_to_colmap_cameras(scene, all_filenames, outdir, scale_low_conf_= 1.0, transform4x4=transform4x4, transformscale=transformscale)

    # import matplotlib.pyplot as plt 
    # plt.imshow(np.moveaxis(np.squeeze(images[0]['img'].numpy()), [0], [2]))
    # plt.imshow(np.squeeze(masks[0]['img'].numpy()))

if __name__ == '__main__':
    main()