import argparse
import logging
import os
import pickle
import time
from dataclasses import dataclass

import open3d as o3d
import numpy as np
import cv2

from pano.stitcher import no_blend, _proj_img_range_border, _add_weights, estimate_resolution
from pano.stitcher import SphProj, no_blend, linear_blend, multiband_blend, no_blend
from pano_tools import PanoImage, _add_weights_single_channel, convert_dust3r_to_pano

def rgbdpano_to_point_cloud(rgb_image, depth_image, conf_image, im_range, resolution):
    """
    Convert RGB and Depth images into a point cloud
    :param rgb_image: HxWx3 numpy array representing the RGB image
    :param depth_image: HxW numpy array representing the depth image
    :param fov_h: Horizontal field of view in radians
    :param fov_v: Vertical field of view in radians
    :return: Open3D point cloud object
    """
    # Get the image dimensions
    height, width = depth_image.shape
    # Calculate the angles for each pixel
    x_indices, y_indices = np.meshgrid(np.arange(width), np.arange(height))
    phi = im_range[0][1] + resolution[1] * y_indices + np.pi / 2
    theta = im_range[0][0] + resolution[0] * x_indices
    # Calculate the 3D coordinates
    x = depth_image * np.sin(phi) * np.cos(theta)
    y = depth_image * np.sin(phi) * np.sin(theta)
    z = depth_image * np.cos(phi)
    # Generate mask using threshold
    th = 0.5
    mask = np.ones((conf_image.shape[0], conf_image.shape[1]))
    mask[conf_image < th] = 0
    # Flatten the coordinates and corresponding color values
    points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=1)
    colors = rgb_image.reshape(-1, 3) / 255.0  # normalize RGB values
    colors = np.flip(colors, axis=1)
    mask_binary = mask.reshape(-1,).astype(bool)
    points_selected = points[mask_binary]
    colors_selected = colors[mask_binary]
    # Create an Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_selected)
    point_cloud.colors = o3d.utility.Vector3dVector(colors_selected)
    return point_cloud

def rgbdpano_to_ptmesh(rgb_image, depth_image, conf_image, im_range, resolution):
    """
    Convert RGB and Depth images into a point cloud
    :param rgb_image: HxWx3 numpy array representing the RGB image
    :param depth_image: HxW numpy array representing the depth image
    :param fov_h: Horizontal field of view in radians
    :param fov_v: Vertical field of view in radians
    :return: Open3D point cloud object
    """
    # Get the image dimensions
    height, width = depth_image.shape
    # Calculate the angles for each pixel
    x_indices, y_indices = np.meshgrid(np.arange(width), np.arange(height))
    phi = im_range[0][1] + resolution[1] * y_indices + np.pi / 2
    theta = im_range[0][0] + resolution[0] * x_indices
    # Calculate the 3D coordinates
    x = depth_image * np.sin(phi) * np.cos(theta)
    y = depth_image * np.sin(phi) * np.sin(theta)
    z = depth_image * np.cos(phi)
    # Generate mask using threshold
    th = 0.5
    mask = np.ones((conf_image.shape[0], conf_image.shape[1]))
    mask[conf_image < th] = 0
    triangles = []
    # Generate vertices
    vertices = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=1)
    colors = rgb_image.reshape(-1, 3) / 255.0  # normalize RGB values
    colors = np.flip(colors, axis=1)
    mask_binary = mask.reshape(-1,).astype(bool)
    # Iterate over all the pixels
    for v in range(height - 1):
        for u in range(width - 1):
            idx = v * width + u
            idx_right = idx + 1
            idx_down = idx + width
            idx_down_right = idx + width + 1
            if mask[v, u] > 0 and mask[v + 1, u] > 0 and mask[v, u + 1] > 0:
                triangles.append([idx, idx_down, idx_right])
            if mask[v, u + 1] > 0 and mask[v + 1, u] > 0 and mask[v + 1, u + 1] > 0:
                triangles.append([idx_right, idx_down, idx_down_right])
    triangles = np.array(triangles)
    # Create the mesh object
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    # Optionally, compute vertex normals
    mesh.compute_vertex_normals()
    return mesh

def pano_stitch(ba_images, blender=no_blend, equalize=False, crop=False):
    # compute range
    for ba_image in ba_images:
        ba_image.range = _proj_img_range_border(ba_image.img.shape[:2], ba_image.hom())
        ba_image.img = _add_weights(ba_image.img)
        ba_image.distimg = _add_weights_single_channel(ba_image.dist())
        ba_image.confidence_mask = _add_weights_single_channel(ba_image.confidence_mask)
    # estimate resolution
    resolution, im_range = estimate_resolution(ba_images)
    target = (im_range[1] - im_range[0]) / resolution

    shape = tuple(int(t) for t in np.round(target))[::-1]  # y,x order
    patches = []
    patches_dist = []
    patches_conf = []

    for reg in ba_images:
        bottom = np.round((reg.range[0] - im_range[0])/resolution)
        top = np.round((reg.range[1] - im_range[0])/resolution)
        bottom, top = bottom.astype(np.int32), top.astype(np.int32)
        hh_, ww_ = reg.img.shape[:2]  # original image shape

        # pad image if multi-band to avoid sharp edges where the image ends
        if blender == multiband_blend:
            bottom = np.maximum(bottom - 10, np.int32([0, 0]))
            top = np.minimum(top + 10, target.astype(np.int32))

        # find pixel coordinates
        y_i, x_i = np.indices((top[1]-bottom[1], top[0]-bottom[0]))
        x_i = (x_i + bottom[0]) * resolution[0] + im_range[0][0]
        y_i = (y_i + bottom[1]) * resolution[1] + im_range[0][1]
        xx_ = SphProj.proj2hom(np.stack([x_i, y_i], axis=-1).reshape(-1, 2))

        # transform to the original image coordinates
        xx_ = reg.proj().dot(xx_.T).T.astype(np.float32)
        xx_ = xx_.reshape(top[1]-bottom[1], top[0]-bottom[0], -1)
        mask = xx_[..., -1] < 0  # behind the screen

        x_pr = xx_[..., :-1] / xx_[..., [-1]] + np.float32([ww_/2, hh_/2])
        mask |= (x_pr[..., 0] < 0) | (x_pr[..., 0] > ww_-1) | \
                (x_pr[..., 1] < 0) | (x_pr[..., 1] > hh_-1)

        # paste only valid pixels
        warped = cv2.remap(reg.img, x_pr[:, :, 0], x_pr[:, :, 1],
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        warped[..., 3] = warped[..., 3] * (~mask)
        irange = np.s_[bottom[1]:top[1], bottom[0]:top[0]]

        # warped dist image
        warped_dist = cv2.remap(reg.distimg, x_pr[:, :, 0], x_pr[:, :, 1],
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        warped_dist[..., 3] = warped_dist[..., 3] * (~mask)

        # warped dist image
        warped_conf = cv2.remap(reg.confidence_mask, x_pr[:, :, 0], x_pr[:, :, 1],
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        warped_conf[..., 3] = warped_conf[..., 3] * (~mask)

        patches.append((warped, mask, irange))
        patches_dist.append((warped_dist, mask, irange))
        patches_conf.append((warped_conf, mask, irange))

    # generate sky
    mosaic_rgb = blender(patches, shape)
    mosaic_rgb = (np.clip(mosaic_rgb, 0.0, 1.0) * 255).astype(np.uint8)
    mosaic_dist = blender(patches_dist, shape)
    mosaic_conf = blender(patches_conf, shape)
    # segment sky
    # convert RGB-Depth image to point clouds
    ptcloud = rgbdpano_to_point_cloud(mosaic_rgb, mosaic_dist[:, :, 0], mosaic_conf[:, :, 0], im_range, resolution)
    mesh = rgbdpano_to_ptmesh(mosaic_rgb, mosaic_dist[:, :, 0], mosaic_conf[:, :, 0], im_range, resolution)

    # save npy
    with open('./data/pano_9001gate.npy', 'wb') as f:
        np.save(f, mosaic_rgb)
        np.save(f, mosaic_dist)
        np.save(f, mosaic_conf)
    # save image
    cv2.imwrite('./data/pano_9001gate.png', mosaic_rgb)
    mosaic_dist_norm = mosaic_dist / np.max(mosaic_dist) * 255
    cv2.imwrite('./data/pano_9001gate_dist.png', mosaic_dist_norm.astype(np.uint8))
    mosaic_conf_norm = mosaic_conf / np.max(mosaic_conf) * 255
    cv2.imwrite('./data/pano_9001gate_conf.png', mosaic_conf_norm.astype(np.uint8))
    # save point cloud
    o3d.io.write_point_cloud("./data/pano_9001gate.ply", ptcloud)
    # save mesh
    o3d.io.write_triangle_mesh("./data/pano_9001gate_mesh.ply", mesh)

    return mosaic_rgb, mosaic_dist, mosaic_conf, resolution, im_range

def main():
    # load data
    with open('./data/9001gate.npy', 'rb') as f:
        imgs = np.load(f)
        focals = np.load(f)
        poses = np.load(f)
        pts3d = np.load(f)
        confidence_masks = np.load(f)
    # generate bundle adjust image
    ba_imgs = convert_dust3r_to_pano(imgs, focals, poses, pts3d, confidence_masks)
    result = pano_stitch(ba_imgs, blender=multiband_blend, equalize=False, crop=False)

    return result

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('numba').setLevel(logging.WARNING)  # silence Numba
    main()

