import argparse
import logging
import os
import pickle
import time
from dataclasses import dataclass

import open3d as o3d
import numpy as np
import cv2

from pano.features import matching
from pano.bundle_adj import _hom_to_from, traverse, Image, rotation_to_mat, intrinsics
from pano.stitcher import no_blend, _proj_img_range_border, _add_weights, estimate_resolution
from pano.stitcher import SphProj, no_blend, linear_blend, multiband_blend, _hat

@dataclass
class PanoImage:
    """Patch with all the informations for stitching."""
    img: np.ndarray
    # extrinsic matrix [R|t]
    rot: np.ndarray # R^-1 of the extrinsic matrix
    trans: np.ndarray # t of the extrinsic matrix
    intr: np.ndarray
    pts3d: np.ndarray
    confidence_mask: np.ndarray
    distimg: np.ndarray
    panocenter: np.ndarray = None
    range: tuple = (np.zeros(2), np.zeros(2))

    def hom(self):
        """Homography from pixel to normalized coordinates."""
        return self.rot.T.dot(np.linalg.inv(self.intr))
    
    def dist(self):
        if self.distimg is None:
            dist_mat = self.pts3d - self.panocenter
            self.distimg = np.sqrt(np.sum(np.square(dist_mat), axis=2))
        return self.distimg

    def proj(self):
        """Return camera projection transform."""
        return self.intr.dot(self.rot)
    
    def cam_center(self):
        camera_center = - self.rot @ self.trans
        return camera_center
    
def pano_segment_sky(image):
    import cv2
    from scipy import ndimage

    # Convert to HSV
    if np.issubdtype(image.dtype, np.floating):
        image = np.uint8(255*image.clip(min=0, max=1))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for blue color and create mask
    lower_blue = np.array([0, 0, 100])
    upper_blue = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue).view(bool)

    # add luminous gray
    mask |= (hsv[:, :, 1] < 10) & (hsv[:, :, 2] > 150)
    mask |= (hsv[:, :, 1] < 30) & (hsv[:, :, 2] > 180)
    mask |= (hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 220)

    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask2 = ndimage.binary_opening(mask, structure=kernel)

    # keep only largest CC
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask2.view(np.uint8), connectivity=8)
    cc_sizes = stats[1:, cv2.CC_STAT_AREA]
    order = cc_sizes.argsort()[::-1]  # bigger first
    i = 0
    selection = []
    while i < len(order) and cc_sizes[order[i]] > cc_sizes[order[0]] / 2:
        selection.append(1 + order[i])
        i += 1
    mask3 = np.in1d(labels, selection).reshape(labels.shape)

    # mask
    mask2 = mask2.astype(np.uint8) * 255
    mask3 = mask3.astype(np.uint8) * 255
    return mask2, mask3

def compute_center_of_camera_arrays(pano_images):
    camera_center = 0
    for pano_image in pano_images:
        camera_center = camera_center + pano_image.cam_center()
    camera_center = camera_center / len(pano_images)
    return camera_center

def convert_dust3r_to_pano(imgs, focals, poses, pts3d, confidence_masks):
    # init images
    pano_images = []
    for idx in range(len(imgs)):
        rotation_mat = poses[idx][:3, :3]
        trans_mat = poses[idx][:3, 3]
        pano_image = PanoImage(img=imgs[idx], rot=rotation_mat.transpose(), trans=trans_mat,
            intr=intrinsics(focals[idx], (0, 0)), pts3d=pts3d[idx], confidence_mask=confidence_masks[idx], distimg=None)
        pano_images.append(pano_image)
    # compute center of all the cameras
    camera_array_center = compute_center_of_camera_arrays(pano_images)
    # compute distance to the center of the camera arrays
    for idx in range(len(pano_images)):
        pano_images[idx].panocenter = camera_array_center
        pano_images[idx].dist()
    return pano_images

def _add_weights_single_channel(img):
    """Add weights scaled as (x-0.5)*(y-0.5) in normalized coordinates."""
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    height, width = img.shape[:2]
    img[..., 3] = _hat(height)[:, None] * _hat(width)[None, :]
    return img

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
    mosaic_sky, mosaic_sky_largest_CC = pano_segment_sky(mosaic_rgb)
    # convert RGB-Depth image to point clouds
    ptcloud = rgbdpano_to_point_cloud(mosaic_rgb, mosaic_dist[:, :, 0], mosaic_conf[:, :, 0], im_range, resolution)

    # save npy
    with open('./data/pano_9001gate.npy', 'wb') as f:
        np.save(f, mosaic_rgb)
        np.save(f, mosaic_dist)
        np.save(f, mosaic_conf)
        np.save(f, mosaic_sky)
    # save image
    cv2.imwrite('./data/pano_9001gate_sky.png', mosaic_sky)
    cv2.imwrite('./data/pano_9001gate.png', mosaic_rgb)
    mosaic_dist_norm = mosaic_dist / np.max(mosaic_dist) * 255
    cv2.imwrite('./data/pano_9001gate_dist.png', mosaic_dist_norm.astype(np.uint8))
    mosaic_conf_norm = mosaic_conf / np.max(mosaic_conf) * 255
    cv2.imwrite('./data/pano_9001gate_conf.png', mosaic_conf_norm.astype(np.uint8))
    # save point cloud
    o3d.io.write_point_cloud("./data/pano_9001gate.ply", ptcloud)

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

def project_point_cloud_to_pano(point_cloud, pano_image, intrinsic_matrix):
    """
    Projects a point cloud onto a panoramic image.

    Args:
        point_cloud (o3d.geometry.PointCloud): The point cloud with RGB data.
        pano_image (np.ndarray): The panoramic image to map the point cloud onto.
        intrinsic_matrix (np.ndarray): The intrinsic matrix for camera projection.

    Returns:
        np.ndarray: The panoramic image with mapped point cloud colors.
    """
    # Extract points and colors from the point cloud
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors) * 255  # Scale RGB values to 0-255

    # Prepare the output image
    h, w, _ = pano_image.shape
    output_image = pano_image.copy()

    # Project points onto the image
    for i in range(points.shape[0]):
        point = np.append(points[i], 1)  # Homogeneous coordinates
        color = colors[i].astype(int)

        # Project 3D point to 2D using intrinsic matrix
        projected_point = intrinsic_matrix @ point
        u = int(projected_point[0] / projected_point[2])
        v = int(projected_point[1] / projected_point[2])

        # Check if the projected point is within the image boundaries
        if 0 <= u < w and 0 <= v < h:
            output_image[v, u] = color

    return output_image
