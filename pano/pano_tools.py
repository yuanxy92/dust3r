import argparse
import logging
import os
import pickle
import time
from dataclasses import dataclass

import open3d as o3d
import numpy as np
import cv2

from .pano.features import matching
from .pano.bundle_adj import _hom_to_from, traverse, Image, rotation_to_mat, intrinsics
from .pano.stitcher import no_blend, _proj_img_range_border, _add_weights, estimate_resolution
from .pano.stitcher import SphProj, no_blend, linear_blend, multiband_blend, _hat

@dataclass
class PanoImage:
    """Patch with all the informations for stitching."""
    img: np.ndarray
    # extrinsic matrix [R|t]
    rot: np.ndarray # R^-1 or R^T of the extrinsic matrix
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
    
def _add_weights_single_channel(img):
    """Add weights scaled as (x-0.5)*(y-0.5) in normalized coordinates."""
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    height, width = img.shape[:2]
    img[..., 3] = _hat(height)[:, None] * _hat(width)[None, :]
    return img
    
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

    # rotate and translate all the cameras
    identity_idx = 0
    rot_to_I = pano_images[identity_idx].rot
    trans_to_I = -pano_images[identity_idx].rot @ pano_images[identity_idx].trans
    # extra rotation
    theta_ = np.radians(-75)  # Convert angle to radians
    c, s = np.cos(theta_), np.sin(theta_)
    R_extra = np.array([[c, 0, s],
                [0, 1, 0],
                [-s, 0, c]])
    transform_mat = np.identity(4, dtype=np.float32)
    transform_mat[:3, :3] = R_extra @ rot_to_I
    transform_mat[:3, 3] = trans_to_I
    # apply to all the cameras
    for idx in range(len(pano_images)):
        extr_mat = np.identity(4, dtype=np.float32)
        extr_mat[:3, :3] = pano_images[idx].rot.transpose()
        extr_mat[:3, 3] = pano_images[idx].trans
        extr_mat_new = transform_mat @ extr_mat
        pano_images[idx].rot = extr_mat_new[:3, :3].transpose()
        pano_images[idx].trans = extr_mat_new[:3, 3]

    return pano_images
    
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
