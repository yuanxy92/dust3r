import open3d as o3d
import numpy as np

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

# load data
with open('./data/9001gate.npy', 'rb') as f:
    imgs = np.load(f)
    focals = np.load(f)
    poses = np.load(f)
    pts3d = np.load(f)
    confidence_masks = np.load(f)

