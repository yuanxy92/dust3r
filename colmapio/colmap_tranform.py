import numpy as np

def apply_4x4_transform(points, transform_matrix):
    """
    Apply a 4x4 transformation matrix to a point cloud.
    
    Parameters:
    - points (np.ndarray): Point cloud, shape (N, 3)
    - transform_matrix (np.ndarray): 4x4 transformation matrix
    
    Returns:
    - transformed_points (np.ndarray): Transformed point cloud, shape (N, 3)
    """
    # Convert points to homogeneous coordinates by adding a 1 in the fourth column
    num_points = points.shape[0]
    homogeneous_points = np.hstack((points, np.ones((num_points, 1))))
    
    # Apply the 4x4 transformation matrix
    transformed_points_homogeneous = np.dot(homogeneous_points, transform_matrix.T)
    
    # Convert back to 3D coordinates by dividing by the homogeneous coordinate
    transformed_points = transformed_points_homogeneous[:, :3] / transformed_points_homogeneous[:, 3].reshape(-1, 1)
    
    return transformed_points

def apply_camera_transform_4x4(extrinsic_matrix, transform_matrix):
    """
    Apply a 4x4 transformation matrix to a camera's extrinsic matrix.
    
    Parameters:
    - extrinsic_matrix (np.ndarray): Camera extrinsic matrix (4, 4)
    - transform_matrix (np.ndarray): 4x4 transformation matrix
    
    Returns:
    - transformed_extrinsic (np.ndarray): Transformed camera extrinsic matrix (4, 4)
    """
    # Apply the transformation matrix to the camera's extrinsic matrix
    transformed_extrinsic = np.dot(transform_matrix, extrinsic_matrix)
    
    return transformed_extrinsic

# Example usage
if __name__ == "__main__":
    # Example point cloud (N x 3)
    point_cloud = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])
    
    # Example camera extrinsic matrix (4x4)
    camera_extrinsic = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    # Example transformation matrix (4x4)
    transform_matrix = np.array([
        [1.0, 0.0, 0.0, 1.0],  # Translation along x-axis
        [0.0, 1.0, 0.0, 2.0],  # Translation along y-axis
        [0.0, 0.0, 1.0, 3.0],  # Translation along z-axis
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    # Apply the 4x4 transformation to the point cloud
    transformed_points = apply_4x4_transform(point_cloud, transform_matrix)
    print("Transformed Point Cloud:\n", transformed_points)
    
    # Apply the 4x4 transformation to the camera extrinsic matrix
    transformed_camera = apply_camera_transform_4x4(camera_extrinsic, transform_matrix)
    print("Transformed Camera Extrinsic Matrix:\n", transformed_camera)
