import argparse
import logging
import os
import pickle
import time
import math
from dataclasses import dataclass
import open3d as o3d
import cv2
import numpy as np

mesh = o3d.io.read_triangle_mesh("/Users/yuanxy/Downloads/LocalSend/data_0829_6_recon/pano/frame_237_mesh.ply")
if not mesh.is_empty():
    print("Successfully loaded mesh!")
else:
    print("Failed to load mesh.")

# mesh.compute_vertex_normals()

# show mesh
vis = o3d.visualization.Visualizer()
vis.create_window(width=1280, height=1280, visible=False)
vis.add_geometry(mesh)

# Set the camera parameters (optional)
ctr = vis.get_view_control()
# ctr.set_front([0, 0, -1])
ctr.set_lookat([0, 0, 0])
ctr.set_up([0, -1, 0])
ctr.set_zoom(0.5)

# vis.run()

# Capture and save the screenshot
vis.poll_events()
vis.update_renderer()

image = vis.capture_screen_float_buffer(False)
image=np.asarray(image)*255
image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
print(image.shape) 
cv2.imwrite("./data/test.png",image)

# Destroy the visualizer window
vis.destroy_window()

# o3d.visualization.draw_geometries([mesh], 
#     window_name="Mesh Viewer", 
#     width=800, 
#     height=600, 
#     left=50, 
#     top=50,
#     mesh_show_back_face=True)

