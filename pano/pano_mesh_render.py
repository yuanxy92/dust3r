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

data_dir = '/Users/yuanxy/Downloads/LocalSend/data_0829_4_recon/pano'
out_dir = '/Users/yuanxy/Downloads/LocalSend/data_0829_4_recon/render/view2'
os.makedirs(out_dir, exist_ok=True)

# front_vec = [-0.038167360946389013, 0.3536034281680413, 0.93461642835240011]
# lookat_vec = [0.057227766485070261, -0.017260430642816252, 0.044081580365883249]
# up_vec = [0.97120364520375169, 0.2332445779368067, -0.048584425550030307]
# zoom_vec = 0.35

# data_0829_4
# front_vec = [ -0.093190958594899398, 0.27184671452844944, 0.9578177326799977 ]
# lookat_vec = [ 0.034970993064742266, 0.070012213894139044, 0.00026119878888348954 ]
# up_vec = [ 0.099209835067291946, -0.95468106816558629, 0.28060909948206575 ]
# zoom_vec = 0.56000000000000005

# front_vec = [ 0.79728443635055979, -0.46115194958867312, 0.38945655334547413 ]
# lookat_vec = [ 0.032927162700970833, 0.098465252919569646, 0.045522668243262933 ]
# up_vec = [ -0.23280186309401779, 0.36036744916918134, 0.90329319388515916 ]
# zoom_vec = 0.6

front_vec = [ 0.28061865046596263, 0.75001568776394234, 0.59894043202863068 ]
lookat_vec = [ 0.032927162700970833, 0.098465252919569646, 0.045522668243262933 ]
up_vec = [ -0.24770074103791054, -0.54628656865138259, 0.8001345685572947 ]
zoom_vec = 0.48


# show mesh
vis = o3d.visualization.Visualizer()
vis.create_window(width=1280, height=1280, visible=True)

for mesh_idx in range(717, 741):
    # load mesh from files
    mesh = o3d.io.read_triangle_mesh(os.path.join(data_dir, f'frame_{mesh_idx}_mesh.ply'))
    if mesh.is_empty():
        continue
    vis.add_geometry(mesh)

    # Set the camera parameters (optional)
    ctr = vis.get_view_control()
    ctr.set_front(front_vec)
    ctr.set_lookat(lookat_vec)
    ctr.set_up(up_vec)
    ctr.set_zoom(zoom_vec)

    # vis.run()

    # Capture and save the screenshot
    vis.poll_events()
    vis.update_renderer()

    # get images from mesh
    image = vis.capture_screen_float_buffer(False)
    image = np.asarray(image) * 255
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(out_dir, f'frame_{mesh_idx}_mesh.png'), image)
    print(f'Finish rendering of frame {mesh_idx} ......')

    # remove gerometry
    vis.remove_geometry(mesh)


# Destroy the visualizer window
vis.destroy_window()


# data_0829_8
# front_vec = [-0.26974646633220895, 0.87459394850815608, 0.40289237909670034]
# lookat_vec = [0.036025796832193625, -0.0076210508984496427, -0.0099135128359533024]
# up_vec = [0.060384630636169868, -0.4022147054091364, 0.91355187435392793]
# zoom_vec = 0.4

# front_vec = [-0.74507661036931139, -0.04827515150966482, 0.66522955017594965]
# lookat_vec = [-0.0022678011691651513, -0.047621474606879891, 0.0056977634580943175]
# up_vec = [0.63159583698853872, 0.26947032576783736, 0.72696110090521537]
# zoom_vec = 0.4

# front_vec = [-0.038167360946389013, 0.3536034281680413, 0.93461642835240011]
# lookat_vec = [0.057227766485070261, -0.017260430642816252, 0.044081580365883249]
# up_vec = [0.97120364520375169, 0.2332445779368067, -0.048584425550030307]
# zoom_vec = 0.35

# data_0829_4
