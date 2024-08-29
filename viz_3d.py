"""
3D visualization based on plotly.
Works for a small number of points and cameras, might be slow otherwise.

1) Initialize a figure with `init_figure`
2) Add 3D points, camera frustums, or both as a pycolmap.Reconstruction

Written by Paul-Edouard Sarlin and Philipp Lindenberger.
"""

from typing import Optional

import numpy as np
import plotly.graph_objects as go
import pycolmap
from matplotlib import pyplot as pl
import torch

def to_homogeneous(points):
    pad = np.ones((points.shape[:-1] + (1,)), dtype=points.dtype)
    return np.concatenate([points, pad], axis=-1)


def init_figure(height: int = 800) -> go.Figure:
    """Initialize a 3D figure."""
    fig = go.Figure()
    axes = dict(
        visible=False,
        showbackground=False,
        showgrid=False,
        showline=False,
        showticklabels=True,
        autorange=True,
    )
    fig.update_layout(
        template="plotly_dark",
        height=height,
        scene_camera=dict(
            eye=dict(x=0.0, y=-0.1, z=-2),
            up=dict(x=0, y=-1.0, z=0),
            projection=dict(type="orthographic"),
        ),
        scene=dict(
            xaxis=axes,
            yaxis=axes,
            zaxis=axes,
            aspectmode="data",
            dragmode="orbit",
        ),
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        legend=dict(orientation="h", yanchor="top", y=0.99, xanchor="left", x=0.1),
    )
    return fig


def plot_points(
    fig: go.Figure,
    pts: np.ndarray,
    color: str = "rgba(255, 0, 0, 1)",
    ps: int = 2,
    colorscale: Optional[str] = None,
    name: Optional[str] = None,
):
    """Plot a set of 3D points."""
    x, y, z = pts.T
    tr = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        name=name,
        legendgroup=name,
        marker=dict(size=ps, color=color, line_width=0.0, colorscale=colorscale),
    )
    fig.add_trace(tr)


def plot_camera(
    fig: go.Figure,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    color: str = "rgb(0, 0, 255)",
    name: Optional[str] = None,
    legendgroup: Optional[str] = None,
    fill: bool = False,
    size: float = 1.0,
    text: Optional[str] = None,
):
    """Plot a camera frustum from pose and intrinsic matrix."""
    W, H = K[0, 2] * 2, K[1, 2] * 2
    corners = np.array([[0, 0], [W, 0], [W, H], [0, H], [0, 0]])
    if size is not None:
        image_extent = max(size * W / 1024.0, size * H / 1024.0)
        world_extent = max(W, H) / (K[0, 0] + K[1, 1]) / 0.5
        scale = 0.5 * image_extent / world_extent
    else:
        scale = 1.0
    corners = to_homogeneous(corners) @ np.linalg.inv(K).T
    corners = (corners / 2 * scale) @ R.T + t
    legendgroup = legendgroup if legendgroup is not None else name

    x, y, z = np.concatenate(([t], corners)).T
    i = [0, 0, 0, 0]
    j = [1, 2, 3, 4]
    k = [2, 3, 4, 1]

    if fill:
        pyramid = go.Mesh3d(
            x=x,
            y=y,
            z=z,
            color=color,
            i=i,
            j=j,
            k=k,
            legendgroup=legendgroup,
            name=name,
            showlegend=False,
            hovertemplate=text.replace("\n", "<br>"),
        )
        fig.add_trace(pyramid)

    triangles = np.vstack((i, j, k)).T
    vertices = np.concatenate(([t], corners))
    tri_points = np.array([vertices[i] for i in triangles.reshape(-1)])
    x, y, z = tri_points.T

    pyramid = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        legendgroup=legendgroup,
        name=name,
        line=dict(color=color, width=1),
        showlegend=False,
        hovertemplate=text.replace("\n", "<br>"),
    )
    fig.add_trace(pyramid)


def plot_camera_colmap(
    fig: go.Figure,
    image: pycolmap.Image,
    camera: pycolmap.Camera,
    name: Optional[str] = None,
    **kwargs
):
    """Plot a camera frustum from PyCOLMAP objects"""
    world_t_camera = image.cam_from_world.inverse()
    plot_camera(
        fig,
        world_t_camera.rotation.matrix(),
        world_t_camera.translation,
        camera.calibration_matrix(),
        name=name or str(image.image_id),
        text=str(image),
        **kwargs
    )


def plot_cameras(fig: go.Figure, reconstruction: pycolmap.Reconstruction, **kwargs):
    """Plot a camera as a cone with camera frustum."""
    for image_id, image in reconstruction.images.items():
        plot_camera_colmap(
            fig, image, reconstruction.cameras[image.camera_id], **kwargs
        )


def plot_reconstruction(
    fig: go.Figure,
    rec: pycolmap.Reconstruction,
    max_reproj_error: float = 6.0,
    color: str = "rgb(0, 0, 255)",
    name: Optional[str] = None,
    min_track_length: int = 2,
    points: bool = True,
    cameras: bool = True,
    points_rgb: bool = True,
    cs: float = 1.0,
):
    # Filter outliers
    bbs = rec.compute_bounding_box(0.001, 0.999)
    # Filter points, use original reproj error here
    p3Ds = [
        p3D
        for _, p3D in rec.points3D.items()
        if (
            (p3D.xyz >= bbs[0]).all()
            and (p3D.xyz <= bbs[1]).all()
            and p3D.error <= max_reproj_error
            and p3D.track.length() >= min_track_length
        )
    ]
    xyzs = [p3D.xyz for p3D in p3Ds]
    if points_rgb:
        pcolor = [p3D.color for p3D in p3Ds]
    else:
        pcolor = color
    if points:
        plot_points(fig, np.array(xyzs), color=pcolor, ps=1, name=name)
    if cameras:
        plot_cameras(fig, rec, color=color, legendgroup=name, size=cs)


def xy_grid(W, H, device=None, origin=(0, 0), unsqueeze=None, cat_dim=-1, homogeneous=False, **arange_kw):
    """ Output a (H,W,2) array of int32 
        with output[j,i,0] = i + origin[0]
             output[j,i,1] = j + origin[1]
    """
    if device is None:
        # numpy
        arange, meshgrid, stack, ones = np.arange, np.meshgrid, np.stack, np.ones
    else:
        # torch
        arange = lambda *a, **kw: torch.arange(*a, device=device, **kw)
        meshgrid, stack = torch.meshgrid, torch.stack
        ones = lambda *a: torch.ones(*a, device=device)

    tw, th = [arange(o, o + s, **arange_kw) for s, o in zip((W, H), origin)]
    grid = meshgrid(tw, th, indexing='xy')
    if homogeneous:
        grid = grid + (ones((H, W)),)
    if unsqueeze is not None:
        grid = (grid[0].unsqueeze(unsqueeze), grid[1].unsqueeze(unsqueeze))
    if cat_dim is not None:
        grid = stack(grid, cat_dim)
    return grid

def draw_scene_ez(scene,pose,pts3d,color="rgba(0,255,0,0.5)"):
    imgs = scene.imgs
    focals = scene.get_focals()
    focals = focals.squeeze().detach().cpu().numpy()
    Rs,ts,Ks = [],[],[]
    for i in range(len(imgs)):
        pose = poses[i]
        Rs.append(pose[:3,:3])
        ts.append(pose[:3,3])
        f = focals[i]
        
        img = imgs[i]
        H,W = img.shape[:2]
        Ks.append(np.array([[f,0,W/2],[0,f,H/2],[0,0,1]]))

    fig = init_figure()
    n_viz = min(40000,pts3d.shape[0])
    idx_to_viz = list(np.round(np.linspace(0, pts3d.shape[0]-1, n_viz)).astype(int))
    plot_points(fig,pts3d[idx_to_viz],color=color,ps=2,name='3dpts')
    for ind,(R,t,K) in enumerate(zip(Rs,ts,Ks)):
        plot_camera(fig,R,t,K,color="rgba(255,0,0,0.5)",text=f'cam_{ind}')
    fig.show()

def save_dust3r_poses_and_depth(scene, filename, pose_refine = None,pts3d = None):
    # get images, focal length
    imgs = scene.imgs
    focals = scene.get_focals()
    focals = focals.squeeze().detach().cpu().numpy()
    if pose_refine is None:
        poses = scene.get_im_poses()
        poses = poses.detach().cpu().numpy()
    else:
        poses = np.asarray(pose_refine)
    if pts3d is None:
        pts3d = scene.get_pts3d()
        pts3d = [pts.detach().cpu().numpy() for pts in pts3d]
    confidence_masks = scene.get_masks()

    with open(filename, 'wb') as f:
        np.save(f, imgs)
        np.save(f, focals)
        np.save(f, poses)
        np.save(f, pts3d)
        np.save(f, confidence_masks)

def draw_dust3r_scene(scene,pose_refine = None,pts3d = None):
    imgs = scene.imgs
    focals = scene.get_focals()
    focals = focals.squeeze().detach().cpu().numpy()

    if pose_refine is None:
        poses = scene.get_im_poses()
        poses = poses.detach().cpu().numpy()
    else:
        poses = np.asarray(pose_refine)
    if pts3d is None:
        pts3d = scene.get_pts3d()
        pts3d = [pts.detach().cpu().numpy() for pts in pts3d]
    confidence_masks = scene.get_masks()
    pts2d_all_list, pts3d_all_list, all_rgbs = [], [], []
    Rs,ts,Ks = [],[],[]
    for i in range(len(imgs)):
        conf_i = confidence_masks[i].cpu().numpy()
        pts2d_all_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])  # imgs[i].shape[:2] = (H, W)
        pts3d_all_list.append(pts3d[i][conf_i])
        
        #pose = np.linalg.inv(poses[i])
        pose = poses[i]
        Rs.append(pose[:3,:3])
        ts.append(pose[:3,3])
        f = focals[i]
        
        img = imgs[i]
        H,W = img.shape[:2]
        Ks.append(np.array([[f,0,W/2],[0,f,H/2],[0,0,1]]))

        for pos in pts2d_all_list[i]:
            x,y = pos
            all_rgbs.append(img[y,x,:])
            

    all_pts3d = np.concatenate(pts3d_all_list,axis=0)

    #plot reconstruction
    fig = init_figure()
    n_viz = min(80000,len(all_rgbs))
    idx_to_viz = list(np.round(np.linspace(0, len(all_rgbs)-1, n_viz)).astype(int))
    vis_rgbs = [all_rgbs[idx] for idx in idx_to_viz]
    plot_points(fig,all_pts3d[idx_to_viz],vis_rgbs,ps=2,name='3dpts')
    for ind,(R,t,K) in enumerate(zip(Rs,ts,Ks)):
        plot_camera(fig,R,t,K,color="rgba(255,0,0,0.5)",text=f'cam_{ind}')
    fig.show()


def draw_dust3r_match(matches_im0,matches_im1,img0,img1,n_viz=20):
    img0 = np.asarray(img0)
    img1 = np.asarray(img1)
    num_matches = len(matches_im0)
    match_idx_to_viz = np.round(np.linspace(0, num_matches-1, n_viz)).astype(int)
    viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]
    H0, W0, H1, W1 = *img0.shape[:2], *img1.shape[:2]
    img0 = np.pad(img0, ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img1 = np.pad(img1, ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img = np.concatenate((img0, img1), axis=1)

    pl.figure()
    pl.imshow(img)
    cmap = pl.get_cmap('jet')
    for i in range(n_viz):
        (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
        pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
    pl.show(block=True)

# def draw_dust3r_match_ez(scene,im1=0,im2=1):
#     scene = scene.mask_sky()
#     imgs = scene.imgs

#     focals = scene.get_focals()
#     focals = focals.squeeze().detach().cpu().numpy()
#     poses = scene.get_im_poses()
#     poses = poses.detach().cpu().numpy()
#     pts3d = scene.get_pts3d()
#     confidence_masks = scene.get_masks()
#     pts2d_list, pts3d_list = [], []
#     for i in range(2):
#         conf_i = confidence_masks[i].cpu().numpy()
#         pts2d_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])  # imgs[i].shape[:2] = (H, W)
#         pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])
#     #plot match_features
#     reciprocal_in_P1, nn1_in_P2,reciprocal_in_P2, nn2_in_P1,num_matches = find_reciprocal_matches(*pts3d_list)

#     print(f'found {num_matches} matches')

#     matches_im0 = pts2d_list[im1][reciprocal_in_P1]
#     matches_im1 = pts2d_list[im2][nn1_in_P2][reciprocal_in_P1]

#     draw_dust3r_match(matches_im0, matches_im1, imgs[im1],imgs[im2])