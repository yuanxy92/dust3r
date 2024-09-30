import os
import numpy as np
import scipy.io as sio
from typing import Optional
import numpy as np
import plotly.graph_objects as go
from plotly.graph_objects import Layout
from matplotlib import pyplot as pl
from plyfile import PlyData, PlyElement

def to_homogeneous(points):
    pad = np.ones((points.shape[:-1] + (1,)), dtype=points.dtype)
    return np.concatenate([points, pad], axis=-1)

def init_figure(height: int = 1200) -> go.Figure:
    """Initialize a 3D figure."""
    layout = Layout(plot_bgcolor='rgba(0,0,0,0)')
    fig = go.Figure(layout=layout)
    axes = dict(
        visible=True,
        showbackground=True,
        showgrid=True,
        showline=True,
        showticklabels=True,
        autorange=True,
        backgroundcolor='white',
        # gridcolor='white',
        zerolinecolor='white',
        linecolor='black',
        linewidth=2,
        tickfont=dict(family='Arial', size=20)
    )
    fig.update_layout(
        height=height,
        scene_camera=dict(
            eye=dict(x=0.0, y=-0.1, z=-2),
            up=dict(x=0, y=-1.0, z=0),
            projection=dict(type="orthographic"),
        ),
        font=dict(family='Arial', size=15),
        scene=dict(
            xaxis=axes,
            yaxis=axes,
            zaxis=axes,
            aspectmode="data",
            dragmode="orbit",
        ),
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        legend=dict(orientation="h", yanchor="top", y=0.99, xanchor="left", x=0.1),

        plot_bgcolor='white',  # Background color of the grid
        paper_bgcolor='white',  # Background color of the entire figure
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
    color: str = "rgb(255, 0, 0)",
    color_fill: str = "rgb(255, 0, 0)",
    name: Optional[str] = None,
    legendgroup: Optional[str] = None,
    fill: bool = False,
    size: float = 1.0,
    text: Optional[str] = None,
    opacity: float = 0.8
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
            color=color_fill,
            i=i,
            j=j,
            k=k,
            legendgroup=legendgroup,
            name=name,
            showlegend=False,
            hovertemplate=text.replace("\n", "<br>"),
            opacity=opacity
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
        line=dict(color=color, width=2),
        showlegend=False,
        hovertemplate=text.replace("\n", "<br>"),
    )
    fig.add_trace(pyramid)

def main():
    # root_dir = '/Users/yuanxy/Downloads/LocalSend/Fig3/20240902/concave_recon'
    # root_dir = '/Users/yuanxy/Downloads/LocalSend/Fig3/20240902/convex_recon'
    # root_dir = '/Users/yuanxy/Downloads/LocalSend/Fig3/20240902/planar_recon'
    root_dir = '/Users/yuanxy/Downloads/LocalSend/Fig3/20240914_phone2/planar_recon'

    point_scale = 50
    # load dust3r results
    dust3r_name = os.path.join(root_dir, 'dust3r_result.npy')
    with open(dust3r_name, 'rb') as f:
        imgs = np.load(f)
        focals = np.load(f)
        poses = np.load(f)
        pts3d = np.load(f)
        confidence_masks = np.load(f)
    # convert K, R, T
    Ks = []
    Rs = []
    Ts = []
    Rs_cam2world = []
    Ts_cam2world = []
    for idx in range(len(imgs)):
        # inverse RT matrix, from cam2world to world2cam 
        Rs_cam2world.append(poses[idx][:3, :3])
        Ts_cam2world.append(poses[idx][:3, 3] * point_scale)
        rotation_mat = poses[idx][:3, :3]
        rotation_mat = rotation_mat.transpose()
        trans_mat = - rotation_mat @ poses[idx][:3, 3] * point_scale
        f = focals[idx]
        W = imgs[idx].shape[1]
        H = imgs[idx].shape[0]
        Ks.append(np.array([[f,0,W/2],[0,f,H/2],[0,0,1]]))
        Rs.append(rotation_mat)
        Ts.append(trans_mat)
    # save to .mat for matlab visualizer
    mdic = {'R':Rs, 'T':Ts}
    sio.savemat(os.path.join(root_dir, 'cameras.mat'), mdic)

    # load clean point clouds
    ply_data = PlyData.read(os.path.join(root_dir, 'points3D_clean.ply'))
    # Access vertex data (including colors if available)
    vertices = ply_data['vertex']
    # Check if the PLY file contains RGB color information
    if all(prop in vertices.data.dtype.names for prop in ['red', 'green', 'blue']):
        # Extract vertex coordinates and colors
        x_coords = vertices['x']
        y_coords = vertices['y']
        z_coords = vertices['z']
        
        red = vertices['red']
        green = vertices['green']
        blue = vertices['blue']
    pts3d = np.stack([x_coords, y_coords, z_coords], axis=1)
    color3d = np.stack([red, green, blue], axis=1)

    # plot reconstruction
    fig = init_figure()
    plot_points(fig, pts3d, color3d, ps=1.5, name='point cloud')
    for ind,(R,t,K) in enumerate(zip(Rs_cam2world, Ts_cam2world, Ks)):
        plot_camera(fig, R, t, K, color="rgba(255,0,0,1)", color_fill="rgba(255,0,0,0.45)",
            fill=True, text=f'cam_{ind}', size=20)
    fig.show()

if __name__ == '__main__':
    main()