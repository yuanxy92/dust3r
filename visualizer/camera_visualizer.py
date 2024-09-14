import os
import numpy as np
import scipy.io as sio
from typing import Optional
import numpy as np
import plotly.graph_objects as go
from matplotlib import pyplot as pl

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


def main():
    root_dir = '/Users/yuanxy/Downloads/LocalSend/concave_recon'
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
    for idx in range(len(imgs)):
        # inverse RT matrix, from cam2world to world2cam 
        rotation_mat = poses[idx][:3, :3]
        rotation_mat = rotation_mat.transpose()
        trans_mat = - rotation_mat @ poses[idx][:3, 3]
        f = focals[idx]
        W = imgs[idx].shape[1]
        H = imgs[idx].shape[0]
        Ks.append(np.array([[f,0,W/2],[0,f,H/2],[0,0,1]]))
        Rs.append(rotation_mat)
        Ts.append(trans_mat)
    # save to .mat for matlab visualizer
    mdic = {'R':Rs, 'T':Ts}
    sio.savemat('/Users/yuanxy/Downloads/LocalSend/concave_recon/cameras.mat', mdic)

    # plot reconstruction
    fig = init_figure()
    # n_viz = min(80000,len(all_rgbs))
    # idx_to_viz = list(np.round(np.linspace(0, len(all_rgbs)-1, n_viz)).astype(int))
    # vis_rgbs = [all_rgbs[idx] for idx in idx_to_viz]
    # plot_points(fig,all_pts3d[idx_to_viz],vis_rgbs,ps=2,name='3dpts')
    for ind,(R,t,K) in enumerate(zip(R, T, Ks)):
        plot_camera(fig,R,t,K,color="rgba(255,0,0,0.5)",text=f'cam_{ind}')
    fig.show()

if __name__ == '__main__':
    main()