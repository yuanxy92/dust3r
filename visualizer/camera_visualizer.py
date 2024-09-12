import numpy as np
import scipy.io as sio

def main():
    # load dust3r results
    dust3r_name = '/Users/yuanxy/Downloads/LocalSend/concave_recon/dust3r_result.npy'
    with open(dust3r_name, 'rb') as f:
        imgs = np.load(f)
        focals = np.load(f)
        poses = np.load(f)
        pts3d = np.load(f)
        confidence_masks = np.load(f)
    
    # save to .mat for matlab visualizer
    R = []
    T = []
    for idx in range(len(imgs)):
        # inverse RT matrix, from cam2world to world2cam 
        rotation_mat = poses[idx][:3, :3]
        rotation_mat = rotation_mat.transpose()
        trans_mat = - rotation_mat @ poses[idx][:3, 3]
        R.append(rotation_mat)
        T.append(trans_mat)
    mdic = {'R':R, 'T':T}
    sio.savemat('/Users/yuanxy/Downloads/LocalSend/concave_recon/cameras.mat', mdic)


if __name__ == '__main__':
    main()