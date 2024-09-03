export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES='6'
python dust3r_pano.py --root /data/xiaoyun/OV6946_Arm_6_cameras/20240829_256px_v2 --dataname data_0829_6 --start 200 --stop 900 --step 5 