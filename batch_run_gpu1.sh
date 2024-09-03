export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES='7'
python dust3r_pano.py --root /data/xiaoyun/OV6946_Arm_6_cameras/20240829_256px_v2 --dataname data_0829_8 --start 560 --stop 900 --step 5