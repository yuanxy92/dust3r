#!/bin/bash

# python convert_dust3r_to_3dgs.py \
#         --root /home/luvision/project/Code/data/Aurora/Fig_4/Apple/capture_20240929_3 \
#         --dataname 003

# for i in $(seq -w 000 007);
# do
#    python convert_dust3r_to_3dgs_soft_gripper.py \
#         --root /home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241002_2 \
#         --dataname $i
# done

# for i in $(seq -w 000 017);
# do
#    python convert_dust3r_to_3dgs_soft_gripper.py \
#         --root /home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241003_1 \
#         --dataname ${i}
#    python convert_dust3r_to_3dgs_soft_gripper.py \
#         --root /home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241003_1 \
#         --dataname ${i}_sr
# done

# for i in $(seq -w 000 004);
# do
#    python convert_dust3r_to_3dgs_soft_gripper.py \
#         --root /home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241004_2 \
#         --dataname ${i}
#    python convert_dust3r_to_3dgs_soft_gripper.py \
#         --root /home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241004_2 \
#         --dataname ${i}_sr
# done

# for i in $(seq -w 000 005);
# do
#    python convert_dust3r_to_3dgs_soft_gripper.py \
#         --root /home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241004_3 \
#         --dataname ${i}
#    python convert_dust3r_to_3dgs_soft_gripper.py \
#         --root /home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241004_3 \
#         --dataname ${i}_sr
# done

for i in $(seq -w 020 024);
do
   python convert_dust3r_to_3dgs_soft_gripper.py \
        --root /home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241004_4 \
        --dataname ${i}
done

# for i in $(seq -w 000 006);
# for i in $(seq -w 011 014);
# do
#    python convert_dust3r_to_3dgs_soft_gripper.py \
#         --root /home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241008_1 \
#         --dataname ${i}
# done