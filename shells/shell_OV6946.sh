#!/bin/bash

# python convert_dust3r_to_3dgs.py \
#         --root /home/luvision/project/Code/data/Aurora/Fig_4/Apple/capture_20240929_3 \
#         --dataname 003

for i in $(seq -w 000 013);
do
   python convert_dust3r_to_3dgs.py \
        --root /home/luvision/project/Code/data/Aurora/Fig_4/Apple/capture_20240929_4 \
        --dataname $i
done


