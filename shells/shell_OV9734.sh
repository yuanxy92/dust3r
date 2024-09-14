#!/bin/bash
for i in {2..10}
do
   python convert_dust3r_to_3dgs.py \
        --root /home/luvision/project/Code/data/Aurora/Fig_3/20240914_OV9734/shape_$i \
        --dataname images
done
    


