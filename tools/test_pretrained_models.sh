#!/bin/bash
DEVICE='0'

printf "\nEvaluate ST-GCN on NTU-RGB+D Cross Subject:\n"
python main.py --config config/NTU-RGB-D/xsub/ST_GCN.yaml --phase test --weights model/ntuxsub-st_gcn.pt --device $Device

printf "\nEvaluate ST-GCN on NTU-RGB+D Cross View:\n"
python main.py --config config/NTU-RGB-D/xview/ST_GCN.yaml --phase test --weights model/ntuxview-st_gcn.pt --device $Device

# printf "\nEvaluate ST-GCN on Kinetics-skeleton:\n"
# python main.py --config config/NTU-RGB-D/xview/TCN.yaml --phase test --weights model/ntuxview-tcn.pt

printf "\nEvaluate TCN on NTU-RGB+D Cross Subject:\n"
python main.py --config config/NTU-RGB-D/xsub/TCN.yaml --phase test --weights model/ntuxsub-tcn.pt --device $Device

printf "\nEvaluate TCN on NTU-RGB+D Cross View:\n"
python main.py --config config/NTU-RGB-D/xview/TCN.yaml --phase test --weights model/ntuxview-tcn.pt --device $Device

printf "\nEvaluate TCN on Kinetics-skeleton:\n"
python main.py --config config/Kinetics/TCN.yaml --phase test --weights model/kinetics-tcn.pt --device $Device