#!/bin/bash

printf "Evaluate TCN on NTU-RGB+D Cross Subject:\n"
python main.py --config config/NTU-RGB-D/xsub/TCN.yaml --phase test --weights model/ntuxsub-tcn.pt
printf "\n"

printf "Evaluate TCN on NTU-RGB+D Cross View:\n"
python main.py --config config/NTU-RGB-D/xview/TCN.yaml --phase test --weights model/ntuxview-tcn.pt
printf "\n"

printf "Evaluate ST-GCN on Kinetics-skeleton:\n"
python main.py --config config/NTU-RGB-D/xview/TCN.yaml --phase test --weights model/ntuxview-tcn.pt
printf "\n"