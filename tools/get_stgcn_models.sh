#!/bin/bash

out_path="checkpoints/"
link="https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmskeleton/models/st-gcn/"
reference_model="./tools/stgcn_models.txt"
mkdir -p $out_path
while IFS='' read -r line || [[ -n "$line" ]]; do
    wget -c $link$line -O $out_path$line
done < "$reference_model"