#!/bin/bash

out_path="model/"
link="https://s3-us-west-1.amazonaws.com/yysijie-data/public/st-gcn/model/"
reference_model="tools/reference_model.txt"

mkdir -p $out_path


while IFS='' read -r line || [[ -n "$line" ]]; do
    wget $link$line -O $out_path$line
done < "$reference_model"