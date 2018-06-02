#!/bin/bash

out_path="models/"
link="https://s3-us-west-1.amazonaws.com/yysijie-data/public/st-gcn/model/"
reference_model="resource/reference_model.txt"

mkdir -p $out_path


while IFS='' read -r line || [[ -n "$line" ]]; do
    wget $link$line -O $out_path$line
done < "$reference_model"