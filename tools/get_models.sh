#!/bin/bash

out_path="models/"
link="https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmskeleton/models/st-gcn/"
reference_model="resource/reference_model.txt"

mkdir -p $out_path
while IFS='' read -r line || [[ -n "$line" ]]; do
    wget -c $link$line -O $out_path$line
done < "$reference_model"


# Downloading models for pose estimation
OPENPOSE_URL="http://posefs1.perception.cs.cmu.edu/OpenPose/models/"
POSE_FOLDER="pose/"

# Body (COCO)
COCO_FOLDER=${POSE_FOLDER}"coco/"
OUT_FOLDER="models/${COCO_FOLDER}"
COCO_MODEL=${COCO_FOLDER}"pose_iter_440000.caffemodel"
wget -c ${OPENPOSE_URL}${COCO_MODEL} -P ${OUT_FOLDER}