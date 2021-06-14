#!/bin/bash

# transform original annos to coco format
cd ./mmdetection/data_process
python panda2coco.py

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
# train 4 classes together with 4 GPU

cd ..

CUDA_VISIBLE_DEVICES=0,1,2,3
PORT=29502 ./tools/dist_train.sh configs/panda/cascade_rcnn_r101_fpn_1x_coco_round1_panda.py 4

# copy the saved model to user_data/model dir
cd ./data_process
python copy_checkpoints.py #copy epoch_100.pth and epoch_120.pth, if you want copy other epoch ,change the source file copy_checkpoints.py