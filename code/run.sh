#!/bin/bash

# test data prepare
cd ./mmdetection/data_process

python MulprocessApply.py

# test
cd ..
# muti GPU
PORT=29505 ./tools/dist_test.sh configs/panda/cascade_rcnn_r101_fpn_1x_coco_round1_panda.py ../../user_data/model_data/cascade_rcnn_r101_fpn_1x_coco_round1_panda/epoch_120.pth 1 --format-only --options "jsonfile_prefix=../../user_data/tmp_data/panda_B_patches_results_rcnn_101_batch120_overlap1200"
# single GPU

# merge submit type JSON data
cd ./data_process

python json_submit.py
