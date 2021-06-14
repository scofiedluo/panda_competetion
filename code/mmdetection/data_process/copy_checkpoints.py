from shutil import copy
import os
#import argparse

#parser = argparse.ArgumentParser(description='Copy model')
#parser.add_argument("-w","--which_epoch",dest="list",nargs='+',help="epoch list to copy")

src1 = '../../../user_data/tmp_data/cascade_rcnn_r101_fpn_1x_coco_round1_panda/epoch_100.pth'
src2 = '../../../user_data/tmp_data/cascade_rcnn_r101_fpn_1x_coco_round1_panda/epoch_120.pth'
save_dir = '../../../user_data/model_data/cascade_rcnn_r101_fpn_1x_coco_round1_panda/'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

#args = parser.parse_args()
#for i in args.which_epoch:
    #src = src1+ i
copy(src1,save_dir)
copy(src2,save_dir)
