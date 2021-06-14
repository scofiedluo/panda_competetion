# -*- coding: utf-8 -*-
# @Time : 2021-03-11 16:38
# @Author : sloan
# @Email : 630298149@qq.com
# @File : MulprocessApply.py
# @Software: PyCharm
from concurrent.futures import ProcessPoolExecutor
from MulprocessBased import SampleGeneratorBase
from sloan_utils import Panda_tool
import os
import time
import cv2
import os.path as osp
import glob
import json
import numpy as np


class GenerateTestImage(SampleGeneratorBase):
    '''
        对原图按宽4096，高3500，步长2048的窗口滑动裁剪；
        会生成裁剪后的子图和对应的未带bbox注释的coco json
        '''
    def __init__(self,workers=4):
        super(SampleGeneratorBase,self).__init__()
        self.workers = workers
        self.threading_num = 0
        self.save_root_path = None
        self.overlap = (2048,2048)
        self.hw = (3500,4096)

    def process_img(self,*args):
        img_path = args[0]
        img_name = osp.split(img_path)[-1]
        src = cv2.imread(img_path)
        overlap_factor_w, overlap_factor_h = self.overlap
        row_cutshape, col_cutshape = self.hw
        crop_img = src.copy()
        crop_img_h,crop_img_w = crop_img.shape[:2]
        rows,cols = int(np.ceil(crop_img_h/row_cutshape)), int(np.ceil(crop_img_w/col_cutshape))
        # 没有超过边界，需要继续补充切片
        rows_gap = rows * row_cutshape - (rows - 1) * overlap_factor_h
        cols_gap = cols * col_cutshape - (cols - 1) * overlap_factor_w
        while rows_gap < crop_img_h:
            rows += 1
            rows_gap = rows * row_cutshape - (rows - 1) * overlap_factor_h
        while cols_gap < crop_img_w:
            cols += 1
            cols_gap = cols * col_cutshape - (cols - 1) * overlap_factor_w
        for row in range(rows):
            for col in range(cols):
                x1, y1, x2, y2 = (col * col_cutshape - col * overlap_factor_w), (
                        row * row_cutshape - row * overlap_factor_h), \
                                 ((col + 1) * col_cutshape - col * overlap_factor_w), (
                                         (row + 1) * (row_cutshape) - row * overlap_factor_h)
                if x2>crop_img_w:   # 切片右边界超出图像
                    offset = x2 - crop_img_w
                    x2 = crop_img_w
                    x1 -= offset
                if y2>crop_img_h:   # 切片下边界超出图像
                    offset = y2 - crop_img_h
                    y2 = crop_img_h
                    y1 -= offset
                img_temp = crop_img[y1:y2, x1:x2]
                im_name = img_name[:-4] + '_{}x{}.jpg'.format(row, col)
                save_img_path = osp.join(self.save_root_path, im_name)
                # print("{} is saved!".format(im_name))
                if not osp.exists(save_img_path):
                    cv2.imwrite(save_img_path, img_temp)
        return self.threading_num



    def batch_sample(self,*args):
        img_root_path,save_img_root_path,save_test_json_path,overlap,hw = args
        os.makedirs(save_img_root_path,exist_ok=True)
        self.save_root_path = save_img_root_path
        self.overlap = overlap
        self.hw = hw
        img_path_list = glob.glob(img_root_path+'/*/*jpg')
        s1 = time.time()
        print(len(img_path_list))
        results = []
        task_pool = ProcessPoolExecutor(max_workers=self.workers)
        count = 0
        for img_path in img_path_list:
            # count += 1
            # if count > 2: break
            if self.needed_to_process(img_path):
                rt = task_pool.submit(self._process_img, img_path)
                results.append(rt)
        results = [rt.result() for rt in results if rt]
        print(len(results))
        Panda_tool.gen_test_json(test_image_path=save_img_root_path,
                                save_json_path=save_test_json_path,
                                hw=self.hw)
        print("-----finished-------")
        print("cost time:{} s".format(time.time() - s1))

class GenerateTrainImage(SampleGeneratorBase):
    '''
    对原图按宽4096，高3500，步长2048的窗口滑动裁剪；
    会生成裁剪后的子图和对应的含注释的coco json
    '''
    def __init__(self,workers=4):
        super(SampleGeneratorBase,self).__init__()
        self.workers = workers
        self.threading_num = 0
        self.save_root_path = None
        self.overlap = (2048,2048)
        self.hw = (3500,4096)


    def process_img(self,*args):
        img_path,v = args[0],args[1]
        bboxes = v['bbox']
        img_name = osp.split(img_path)[-1]
        src = cv2.imread(img_path)
        src_h,src_w = src.shape[:2]

        # some param and var
        overlap_factor_w, overlap_factor_h = self.overlap

        row_cutshape, col_cutshape = self.hw
        images_dict = {}
        img_idx,anno_idx = 0, 0
        images, annotations = [], []

        crop_img = src.copy()
        crop_img_h,crop_img_w = crop_img.shape[:2]
        rows,cols = int(np.ceil(crop_img_h/row_cutshape)), int(np.ceil(crop_img_w/col_cutshape))

        # 没有超过边界，需要补充一个切片
        # if rows * row_cutshape - (rows - 1) * overlap_factor < crop_img_h:
        #     rows += 1
        # if cols * col_cutshape - (cols - 1) * overlap_factor < crop_img_w:
        #     cols += 1
        rows_gap = rows * row_cutshape - (rows - 1) * overlap_factor_h
        cols_gap = cols * col_cutshape - (cols - 1) * overlap_factor_w
        while rows_gap < crop_img_h:
            rows += 1
            rows_gap = rows * row_cutshape - (rows - 1) * overlap_factor_h
        while cols_gap < crop_img_w:
            cols += 1
            cols_gap = cols * col_cutshape - (cols - 1) * overlap_factor_w
        # 开始按n行m列切图
        for row in range(rows):
            for col in range(cols):
                cut_x1, cut_y1, cut_x2, cut_y2 = (col * col_cutshape - col * overlap_factor_w), (
                        row * row_cutshape - row * overlap_factor_h), \
                                                 ((col + 1) * col_cutshape - col * overlap_factor_w), (
                                                         (row + 1) * (row_cutshape) - row * overlap_factor_h)
                if cut_x2 > crop_img_w:  # 切片右边界超出图像
                    col_offset = cut_x2 - crop_img_w
                    cut_x2 = crop_img_w
                    cut_x1 -= col_offset
                if cut_y2 > crop_img_h:  # 切片下边界超出图像
                    row_offset = cut_y2 - crop_img_h
                    cut_y2 = crop_img_h
                    cut_y1 -= row_offset
                crop_img_cp = crop_img.copy()
                # 类别位置与裁剪位置判断
                for bbox in bboxes:
                    dst_cat, dst_w, dst_h = bbox['category_id'], bbox['w'], bbox['h']
                    dst_x1, dst_y1 = bbox['x1'], bbox['y1']
                    dst_x2, dst_y2 = dst_x1 + dst_w, dst_y1 + dst_h
                    if Panda_tool.judge_saved(src_pos=(cut_x1, cut_y1, cut_x2, cut_y2),
                                             dst_pos=(dst_x1, dst_y1, dst_x2, dst_y2),
                                             iof_thr=0.5):


                        # 裁剪过程可能导致类别被切分，类别需重新计算位置并考虑边界切分情况
                        dst_x1, dst_y1, dst_x2, dst_y2 = max(dst_x1 - cut_x1, 0), max(dst_y1 - cut_y1, 0), \
                                                         min(dst_x2 - cut_x1, cut_x2), min(dst_y2 - cut_y1, cut_y2)
                        # 写入coco格式
                        im_name = osp.split(img_path)[-1][:-4] + '_{}x{}.jpg'.format(row, col)
                        if im_name not in images_dict.keys():
                            img_temp = crop_img_cp[cut_y1:cut_y2, cut_x1:cut_x2]
                            img_temp_h, img_temp_w = img_temp.shape[:2]
                            save_img_path = osp.join(self.save_root_path, im_name)
                            if not osp.exists(save_img_path):
                                cv2.imwrite(save_img_path, img_temp)
                            images_dict[im_name] = img_idx
                            image = {}
                            image['file_name'] = im_name
                            image['width'] = img_temp_w
                            image['height'] = img_temp_h
                            image['id'] = img_idx
                            images.append(image)
                            img_idx += 1
                        annotation = {}
                        dst_w, dst_h = dst_x2 - dst_x1, dst_y2 - dst_y1
                        box = [dst_x1, dst_y1, dst_w, dst_h]
                        annotation['bbox'] = box
                        annotation['area'] = dst_w * dst_h
                        annotation['iscrowd'] = 0
                        annotation['image_id'] = images_dict[im_name]
                        annotation['category_id'] = dst_cat
                        annotation['id'] = anno_idx
                        anno_idx += 1
                        annotations.append(annotation)
        return self.threading_num,(images,annotations)

    def batch_sample(self,*args):
        img_root_path,save_img_root_path,save_anno_path,src_anno_json,overlap,hw = args
        os.makedirs(save_img_root_path,exist_ok=True)
        self.save_root_path = save_img_root_path
        self.overlap = overlap
        self.hw = hw
        all_instance, cla_instance, img_instance, defect, categories = Panda_tool._create_data_dict(
            json_paths=[src_anno_json],
            img_path=[img_root_path])
        s1 = time.time()
        print(len(img_instance.keys()))
        results = []
        task_pool = ProcessPoolExecutor(max_workers=self.workers)
        count = 0
        for img_path,v in img_instance.items():
            # count += 1
            # if count > 3: break
            if self.needed_to_process(img_path):
                rt = task_pool.submit(self._process_img, img_path,v)
                results.append(rt)
        meta = {}
        img_idx = 0
        anno_idx = 0
        new_images,new_annotations = [],[]
        for rt in results:
            if rt:
                images,annotations = rt.result()[1]
                img_old2new = {}
                for image in images:
                    img_old2new[image['id']] = img_idx
                    image['id'] = img_idx
                    img_idx += 1
                    new_images.append(image)
                for annotation in annotations:
                    annotation['image_id'] = img_old2new[annotation['image_id']]
                    annotation['id'] = anno_idx
                    anno_idx += 1
                    new_annotations.append(annotation)
        # 保存至coco格式
        meta['images'] = new_images
        meta['annotations'] = new_annotations
        meta['categories'] = Panda_tool.CATEGOTIES
        Panda_tool.write2result(meta, save_anno_path)
        print(len(results))
        print("-----finished-------")
        print("cost time:{} s".format(time.time() - s1))


if __name__ == "__main__":
    PATCH_W, PATCH_H = 3000,3000
    # TODO: adjust back to 1500
    PATCH_OVERLAP = (1200,1200)

    # TODO: test data A/B adjust
    GTI_FC = GenerateTestImage(workers=8)
    GTI_FC.batch_sample('../../../tcdata/panda_round1_test_202104_B/',
                        '../../../user_data/tmp_data/panda_round1_test_202104_B_patches_{}_{}'.format(PATCH_W,PATCH_H),
                        '../../../user_data/tmp_data/panda_round1_coco_full_patches_wh_{}_{}_testB.json'.format(PATCH_W,PATCH_H),
                        PATCH_OVERLAP,(PATCH_H,PATCH_W))

    # GTI_FC = GenerateTrainImage(workers=16)
    # GTI_FC.batch_sample('panda_round1_train_202104/',
    #                     'panda_round1_train_202104_patches_{}_{}/'.format(PATCH_W,PATCH_H),
    #                     'panda_round1_coco_full_patches_wh_{}_{}.json'.format(PATCH_W,PATCH_H),
    #                     'panda_round1_coco_full.json',
    #                     PATCH_OVERLAP,(PATCH_H,PATCH_W)
    #                     )
