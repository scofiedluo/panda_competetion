# -*- coding: utf-8 -*-
# @Time : 2021-03-11 11:22
# @Author : sloan
# @Email : 630298149@qq.com
# @File : sloan_utils.py
# @Software: PyCharm
import json
from tqdm import tqdm
from easydict import EasyDict as edict
import os.path as osp
import os
import random
import torch
import cv2
import glob
import time
import numpy as np
import matplotlib.pyplot as plt

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1+1) * (y2 - y1+1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1+1)
        h = np.maximum(0.0, yy2 - yy1+1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def box_iou(box1, box2, use_iof=False,order='xyxy'):
    # box1 shape: [N,5]   box2 shape:[N,5]  np.ndarray
    box1=torch.from_numpy(box1).float()
    box2=torch.from_numpy(box2).float()

    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(box1[:,None,:2], box2[:,:2])  # [N,M,2]
    rb = torch.min(box1[:,None,2:], box2[:,2:])  # [N,M,2]

    wh = (rb-lt+1).clamp(min=0)      # [N,M,2]
    inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

    area1 = (box1[:,2]-box1[:,0]+1) * (box1[:,3]-box1[:,1]+1)  # [N,]
    area2 = (box2[:,2]-box2[:,0]+1) * (box2[:,3]-box2[:,1]+1)  # [M,]
    if use_iof:
        union = torch.min(area1[:,None],area2)
    else:
        union = (area1[:,None] + area2 - inter)
    iou = inter / union
    return iou.numpy()

def box_voting(top_dets, all_dets, thresh, scoring_method='ID', beta=1.0):
    """Apply bounding-box voting to refine `top_dets` by voting with `all_dets`.
    See: https://arxiv.org/abs/1505.01749. Optional score averaging (not in the
    referenced  paper) can be applied by setting `scoring_method` appropriately.
    """
    # top_dets is [N, 5] each row is [x1 y1 x2 y2, sore]
    # all_dets is [N, 5] each row is [x1 y1 x2 y2, sore]
    top_dets_out = top_dets.copy()
    top_boxes = top_dets[:, :4]
    all_boxes = all_dets[:, :4]
    all_scores = all_dets[:, 4]
    top_to_all_overlaps = box_iou(top_boxes, all_boxes)
    for k in range(top_dets_out.shape[0]):
        inds_to_vote = np.where(top_to_all_overlaps[k] >= thresh)[0]
        boxes_to_vote = all_boxes[inds_to_vote, :]
        ws = all_scores[inds_to_vote]
        np.where(ws==0,0.001,ws)
        top_dets_out[k, :4] = np.average(boxes_to_vote, axis=0, weights=ws)
        #top_dets_out[k, :4] = np.average(boxes_to_vote, axis=0)
        if scoring_method == 'ID':
            # Identity, nothing to do
            pass
        elif scoring_method == 'SCORE_SUM':
            # Combine new probs from overlapping boxes
            top_dets_out[k, 4] = ws.sum()
        elif scoring_method == 'TEMP_AVG':
            # Average probabilities (considered as P(detected class) vs.
            # P(not the detected class)) after smoothing with a temperature
            # hyperparameter.
            P = np.vstack((ws, 1.0 - ws))
            P_max = np.max(P, axis=0)
            X = np.log(P / P_max)
            X_exp = np.exp(X / beta)
            P_temp = X_exp / np.sum(X_exp, axis=0)
            P_avg = P_temp[0].mean()
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'AVG':
            # Combine new probs from overlapping boxes
            top_dets_out[k, 4] = ws.mean()
        elif scoring_method == 'IOU_AVG':
            P = ws
            ws = top_to_all_overlaps[k, inds_to_vote]
            P_avg = np.average(P, weights=ws)
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'GENERALIZED_AVG':
            P_avg = np.mean(ws**beta)**(1.0 / beta)
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'QUASI_SUM':
            top_dets_out[k, 4] = ws.sum() / float(len(ws))**beta
        else:
            raise NotImplementedError(
                'Unknown scoring method {}'.format(scoring_method)
            )

    return top_dets_out

class Panda_tool:

    CLASSES = {
        "1": "人体可见部分",
        "2": "全人体",
        "3": "人头",
        "4": "车辆可见部分",
        # 考虑是否加入5和6，异常行人和异常车辆
    }

    CATEGORIES = [{'id': 1, 'name': 'visible body'},  # 人体可见部分
                  {'id': 2, 'name': 'full body'},  # 全人体
                  {'id': 3, 'name': 'head'},  # 人头
                  {'id': 4, 'name': 'visible car'},  # 车辆可见部分
                  # {'id': 5, 'name': 'ignore person'},  # 人体忽略部分
                  # {'id': 6, 'name': 'ignore car'},  # 车辆忽略部分
                  ]

    @classmethod
    def _create_data_dict(cls, json_paths, img_path):
        '''
        :return:
        ins
            {'im_name': abs_path/xxx.jpg, 'bbox': {'x1': 165.14,'y1': 53.71,'w': 39.860000000000014,'h': 63.29,'category_id': 2},'width': 20, 'height': 20}

        all_instance
            [ ins1,ins2,...]

        cla_instance
            {'1':[ins1,ins2,...], '2'[ins1,ins2,...]}

        instance
            {
            'im_name': 'xxx/xxx.jpg',
             'bbox': [{'x1': 165.14,'y1': 53.71,'w': 39.860000000000014,'h': 63.29,'category_id': 2}],
             'width': 492,
             'height': 658
             }

        img_instance
            {'xx1.jpg':instance  ,'xxx.jpg':instance}

        '''
        '''
        json format
        {
            "images":
                [
                    {"file_name":"cat.jpg", "id":1, "height":1000, "width":1000},
                    {"file_name":"dog.jpg", "id":2, "height":1000, "width":1000},
                    ...
                ]
            "annotations":
                [
                    {"image_id":1, "bbox":[100.00, 200.00, 10.00, 10.00], "category_id": 1}
                    {"image_id":2, "bbox":[150.00, 250.00, 20.00, 20.00], "category_id": 2}
                    ...
                ]
            "categories":
                [
                    {"id":0, "name":"bg"}
                    {"id":1, "name":"cat"}
                    {"id":1, "name":"dog"}
                    ...
                ]
        }
        '''
        all_instance = []
        cla_instance = edict()
        img_instance = edict()
        defect = edict()
        info_temp = {}
        im_name_to_index = {}
        categories = ""
        for i in range(len(json_paths)):
            anno_path = json_paths[i]
            anno_o = open(anno_path, 'r')
            anno = json.load(anno_o)
            # get image info
            '''
            [1:'image1',
             2:'imag2',
             ...
            ]
            '''
            for im_info in anno['images']:
                file_name = im_info['file_name']
                id = im_info['id']

                height = im_info['height']
                width = im_info['width']

                file_path = osp.join(img_path[i], file_name)
                im_name_to_index[id] = [file_path, height, width]
            # get anno info
            '''
            {
                1:[{x1: 1,y1: 2,w: 3,h: 4,category_id:3}, {x1: 1,y1: 2,w: 3,h: 4,category_id:3}, ...],
                2:[{x1: 1,y1: 2,w: 3,h: 4,category_id:3}, ...]
                ...
            }
            '''
            for im_info in anno['annotations']:
                image_id = im_info['image_id']
                bbox = im_info['bbox']
                category_id = im_info['category_id']
                bbox.append(category_id)
                ins = edict()
                ins.x1 = bbox[0]
                ins.y1 = bbox[1]
                ins.w = bbox[2]
                ins.h = bbox[3]
                ins.category_id = category_id
                if image_id in info_temp.keys():
                    info_temp[image_id].append(ins)
                else:
                    info_temp[image_id] = [ins]
            # get defect info

            categories = anno['categories']
            for im_info in anno['categories']:
                defect[str(im_info['id'])] = im_info['name']

            # for im_info in anno['categories']:
            #     defect[str(im_info['id'])] = im_info['name']
            for index in im_name_to_index.keys():
                if index not in info_temp.keys():
                    continue
                im_name = im_name_to_index[index][0]

                w = im_name_to_index[index][2]
                h = im_name_to_index[index][1]
                img_instance[im_name] = {}
                img_instance[im_name].bbox = info_temp[index]
                img_instance[im_name].width = w
                img_instance[im_name].height = h
                for box in img_instance[im_name].bbox:
                    ins = edict()
                    ins.im_name = im_name
                    ins.bbox = box
                    ins.width = img_instance[im_name].width
                    ins.height = img_instance[im_name].height
                    cat = box['category_id']
                    if str(cat) in cla_instance.keys():
                        cla_instance[str(cat)].append(ins)
                    else:
                        cla_instance[str(cat)] = [ins]
                    all_instance.append(ins)
        return all_instance, cla_instance, img_instance, defect, categories

    @classmethod
    def data2coco(cls, src_json_paths: list, save_json_path: str = 'coco_full.json') -> None:
        '''
        将官方json转为coco格式
        :param src_json_paths:
        :param save_json_path:
        :return:
        '''
        if isinstance(src_json_paths,str):
            src_json_paths = [src_json_paths]
        json_files = []
        for src_json_path in src_json_paths:
            with open(src_json_path) as fr:
                json_file = json.load(fr)
                json_files.append(json_file)

        meta = {}
        images = []
        annotations = []
        num_box = 0
        cur_img_idx = 0
        images_dict = {}
        for json_file in json_files:
            for im_name,v in json_file.items():
                width = v['image size']['width']
                height = v['image size']['height']
                if im_name not in images_dict.keys():
                    images_dict[im_name] = cur_img_idx
                    image = {}
                    image['file_name'] = im_name
                    image['width'] = width
                    image['height'] = height
                    image['id'] = cur_img_idx
                    images.append(image)
                    cur_img_idx += 1
                for objs in v['objects list']:
                    category = objs['category']
                    if category == 'person':
                        for (cat_id, key) in [(1,'visible body'),(2,'full body'),(3,'head')]:
                            annotation = {}
                            cur_pos = objs['rects'][key]
                            x1, y1, x2, y2 = cur_pos['tl']['x']*width, cur_pos['tl']['y']*height,\
                                             cur_pos['br']['x']*width, cur_pos['br']['y']*height
                            w, h = x2 - x1, y2 - y1
                            box = [x1, y1, w, h]
                            annotation['bbox'] = box
                            annotation['area'] = w * h
                            annotation['iscrowd'] = 0
                            annotation['image_id'] = images_dict[im_name]
                            annotation['category_id'] = cat_id
                            annotation['id'] = num_box
                            num_box += 1
                            annotations.append(annotation)
                    elif category in ['small car', 'midsize car', 'large car', 'bicycle', 'motorcycle', 'tricycle',
                                    'electric car', 'baby carriage']:
                        annotation = {}
                        cur_pos = objs['rect']
                        x1, y1, x2, y2 = cur_pos['tl']['x'] * width, cur_pos['tl']['y'] * height, \
                                         cur_pos['br']['x'] * width, cur_pos['br']['y'] * height
                        w, h = x2 - x1, y2 - y1
                        box = [x1, y1, w, h]
                        annotation['bbox'] = box
                        annotation['area'] = w * h
                        annotation['iscrowd'] = 0
                        annotation['image_id'] = images_dict[im_name]
                        annotation['category_id'] = 4
                        annotation['id'] = num_box
                        num_box += 1
                        annotations.append(annotation)

                    # elif category in ['fake person', 'ignore', 'crowd']:
                    #     annotation = {}
                    #     cur_pos = objs['rect']
                    #     x1, y1, x2, y2 = cur_pos['tl']['x'] * width, cur_pos['tl']['y'] * height, \
                    #                      cur_pos['br']['x'] * width, cur_pos['br']['y'] * height
                    #     w, h = x2 - x1, y2 - y1
                    #     box = [x1, y1, w, h]
                    #     annotation['bbox'] = box
                    #     annotation['area'] = w * h
                    #     annotation['iscrowd'] = 0
                    #     annotation['image_id'] = images_dict[im_name]
                    #     annotation['category_id'] = 5
                    #     annotation['id'] = num_box
                    #     num_box += 1
                    #     annotations.append(annotation)
                    # elif category in ['vehicles', 'unsure']:
                    #     annotation = {}
                    #     cur_pos = objs['rect']
                    #     x1, y1, x2, y2 = cur_pos['tl']['x'] * width, cur_pos['tl']['y'] * height, \
                    #                      cur_pos['br']['x'] * width, cur_pos['br']['y'] * height
                    #     w, h = x2 - x1, y2 - y1
                    #     box = [x1, y1, w, h]
                    #     annotation['bbox'] = box
                    #     annotation['area'] = w * h
                    #     annotation['iscrowd'] = 0
                    #     annotation['image_id'] = images_dict[im_name]
                    #     annotation['category_id'] = 6
                    #     annotation['id'] = num_box
                    #     num_box += 1
                    #     annotations.append(annotation)

        meta['images'] = images
        meta['annotations'] = annotations
        meta['categories'] = cls.CATEGORIES
        cls.write2result(meta, save_json_path)

    @classmethod
    def write2result(cls, obj_ins: (list, dict), save_path: str = 'result.json') -> None:
        '''
        将内容保存至json文件
        :param obj_ins:
        :return:
        '''
        with open(save_path, 'w') as fp:
            json.dump(obj_ins, fp, indent=4, ensure_ascii=False)
            # json.dump(obj_ins, fp, indent=4)

    @staticmethod
    def split_train_val(json_path: str, pic_path: str, ratio: float = 0.2):
        '''
        设定随机种子，根据比例划分训练集与验证集
        :param json_path:
        :param pic_path:
        :param ratio:
        :return:
        '''
        random.seed(2021)
        f = json.load(open(json_path, 'r'))
        categories = f['categories']
        _, _, img_instance, _, _ = Panda_tool._create_data_dict([json_path], [pic_path])
        all_images = img_instance.keys()
        print("all imgs:",len(all_images))
        val_images_split = random.sample(all_images, int(len(all_images) * ratio))
        print("val imgs:",len(val_images_split))

        train_annotations = []
        val_annotations = []
        train_images = []
        val_images = []
        train_id = 0
        val_id = 0
        train_box_id = 0
        val_box_id = 0
        for image_name in tqdm(img_instance.keys()):
            flag_val = False
            if image_name in val_images_split:
                flag_val = True
                val_id += 1
            else:
                train_id += 1

            image_anno = {}
            image_anno['file_name'] = image_name.split('/')[-1]
            image_anno['width'] = img_instance[image_name]['width']
            image_anno['height'] = img_instance[image_name]['height']
            if flag_val:
                image_anno['id'] = val_id
                val_images.append(image_anno)
            else:
                image_anno['id'] = train_id
                train_images.append(image_anno)
            # print(image_name)
            # print(img_instance[image_name]['bbox'])
            # print("flag,val_id,train_id:",flag_val,val_id,train_id)

            bboxes = img_instance[image_name]['bbox']
            for bbox in bboxes:
                bbox_anno = {}
                if flag_val:
                    val_box_id += 1
                    bbox_anno['id'] = val_box_id
                    bbox_anno['image_id'] = val_id
                else:
                    train_box_id += 1
                    bbox_anno['id'] = train_box_id
                    bbox_anno['image_id'] = train_id
                x1 = bbox['x1']
                y1 = bbox['y1']
                w = bbox['w']
                h = bbox['h']
                category_id = bbox['category_id']
                bbox_anno['bbox'] = [x1, y1, w, h]
                bbox_anno['category_id'] = category_id
                bbox_anno['area'] = w * h
                bbox_anno['iscrowd'] = 0
                if flag_val:
                    val_annotations.append(bbox_anno)
                else:
                    train_annotations.append(bbox_anno)
            #     print('val_box_id,train_box_id:',val_box_id,train_box_id)
            #     print("bbox_anno:",bbox_anno)
            #     print("train_anno:",train_annotations)
            # input()
        train_meta = {'categories': categories, 'images': train_images, 'annotations': train_annotations}
        val_meta = {'categories': categories, 'images': val_images, 'annotations': val_annotations}
        train_path = json_path.split('.')[0] + '_train.json'
        val_path = json_path.split('.')[0] + '_val.json'
        Panda_tool.write2result(train_meta, train_path)
        Panda_tool.write2result(val_meta, val_path)
        # with open(train_path,'w')as f:
        #     json.dump(train_meta, f)
        # with open(val_path,'w')as f1:
        #     json.dump(val_meta, f1)

    @staticmethod
    def vis_gt(json_paths: list, img_paths: list):
        '''
        可视化ground truth
        :param json_paths:
        :param img_paths:
        :return:
        '''
        all_instance, cla_instance, img_instance, defect, categories = Panda_tool._create_data_dict(
            json_paths=json_paths, img_path=img_paths)
        print(len(img_instance.keys()))
        for img_name, v in img_instance.items():
            bboxes = v['bbox']
            src = cv2.imread(img_name)
            for bbox in bboxes:
                x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x1'] + bbox['w']), int(
                    bbox['y1'] + bbox['h'])
                label = bbox['category_id']
                cv2.rectangle(src, (x1, y1), (x2, y2), (255, 0, 0), 4)
                cv2.putText(src, '%d' % label,
                            (x2, y2), cv2.FONT_HERSHEY_COMPLEX,
                            0.8,
                            (0, 0, 255), 4)
            print(img_name)
            cv2.imwrite('./visual_results/'+osp.basename(img_name),src)
            input()
            # cv2.namedWindow("im", cv2.WINDOW_NORMAL)
            # cv2.imshow('im', src)
            # key = cv2.waitKey(0) & 0xFF
            # if key == ord('q'):
            #     break

    @staticmethod
    def vis_gt_offical(json_path:str, img_root_path:str):
        '''
        针对生成提交的panda结果json可视化
        :param json_path:
        :param img_path:
        :return:
        '''
        with open('panda_round1_test_A_annos_202104/person_bbox_test_A.json','r') as fr:
            img2wh_id = json.load(fr)
        with open(json_path,'r') as fr:
            commited_results = json.load(fr)
        id2img_name = {}
        for k,v in img2wh_id.items():
            id2img_name[v['image id']] = k
        pred_results = {}
        for ins in commited_results:
            img_id = ins['image_id']
            cat_id = ins['category_id']
            x1,y1,w,h = ins['bbox_left'],ins['bbox_top'],ins['bbox_width'],ins['bbox_height']
            x2,y2 = x1+w,y1+h
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            score = ins['score']
            pred_results[img_id] = pred_results.get(img_id,[]) + [[x1,y1,x2,y2,cat_id,score]]
        for k,v in pred_results.items():
            cur_img_path = osp.join(img_root_path,id2img_name[k])
            src = cv2.imread(cur_img_path)
            for bbox in v:
                x1,y1,x2,y2,label,score = bbox
                cv2.rectangle(src, (x1, y1), (x2, y2), (255, 0, 0), 6)
                cv2.putText(src, '%d|%.3f' % (label,score),
                            (x2, y2), cv2.FONT_HERSHEY_COMPLEX,
                            0.8,
                            (0, 0, 255), 6)
            cv2.imwrite('./visual_results/'+osp.basename(cur_img_path),src)
            print(cur_img_path)
            input()

    @staticmethod
    def gen_test_json(test_image_path: str, save_json_path: str = 'coco_test.json',
                      hw=None) -> None:
        '''
        将官方测试集转为coco格式，以便调用dist_test.py
        :param src_json_path:
        :param save_json_path:
        :return:
        '''
        meta = {}
        images = []
        annotations = []
        img_paths = glob.glob(test_image_path + '/*jpg')
        for img_idx, img_path in tqdm(enumerate(img_paths)):
            if not hw:
                src = cv2.imread(img_path, 0)
                h, w = src.shape[:2]
            else:
                h,w = hw
            im_name = osp.split(img_path)[-1]
            image = {}
            image['file_name'] = im_name
            image['width'] = w
            image['height'] = h
            image['id'] = img_idx
            images.append(image)
        meta['images'] = images
        meta['annotations'] = annotations
        meta['categories'] = Panda_tool.CATEGORIES
        Panda_tool.write2result(meta, save_json_path)

    @staticmethod
    def bbox_json2commited_4divided(source_test_anos: str, coco_test_path: str, bbox_json_path: str,
                                    save_json_path: str = '',overlap_factor: list = [2048,2048],
                                    nms_thresh:float=0.25,
                                    use_box_voting:bool = True,
                                    show_flag=True) -> None:
        '''
        从dist_test.py生成的json转换成官方提交格式，针对切图场景的还原
        :return:
        '''
        root_path_dict = {'14': '14_OCT_Habour', '15': '15_Nanshani_Park', '16': '16_Primary_School',
                          '17': '17_New_Zhongguan', '18': '18_Xili_Street'}

        # 首先建立image_id 与 image_name和宽高的索引关系
        img_id2name = {}
        with open(coco_test_path) as fr:
            coco_test_json_file = json.load(fr)
        imgs_list = coco_test_json_file['images']
        for imgs_ins in imgs_list:
            img_id2name[imgs_ins['id']] = [imgs_ins['file_name'], (imgs_ins['width'], imgs_ins['height'])]
        with open(source_test_anos,'r') as fr:  #FIXME:may have some problem
            img2wh_id = json.load(fr)

        images_ins = {}
        with open(bbox_json_path) as fr:
            bbox_json_file = json.load(fr)

        overlap_factor_w, overlap_factor_h = overlap_factor
        for bbox_ins in bbox_json_file:
            file_name, (col_cutshape, row_cutshape) = img_id2name[bbox_ins['image_id']]

            src_file_name = '_'.join(file_name.split('_')[:3])+'.jpg'
            src_file_name = osp.join(root_path_dict[src_file_name.split('_')[1]],src_file_name)
            full_img_w, full_img_h = img2wh_id[src_file_name]['image size']['width'],\
                                     img2wh_id[src_file_name]['image size']['height']

            row, col = list(map(int, file_name[:-4].split('_')[-1].split('x')))

            category = bbox_ins['category_id']
            x1, y1, w, h = bbox_ins['bbox']
            score = bbox_ins['score']
            start_x1, start_y1 = (col * col_cutshape - col * overlap_factor_w), \
                                 (row * row_cutshape - row * overlap_factor_h)
            end_x1,end_y1 = ((col + 1) * col_cutshape - col * overlap_factor_w), (
                                             (row + 1) * (row_cutshape) - row * overlap_factor_h)
            if end_x1>full_img_w:
                offset = end_x1 - full_img_w
                start_x1 -= offset
                end_x1 = full_img_w
            if end_y1>full_img_h:
                offset = end_y1 - full_img_h
                start_y1 -= offset
                end_y1 = full_img_h

            # 映射回原图
            x1, y1 = start_x1 + x1, start_y1 + y1
            x2, y2 = x1 + w, y1 + h

            bbox = [x1, y1, x2, y2, score]
            if src_file_name not in images_ins.keys():
                images_ins[src_file_name] = [[] for _ in range(len(Panda_tool.CATEGORIES))]
            images_ins[src_file_name][category - 1].append(bbox)
        print(len(images_ins.keys()))  # Maybe equal to 80 if every image has object
        if show_flag:
            # 可视化看看
            for img_name, bboxes in images_ins.items():
                print(img_name)
                src = cv2.imread(osp.join('panda_round1_test_202104_A/{}'.format(root_path_dict[img_name.split('_')[1]]), img_name))
                print(bboxes)
                for cat_idx, bbox in enumerate(bboxes):
                    if bbox:
                        for bb in bbox:
                            x1, y1, x2, y2, score = bb
                            cv2.rectangle(src, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 5)
                            cv2.putText(src, '{}|{:.3f}'.format(cat_idx + 1, score),
                                        (int(x1 - 24), int(y1 - 24)),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 3)
                cv2.imwrite('visual_results/' + img_name, src)
                input()
        # 生成官方提交文件
        result = []
        # 对每张图的每个类别做NMS
        for img_name, bboxes in images_ins.items():
            for cat_idx, bbox in enumerate(bboxes):
                if len(bbox) > 1:
                    # print("original:",len(bbox),bbox)
                    if use_box_voting:
                        # use box_voting
                        nms_in = np.array(bbox,dtype=np.float32,copy=True)
                        keep = py_cpu_nms(nms_in,thresh=nms_thresh)
                        nms_out = nms_in[keep,:]
                        vote_out = box_voting(nms_out,nms_in,thresh=0.8,scoring_method='TEMP_AVG')
                        bboxes[cat_idx] = vote_out.tolist()
                        # print(vote_out,type(vote_out),len(vote_out))
                    else:
                        keep = py_cpu_nms(np.array(bbox), thresh=nms_thresh)
                        bboxes[cat_idx] = np.array(bbox)[keep, :].tolist()
                        # print("after nms:",len(bboxes[cat_idx]),bboxes[cat_idx])
                        # input()
                for bb in bboxes[cat_idx]:
                    x1, y1, x2, y2, score = bb
                    x1, y1, x2, y2 = np.round(x1, 2), np.round(y1, 2), np.round(x2, 2), np.round(y2, 2)
                    w, h = x2 - x1, y2 - y1
                    cur_ins = {}
                    cur_ins['image_id'] = img2wh_id[img_name]['image id']
                    cur_ins['category_id'] = cat_idx + 1
                    cur_ins['bbox_left'] = x1
                    cur_ins['bbox_top'] = y1
                    cur_ins['bbox_width'] = w
                    cur_ins['bbox_height'] = h
                    cur_ins['score'] = score
                    result.append(cur_ins)
        if not save_json_path:
            save_name = "result_" + time.strftime('%Y%m%d_%H_%M_%S', time.localtime(time.time())) + '.json'
            save_json_path = osp.join('panda_results/', save_name)
        Panda_tool.write2result(result, save_json_path)

    @staticmethod
    def judge_saved(src_pos: tuple, dst_pos: tuple, iof_thr: float = 0.5) -> bool:
        '''
        根据类别位置与裁剪图的iof，判断裁剪图上的类别是否应该保留
        :param src_pos:
        :param dst_pos:
        :return:
        '''
        src_x1, src_y1, src_x2, src_y2 = src_pos
        dst_x1, dst_y1, dst_x2, dst_y2 = dst_pos
        xx1, yy1 = max(src_x1, dst_x1), max(src_y1, dst_y1)
        xx2, yy2 = min(src_x2, dst_x2), min(src_y2, dst_y2)
        w, h = max(xx2 - xx1 + 1,0), max(yy2 - yy1 + 1,0)
        area = w*h
        dst_area = (dst_x2 - dst_x1 + 1) * (dst_y2 - dst_y1 + 1)
        iof = area / dst_area

        return iof >= iof_thr


    @staticmethod
    def run():
        '''
        1.将官方格式转为coco格式
        2.生成切片图和对应的coco json
        3.生成训练集与测试集json
        4.训练模型，调参
        5.生成测试集对应的切片图和coco json
        6.调用dist_test.py前传测试集
        7.将生成的bbox.json转为官方提交格式，注意切图场景
        '''
        Panda_tool.data2coco(src_json_paths=['./panda_round1_train_annos_202104/person_bbox_train.json',
                                             './panda_round1_train_annos_202104/vehicle_bbox_train.json'],
                            save_json_path='panda_round1_coco_full.json')
        # Panda_tool.split_train_val(json_path='panda_round1_coco_full_patches_wh_4096_3500.json',
        #                           pic_path='panda_round1_train_202104_patches_4096_3500')
        # Panda_tool.split_train_val(json_path='panda_round1_coco_full.json',
        #                           pic_path='panda_round1_train_202104')
        # Panda_tool.vis_gt(json_paths=['panda_round1_coco_full_only1.json'],img_paths=['panda_round1_train_202104'])
        # Panda_tool.vis_gt(json_paths=['panda_round1_coco_full_patches_wh_4096_3500.json'],img_paths=['panda_round1_train_202104_patches_4096_3500'])

        #从切分图还原到原图
        # Panda_tool.bbox_json2commited_4divided(coco_test_path='panda_round1_coco_full_patches_wh_6000_4000_testA.json',
        #                                                  bbox_json_path='./bbox_jsons/panda_patches_results_only4_1.bbox.json',
        #                                                  save_json_path='panda_results/only4_1.json',overlap_factor=[3000,2000],
        #                                                  nms_thresh=0.5,use_box_voting=True,
        #                                                  show_flag=False)

        # Panda_tool.vis_gt_offical(json_path='panda_results/only4_1.json',img_root_path='panda_round1_test_202104_A')

if __name__ == "__main__":
    Panda_tool.run()
