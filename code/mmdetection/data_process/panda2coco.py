from sloan_utils import Panda_tool

Panda_tool.data2coco(src_json_paths=['../../../tcdata/panda_round1_train_annos_202104/person_bbox_train.json',
                                     '../../../tcdata/panda_round1_train_annos_202104/vehicle_bbox_train.json'],
                    save_json_path='../../../user_data/tmp_data/panda_round1_coco_full.json')