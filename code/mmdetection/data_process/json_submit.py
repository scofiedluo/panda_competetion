from sloan_utils import Panda_tool

#TODO:adjust the overlap_factor back to 1500
Panda_tool.bbox_json2commited_4divided(source_test_anos = '../../../tcdata/panda_round1_test_B_annos_20210222/person_bbox_test_B.json',
                                       coco_test_path='../../../user_data/tmp_data/panda_round1_coco_full_patches_wh_3000_3000_testB.json',
                                       bbox_json_path='../../../user_data/tmp_data/panda_B_patches_results_rcnn_101_batch120_overlap1200.bbox.json',
                                       save_json_path='../../../prediction_result/det_results.json',overlap_factor=[1200,1200],
                                       nms_thresh=0.5,use_box_voting=True,
                                       show_flag=False)
