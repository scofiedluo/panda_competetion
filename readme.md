# Introduction

This repository is for the round 1 competition of [TIANCI ](https://tianchi.aliyun.com/competition/entrance/531855/introduction?spm=5176.12281949.1003.5.6d8e2448XfQEXk), an object detection task on [PANDA Dataset](http://www.panda-dataset.com/). And this repository  is based on [MMDetection](https://github.com/open-mmlab/mmdetection).

# Solution

Here we need to detect 4 classes object:` 'visible body','full body','head','visible car'`. 

## Challenge

* Since we have almost 1 billion pixels in just one image, we can't use the original image as input to train an object detection. If we directly zoom out the original image, much information will lose.  
* The 4 classes object have various size.
* For one original image, we need to finish inference in 90s.

Considering the above challenges, image cropping strategy will be used in training and testing. And in order to balancing the inference time and testing result, we choose to train just one model to detect the whole 4 classes. Here we use **Cascade R-CNN R101_FPN** as backbone.

## Training

Online image cropping is used during training. In other words, we use the original image as input, then select a target ground truth(GT) in the input image, and select a window of a **specified size**($3000\times3000$) (including the selected target GT) at a random location near its GT for cropping.  If the window include other targets, we will retain target GT larger than 0.5iof is and limit it in the window. If the currently selected target is larger than the specified window size, the target crop will be limited to the window size. After the online cropping, we will resize the window to $1500\times1500$.

## Testing

We will crop the image with overlap. The cropping size is same as the training stage, which is $3000\times3000$. Then the cropped image will be used for inference.

## Hyperparameters

Hyperparameters can be tuned in `code/mmdetection/configs/panda/cascade_rcnn_r101_fpn_1x_coco_round1_panda.py`. Here we show those parameters that is vital to our final result.

* `learning_rate`: 0.05
* `samples_per_gpu`: 1
* `optimizer`:  SGD with momentum 0.9 and weight_decay 0.0001
* `score_thr`: 0.03
* `overlap`: training $1500 \times 1500$, testing $1200 \times 1200$.
* `ratios`: ratios in anchor_generator $[0.2, 0.5, 1.0, 2.0]$.

# Getting start

## Prerequisites

1. Environment: **Python 3**

   * Create a conda virtual environment and activate it.

     ```
     conda create -n PANDA_tianchi python=3.7 -y
     conda activate PANDA_tianchi
     ```

   * Install PyTorch and torchvision following the [official instructions](https://pytorch.org/),

     ```
     conda install pytorch=1.7.0 cudatoolkit=10.1 torchvision=0.8.1 -c pytorch
     ```

2. Install the **MMDetection**, referring to [this page](https://github.com/open-mmlab/mmdetection). Details as follows:

   * Install mmcv-full, we recommend you to install the pre-build package as below.

     ```
     pip install mmcv-full==1.2.7 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html
     ```

     **Note:**  The pre-build package link is based on the version of pytorch and cudatoolkit required above.

   * Clone the MMDetection repository.

     ```
     git clone https://github.com/open-mmlab/mmdetection.git
     cd mmdetection
     ```

   * Install build requirements and then install MMDetection.

     ```
     pip install -r requirements/build.txt
     pip install -v -e .  # or "python setup.py develop"
     ```

## Using of this repository

Our local training and testing OS  information as:

```
LSB Version:    :core-4.1-amd64:core-4.1-noarch
Distributor ID: CentOS
Description:    CentOS Linux release 7.7.1908 (Core)
Release:        7.7.1908
Codename:       Core
```

You may use others Linux OS.

1. Put this repository as:

   ```
   |--panda_project
   |--mmdetection
   ```

   **Note:** The panda_project can be **git clone from github** in this [link](https://github.com/scofiedluo/panda_competetion.git), or unzip from our submitting compressed file.

2. Install the dependencies:

   ```
   cd panda_project
   pip install -r requirements.txt
   ```

3. **Prepare the training and testing datasets as :**

   ```
   panda_project
   	|-tcdata
   		|--panda_round1_test_202104_A   
   		|--panda_round1_test_202104_B 
                   |--panda_round1_test_A_annos_202104
                   |--panda_round1_test_B_annos_20210222
                   |--panda_round1_train_202104
                   |--panda_round1_train_annos_202104
   	...
   ```

   **Note:** 

   * All data should download from this [link](https://tianchi.aliyun.com/competition/entrance/531855/information).
   * **All the training data should put in one folder, that is you should merge the `panda_round1_train_202104_part1` and `panda_round1_train_202104_part2` in `panda_round1_train_202104`.**
   * If you just want to test `panda_round1_test_202104_B`, you can just prepare `panda_round1_test_202104_B` and `panda_round1_test_B_annos_20210222`, we have save the pretrained checkpoints in `./panda_project/user_data/model_data/cascade_rcnn_r101_fpn_1x_coco_round1_panda/epoch_120.pth`

4. Downloaded the pretrained model of **Cascade R-CNN R101_FPN** in this [link](http://download.openmmlab.com/mmdetection/v2.0/dcn/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200203-3b2f0594.pth), and put it in `./panda_project/code/mmdetection/checkpoints/`, or use the following url.

```
http://download.openmmlab.com/mmdetection/v2.0/dcn/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200203-3b2f0594.pth
```
   
5. **Training:**

   Suppose you are in folder `panda_project`,  then run the following to start training:

   ```
   cd code
   sh train.sh
   ```

   **Note:** 

   * We use **4 Tesla V100 GPU** for training in our local environment.
   * If you do not have enough GPU memory, you may try to resize the cropped image to small image, such as $1000\times1000$, or change the hyperparameter `ratios` with less anchor, e.g,  $[0.5, 1.0, 2.0]$.
   * You can choose which epoch model you want to save by modify the source code in `./code/mmdetection/data_process/copy_checkpoints.py`, e.g, `epoch_120.pth` , `epoch_100.pth` .

6. **Testing:**

   Suppose you are in folder `panda_project`,  then run the following to start testing:

   ```
   cd code
   sh run.sh
   ```

   **Note:** 

   * We use **1 Tesla V100 GPU** for testing in our local environment.
   * **The command above will save the testing result of  `panda_round1_test_202104_B` in `./panda_project/prediction_result/` as `det_results.json`. And the `det_results.json` is what we submit to TIANCI system.**
   * We suppose you use `epoch_120.pth` for testing. You can use different trained model for testing by modify the shell scripts `run.sh`.
   * We have test the above training and testing , you can find the log in `./panda_project/code/slurm/`.

# Acknowledgements

This repo is based on 

* [MMDetection](https://github.com/open-mmlab/mmdetection)

Thanks for their great work.

# Contact

If you have any question to reproduce the result `det_results.json`, please email 

```
luotongyi@sjtu.edu.cn
```
