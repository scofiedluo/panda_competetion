# -*- coding: utf-8 -*-
# @Time : 2021-03-11 16:30
# @Author : sloan
# @Email : 630298149@qq.com
# @File : sloan_utils.py
# @Software: PyCharm
from concurrent.futures import ProcessPoolExecutor
from threading import Lock
import os
import traceback
import shutil
import glob
import time
import os.path as osp


def mark_succeeded(img_path):
    img_dir, img_name = osp.split(img_path)
    mark_dir = osp.join("succeeded")
    succeed_img = osp.join(mark_dir, img_name)
    os.makedirs(mark_dir, exist_ok=True)
    processing_img = osp.join("processing", img_name)
    shutil.move(processing_img, succeed_img)


def mark_failed(img_path):
    img_dir, img_name = osp.split(img_path)
    mark_dir = osp.join("failed")
    os.makedirs(mark_dir, exist_ok=True)
    failed_img = osp.join(mark_dir, img_name)
    processing_img = osp.join("processing", img_name)
    shutil.move(processing_img, failed_img)


def mark_processing(img_path):
    img_dir, img_name = osp.split(img_path)
    mark_dir = osp.join("processing")
    os.makedirs(mark_dir, exist_ok=True)
    processing_img = osp.join(mark_dir, img_name)
    open(processing_img, "w").close()


class SampleGeneratorBase(object):
    '''
    多进程基类，针对图片，视频的密集计算
    '''

    _task_lock = Lock()
    def __init__(self):
        pass

    def needed_to_process(self,local_path):
        """并发处理时，获取未处理的图片/视频
        """
        with self._task_lock:
            img_dir, img_name = osp.split(local_path)
            check_1 = osp.join("succeeded", img_name)
            check_2 = osp.join("processing", img_name)
            check_3 = osp.join("failed", img_name)
            if not osp.exists(check_1) and not osp.exists(check_2) and not osp.exists(check_3):
                mark_processing(local_path)
                return True
            return False

    def process_img(self,*args):
        pass

    def _process_img(self,*args):
        """单图片/视频标记
        arg[0]:为图片或视频路径
        """
        try:
            rt = self.process_img(*args)
        except:
            mark_failed(args[0])
            traceback.print_exc()
            return None
        else:
            print("[succeeded]: {} [left]: {}".format(args[0],len(glob.glob('processing'+'/*.jpg'))-1))
            self.threading_num += 1
            mark_succeeded(args[0])
            return rt

    def batch_sample(self,*args):
        '''
        # 来试试多进程吧！！！
        # 多进程访问图片文件夹目录
        img_dir,ext = args
        img_path_list = glob.glob(img_dir + "/*{}".format(ext))
        s1 = time.time()
        print(img_path_list)
        results = []
        task_pool = ProcessPoolExecutor(max_workers=4)
        for img_path in img_path_list[:100]:
            if self.needed_to_process(img_path):
                rt = task_pool.submit(self._process_img, img_path,ext)
                results.append(rt)
        results = [rt.result() for rt in results if rt]
        print("finished")
        print("cost time:",time.time()-s1)
        print(len(results),results)
        '''

        pass

# Example Template
class child(SampleGeneratorBase):
    def __init__(self,workers=4):
        super(SampleGeneratorBase,self).__init__()
        self.workers = workers
        self.threading_num = 0

    # 重新设计构造函数process_img
    def process_img(self,*args):
        print("child:",args)
        return self.threading_num

    # 重新设计函数batch_sample
    def batch_sample(self,*args):
        img_dir, ext = args
        img_path_list = glob.glob(img_dir + "/*{}".format(ext))
        s1 = time.time()
        results = []
        task_pool = ProcessPoolExecutor(max_workers=self.workers)
        for img_path in img_path_list[:100]:
            if self.needed_to_process(img_path):
                rt = task_pool.submit(self._process_img, img_path, ext)
                results.append(rt)
        results = [rt.result() for rt in results if rt] # rt.result()为process_img成功返回内容
        print("finished")
        print("cost time:", time.time() - s1)
        print(len(results), results)

if __name__=="__main__":
    # sample_generator = SampleGeneratorBase()
    # sample_generator.batch_sample('./train_imgs','jpg')
    child_sample_generator = child(workers=8)
    child_sample_generator.batch_sample('./train_imgs','jpg')