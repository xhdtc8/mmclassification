import os.path as osp
import xml.etree.ElementTree as ET

import mmcv
import numpy as np

from .builder import DATASETS
# from .multi_label import MultiLabelDataset
from .base_dataset import BaseDataset
from .multi_label import MultiLabelDataset
from .imagenet import ImageNet



@DATASETS.register_module()
class CUSTOMCLS(BaseDataset):
    # CLASSES = ['nonanfang','anfang']
    CLASSES = ['female','male']
    def load_annotations(self):
        """Load annotations
            其实和imagenet基本一样，就是有个annfile自己输入了哈哈
        """
        data_infos = []
        lines = mmcv.list_from_file(self.ann_file)
        for line in lines:
            line=line.split(' ')
            filename = line[0] +'.png' # VOC没有后缀，需要补充
            gt_label=np.array(int(line[1])) if int(line[1])>0 else np.array(0)

            info = dict(
                img_prefix=self.data_prefix, #绝对路径不需要
                img_info=dict(filename=filename),
                gt_label=gt_label.astype(np.int64))
            data_infos.append(info)

        return data_infos

@DATASETS.register_module()
class CUSTOMMULTILABEL(MultiLabelDataset):
    # CLASSES = ['female','old','young']
    CLASSES = ['female']

    def load_annotations(self):
        """Load annotations
            其实和imagenet基本一样，就是有个annfile自己输入了哈哈
        """
        data_infos = []
        lines = mmcv.list_from_file(self.ann_file)
        for line in lines:
            line=line.split(' ')
            filename = line[0]
            gt_label = np.zeros(len(self.CLASSES),dtype=np.int64)
            for i in range(len(self.CLASSES)):
                gt_label[i]=int(line[i+1])
            info = dict(
                img_prefix=self.data_prefix, 
                img_info=dict(filename=filename),
                gt_label=gt_label.astype(np.int64))
            data_infos.append(info)

        return data_infos
    
# @DATASETS.register_module()
# class HandDataset(ImageNet):
#     CLASSES = ['pos','neg']

# @DATASETS.register_module()
# class SmokeDataset(ImageNet):
#     CLASSES = ['smoking','normal']

@DATASETS.register_module()
class TwowayDataset(ImageNet):
    CLASSES = ['pos','neg']





