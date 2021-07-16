import os.path as osp
import xml.etree.ElementTree as ET

import mmcv
import numpy as np

from .builder import DATASETS
# from .multi_label import MultiLabelDataset
from .base_dataset import BaseDataset


# @DATASETS.register_module()
# class HJQ(MultiLabelDataset):
#     CLASSES = ('indoor','outdoor','surveil')

#     def load_annotations(self):
#         """Load annotations

#         """
#         data_infos = []
#         lines = mmcv.list_from_file(self.ann_file,self.data_prefix)
#         for line in lines:
#             line=line.split(' ')
#             filename = line[0]

#             gt_label = np.zeros(len(self.CLASSES))
#             for i in range(len(self.CLASSES)):
#                 if int(line[i+1]) == 1:
#                     gt_label[i]=1

#             info = dict(
#                 img_prefix=self.data_prefix,
#                 img_info=dict(filename=filename),
#                 gt_label=gt_label.astype(np.int8))
#             data_infos.append(info)

#         return data_infos


@DATASETS.register_module()
class HJQ(BaseDataset):
    CLASSES = ['indoor','outdoor']

    def load_annotations(self):
        """Load annotations

        """
        data_infos = []
        lines = mmcv.list_from_file(self.ann_file,self.data_prefix)
        for line in lines:
            line=line.split(' ')
            filename = line[0]
            # gt_label = np.zeros(len(self.CLASSES))
            # for i in range(len(self.CLASSES)):
            #     if int(line[i+1]) == 1:
            #         gt_label[i]=1
            #     else:
            #         gt_label[i]=0
            gt_label=np.array(int(line[2]))

            info = dict(
                img_prefix=self.data_prefix,
                img_info=dict(filename=filename),
                gt_label=gt_label.astype(np.int64))
            data_infos.append(info)

        return data_infos





