import os
import os.path as osp

'''
本文档适用于单标签的2分类、多分类，绝对路径imglist生成
需要配置：
--prefix 数据root路径
--CLASS  类别名称和id的映射
--exts   文件后缀

文件结构：
--dataset
    --train
        class1
            *.jpg
            */*.jpg
            ...
        class2
    --test
    train.txt  待生成
    test.txt   待生成
'''


prefix = '/data/xuhua/data/kitchen/smoking-calling/smoke-hard-head/'  #注意斜杠结尾
CLASSES={
    'pos':'0',
    'neg':'1'
}
exts=['jpg','jpeg','bmp','png']

def get_all_imgs(classid,imgdir):
    clsimglist = []
    for root,dirs_name,files_name in os.walk(imgdir):
        for i in files_name:
            if i.split('.')[-1] not in exts:
                continue
            file_name = os.path.join(root, i)
            clsimglist.append([file_name,classid])
    return clsimglist

def write2txt(prefix,mode,modelist):
    with open(prefix+mode+'.txt','w') as f:
        for line in modelist:
            f.writelines(line[0]+' '+line[1]+'\n')

for mode in os.listdir(prefix):
    if mode=='train':
        datadir = osp.join(prefix,mode)
        modelist=[]
        for classes in os.listdir(datadir):
            classid = CLASSES[classes]
            clsimglist = get_all_imgs(classid,osp.join(prefix,mode,classes))
            if len(modelist)>0:
                modelist.extend(clsimglist)
            else:
                modelist=clsimglist
        write2txt(prefix,mode,modelist)
    if mode=='test':
        datadir = osp.join(prefix,mode)
        modelist=[]
        for classes in os.listdir(datadir):
            classid = CLASSES[classes]
            clsimglist = get_all_imgs(classid,osp.join(prefix,mode,classes))
            if len(modelist)>0:
                modelist.extend(clsimglist)
            else:
                modelist=clsimglist
        write2txt(prefix,mode,modelist)
