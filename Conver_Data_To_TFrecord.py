'''
Created on 2017-4-11
# data: Parking

@author: XuTing
'''

import tensorflow as tf
import numpy as np
import os
from PIL import Image
import math


#%%

def get_file(file_dir, ratio):
    '''Get full image directory and corresponding labels
    Args:
        file_dir: file directory
    Returns:
        images: image directories, list, string   只是list不是数据矩阵
        labels: label, list, int
    '''

    images = []
    temp = []
    for root, sub_folders, files in os.walk(file_dir):
        # image directories
        for name in files:
            images.append(os.path.join(root, name))
        # get sub-folder names
        for name in sub_folders:
            temp.append(os.path.join(root, name))
            
    # assign labels based on the folder names
    print('子文件数',len(temp))
    print('总原始样本',len(images))
    labels = []        
    for one_folder in temp:        
        n_img = len(os.listdir(one_folder))
        letter = one_folder.split('/')[-1]
            
        if letter=='0':
            labels = np.append(labels, n_img*[0])#赋值为0标签
            print('Reading 0......')
        else:
            labels = np.append(labels, n_img*[1])
            print('Reading 1......')
    
    # shuffle
    temp = np.array([images, labels])
    temp = temp.transpose()#column?
    np.random.shuffle(temp)
    
    all_image_list = list(temp[:, 0])#只是list不是数据矩阵
    all_label_list = list(temp[:, 1])
    
    n_sample = len(all_image_list)
    n_val = math.ceil(n_sample*ratio) # number of validation samples=5000*0.2=1000
    n_train = n_sample - n_val # number of trainning samples=4000
    
    tra_imageslist = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    
    val_imageslist = all_image_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]
    
             
    return tra_imageslist, tra_labels,val_imageslist,val_labels


#%%

def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
def convert_to_tfrecord(images, labels, save_dir, name):
    '''convert all images and labels to one tfrecord file.
    Args:
        images: list of image directories, string type
        labels: list of labels, int type
        save_dir: the directory to save tfrecord file, e.g.: '/home/folder1/'
        name: the name of tfrecord file, string type, e.g.: 'train'
    Return:
        no return
    Note:
        converting needs some time, be patient...
    '''
        #方法2
    #             image = io.imread(images[i]) # type(image) must be array!
    #             image_raw = image.tostring()
    
    filename = os.path.join(save_dir, name + '.tfrecords')
    n_samples = len(labels)
    num = 0  
    if np.shape(images)[0] != n_samples:
        raise ValueError('Images size %d does not match label size %d.' %(images.shape[0], n_samples))
    # wait some time here, transforming need some time based on the size of your data.
    writer = tf.python_io.TFRecordWriter(filename)
    print('\nTransform start......')
    for i in np.arange(0, n_samples):
        try:            
#方法1
            img=Image.open(images[i])
            if (min(img.size)<min(H,W)) or (os.path.getsize(images[i]) >=479000):
                print('~~error，Ruined Images!: %s'%(images[i]))
            else:
                num += 1
                img=img.resize((W,H))#resize (W,H) = 200×150
                image_raw=img.tobytes()#将图片转化为二进制格式  
                label = int(labels[i])
                example = tf.train.Example(features=tf.train.Features(feature={
                                'label':int64_feature(label),
                                'image_raw': bytes_feature(image_raw)}))
                writer.write(example.SerializeToString())
        except IOError as e:
            print('Could not read:', images[i])
            print('error: %s' %e)
            print('Skip it!\n')
    writer.close()
    print('Transform done!')
    print('总可用样本：',num)
    
  
#%% Convert data to TFRecord
train_InitData_dir ='D://AutoSparePart//Reshape//ToFinall_Data//'
save_dir = 'D:\\AutoSparePart\\Train_Test_TF\\'


#Convert test data: you just need to run it ONCE !
W = 200
H = 150
tra_imageslist,tra_labels,val_imageslist,val_labels = get_file(train_InitData_dir,ratio=0.2)
convert_to_tfrecord(tra_imageslist, tra_labels, save_dir, 'train')
convert_to_tfrecord(val_imageslist, val_labels, save_dir, 'val')
