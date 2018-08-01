'''
包含读取train_data.npz，train_lable.npz,test_data.npz,weigh.h5文件的函数
'''

import numpy as np
import os

def get_data(file_path,img_size,channel=1):
    '''
    目的是读取指定文件，reshape、求mean，std，对读取的数据稍微处理，返回值为input_image

    :param file_path: 文件路径
    :param img_size: 图片尺寸，height,weight
    :return: input_image
    :channel: 通道数
    '''
    input = np.load(file_path)
    att1 = input.size // img_size[0] //img_size[1] // channel
    input = input.reshape((att1,img_size[0],img_size[1],channel))

    input = input.astype('float32')
    input/=255
    #mean = np.mean(input)
    #std = np.std(input)

    #input -=mean
    #input/=std

    return input
