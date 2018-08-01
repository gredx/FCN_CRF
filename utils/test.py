'''
使用得到的权值和网络进行图像分割并保存
'''

import os
from Unet import Unet,dice_coef_loss,dice_coef
import matplotlib.pyplot as plt
from skimage.io import imsave
from keras.optimizers import Adam
import numpy as np
from utils.get_data import get_data
from utils.datapacking import get_randstr

def predict():
    # get model
    model = Unet(512,512,optimizer=Adam(1e-4),loss=dice_coef_loss,metrics=dice_coef)
    # get test images
    print("getting test images...")
    testImagesPath = 'E:\ljs\\u-net\\test_data.npy'
    imgs_test = get_data(testImagesPath,(512,512))
    # load weighs
    print("loading weighs...")
    model.load_weights('weighs.h5')
    # predict
    print("predicting ...")
    imgs_predict = model.predict(imgs_test,batch_size=1,verbose=1)

    # 把输出结果保存到一个文件夹里
    print('saving output images...')
    dir = get_randstr()
    os.mkdir(dir)

    for id,image in enumerate(imgs_predict):
        image = (image[:, :, 0]*255.).astype(np.uint8)
        # plt 得到的结果默认是彩色存储
        #plt.imsave(os.path.join(dir,str(id)+'.png'),image)
        # skimage存灰度
        imsave(os.path.join(dir,str(id)+'.png'),image)

if __name__ == '__main__':
    predict()