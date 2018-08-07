import os,sys
path = "F:\liuyang\\U-net数据"
os.chdir(path)
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf
from skimage.io import imshow
from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian, softmax_to_unary,unary_from_softmax

from utils.datapacking import unpackImage
import skimage.io as io

print('loading data...')
lables=np.load('train_result.npz')
lables = lables['arr_0.npy']
one=np.ones((512,512))
print('processing...')
lables=lables.reshape(100,512,512)

lable2=np.array([],dtype=np.float32)
for j in range(100):
    a=lables[j]
    b=one-a
    lable1=np.array([a,b])
    lable2=np.append(lable2,lable1)
_lable=lable2.reshape(100,2,512, 512)

result = np.array([], dtype=int)
for i in range(100):

    tensor = _lable[i]
   # print(tensor)
    #lable = -np.log(tensor)

    lable = tensor
    unary = unary_from_softmax(lable)
    d = dcrf.DenseCRF(512* 512, 2)
    d.setUnaryEnergy(unary)
    feats = create_pairwise_gaussian(sdims=(-1, -1), shape=(512,512))
    d.addPairwiseEnergy(feats, compat=2.5,
                        kernel=dcrf.FULL_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(2)
    res = np.argmax(Q, axis=0).reshape((512, 512))

    # res = np.where(res==1, 0, 1)
    result = np.append(result, res)

print('saving...')
#result = result.astype(np.uint8)
unet_crf = result.reshape(100, 512, 512)

unet_crf = unet_crf^1
unet_crf*=255
np.save('unet_crf.npy',unet_crf)
unpackImage('F:\liuyang\\U-net数据\\unet_crf.npy', imageSize=(512,512), channel=1)
print('done')
