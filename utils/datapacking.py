# 把数据打包
from PIL import Image
from utils import  searchFile
import matplotlib.pyplot as plt
from skimage.io import imsave
import numpy as np
import os,random
def get_randstr(len=20):
    bas_str = 'ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789'
    tar_str=''
    strlen = bas_str.__len__()
    for i in range(len):
        tar_str +=bas_str[random.randint(0,strlen-1)]
    return tar_str

def packImage(path,format='jpg',saved_name=None,size=(0,0,512,512)):
    '''把指定目录下的图片打包成npz文件

    会把目录下文件和文件名一同打包
    r = np.load(file)   \n
    r[arr_0.npy] # 图片的包
    r[arr_1.npy] # 文件名的包


    :param path: 要打包图片的路径
    :param format: 图片的格式，默认是'bmp'
    :param saved_name: 输出文件的名字，缺省是一个长度为20的随机字符串
    :param size: a tuple.对图片裁剪的参数，(left,upper,right,lower),这个参数绝对不能超出图片的大小，否则出现意料之外的错误
    :return: 无返回值

    可能的异常结果:
        输出图片不正常：可能是参数size出错，查看是否超出了图片的大小
    '''
    lis = searchFile.search_spe_file(path,format)
    packfile =[]
    filesname =[]
    if saved_name is None:
        saved_name = get_randstr()
    i=0
    for filepath in lis:
        i=i+1
        if i>100:
            break
        filename = filepath.split('\\')[-1]
        filesname.append(filename)

        image = Image.open(filepath)
        image = image.crop(size)    # PIL 中的Image.crop参数是(left,upper,right,lower) 列，行
                                    # matplotplib.pyplot as plt 中的 plt.imread对象裁剪image[:800,:1000,:] 行，列
        image = np.array(image)
        packfile.append(image)

    np.savez(saved_name,packfile,filesname)

def unpackImage(file, imageSize, channel, output_dir=None):
    '''把npz图片还原为图片

    默认npz文件中arr_1是文件名，arr_0是图片，输入npz文件的路径，图片的尺寸，要输出的文件夹

    :param file: npz文件的路径
    :param imageSize: a tuple图片的尺寸 (hight,weight)
    :param channel: 图像的通道数，彩色3，灰度1
    :param output_dir: 输出文件夹的路径，若为空则创建一个随机名字的文件夹
    :return: 无返回值
    '''
    if output_dir is None:
        output_dir = get_randstr(20)
        os.mkdir(output_dir)
    os.chdir(output_dir)

    r = np.load(file)
    images = r


    filesname=[]
    if file.split('.')[-1] == 'npz':
        images = r['arr_0.npy']
        filesname = r['arr_1.npy']

    else :
        att1 = images.size // imageSize[0] // imageSize[1] // channel
        for i in range(att1):
            filesname.append(str(i)+'.jpg')
    att1 = images.size // imageSize[0] // imageSize[1] // channel
    if channel>1:
        images = np.reshape(images, (att1, imageSize[0], imageSize[1], channel))
    else :
        images = images.astype(np.uint8)
        images = images.reshape((att1, imageSize[0], imageSize[1]))
    for image,name in zip(images,filesname):
        imsave(name,image)


if __name__ == '__main__':
    path = 'F:\liuyang\\U-net数据\\train_result'
    saved_name = 'train_result'
    #packImage(path,format='png',size=(0,0,512,512),saved_name=saved_name)
    unpackImage('F:\liuyang\\U-net数据\\train_result.npz', imageSize=(512,512), channel=1)
